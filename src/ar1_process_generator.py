"""
ar1_process_generator.py
========================
Generates first-order autoregressive AR(1) stochastic processes for use as
time-varying model parameters (NH and φ) in the X-ray count simulator.

Three functions are provided, in order of increasing abstraction:

``ar_one``
    Generate an ensemble of zero-mean AR(1) realisations with a fixed
    autocorrelation coefficient α and Gaussian noise.

``stochastic_based_nh_phi_simulator``
    Wrapper around ``ar_one`` that applies a linear scaling (φ) and shift
    to produce NH or φ time series with physical units.

``stochastic_based_nh_phi_simulator_v2``
    **Primary function used by the pipeline.**  Generates a single mean-reverting
    AR(1) process::

        x(t) = α · x(t−1) + (1 − α) · μ + ε(t),   ε ~ N(0, σ)

    where α is the autocorrelation coefficient, μ is the long-run mean
    (``shift_term``), and σ is the innovation standard deviation.  The
    process reverts toward μ at a rate governed by (1 − α).

Notes
-----
All log-space parameters (NH, φ) are generated on a logarithmic scale so
that the physical quantities are lognormally distributed; exponentiation
(``10**process``) is applied in the calling code.
"""

import numpy as np


def ar_one(
    sigma: float,
    mean: float = 0.0,
    alpha_1: float = 0.9,
    length: int = 500,
    simul_number: int = 100,
) -> tuple:
    """Generate an ensemble of zero-mean AR(1) time series.

    Each realisation is initialised with a draw from N(0, 1) and evolved as::

        x(t) = α · x(t−1) + ε(t),   ε ~ N(mean, sigma)

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian innovation noise ε(t).
    mean : float, optional
        Mean of the Gaussian innovation noise (default: 0.0).
    alpha_1 : float, optional
        Autocorrelation coefficient α, controlling the persistence of the
        process (default: 0.9).  Must satisfy |α| < 1 for stationarity.
    length : int, optional
        Number of time steps per realisation (default: 500).
    simul_number : int, optional
        Number of independent realisations to generate (default: 100).

    Returns
    -------
    time_steps : np.ndarray, shape (length,)
        Integer time indices 0, 1, …, length−1.
    realisations : np.ndarray, shape (simul_number, length)
        Array of AR(1) time series, one row per realisation.

    Examples
    --------
    >>> t, x = ar_one(sigma=0.1, alpha_1=0.9, length=200, simul_number=10)
    >>> x.shape
    (10, 200)
    """
    realisations = []

    for _ in range(simul_number):
        x_prev  = np.random.normal(0, 1)
        ar_list = []

        for _ in range(length):
            z_t    = np.random.normal(mean, sigma)
            x_curr = alpha_1 * x_prev + z_t
            x_prev = x_curr
            ar_list.append(x_curr)

        realisations.append(ar_list)

    return np.arange(length), np.array(realisations)


def stochastic_based_nh_phi_simulator(
    error_mean: float,
    error_var: float,
    phi: float,
    shift: float,
    t_length: int,
) -> tuple:
    """Simulate an ensemble of NH or φ processes via AR(1) with linear transform.

    Generates 100 AR(1) realisations using ``ar_one`` and then applies the
    affine transform::

        output(t) = phi · x(t) + shift

    to convert the zero-mean AR(1) into a physically meaningful parameter.

    Parameters
    ----------
    error_mean : float
        Mean of the Gaussian innovation noise passed to ``ar_one``.
    error_var : float
        Standard deviation of the Gaussian innovation noise passed to ``ar_one``.
    phi : float
        Scaling factor applied to the AR(1) output.
    shift : float
        Additive shift applied after scaling (sets the long-run mean in
        log space, e.g. ``log10(NH_mean)``).
    t_length : int
        Number of time steps per realisation.

    Returns
    -------
    time_steps : np.ndarray, shape (t_length,)
        Integer time indices 0, 1, …, t_length−1.
    nh_phi_output : np.ndarray, shape (100, t_length)
        Array of scaled and shifted AR(1) processes.

    See Also
    --------
    stochastic_based_nh_phi_simulator_v2 : preferred function for the pipeline.
    """
    time_steps, ar1 = ar_one(sigma=error_var, mean=error_mean, length=t_length)
    nh_phi_output   = phi * ar1 + shift
    return time_steps, nh_phi_output


def stochastic_based_nh_phi_simulator_v2(
    scale_factor: float,
    shift_term: float,
    error_sigma: float,
    process_length: int,
    reproducibility: bool = True,
) -> tuple:
    """Generate a single mean-reverting AR(1) process (primary pipeline function).

    Evolves a scalar state variable according to the mean-reverting AR(1)
    recursion::

        x(t) = α · x(t−1) + (1 − α) · μ + ε(t),   ε ~ N(0, σ)

    where α = ``scale_factor``, μ = ``shift_term``, and σ = ``error_sigma``.
    The process is initialised at x(0) = μ and reverts toward μ over time at
    a rate controlled by (1 − α).

    This function is the primary AR(1) generator used by the synthetic data
    pipeline (``generate_synthetic_data.py``) to produce time series of NH and
    φ in log space.

    Parameters
    ----------
    scale_factor : float
        Autocorrelation coefficient α, controlling persistence.  Typical
        range: [0.5, 0.99].  Must satisfy |α| < 1 for stationarity.
    shift_term : float
        Long-run mean μ of the process (in log space, e.g.
        ``log10(NH_mean)``).  Also used as the initial value x(0).
    error_sigma : float
        Standard deviation σ of the Gaussian innovation noise ε(t).
    process_length : int
        Number of time steps T to simulate.
    reproducibility : bool, optional
        If ``True`` (default), the innovation noise is drawn with a fixed
        random seed (42) so that results are reproducible across runs.
        Set to ``False`` for fully stochastic realisations.

    Returns
    -------
    time_steps : np.ndarray, shape (T,)
        Integer time indices 0, 1, …, T−1.
    process : np.ndarray, shape (T,)
        AR(1) time series in log space.  Exponentiate (``10**process``) to
        recover NH or φ in physical units.

    Examples
    --------
    >>> t, x = stochastic_based_nh_phi_simulator_v2(
    ...     scale_factor=0.95, shift_term=21.5, error_sigma=0.1,
    ...     process_length=100, reproducibility=True,
    ... )
    >>> x.shape
    (100,)
    >>> import numpy as np
    >>> np.isclose(x.mean(), 21.5, atol=1.0)  # mean-reverting toward shift_term
    True
    """
    previous_step = shift_term
    process_list  = []

    if reproducibility:
        rng = np.random.default_rng(42)
        for _ in range(process_length):
            current_step  = (scale_factor * previous_step
                             + (1 - scale_factor) * shift_term
                             + rng.normal(0.0, error_sigma))
            previous_step = current_step
            process_list.append(current_step)
    else:
        for _ in range(process_length):
            current_step  = (scale_factor * previous_step
                             + (1 - scale_factor) * shift_term
                             + np.random.normal(0.0, error_sigma))
            previous_step = current_step
            process_list.append(current_step)

    return np.arange(process_length), np.array(process_list)
