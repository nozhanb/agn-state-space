"""
count_simulator.py
==================
Simulates X-ray photon count spectra from a physical source model.

The simulation pipeline follows Equations 1 and 3 in Buchner & Boorman (2023):

1. **Intrinsic flux** — combine three XSPEC spectral components::

       F(E) = ZPOWERLW(E) × ZWABS(E) × WABS(E)

   where ZPOWERLW models the redshifted power-law continuum, ZWABS applies
   intrinsic absorption at the source redshift, and WABS accounts for
   Galactic foreground absorption.

2. **Instrument response** — fold the flux through the ARF and RMF matrices::

       λᵢ = Σⱼ F(Eⱼ) · Δt · ΔEⱼ · ARFⱼ · RMFⱼᵢ

3. **Poisson sampling** — draw integer counts from the predicted rate::

       Cᵢ ~ Poisson(λᵢ)

References
----------
Buchner, J. & Boorman, P. (2023), arXiv:2309.05705.
XSPEC model definitions:
    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node220.html
"""

import numpy as np
import jax.numpy as jnp
import xraystan
import astropy.io.fits as fits


def flux_multi_comp(
    gamma: float,
    scaling_factor: np.ndarray,
    hydrogen_column_density: np.ndarray,
    galactic_hydrogen_column_density: float,
    photo_electric_sigma: np.ndarray,
    photo_electric_sigma_redshift: np.ndarray,
    redshift: float,
    energy: np.ndarray,
) -> jnp.ndarray:
    """Compute the X-ray source flux via ZPOWERLW × ZWABS × WABS.

    Combines three XSPEC spectral components to produce a time-resolved flux
    array.  All three components are evaluated simultaneously across the full
    instrument energy grid using JAX array operations.

    Model components
    ----------------
    - **ZPOWERLW**: redshifted power-law photon continuum.
    - **ZWABS**: intrinsic absorption by the source hydrogen column.
    - **WABS**: Galactic foreground absorption (constant in time).

    Parameters
    ----------
    gamma : float
        Photon index Γ of the power-law continuum.
    scaling_factor : np.ndarray, shape (T,)
        Power-law normalisation at each time step.
    hydrogen_column_density : np.ndarray, shape (T,)
        Intrinsic NH at each time step, in units of 10²² atoms cm⁻².
    galactic_hydrogen_column_density : float
        Galactic foreground NH toward the source, in units of 10²² atoms cm⁻².
    photo_electric_sigma : np.ndarray, shape (N_full,)
        Photoelectric cross-sections at rest-frame channel energies
        (cm² per H atom).
    photo_electric_sigma_redshift : np.ndarray, shape (N_full,)
        Photoelectric cross-sections at redshifted energies (cm² per H atom).
    redshift : float
        Cosmological redshift z of the source.
    energy : np.ndarray, shape (N_full,)
        Mid-point energies of the instrument energy channels (keV).

    Returns
    -------
    out_flux : jnp.ndarray, shape (T, N_full)
        Source flux in each energy channel at each time step
        (photons cm⁻² s⁻¹ keV⁻¹).

    Notes
    -----
    The three model components evaluate as::

        ZPOWERLW = sf · [E · (1 + z)]^{−Γ}          (T, N_full)
        ZWABS    = exp(−NH · σ_z)                    (T, N_full)
        WABS     = exp(−NH_gal · 10²² · σ)           (N_full,)

    where σ and σ_z are the pre-computed photoelectric cross-sections at
    rest-frame and redshifted energies respectively.  ``galactic_hydrogen_column_density``
    is multiplied by 10²² to convert from units of 10²² atoms cm⁻² to
    atoms cm⁻², matching the WABS model convention.

    References
    ----------
    XSPEC manual: https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node220.html
    """
    sf = scaling_factor[:, None]               # (T, 1)
    nh = hydrogen_column_density[:, None]      # (T, 1)

    E = energy[None, :] * (1.0 + redshift)    # (1, N_full)

    zpowerlw = sf * E ** -gamma                                                       # (T, N_full)
    zwabs    = jnp.exp(-nh * photo_electric_sigma_redshift[None, :])                 # (T, N_full)
    wabs     = jnp.exp(-(galactic_hydrogen_column_density * 1e22) *
                       photo_electric_sigma)                                          # (N_full,)

    out_flux = zpowerlw * zwabs * wabs[None, :]                                      # (T, N_full)
    return out_flux


def count_generator(
    pi_filename: str,
    rmf_filename: str,
    photo_sigma: np.ndarray,
    photo_sigma_red: np.ndarray,
    low_energy_limit: float = 0.3,
    high_energy_limit: float = 8.0,
    gamma_index: float = 2.0,
    nh_density: np.ndarray = None,
    scaling_val: np.ndarray = None,
    galactic_nh: float = 0.01,
    redshift: float = 0.0,
    simulation_count: int = 1,
    reproducibility: bool = True,
) -> tuple:
    """Generate synthetic X-ray count spectra from a physical source model.

    Implements the simulation pipeline described in Buchner & Boorman (2023):

    1. Load the observed PHA/RMF/ARF instrument files for the target source.
    2. Evaluate the ZPOWERLW × ZWABS × WABS flux model over the full
       instrument energy grid (``N_full`` channels).
    3. Fold the flux through the ARF and RMF response matrices to obtain
       the predicted count expectation λᵢ per detector channel.
    4. Draw Poisson-distributed integer counts Cᵢ ~ Poisson(λᵢ) for each
       requested simulation realisation.

    Parameters
    ----------
    pi_filename : str
        Path to the source PHA file (CIAO/SHERPA output, ".pi" extension).
    rmf_filename : str
        Path to the redistribution matrix file (CIAO output, ".rmf" extension).
    photo_sigma : np.ndarray, shape (N_full,)
        Pre-computed photoelectric cross-sections at rest-frame energies
        (cm² per H atom); covers the full instrument energy grid.
    photo_sigma_red : np.ndarray, shape (N_full,)
        Pre-computed photoelectric cross-sections at redshifted energies
        (cm² per H atom).
    low_energy_limit : float, optional
        Lower energy cut-off in keV (default: 0.3).
    high_energy_limit : float, optional
        Upper energy cut-off in keV (default: 8.0).
    gamma_index : float, optional
        Photon index Γ of the power-law continuum (default: 2.0).
    nh_density : np.ndarray, shape (T,)
        Time series of intrinsic NH values in units of 10²² atoms cm⁻²,
        passed to the ZWABS component.  Must be a 1-D array of length T.
    scaling_val : np.ndarray, shape (T,)
        Time series of power-law normalisations passed to ZPOWERLW.
        Must be a 1-D array of length T.
    galactic_nh : float, optional
        Galactic foreground NH toward the source in units of 10²² atoms cm⁻²
        (default: 0.01, appropriate for SDSSJ0932+0405).
    redshift : float, optional
        Cosmological redshift z of the source (default: 0.0).
    simulation_count : int, optional
        Number of independent Poisson realisations to draw (default: 1).
        Each realisation represents a distinct synthetic observation of the source.
    reproducibility : bool, optional
        If ``True`` (default), Poisson sampling uses a fixed random seed (42)
        so that results are reproducible across runs.  Set to ``False`` for
        fully stochastic realisations.

    Returns
    -------
    pred_count : np.ndarray, shape (simulation_count, T, N_chan)
        Integer Poisson counts per simulation realisation, time step, and
        energy channel.
    count_rate_list : np.ndarray, shape (1, simulation_count, T, N_chan)
        Predicted count rate in count s⁻¹ keV⁻¹ for each realisation and
        time step.
    observed_rate : np.ndarray, shape (N_chan,)
        Observed count rate from the input PHA file, masked to the energy
        range [``low_energy_limit``, ``high_energy_limit``].
    chan_e : np.ndarray, shape (N_chan,)
        Mid-point energies of the detector channels after energy filtering
        (keV).

    Notes
    -----
    ``N_full`` is the total number of instrument energy channels used to
    evaluate the flux model (typically 1070).  ``N_chan`` is the number of
    detector channels within the specified energy range (typically 526 for
    the standard 0.3–8.0 keV band).

    References
    ----------
    Buchner, J. & Boorman, P. (2023), arXiv:2309.05705.
    """
    # ------------------------------------------------------------------
    # Load instrument response (ARF, RMF, PHA)
    # ------------------------------------------------------------------
    data = xraystan.load_pha(pi_filename, low_energy_limit, high_energy_limit, fitbkg=False)

    pha = fits.open(pi_filename)
    rmf = fits.open(rmf_filename)

    # Channel energy bounds from the RMF EBOUNDS extension
    ebounds    = rmf["EBOUNDS"]
    chan_e_min = ebounds.data['E_MIN']
    chan_e_max = ebounds.data['E_MAX']

    # Mask to select channels within the analysis energy range
    mask = np.logical_and(chan_e_min > low_energy_limit, chan_e_max < high_energy_limit)

    # Observed count rate and exposure time from the PHA file
    count_rate = pha["SPECTRUM"].data["COUNT_RATE"]
    exposure   = pha["SPECTRUM"].header["EXPOSURE"]

    # ------------------------------------------------------------------
    # Energy grid quantities
    # ------------------------------------------------------------------
    # Full instrument energy grid (N_full channels)
    e_mid   = (data['e_hi'] + data['e_lo']) / 2.0   # channel midpoints (keV)
    e_width = data['e_hi'] - data['e_lo']             # channel widths (keV)

    # Filtered channel grid within the analysis energy range (N_chan channels)
    chan_e = (data['chan_e_min'] + data['chan_e_max']) / 2.0  # midpoints (keV)
    de     = data['chan_e_max'] - data['chan_e_min']           # widths (keV)

    # ------------------------------------------------------------------
    # Flux model → predicted counts via ARF/RMF convolution (Eq. 1, BB2023)
    # ------------------------------------------------------------------
    fluxmodel = flux_multi_comp(
        gamma=gamma_index,
        scaling_factor=scaling_val,
        hydrogen_column_density=nh_density,
        galactic_hydrogen_column_density=galactic_nh,
        photo_electric_sigma=photo_sigma,
        photo_electric_sigma_redshift=photo_sigma_red,
        redshift=redshift,
        energy=e_mid,
    )                                                       # (T, N_full)

    flux_temp = fluxmodel * exposure * e_width              # (T, N_full) — photons cm⁻²
    flux_arf  = flux_temp * jnp.asarray(data["ARF"].astype("float32"))   # (T, N_full)
    lambda_i  = jnp.dot(flux_arf, jnp.asarray(data['RMF'].astype("float32")))  # (T, N_chan)

    # ------------------------------------------------------------------
    # Poisson sampling (Eq. 3, BB2023)
    # ------------------------------------------------------------------
    out_shape = (simulation_count, lambda_i.shape[0], lambda_i.shape[1])
    if reproducibility:
        rng = np.random.default_rng(42)
        pred_count = rng.poisson(lambda_i, out_shape)
    else:
        pred_count = np.random.poisson(lambda_i, out_shape)

    # ------------------------------------------------------------------
    # Derived count-rate quantities
    # ------------------------------------------------------------------
    pred_rate  = pred_count / exposure   # count s⁻¹
    pred_dt_de = pred_rate / de          # count s⁻¹ keV⁻¹
    count_rate_list = [pred_dt_de]

    return (
        np.array(pred_count),
        np.array(count_rate_list),
        np.array(count_rate[mask]),
        np.array(chan_e),
    )
