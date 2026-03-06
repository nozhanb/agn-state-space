"""
run_inference.py
================
Runs Bayesian inference on a synthetic X-ray count spectrum using Hamiltonian
Monte Carlo (HMC/NUTS) via NumPyro.

The probabilistic model jointly infers:

- **γ** (``gamma_value``) — photon index of the power-law continuum.
- **NH AR(1) parameters** — decorrelation timescale τ_NH, long-run mean μ_NH,
  and innovation std σ_NH for the hydrogen column density process.
- **φ AR(1) parameters** — decorrelation timescale τ_φ, long-run mean μ_φ,
  and innovation std σ_φ for the power-law normalisation process.
- **NH(t)** and **φ(t)** — the full time-resolved latent trajectories,
  recovered via a NumPyro ``scan`` over the AR(1) recursion.

The likelihood is Poisson-distributed count data produced by convolving the
physical flux model (ZPOWERLW × ZWABS × WABS) with the instrument ARF and RMF.

Outputs
-------
``output/inference/<output_file>.npz``
    Posterior samples for all parameters and posterior predictive counts.
``output/summary/<summary_file>.csv``
    Per-parameter MCMC diagnostics (mean, std, ESS, R-hat, runtime).

Usage
-----
Run from the repository root::

    python src/run_inference.py

Edit the ``CONFIGURATION`` block below to select a different synthetic dataset.

References
----------
Buchner, J. & Boorman, P. (2023), arXiv:2309.05705.
NumPyro: https://num.pyro.ai
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.contrib.control_flow import scan
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, Predictive
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION — edit here to select the input dataset and MCMC settings
# ---------------------------------------------------------------------------

# Ground-truth parameters used when generating the synthetic dataset
NH_LABEL:  int = 19    # log10(NH) grid value used in generate_synthetic_data.py
TAU_LABEL: int = 38    # τ grid value used in generate_synthetic_data.py
PHI_LABEL: int = -4    # log10(φ) grid value used in generate_synthetic_data.py

# Index of the simulation realisation to use as observed data (0-indexed)
# The synthetic count array has shape (ITER_NUMBER, T, N_chan); we pick one
# realisation along the first axis to serve as the "observed" spectrum.
SIM_INDEX: int = 500

# Source parameters (must match those used in generate_synthetic_data.py)
REDSHIFT:   float = 0.0108
GALACTIC_NH: float = 0.01   # in units of 10²² atoms cm⁻²

# NUTS/MCMC settings
NUM_WARMUP:  int = 2000
NUM_SAMPLES: int = 6000
NUM_CHAINS:  int = 1
RNG_SEED:    int = 42

# Number of host devices for parallel chains (set ≥ NUM_CHAINS)
N_DEVICES: int = 5

# ---------------------------------------------------------------------------
# PATH RESOLUTION — all paths relative to this file's location
# ---------------------------------------------------------------------------
_SRC_DIR    = Path(__file__).parent
_REPO_DIR   = _SRC_DIR.parent
_DATA_DIR   = _REPO_DIR / "data"
_INFER_DIR  = _REPO_DIR / "output" / "inference"
_SUMMARY_DIR = _REPO_DIR / "output" / "summary"


# ---------------------------------------------------------------------------
# PHYSICS MODEL
# ---------------------------------------------------------------------------

def flux_multi_comp(
    gamma: float,
    scaling_factor: jnp.ndarray,
    hydrogen_column_density: jnp.ndarray,
    galactic_hydrogen_column_density: float,
    photo_electric_sigma: jnp.ndarray,
    photo_electric_sigma_redshift: jnp.ndarray,
    redshift: float,
    energy: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the X-ray source flux via ZPOWERLW × ZWABS × WABS.

    This is the in-model variant of the flux computation used within the
    NumPyro probabilistic model.  Shapes are determined by NumPyro's scan,
    so ``scaling_factor`` and ``hydrogen_column_density`` arrive pre-shaped
    as (T, 1) column vectors that broadcast against the energy grid (N_chan,).

    Parameters
    ----------
    gamma : float
        Photon index Γ of the power-law continuum.
    scaling_factor : jnp.ndarray, shape (T, 1)
        Power-law normalisation at each time step.
    hydrogen_column_density : jnp.ndarray, shape (T, 1)
        Intrinsic NH at each time step, in units of 10²² atoms cm⁻².
    galactic_hydrogen_column_density : float
        Galactic foreground NH in units of 10²² atoms cm⁻².
    photo_electric_sigma : jnp.ndarray, shape (N_chan,)
        Photoelectric cross-sections at rest-frame energies (cm² per H atom).
    photo_electric_sigma_redshift : jnp.ndarray, shape (N_chan,)
        Photoelectric cross-sections at redshifted energies (cm² per H atom).
    redshift : float
        Cosmological redshift z of the source.
    energy : jnp.ndarray, shape (N_chan,)
        Mid-point energies of the instrument channels (keV).

    Returns
    -------
    out_flux : jnp.ndarray, shape (T, N_chan)
        Source flux per energy channel at each time step.
    """
    zpowerlw = scaling_factor * (energy * (1 + redshift)) ** -gamma
    zwabs    = jnp.exp(-hydrogen_column_density * photo_electric_sigma_redshift)
    wabs     = jnp.exp(-(galactic_hydrogen_column_density * 1e22) * photo_electric_sigma)
    return zpowerlw * zwabs * wabs


# ---------------------------------------------------------------------------
# NUMPYRO PROBABILISTIC MODEL
# ---------------------------------------------------------------------------

def model(
    count: jnp.ndarray,
    e_mid: jnp.ndarray,
    rmf: jnp.ndarray,
    arf: jnp.ndarray,
    exposure: jnp.ndarray,
    e_width: jnp.ndarray,
    sigma: jnp.ndarray,
    sigma_redshift: jnp.ndarray,
    array_length: int,
) -> None:
    """NumPyro probabilistic model for joint X-ray spectral-temporal inference.

    Samples the following latent variables:

    - ``gamma_value``  — photon index Γ ~ Uniform(1, 3).
    - ``nh_log_tau``   — log(τ_NH) ~ Uniform(log 2, log 80).
    - ``nH_shift``     — long-run mean μ_NH in log10 space ~ Uniform(15, 25).
    - ``phi_log_tau``  — log(τ_φ) ~ Uniform(log 2, log 80).
    - ``phi_shift``    — long-run mean μ_φ in log10 space ~ Uniform(−6, −2).
    - ``nh_sigma``     — AR(1) innovation std σ_NH ~ HalfNormal(0.5).
    - ``phi_sigma``    — AR(1) innovation std σ_φ ~ HalfNormal(0.5).

    Two AR(1) trajectories are generated via ``scan``:

    - ``nh_y``  — log10(NH(t)) time series.
    - ``phi_y`` — log10(φ(t)) time series.

    The likelihood is Poisson-distributed counts observed at each time step
    and energy channel, given the folded flux model.

    Parameters
    ----------
    count : jnp.ndarray, shape (T, N_chan) or ``None``
        Observed count data.  Pass ``None`` during posterior predictive sampling.
    e_mid : jnp.ndarray, shape (N_full,)
        Mid-point energies of the full instrument grid (keV).
    rmf : jnp.ndarray, shape (N_full, N_chan)
        Redistribution matrix (RMF).
    arf : jnp.ndarray, shape (N_full,)
        Ancillary response function (ARF).
    exposure : float
        Exposure time in seconds.
    e_width : jnp.ndarray, shape (N_full,)
        Width of each energy channel (keV).
    sigma : jnp.ndarray, shape (N_full,)
        Photoelectric cross-sections at rest-frame energies.
    sigma_redshift : jnp.ndarray, shape (N_full,)
        Photoelectric cross-sections at redshifted energies.
    array_length : int
        Number of time steps T.
    """
    # ── Global parameter priors ──────────────────────────────────────────────
    gamma_value_in = numpyro.sample("gamma_value", dist.Uniform(1.0, 3.0))
    nh_log_tau     = numpyro.sample("nh_log_tau",  dist.Uniform(jnp.log(2),  jnp.log(80)))
    nH_shift_in    = numpyro.sample("nH_shift",    dist.Uniform(15.0, 25.0))
    phi_log_tau    = numpyro.sample("phi_log_tau", dist.Uniform(jnp.log(2),  jnp.log(80)))
    phi_shift_in   = numpyro.sample("phi_shift",   dist.Uniform(-6, -2))
    nh_sigma       = numpyro.sample("nh_sigma",    dist.HalfNormal(0.5))
    phi_sigma      = numpyro.sample("phi_sigma",   dist.HalfNormal(0.5))

    # Convert log-tau to alpha (AR(1) autocorrelation coefficient)
    nh_alpha  = jnp.exp(-1.0 / jnp.exp(nh_log_tau))
    phi_alpha = jnp.exp(-1.0 / jnp.exp(phi_log_tau))

    # ── AR(1) latent trajectories ────────────────────────────────────────────
    def ar1_model(name: str, T: int, mu: float, alpha: float, sigma: float) -> jnp.ndarray:
        """Generate a mean-reverting AR(1) trajectory via NumPyro scan.

        Draws T i.i.d. standard-normal innovations and propagates them through
        the AR(1) recursion::

            y(t) = α · y(t−1) + (1 − α) · μ + σ · ε(t)

        Returns the full trajectory of length T.
        """
        eps = numpyro.sample(f"{name}_eps", dist.Normal(0.0, 1.0).expand([T]).to_event(1))
        y0  = numpyro.sample(f"{name}_y0",  dist.Normal(mu, sigma / jnp.sqrt(1 - alpha ** 2)))

        def step(prev: float, e: float):
            y = alpha * prev + (1 - alpha) * mu + sigma * e
            return y, y

        _, ys = scan(step, y0, eps)
        numpyro.deterministic(f"{name}_y", ys)
        return ys

    y_t_nh  = ar1_model("nh",  array_length, nH_shift_in,  nh_alpha,  nh_sigma)
    y_t_phi = ar1_model("phi", array_length, phi_shift_in, phi_alpha, phi_sigma)

    # ── Physical flux model ──────────────────────────────────────────────────
    flux = flux_multi_comp(
        gamma=gamma_value_in,
        scaling_factor=10 ** y_t_phi[:, None],
        hydrogen_column_density=10 ** y_t_nh[:, None],
        galactic_hydrogen_column_density=GALACTIC_NH,
        photo_electric_sigma=sigma,
        photo_electric_sigma_redshift=sigma_redshift,
        redshift=REDSHIFT,
        energy=e_mid,
    )                                                   # (T, N_full)

    # Fold through instrument response → expected counts per channel
    lambda_i = jnp.dot(flux * arf * exposure * e_width, rmf)   # (T, N_chan)

    # ── Likelihood ───────────────────────────────────────────────────────────
    numpyro.sample("count", dist.Poisson(lambda_i), obs=count)


# ---------------------------------------------------------------------------
# INFERENCE DRIVER
# ---------------------------------------------------------------------------

def run_inference_and_predictive(
    count: jnp.ndarray,
    e_mid: jnp.ndarray,
    rmf: jnp.ndarray,
    arf: jnp.ndarray,
    exposure: jnp.ndarray,
    e_width: jnp.ndarray,
    sigma: jnp.ndarray,
    sigma_redshift: jnp.ndarray,
    array_length: int,
    output_file: str,
    summary_file: str,
) -> tuple:
    """Run NUTS/HMC inference and posterior predictive sampling.

    Executes the full Bayesian workflow:

    1. Warm up and run the NUTS sampler.
    2. Compute per-parameter MCMC diagnostics and save to CSV.
    3. Generate posterior predictive count samples.
    4. Save all results to ``.npz``.

    Parameters
    ----------
    count : jnp.ndarray, shape (T, N_chan)
        Observed count data used as the likelihood target.
    e_mid, rmf, arf, exposure, e_width : jnp.ndarray
        Instrument response quantities (see ``model`` for shapes).
    sigma, sigma_redshift : jnp.ndarray
        Photoelectric cross-sections at rest-frame and redshifted energies.
    array_length : int
        Number of time steps T in the count spectrum.
    output_file : str
        Filename stem for the inference ``.npz`` (saved to ``output/inference/``).
    summary_file : str
        Filename for the diagnostics CSV (saved to ``output/summary/``).

    Returns
    -------
    posterior_samples : dict
        Dictionary of posterior sample arrays keyed by parameter name.
    predictive_samples : dict
        Dictionary of posterior predictive sample arrays.
    """
    _INFER_DIR.mkdir(parents=True, exist_ok=True)
    _SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ── NUTS / HMC ───────────────────────────────────────────────────────────
    rng_key = jax.random.PRNGKey(RNG_SEED)
    sampler = NUTS(model)
    mcmc    = MCMC(
        sampler,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        progress_bar=True,
        jit_model_args=False,
    )

    mcmc.run(
        rng_key, count, e_mid, rmf, arf, exposure, e_width,
        sigma, sigma_redshift, array_length,
    )
    mcmc.print_summary()

    runtime_minutes = (time.time() - start_time) / 60.0
    print(f"\nMCMC elapsed time: {runtime_minutes:.2f} minutes")

    # ── MCMC diagnostics → CSV ───────────────────────────────────────────────
    posterior_samples = mcmc.get_samples()
    summ = summary(posterior_samples, group_by_chain=False)
    df   = pd.DataFrame(summ).T
    df.loc["runtime_minutes", "mean"] = runtime_minutes
    df.to_csv(_SUMMARY_DIR / summary_file)
    print(f"Diagnostics saved → {_SUMMARY_DIR / summary_file}")

    # ── Posterior predictive ─────────────────────────────────────────────────
    print("\nGenerating posterior predictive samples…")
    predictive = Predictive(model, posterior_samples, return_sites=["count"])
    predictive_samples = predictive(
        jax.random.PRNGKey(123),
        count=None,          # remove observation to sample new counts
        e_mid=e_mid,
        rmf=rmf,
        arf=arf,
        exposure=exposure,
        e_width=e_width,
        sigma=sigma,
        sigma_redshift=sigma_redshift,
        array_length=array_length,
    )

    pp_counts = predictive_samples["count"]   # (samples, T, N_chan)
    pp_total  = pp_counts.sum(axis=2)         # (samples, T)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = _INFER_DIR / output_file
    np.savez(
        out_path,
        gamma_values=posterior_samples["gamma_value"],
        nh_tau=posterior_samples["nh_log_tau"],
        nH_shift=posterior_samples["nH_shift"],
        phi_tau=posterior_samples["phi_log_tau"],
        phi_shift=posterior_samples["phi_shift"],
        nh_sigma=posterior_samples["nh_sigma"],
        phi_sigma=posterior_samples["phi_sigma"],
        nh_y=posterior_samples["nh_y"],
        phi_y=posterior_samples["phi_y"],
        posterior_predictive_counts=pp_counts,
        posterior_predictive_total=pp_total,
    )
    print(f"Results saved → {out_path}.npz")
    print("Done.")

    return posterior_samples, predictive_samples


# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """Load data and run the inference pipeline."""
    numpyro.set_host_device_count(N_DEVICES)

    # ── Load instrument response ─────────────────────────────────────────────
    instr       = np.load(_DATA_DIR / "fake_count.npz")
    sigma_data  = np.load(_DATA_DIR / "photo_electric_sigma_redshift_0108.npz")

    e_mid     = jnp.asarray(instr["energy"].astype("float32"))
    de        = jnp.asarray(instr["dE"].astype("float32"))
    rmf       = jnp.asarray(instr["rmf_matrix"].astype("float32"))
    arf       = jnp.asarray(instr["arf_matrix"].astype("float32"))
    e_width   = jnp.asarray(instr["ener_width"].astype("float32"))
    exposure  = jnp.asarray(instr["expo"].astype("float32"))

    sigma          = jnp.asarray(sigma_data["sigma"].astype("float32"))
    sigma_redshift = jnp.asarray(sigma_data["sigma_redshift"].astype("float32"))

    # ── Load synthetic count data ────────────────────────────────────────────
    data_file = (
        _DATA_DIR /
        f"synthetic_count_NH_and_phi_spec_visualisation_"
        f"SDSSJ0932+0405_multiple_simulated_stochastic_"
        f"Nh_{NH_LABEL}_and_Tau_{TAU_LABEL}_and_phi_{PHI_LABEL}.npz"
    )
    print(f"Loading synthetic data: {data_file}")
    data2 = np.load(data_file)

    # Select one simulation realisation as the "observed" spectrum
    count = jnp.asarray(data2["all_simulated_count"][SIM_INDEX, :, :].astype("int32"))
    print(f"Count spectrum shape: {count.shape}   (T × N_chan)")

    # ── Output filenames ─────────────────────────────────────────────────────
    tag = f"Nh_{NH_LABEL}_and_Tau_{TAU_LABEL}_and_phi_{PHI_LABEL}"
    output_file  = (
        f"synthetic_count_spec_visualisation_SDSSJ0932+0405_"
        f"multiple_simulated_stochastic_{tag}_"
        f"steps_100_iter_{NUM_SAMPLES}_spectrum_{SIM_INDEX}_redshift_0108"
    )
    summary_file = f"summary_{tag}.csv"

    # ── Run inference ────────────────────────────────────────────────────────
    run_inference_and_predictive(
        count=count,
        e_mid=e_mid,
        rmf=rmf,
        arf=arf,
        exposure=exposure,
        e_width=e_width,
        sigma=sigma,
        sigma_redshift=sigma_redshift,
        array_length=int(count.shape[0]),
        output_file=output_file,
        summary_file=summary_file,
    )


if __name__ == "__main__":
    main()
