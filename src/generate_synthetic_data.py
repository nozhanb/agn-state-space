"""
generate_synthetic_data.py
==========================
Grid-search script that generates synthetic X-ray count spectra for a range
of AR(1) parameter combinations and saves each dataset to a ``.npz`` file.

For every combination of (NH_mean, tau, phi_mean) the script:

1. Draws two independent AR(1) time series in log space — one for the
   hydrogen column density NH and one for the power-law normalisation φ::

       log10(NH(t)) ~ AR(1, α, NH_mean, σ=0.1)
       log10(φ(t))  ~ AR(1, α, phi_mean, σ=0.1)

   where the autocorrelation coefficient is α = exp(−1 / tau).

2. Simulates ``ITER_NUMBER`` independent Poisson count realisations at each
   time step using ``count_generator`` (Buchner & Boorman 2023, Eqs. 1 & 3).

3. Saves the count array, the NH and φ processes, and summary statistics to
   ``data/synthetic_count_NH_and_phi_spec_<tag>.npz``.

Usage
-----
Run from the repository root::

    python src/generate_synthetic_data.py

Configuration
-------------
Edit the constants in the ``CONFIGURATION`` block below to change the
parameter grid, time-series length, or number of Poisson realisations.

Output files
------------
One ``.npz`` per parameter combination, saved to ``data/``.  Keys:

- ``all_simulated_count`` — shape (ITER_NUMBER, T, N_chan), int32
- ``nh_process``  — shape (T,), log10(NH) time series
- ``phi_process`` — shape (T,), log10(φ) time series

References
----------
Buchner, J. & Boorman, P. (2023), arXiv:2309.05705.
"""

import time
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from ar1_process_generator import stochastic_based_nh_phi_simulator_v2
from count_simulator import count_generator


# ---------------------------------------------------------------------------
# CONFIGURATION — edit here to change the parameter grid
# ---------------------------------------------------------------------------

# Log10(NH) grid values [log10(atoms cm⁻²)]
NH_MEAN_GRID: list = [19, 20, 21, 22, 23, 24]

# AR(1) decorrelation timescales τ; autocorrelation α = exp(−1/τ)
TAU_GRID: list = [2, 38, 50]

# Log10(φ) grid values (power-law normalisation)
PHI_MEAN_GRID: list = [-3, -4, -5]

# Fixed spectral parameters
GAMMA_INDEX: float = 2.0     # Photon index Γ of the ZPOWERLW continuum
REDSHIFT: float    = 0.0108  # Cosmological redshift of SDSSJ0932+0405

# AR(1) process settings
PROCESS_STEPS: int  = 100    # Number of time steps T per AR(1) realisation
ERROR_SIGMA: float  = 0.1    # Innovation std σ for both NH and φ processes

# Number of independent Poisson realisations per (NH, τ, φ) combination
ITER_NUMBER: int = 1000

# Energy band for count extraction (keV)
LOW_ENERGY_CUT:  float = 0.3
HIGH_ENERGY_CUT: float = 8.0

# ---------------------------------------------------------------------------
# PATH RESOLUTION — all paths are relative to this file's location
# ---------------------------------------------------------------------------
_SRC_DIR  = Path(__file__).parent
_REPO_DIR = _SRC_DIR.parent
_DATA_DIR = _REPO_DIR / "data"


def main() -> None:
    """Run the full parameter-grid synthetic data generation."""
    # Load pre-computed photoelectric cross-sections
    sigma_data           = np.load(_DATA_DIR / "photo_electric_sigma_redshift_0108.npz")
    photo_sigma          = sigma_data["sigma"].astype("float32")
    photo_sigma_redshift = sigma_data["sigma_redshift"].astype("float32")

    pi_file  = _DATA_DIR / "SDSSJ0932+0405.pi"
    rmf_file = _DATA_DIR / "SDSSJ0932+0405.rmf"

    start = time.time()

    for nh_mean in NH_MEAN_GRID:
        for tau in TAU_GRID:
            alpha = float(jnp.exp(-1.0 / tau))   # AR(1) autocorrelation coefficient

            for phi_mean in PHI_MEAN_GRID:
                print(f"Running  NH_mean={nh_mean}  tau={tau}  phi_mean={phi_mean}")

                # Generate log-space AR(1) time series for NH and φ
                _, nh_process  = stochastic_based_nh_phi_simulator_v2(
                    scale_factor=alpha, shift_term=nh_mean,
                    error_sigma=ERROR_SIGMA, process_length=PROCESS_STEPS,
                    reproducibility=False,
                )
                _, phi_process = stochastic_based_nh_phi_simulator_v2(
                    scale_factor=alpha, shift_term=phi_mean,
                    error_sigma=ERROR_SIGMA, process_length=PROCESS_STEPS,
                    reproducibility=False,
                )

                print(f"  nh_process  shape={nh_process.shape}  mean={nh_process.mean():.4f}")

                # Simulate count spectra (shape: ITER_NUMBER × T × N_chan)
                count, _, _, _ = count_generator(
                    str(pi_file), str(rmf_file),
                    photo_sigma, photo_sigma_redshift,
                    low_energy_limit=LOW_ENERGY_CUT,
                    high_energy_limit=HIGH_ENERGY_CUT,
                    gamma_index=GAMMA_INDEX,
                    nh_density=10 ** nh_process,
                    scaling_val=10 ** phi_process,
                    redshift=REDSHIFT,
                    simulation_count=ITER_NUMBER,
                    reproducibility=False,
                )

                # Summary statistics over all channels and realisations
                total = count.sum(axis=(1, 2))
                print(
                    f"  count stats — mean={total.mean():.2f}  "
                    f"std={total.std():.2f}  "
                    f"[mean±2σ] = [{total.mean() - 2*total.std():.2f}, "
                    f"{total.mean() + 2*total.std():.2f}]"
                )

                # Save output
                tag = (
                    f"synthetic_count_NH_and_phi_spec_visualisation_"
                    f"SDSSJ0932+0405_multiple_simulated_stochastic_"
                    f"Nh_{nh_mean}_and_Tau_{tau}_and_phi_{phi_mean}"
                )
                out_path = _DATA_DIR / tag
                np.savez(
                    out_path,
                    all_simulated_count=count,
                    nh_process=nh_process,
                    phi_process=phi_process,
                )
                print(f"  Saved → {out_path}.npz")

    elapsed = (time.time() - start) / 60.0
    print(f"\nTotal time: {elapsed:.2f} min")


if __name__ == "__main__":
    main()
