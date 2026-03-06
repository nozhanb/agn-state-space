# AGN X-ray Variability: Simulation-Based Inference

Bayesian inference of stochastic X-ray variability in Active Galactic Nuclei
(AGNs) via a simulation-based approach.  The pipeline jointly recovers the
hydrogen column density NH(t) and the power-law normalisation φ(t) from
observed X-ray count spectra using an AR(1) latent process model and
Hamiltonian Monte Carlo (HMC/NUTS) implemented in NumPyro.

---

## Overview

X-ray variability in AGNs arises from two sources:

| Source | Physical origin | Model component |
|---|---|---|
| **Intrinsic** | Coronal fluctuations | Power-law normalisation φ(t) |
| **External** | Absorbing winds / outflows | Hydrogen column density NH(t) |

This pipeline models both simultaneously by embedding an AR(1) stochastic
process for each parameter inside a full X-ray spectral forward model:

```
F(E, t) = ZPOWERLW(E; γ, φ(t)) × ZWABS(E; NH(t)) × WABS(E; NH_gal)
```

Photon counts are then predicted by folding F through the instrument ARF and
RMF, and a Poisson likelihood connects the model to observed counts:

```
λᵢ(t) = Σⱼ F(Eⱼ, t) · Δt · ΔEⱼ · ARFⱼ · RMFⱼᵢ
Cᵢ(t) ~ Poisson(λᵢ(t))
```

See [Buchner & Boorman (2023)](https://arxiv.org/abs/2309.05705) for the
theoretical background.

---

## Repository Structure

```
.
├── src/
│   ├── photo_electric_absorption.py   # Morrison & McCammon (1983) cross-sections
│   ├── ar1_process_generator.py       # AR(1) stochastic process generators
│   ├── count_simulator.py             # X-ray count spectrum simulator
│   ├── generate_synthetic_data.py     # Grid-search synthetic data generation
│   ├── run_inference.py               # HMC/NUTS Bayesian inference
│   └── visualise_results.py           # Posterior trace and histogram plots
│
├── data/
│   ├── fake_count.npz                 # Pre-processed instrument response grid
│   ├── photo_electric_sigma_redshift_0108.npz   # Pre-computed cross-sections
│   └── SDSSJ0932+0405.*              # Chandra observation files (not tracked)
│
├── output/
│   ├── inference/                     # Saved posterior sample arrays (.npz)
│   ├── summary/                       # MCMC diagnostic CSVs
│   └── plots/                         # Saved figures (.png)
│
├── requirements.txt
└── README.md
```

---

## Pipeline

The pipeline runs in three stages:

### Stage 1 — Pre-compute cross-sections

```bash
python src/photo_electric_absorption.py
```

Computes photoelectric absorption cross-sections σ(E) using the
[Morrison & McCammon (1983)](https://doi.org/10.1086/161108) piecewise
polynomial model at both rest-frame and redshifted energies, and saves
them to `data/photo_electric_sigma_redshift_<tag>.npz`.

### Stage 2 — Generate synthetic count spectra

```bash
python src/generate_synthetic_data.py
```

Loops over a grid of (NH_mean, τ, φ_mean) parameter combinations.  For each
combination it:

1. Draws two AR(1) time series in log space for NH(t) and φ(t).
2. Simulates 1000 independent Poisson count realisations via `count_generator`.
3. Saves each dataset to `data/synthetic_count_NH_and_phi_spec_<tag>.npz`.

Edit the `CONFIGURATION` block at the top of the script to change the
parameter grid.  **Expected runtime: several hours for the full 54-combination
grid.** Run a single combination for a quick test by temporarily setting each
list to one element.

### Stage 3 — Run Bayesian inference

```bash
python src/run_inference.py
```

Selects one synthetic dataset (configured via `NH_LABEL`, `TAU_LABEL`,
`PHI_LABEL` at the top of the script), runs NUTS with 2000 warm-up and 6000
posterior samples, and saves:

- `output/inference/<tag>.npz` — posterior samples and posterior predictive counts.
- `output/summary/<tag>.csv` — per-parameter R-hat, ESS, mean, std.

**Expected runtime: 30–120 min** depending on the time-series length and
hardware.

### Stage 4 — Visualise results

```bash
python src/visualise_results.py
```

Loads the inference output and plots:

- **Trace comparison**: posterior NH(t) and φ(t) trajectories vs ground truth,
  plus posterior predictive total counts.
- **Posterior histograms**: marginals for μ_NH and μ_φ with 95% credible
  intervals and true values marked.

---

## Installation

```bash
conda create -n agn-sbi python=3.11
conda activate agn-sbi
pip install -r requirements.txt
```

`xraystan` is not on PyPI; install from source:

```bash
pip install git+https://github.com/JohannesBuchner/xraystan
```

---

## Data

The Chandra observation files for SDSSJ0932+0405
(`SDSSJ0932+0405.pi`, `.rmf`, `.corr.arf`, `_bkg.*`)
are not tracked in this repository (they are not freely redistributable).
Place them in the `data/` directory before running Stage 2 or 3.

The two pre-processed files that are tracked:

| File | Contents |
|---|---|
| `data/fake_count.npz` | Instrument energy grid, ARF, RMF (1070 channels) |
| `data/photo_electric_sigma_redshift_0108.npz` | Pre-computed σ(E) at z = 0.0108 |

---

## Key Parameters

| Symbol | Description | Default |
|---|---|---|
| γ | Power-law photon index | inferred |
| NH(t) | Intrinsic hydrogen column density (log₁₀, 10²² cm⁻²) | inferred |
| φ(t) | Power-law normalisation (log₁₀) | inferred |
| α | AR(1) autocorrelation coefficient (= exp(−1/τ)) | inferred |
| τ | AR(1) decorrelation timescale | inferred |
| σ | AR(1) innovation standard deviation | inferred |
| NH_gal | Galactic foreground NH (fixed) | 0.01 × 10²² cm⁻² |
| z | Source redshift (fixed) | 0.0108 |

---

## References

- Morrison, R. & McCammon, D. (1983), *ApJ*, **270**, 119.
  [doi:10.1086/161108](https://doi.org/10.1086/161108)
- Buchner, J. & Boorman, P. (2023), *arXiv*:2309.05705.
  [arXiv](https://arxiv.org/abs/2309.05705)
- Phan, D. et al. (2019), *arXiv*:1912.11554 — NumPyro.
  [arXiv](https://arxiv.org/abs/1912.11554)
- XSPEC model documentation:
  <https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node220.html>

---

## Author

**Nozhan Balafkan**
In collaboration with Dr. Johannes Buchner (Max Planck Institute for
Extraterrestrial Physics, Garching, Germany).
