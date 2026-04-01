# psd_beta_recovery_tau2

## Question
Can Method B recover the PSD power-law slope β from posterior trajectory samples of an AR(1) process with τ=2?

## Motivation
Validate that the SBI inference correctly recovers the temporal variability structure of both the N_H and log K processes when the true timescale is short (τ=2). Method B computes the periodogram for every posterior sample individually (rather than just the posterior mean), enabling a full posterior distribution over β.

## Parameters
| Parameter | Value |
|-----------|-------|
| NH (mean log N_H) | 21 |
| TAU (AR(1) timescale τ) | 2 |
| PHI (mean log K) | -3 |
| AR(1) persistence φ | exp(-1/2) ≈ 0.607 |
| Time series length T | 100 |
| Total HMC samples | 6000 |
| Samples used for Method B | 1000 (random seed 42) |
| Frequency bins (Vaughan) | 7 |
| Power-law model | a·f^{-β} + c |

## Script
`nh_phi_psd_beta_recovery_methodB_paper.py`

## Data Files

### In `data/` (copied — both < 100 MB)
| File | Size | Contents |
|------|------|----------|
| `cache_sample_psds_Nh21_Tau2_Phi-3_n1000_seed42.npz` | 896 KB | Pre-computed periodograms and Vaughan-binned PSDs for 1000 posterior samples; β values for each sample. Arrays: `nh_freq_nz`, `nh_all_psds` (1000×n_freq), `nh_bin_centers`, `nh_binned_means` (1000×6), `nh_betas`, analogous `phi_*` arrays. |
| `cache_true_fits_Nh21_Tau2_Phi-3.npz` | 8 KB | Vaughan-binned PSD and power-law fit results for the true (generating) N_H and log K trajectories. |

### Not copied (> 100 MB) — required to regenerate caches
| File | Size | Path |
|------|------|------|
| Inference NPZ (posterior trajectories) | 1.2 GB | `/Users/home/Documents/Claude/project/agn_sbi/code_backup/Claude_Beta_Recovery/inference_data/synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_stochastic_Nh_21_and_Tau_2_and_phi_-3_steps_100_iter_6000_spectrum_500_redshift_0108.npz` |
| Simulation NPZ (true trajectories) | 401 MB | `/Users/home/Documents/Claude/project/agn_sbi/code_backup/Claude_Beta_Recovery/simulation_data/synthetic_count_NH_and_phi_spec_visualisation_SDSSJ0932+0405_multiple_simulated_stochastic_Nh_21_and_Tau_2_and_phi_-3.npz` |

The script loads caches automatically if present; the large NPZ files are only needed on first run.

## Outputs
| File | Description |
|------|-------------|
| `psd_beta_recovery_Nh21_Tau2_Phi-3_methodB_paper.pdf` | Publication-quality PDF |
| `psd_beta_recovery_Nh21_Tau2_Phi-3_methodB_paper.png` | 300 dpi PNG |

Figure layout: 3 rows × 2 columns.
- Row 1: Time series — 200 posterior sample traces (silver), posterior mean (black), true trajectory (crimson dashed).
- Row 2: PSD — 5th–95th percentile band of 1000 binned periodograms (grey), true periodogram + binned PSD + power-law fit, posterior median binned PSD (orange circles).
- Row 3: β histogram — distribution of β values from all 1000 posterior samples, with 5–95% credible interval, posterior median, and true β marked.

## Outcome
Both processes recover β ≈ 1 well. The N_H β distribution shows a heavy tail toward β ≈ 4 caused by a single-point first bin in the Vaughan log-frequency binning creating high periodogram variance at low frequencies (the first bin in a 7-bin Vaughan scheme over T=100 steps contains only 1 frequency point, so CV=100%).

## LaTeX Caption
```latex
\textbf{Method B PSD $\beta$ recovery for $N_H = 10^{21}$\,cm$^{-2}$, $\tau = 2$, $\log K = -3$.}
Top row: posterior trajectory samples (silver), posterior mean (black), and true generating trajectory (crimson dashed) for the $N_H$ (left) and $\log K$ (right) AR(1) processes.
Middle row: log--log PSD panels showing the 5th--95th percentile band of 1\,000 individual posterior sample periodograms (grey), the true binned PSD (filled squares) with power-law fit (black dashed), and the posterior-median binned PSD (orange circles).
Bottom row: posterior distribution of the fitted power-law slope $\beta$ for each process, with 5--95\% credible interval (shaded), posterior median (solid line), and true $\beta$ (crimson dashed line).
Both processes recover $\beta \approx 1$ correctly; the heavy high-$\beta$ tail visible in the $N_H$ panel arises from finite-sample periodogram variance in the single-point lowest-frequency bin of the Vaughan scheme.
```
