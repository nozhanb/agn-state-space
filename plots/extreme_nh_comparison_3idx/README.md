# extreme_nh_comparison_3idx

## Question
Is the β ≈ 4 artifact for index 382 reproducible? Do other extreme-β samples show the same low-frequency anchor mechanism?

## Motivation
Test the hypothesis that high-β outliers share a common cause (anomalous single-point bin-1 draw) rather than representing a genuine second mode in the posterior. By placing three independently selected extreme-β samples side by side, the figure allows visual comparison of both their time series character and PSD structure.

## Parameters
| Parameter | Value |
|-----------|-------|
| NH (mean log N_H) | 21 |
| TAU (AR(1) timescale τ) | 2 |
| PHI (mean log K) | -3 |
| Sample indices compared | 382 (β ≈ 4.0), 219, 944 |
| Grey trace count per column | 50 random non-extreme samples |
| Random seed | 42 (draw selection), 99 (grey trace selection) |
| Frequency bins (Vaughan) | 7 |

## Script
`nh_extreme_comparison.py`

## Data Files

### In `data/` (copied — both < 100 MB)
| File | Size | Contents |
|------|------|----------|
| `cache_sample_psds_Nh21_Tau2_Phi-3_n1000_seed42.npz` | 896 KB | Periodograms, Vaughan-binned PSDs, and β values for 1000 posterior samples. Provides trajectories, raw PSDs, and binned PSDs for all three indices. |
| `cache_true_fits_Nh21_Tau2_Phi-3.npz` | 8 KB | True trajectory Vaughan-binned PSD and power-law fit parameters used as reference in every PSD panel. |

### Not copied (> 100 MB) — required to regenerate caches
| File | Size | Path |
|------|------|------|
| Inference NPZ (posterior trajectories) | 1.2 GB | `/Users/home/Documents/Claude/project/agn_sbi/code_backup/Claude_Beta_Recovery/inference_data/synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_stochastic_Nh_21_and_Tau_2_and_phi_-3_steps_100_iter_6000_spectrum_500_redshift_0108.npz` |
| Simulation NPZ (true trajectory) | 401 MB | `/Users/home/Documents/Claude/project/agn_sbi/code_backup/Claude_Beta_Recovery/simulation_data/synthetic_count_NH_and_phi_spec_visualisation_SDSSJ0932+0405_multiple_simulated_stochastic_Nh_21_and_Tau_2_and_phi_-3.npz` |

## Outputs
| File | Description |
|------|-------------|
| `nh_extreme_comparison_Nh21_Tau2_Phi-3.pdf` | Publication-quality PDF |
| `nh_extreme_comparison_Nh21_Tau2_Phi-3.png` | 300 dpi PNG |

Figure layout: 2 rows × 3 columns (one column per extreme sample index).
- Row 1: N_H time series for each sample — 50 grey posterior traces, posterior mean (black), true trajectory (crimson dashed), target extreme sample (orange thick).
- Row 2: log-log PSD for each sample — grey halo (5–95% binned), true binned PSD + fit (crimson/black), extreme sample raw periodogram + binned PSD + fit (orange).

## Outcome
All three extreme samples show elevated low-frequency power relative to high-frequency power, consistent with the single-point bin-1 variance mechanism. High-frequency bins are similar across all samples, confirming the high-frequency anchor is stable and the spread in β is caused entirely by variance in the lowest-frequency bin.

## LaTeX Caption
```latex
\textbf{Comparison of three extreme-$\beta$ posterior NH samples (draw indices 382, 219, 944), $N_H = 10^{21}$\,cm$^{-2}$, $\tau = 2$, $\log K = -3$.}
Each column corresponds to one high-$\beta$ sample.
\textit{Top row:} $N_H$ time series; 50 representative posterior traces (silver), posterior mean (black), true trajectory (crimson dashed), and the target sample (orange).
\textit{Bottom row:} log--log PSD; grey band = 5th--95th percentile of all 1\,000 posterior binned PSDs; crimson squares and black dashed line = true binned PSD and power-law fit; orange symbols and line = target sample binned PSD and fit.
All three extreme samples exhibit elevated lowest-frequency power (single-point Vaughan bin, CV = 100\%) while sharing similar high-frequency behaviour, confirming that high-$\beta$ outliers are driven by bin-1 periodogram variance rather than a genuine long-correlation mode.
```
