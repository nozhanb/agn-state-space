# extreme_nh_sample_diagnostic_idx382

## Question
What does the extreme β ≈ 4 posterior sample (index 382) look like in time series and PSD space? Does its trajectory explain the anomalous β value?

## Motivation
Diagnose whether β ≈ 4 is caused by (a) near-unit-root τ posterior samples, (b) inference distortion from the nonlinear absorption model, or (c) finite-sample periodogram variance from a single-point first bin. The diagnostic compares the flagged sample's time series and PSD directly against the posterior ensemble and true trajectory, making the mechanism visually explicit.

## Parameters
| Parameter | Value |
|-----------|-------|
| NH (mean log N_H) | 21 |
| TAU (AR(1) timescale τ) | 2 |
| PHI (mean log K) | -3 |
| Extreme sample index | 382 (β ≈ 4.0, closest to 4.0 among 1000 draws) |
| Grey trace count | 50 random non-extreme samples |
| Random seed | 42 (draw selection), 99 (grey trace selection) |
| Frequency bins (Vaughan) | 7 |

## Script
`nh_extreme_sample_diagnostic.py`

## Data Files

### In `data/` (copied — both < 100 MB)
| File | Size | Contents |
|------|------|----------|
| `cache_sample_psds_Nh21_Tau2_Phi-3_n1000_seed42.npz` | 896 KB | Periodograms, Vaughan-binned PSDs, and β values for 1000 posterior samples. Used to identify the extreme index and reconstruct its PSD panels. |
| `cache_true_fits_Nh21_Tau2_Phi-3.npz` | 8 KB | True trajectory Vaughan-binned PSD and power-law fit parameters. |

### Not copied (> 100 MB) — required to regenerate caches
| File | Size | Path |
|------|------|------|
| Inference NPZ (posterior trajectories) | 1.2 GB | `/Users/home/Documents/Claude/project/agn_sbi/code_backup/Claude_Beta_Recovery/inference_data/synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_stochastic_Nh_21_and_Tau_2_and_phi_-3_steps_100_iter_6000_spectrum_500_redshift_0108.npz` |
| Simulation NPZ (true trajectory) | 401 MB | `/Users/home/Documents/Claude/project/agn_sbi/code_backup/Claude_Beta_Recovery/simulation_data/synthetic_count_NH_and_phi_spec_visualisation_SDSSJ0932+0405_multiple_simulated_stochastic_Nh_21_and_Tau_2_and_phi_-3.npz` |

## Outputs
| File | Description |
|------|-------------|
| `nh_extreme_sample_diagnostic_Nh21_Tau2_Phi-3.pdf` | Publication-quality PDF |
| `nh_extreme_sample_diagnostic_Nh21_Tau2_Phi-3.png` | 300 dpi PNG |

Figure layout: 1 row × 3 panels.
- Panel 1: N_H time series — 50 normal posterior traces (silver), posterior mean (black), true trajectory (crimson dashed), extreme sample trajectory (orange thick).
- Panel 2: PSD log-log — grey halo of binned PSDs (5–95%), true binned PSD + fit (crimson/black), extreme sample raw periodogram + binned PSD + power-law fit (orange).
- Panel 3: β histogram — all 1000 posterior β values (steelblue), with posterior median, true β, and extreme sample β (orange vertical line) marked.

## Outcome
The extreme sample trajectory shows an apparent monotonic drift, concentrating power at the lowest frequency (single-point bin 1, CV=100%). The high-frequency bins are well-constrained (bin 6 has 24 points, CV≈20%), confirming that the steep β is anchored by an anomalously high bin-1 draw rather than genuine long-range correlation.

## LaTeX Caption
```latex
\textbf{Diagnostic for the extreme $\beta \approx 4$ posterior sample (draw index 382), $N_H = 10^{21}$\,cm$^{-2}$, $\tau = 2$, $\log K = -3$.}
\textit{Left:} $N_H$ time series showing 50 representative posterior traces (silver), the posterior mean (black), the true generating trajectory (crimson dashed), and the extreme-$\beta$ sample (orange).
\textit{Centre:} log--log PSD panel; the grey band is the 5th--95th percentile of all 1\,000 binned posterior PSDs, the crimson squares and dashed line show the true binned PSD and its power-law fit ($\beta \approx 1$), and the orange symbols show the extreme sample's raw periodogram, binned PSD, and fit ($\beta \approx 4$).
\textit{Right:} Posterior $\beta$ histogram with the extreme sample highlighted (orange line).
The apparent monotonic drift in the extreme trajectory produces anomalously high power in the single-point lowest-frequency Vaughan bin (CV = 100\%), driving the steep fitted slope.
```
