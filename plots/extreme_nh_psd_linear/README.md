# extreme_nh_psd_linear

## Question
Is the β ≈ 4 fit visually apparent when the PSD y-axis is shown in linear scale (x-axis remains log)?

## Motivation
In log-log scale, the difference between β ≈ 1 and β ≈ 4 power-law fits can be hard to distinguish visually because data points overlap and the eye perceives slope differences poorly at extreme scales. Linear PSD scale should make the steepness of the extreme sample's fit more apparent, providing a clearer intuitive illustration of the artifact's magnitude.

## Parameters
| Parameter | Value |
|-----------|-------|
| NH (mean log N_H) | 21 |
| TAU (AR(1) timescale τ) | 2 |
| PHI (mean log K) | -3 |
| Extreme sample index | 382 (β ≈ 4.0, closest to 4.0 among 1000 draws) |
| PSD x-axis | logarithmic |
| PSD y-axis | linear |
| Frequency bins (Vaughan) | 7 |

## Script
`nh_extreme_psd_linear.py`

## Data Files

### In `data/` (copied — both < 100 MB)
| File | Size | Contents |
|------|------|----------|
| `cache_sample_psds_Nh21_Tau2_Phi-3_n1000_seed42.npz` | 896 KB | Binned PSDs and β values for 1000 posterior samples. Used to identify index 382 and retrieve its binned PSD. |
| `cache_true_fits_Nh21_Tau2_Phi-3.npz` | 8 KB | True trajectory binned PSD and power-law fit parameters. |

### Not copied (> 100 MB) — required to regenerate caches
| File | Size | Path |
|------|------|------|
| Inference NPZ (posterior trajectories) | 1.2 GB | `/Users/home/Documents/Claude/project/agn_sbi/code_backup/Claude_Beta_Recovery/inference_data/synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_stochastic_Nh_21_and_Tau_2_and_phi_-3_steps_100_iter_6000_spectrum_500_redshift_0108.npz` |

Note: the inference NPZ is listed in the script's config but is NOT actually opened at runtime — the script loads caches only. It is retained in the config for consistency with the other scripts.

## Outputs
| File | Description |
|------|-------------|
| `nh_extreme_psd_linear_Nh21_Tau2_Phi-3.pdf` | Publication-quality PDF |
| `nh_extreme_psd_linear_Nh21_Tau2_Phi-3.png` | 300 dpi PNG |

Figure layout: single panel (7 × 5 inches).
- Red squares + black dashed line: true binned PSD and power-law fit (β ≈ 1).
- Orange circles + orange solid line: extreme sample binned PSD and power-law fit (β ≈ 4).
- x-axis: log-scale frequency; y-axis: linear-scale PSD (bottom fixed at 0).

## Outcome
The linear PSD y-axis makes the contrast between the flat (β ≈ 1) true spectrum and the steeply falling (β ≈ 4) extreme sample immediately visible: the extreme fit drops sharply from an elevated low-frequency value and becomes negligible at high frequencies, while the true spectrum remains nearly flat across the full frequency range. This confirms that the β ≈ 4 fit is not a marginal deviation but a qualitatively different spectral shape anchored at a single anomalously high low-frequency bin.

## LaTeX Caption
```latex
\textbf{Linear-scale PSD comparison for the extreme-$\beta$ NH sample (draw index 382) vs.\ the true spectrum, $N_H = 10^{21}$\,cm$^{-2}$, $\tau = 2$, $\log K = -3$.}
Frequency axis is logarithmic; PSD amplitude axis is linear (bottom fixed at zero).
Crimson squares and black dashed line: Vaughan-binned true PSD and power-law fit ($\beta \approx 1$).
Orange circles and solid line: binned PSD and fit for the extreme-$\beta$ sample ($\beta \approx 4$).
The linear amplitude scale reveals that the steep extreme fit is anchored by a single anomalously high point at the lowest frequency bin, with the fitted curve collapsing to near zero at high frequencies — behaviour inconsistent with the nearly flat true spectrum.
```
