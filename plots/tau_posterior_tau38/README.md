# tau_posterior_tau38

## Question
What do the posterior distributions for τ_NH and τ_φ look like for a long-timescale AGN (τ=38)?

## Motivation
Show that credible intervals for τ are wide when τ is large relative to the series length. For τ=38 and T=100, the effective number of independent samples is n_eff ≈ T/(2τ) ≈ 1.3, meaning the data contain very little information about the timescale. The figure documents this prior-dominated regime and calibrates the reader's expectation for uncertainty in the recovered τ values.

## Parameters
| Parameter | Value |
|-----------|-------|
| NH (mean log N_H) | 22 |
| TAU (AR(1) true timescale τ) | 38 |
| PHI (mean log K) | -4 |
| Prior on τ | Log-Uniform on [2, 80] → p(τ) = 1/(τ ln(80/2)) |
| n_eff ≈ T/(2τ) | ≈ 1.3 |
| HMC samples | 6000 |
| KDE bandwidth | Scott's rule |

## Script
`tau_posterior_paper.py`

## Data Files

### In `data/` (copied — < 100 MB)
| File | Size | Contents |
|------|------|----------|
| `summary_Nh_22_and_Tau_38_and_phi_-4.csv` | 36 KB | Authoritative posterior summary table with 5%, median, and 95% quantiles for all parameters (stored as ln τ). The script reads `nh_log_tau` and `phi_log_tau` rows to extract CI markers for the plot, ensuring the figure matches the paper table exactly. |

### Not copied (> 100 MB) — required to regenerate KDE
| File | Size | Path |
|------|------|------|
| Inference NPZ (posterior τ samples) | 1.2 GB | `/Users/home/Documents/Claude/project/agn_sbi/code_backup/remote/synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_stochastic_Nh_22_and_Tau_38_and_phi_-4_steps_100_iter_6000_spectrum_500_redshift_0108.npz` |

The NPZ stores τ as `nh_tau` and `phi_tau` in ln-τ space; the script converts via `np.exp(...)`. The KDE is computed on-the-fly from these samples. The CSV credible interval markers use a separate authoritative run so that CI lines match the paper table exactly (both runs are converged, r_hat ≈ 1).

## Outputs
| File | Description |
|------|-------------|
| `tau_posterior_Nh22_Tau38_phi-4_paper.pdf` | Publication-quality PDF |
| `tau_posterior_Nh22_Tau38_phi-4_paper.png` | 300 dpi PNG |

Figure layout: 2 rows × 1 column, shared x-axis (τ ∈ [0.5, 100]).
- Top panel: τ_NH — prior (steelblue), posterior KDE (orange), 5–95% CI shading, posterior median (dashed), true τ=38 (crimson dotted).
- Bottom panel: τ_φ — same format.
Both panels have linear axes.

## Outcome
The posterior for both τ_NH and τ_φ is broad and shifted relative to the prior, covering a wide range that includes the true τ=38. The 5–95% credible intervals are large, consistent with n_eff ≈ 1.3. The posterior is non-negligibly influenced by the log-uniform prior, as expected in this low-information regime.

## LaTeX Caption
```latex
\textbf{Prior and posterior distributions for the AR(1) timescale $\tau$ at $N_H = 10^{22}$\,cm$^{-2}$, $\tau_\mathrm{true} = 38$, $\log K = -4$.}
Top panel: $\tau_{N_H}$; bottom panel: $\tau_{\phi}$.
Blue curve: log-uniform prior $p(\tau) \propto \tau^{-1}$ on $[2,\,80]$.
Orange curve and shading: posterior KDE and 5th--95th percentile credible interval.
Orange dashed line: posterior median.
Crimson dotted line: true injected timescale ($\tau = 38$).
For $T = 100$ and $\tau = 38$, the effective sample size $n_\mathrm{eff} \approx T/(2\tau) \approx 1.3$, leading to the wide, prior-influenced credible intervals seen here.
```
