# appendix_b

Fig. B.1 for Appendix B ("Intrinsic absorption behaviour for different
values of $N_H$", `app:optical_depth` in `tex/july_16_2026.tex`) — the
only figure in that appendix.

## Question
Below what hydrogen column density $N_H$ does intrinsic photoelectric
absorption become negligible in the observed X-ray band, i.e. where does
the data stop carrying information to constrain $N_H$?

## Motivation
Used in Sect. `sec:verification_results_parameter` to explain the
systematic underestimation of $N_H$ at low column density in the
recovery tests (Table `tab:light_curve_sim_table`): the model performs
poorly for $N_H = 10^{19}\,\mathrm{cm}^{-2}$ because the spectrum at that
column density is visually indistinguishable from the unabsorbed case,
so there is essentially no signal for the inference to latch onto.

## Parameters
- $\Gamma = 2.0$ (photon index)
- $K = 10^{-4}$ (power-law normalisation)
- $N_H$ = $10^{21}$, $10^{19}$, $10^{15}$, $10^{10}$ cm$^{-2}$
- No Galactic foreground absorption — intrinsic source absorption only,
  rest frame ($z = 0$)
- Absorption cross-section: Wisconsin cross-sections at $z = 0.0108$
  (\citealt{wilms2000}), from `data/photo_electric_sigma_redshift_0108.npz`

## Script
`photo_electric_absorption_visual_paper.py`

Run with:
```bash
python photo_electric_absorption_visual_paper.py
```

Originally `code_backup/photo_electric_absorption_visual_paper.py`
(STEP 3d in `code_backup/execution_order.txt`); relocated here
2026-07-29 and its `DATA_DIR`/`RESULTS_DIR` paths updated to be
self-contained in this directory. Verified to regenerate a plot
visually identical to the one already compiled into the paper
(`tex/observed_flux_NH_comparison.pdf`, untouched, still used by
pdflatex to compile the document).

## Data
| File | Source | Description |
|------|--------|-------------|
| `data/photo_electric_sigma_redshift_0108.npz` | `code_backup/data/` (copied, not moved — shared by ~20 other scripts) | Wisconsin photoelectric absorption cross-sections at $z=0.0108$ |
| `data/fake_count.npz` | `code_backup/data/` (copied, not moved — shared by ~20 other scripts) | Energy grid |

## Outputs
- `observed_flux_NH_comparison.pdf`
- `observed_flux_NH_comparison.png`

## Outcome
Below $N_H \sim 10^{20}\,\mathrm{cm}^{-2}$ the absorbed spectra
($N_H = 10^{10}, 10^{15}, 10^{19}$) are visually indistinguishable from
one another and from the unabsorbed case across the observed bandpass.
Only at $N_H = 10^{21}\,\mathrm{cm}^{-2}$ does soft-band ($E \lesssim
1$ keV) suppression become detectable. This defines the lower
sensitivity boundary of the inference framework and explains the
systematic underestimation of $N_H$ observed in simulations with true
$N_H = 10^{19}\,\mathrm{cm}^{-2}$.

## LaTeX caption
```latex
Intrinsic absorbed power-law flux $F_E$ as a function of
photon energy for four values of hydrogen column density $N_H$,
illustrating the effective sensitivity threshold of X-ray
photoelectric absorption. The spectral model is a power law with
photon index $\Gamma = 2.0$ and normalisation $K = 10^{-4}$,
attenuated by $\exp(-N_H\,\sigma(E))$, where $\sigma(E)$ is the
photoelectric absorption cross-section from
\citet{wilms2000}. No Galactic foreground absorption is included;
all curves represent intrinsic absorption at the source in the
rest frame ($z = 0$). The three curves for $N_H = 10^{10}$,
$10^{15}$, and $10^{19}$\,cm$^{-2}$ are indistinguishable from
one another across the full energy range, demonstrating that below
$N_H \sim 10^{20}$\,cm$^{-2}$ the attenuation factor
$e^{-N_H\sigma(E)} \approx 1$ and the spectrum carries essentially
no information about $N_H$. Only at $N_H = 10^{21}$\,cm$^{-2}$
does the soft-band ($E \lesssim 1$\,keV) suppression become
detectable. This defines the lower sensitivity boundary of our
inference framework and explains the systematic underestimation of
$N_H$ observed in simulations with true $N_H = 10^{19}$\,cm$^{-2}$
(Table~\ref{tab:light_curve_sim_table}).
```
