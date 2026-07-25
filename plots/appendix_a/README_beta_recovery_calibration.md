# beta_recovery_calibration

## Question
Across the three colorednoise validation cases in Table A.1 (true
beta = 1.7, 3.0, 4.0), how well does the recovered beta actually match
the true value, and how does the answer change depending on what kind
of uncertainty you measure?

## Why this plot exists (Johannes's original request)
This is the plot Johannes asked for in his Appendix A review comments,
never previously built as a literal figure. His comments (numbered as
given, full list recovered from the session transcript and re-listed in
chat on request):

> 12. "Please make several simulations (maybe 20)."
> 13. "Maybe you can add a plot with beta as the y-axis, with the true
>     value as a horizontal dashed line, and the inferred value as an
>     error bar."
> 14. "Please also vary the 'signal-to-noise' ratios (normalisations).
>     This can be the x-axis on the plot."

Comment 14 was answered separately by the beta=3/4 SNR sweep
(`plots/beta3_4_snr_trend/`) -- that plot's x-axis is SNR, not true
beta, and only covers beta=3/4. Comments 12-13 were never built as a
standalone figure until this one. Here we used true beta on the x-axis
(one panel, all three cases, one 1:1 reference line) instead of a
separate horizontal line per case -- a design choice by the paper's
author (Nozhan), not Johannes's literal request, made because it shows
calibration across all three tested regimes at a glance.

## Two markers per beta -- why
This plot deliberately shows **two different kinds of uncertainty** for
every beta value, because collapsing them into one number would
misrepresent what we actually know:

1. **Single realization** (blue circles) -- the [5%,95%] spread of
   per-draw beta fits *within one HMC run*. This is exactly what's
   published in Table A.1 (`tab:appendix_params` in
   `tex/july_16_2026.tex`). It answers "how much does the reconstructed
   light curve vary from draw to draw within a single posterior."

2. **N=20 realization ensemble** (orange squares) -- mean +/- std of
   the recovered-beta point estimate across 20 independent colorednoise
   realizations, each run through the full real pipeline (Poisson
   counts -> AR(1)/HMC -> periodogram fit). This answers a completely
   different question: "how much would my answer change if I generated
   the data over again with a new random seed." This is what Johannes's
   comment 12 was actually asking about.

Originally (as of 2026-07-25 afternoon) only beta=1.7 had this second
marker, since building the beta=3/4 equivalent required rerunning the
full pipeline 20x per beta. That was completed the same day -- see
`beta3_4_N20_ensemble/README.md` for the full method and statistical
analysis. **Do not average or merge the two markers for any given
beta** -- they are answers to different questions, not two estimates
of the same thing.

Nozhan considered also adding an N=40 version for beta=1.7 (an earlier,
cheaper 40-realization test exists, but it fits the raw pre-AR(1) flux
directly rather than running the actual inference pipeline -- see the
existing Appendix A footnote on this) -- decided N=20 (the full-pipeline
version) is the more defensible one to show, so only N=20 is plotted,
for all three beta values.

## Parameters
- beta_true: 1.7, 3.0, 4.0
- Single-realization data: T=1000, shift=7 (same posteriors used for
  Fig. A.1/A.2 and the Table A.1 rebuild)
- N=20 ensembles, all three beta values: T=1000, shift=7, 20 independent
  colorednoise realizations each, through the full corrected AR(1)/HMC
  pipeline (`ar1_hmc_v2.py` for beta=1.7, `ar1_hmc_v2_widerprior.py` for
  beta=3/4 -- the wider-prior version was needed at shift=7 for beta=3/4
  specifically; see `beta3_4_N20_ensemble/README.md`)

## Script
`plot_beta_recovery_calibration.py`

Run with:
```bash
/opt/anaconda3/envs/pub_one/bin/python3 plot_beta_recovery_calibration.py
```

Reads directly from `results_table_A1_beta1.7_longchain.txt` and
`results_table_A1_corrected.txt` (single-realization numbers, already in
this directory) plus the three ensemble `.npy` files -- so re-running
this script after any future update to those sources will automatically
pick up new numbers.

## Data
| File | Source | Description |
|------|--------|-------------|
| `results_table_A1_beta1.7_longchain.txt` | already in this directory (`generate_table_A1_beta1.7_longchain.py`'s output) | beta=1.7 single-realization p5/median/p95 |
| `results_table_A1_corrected.txt` | already in this directory (`generate_table_A1.py`'s output) | beta=3.0, beta=4.0 single-realization p5/median/p95 |
| `data/beta1.7_N20_realization_medians.npy` | `code_backup/.../reproduction_2026-07-20/beta1.7_N20_realization_medians.npy` | 20 per-realization recovered-beta medians, beta=1.7 |
| `.../reproduction_2026-07-20/beta3_N20_realization_medians.npy` | `beta3_4_N20_ensemble/fit_beta34_N20_ensemble.py`'s output | 20 per-realization recovered-beta medians, beta=3.0 |
| `.../reproduction_2026-07-20/beta4_N20_realization_medians.npy` | `beta3_4_N20_ensemble/fit_beta34_N20_ensemble.py`'s output | 20 per-realization recovered-beta medians, beta=4.0 |

## Outputs
- `beta_recovery_calibration.pdf`
- `beta_recovery_calibration.png`

## Outcome
All three single-realization points fall below the 1:1 perfect-recovery
line, with none of their [5%,95%] intervals reaching the true value.
**The N=20 ensembles reveal this is NOT the same story for all three
beta values:**

- **beta=1.7**: the ensemble mean (1.710) lands almost exactly on the
  true value (1.70), with a much wider, honest error bar (std=0.159 vs.
  the single-realization std~0.004) -- confirming the single-realization
  "miss" was ordinary periodogram sampling noise, not a systematic bias.
- **beta=4**: the ensemble mean (3.534) remains clearly below the true
  value (4.00) even after averaging 20 independent realizations. A sign
  test shows 20/20 realizations underestimate beta (p=0.000002) --
  overwhelming evidence this is a real, persistent bias, not noise.
- **beta=3**: an intermediate, more ambiguous case. 13/20 realizations
  underestimate beta (sign test p=0.26, not significant on its own),
  but a t-test against the true value is marginally significant
  (p=0.039). Some real bias is plausible but far less clear-cut than
  beta=4's case.

Full statistical detail (sign tests, t-tests, per-realization values)
in `beta3_4_N20_ensemble/README.md`. **Do not caption this plot as
uniformly "confirms recovery is unbiased on average" or uniformly
"confirms a structural bias" across all three beta values -- the
evidence differs meaningfully between beta=1.7, beta=3, and beta=4, and
the caption/paper text should say so.**

## LaTeX caption (draft, not yet inserted into the paper)
```latex
Recovered vs. true PSD power-law index $\beta$ for the three
\texttt{colorednoise} validation cases (Table~\ref{tab:appendix_params}).
Blue circles: median and 5--95\% credible interval of per-draw $\beta$
fits within a single HMC run. Orange squares: mean $\pm$ std of the
recovered-$\beta$ point estimate across 20 independent realizations per
$\beta$ value (full pipeline, not the raw-flux shortcut used elsewhere
in this appendix). For $\beta=1.7$ the ensemble mean recovers the true
value almost exactly, confirming the single-realization miss is ordinary
periodogram sampling noise. For $\beta=4$, all 20 realizations
underestimate $\beta$ (sign test $p=2\times10^{-6}$), confirming a
genuine, persistent bias consistent with the AR(1) model's structural
PSD ceiling. $\beta=3$ is an intermediate case with weaker statistical
support for a real bias (13/20 realizations below true, sign test
$p=0.26$; one-sample $t$-test $p=0.039$). Dashed line: perfect recovery.
```

## Status
Plot updated with all three beta=3/4 ensemble markers 2026-07-25.
**Not yet inserted into the paper** -- awaiting Nozhan's review of the
beta=3/4 ensemble finding, which is more nuanced than initially expected
(see `beta3_4_N20_ensemble/README.md`) and may need careful, asymmetric
wording between beta=3 and beta=4 rather than treating them as one case.
