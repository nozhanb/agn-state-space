# beta_recovery_calibration_squares_only

## What changed from the two-marker version, and why

The original `plot_beta_recovery_calibration.py` showed two marker types
per beta value: blue circles (single-realization, from Table A.1) and
orange squares (N=20 realization-ensemble mean/std). This version drops
the single-realization markers entirely and shows only the ensemble
result, now upgraded to N=40 realizations per beta and extended to a
fourth beta value (8.0).

**The reason for dropping the single-realization markers is not just
visual simplicity -- it's that the two marker types are not the same
kind of statistical interval, and plotting them together invites a
comparison that doesn't mean what it looks like it means.**

- The blue circles are a **Bayesian credible interval**: the [5%,95%]
  spread of per-draw beta fits computed *within one posterior*, given
  one fixed, already-observed dataset. It answers "given the data we
  have, what range of beta values does the posterior support."
- The orange/red squares are an **empirical, frequentist-style interval**:
  the spread of independent point estimates across many separately
  generated datasets (repeated realizations). It answers a different
  question entirely -- "if I regenerated the data with a new random
  seed, how much would my answer change." This is much closer in spirit
  to a confidence interval (a statement about the sampling procedure's
  variability across hypothetical repeated datasets) than to a credible
  interval (a statement about belief given one dataset).

These two intervals are not interchangeable, do not have the same
coverage interpretation, and are not expected to relate to each other in
any simple way -- one single-realization point falling outside the
other's band is not evidence of a problem (see the base-rate check in
the git history of the two-marker version: ~25-30% of the ensemble's own
realizations already fall outside its own +/-1 std band, which is
ordinary sampling behaviour, not an anomaly). Putting a credible interval
and a confidence-interval-like quantity on the same axis, styled the
same way (both as "error bars"), invites a reader to compare them as if
they were the same kind of object. They aren't, and that comparison
doesn't actually mean anything rigorous. Dropping to ensemble-only
removes the temptation entirely and keeps the figure to one
well-defined, consistently-interpreted statistical quantity throughout.

## Why N=40 (not N=20) for beta=1.7/3.0/4.0, and why beta=8.0 was added

Both extensions were proposed by Nozhan in the same request:

1. **N=40, not N=20, for all of beta=1.7/3.0/4.0.** The original
   ensembles (still available, see `beta{1.7,3,4}_N20_realization_medians.npy`)
   used 20 realizations each. Topped up with 20 further realizations per
   beta (same model/config, fresh seeds -- see
   `regenerate_ensemble_realizations.py`) to reach N=40, giving tighter,
   more defensible statistics for the same three cases.

2. **beta=8.0 added, N=40 fresh.** Motivated by a visual observation on
   the N=20 three-point plot: recovered beta appeared to be flattening
   relative to true beta as true beta increased past 2 (slope
   beta=1.7->3.0 was ~0.87, slope beta=3.0->4.0 was ~0.70 -- see the
   git history of the two-marker version for this calculation). The
   question was whether this flattening continues, saturates, or does
   something else at a much steeper true beta.

## Result

| true beta | N | mean | std | median | below true / N | sign-test p | t-test p |
|---|---|---|---|---|---|---|---|
| 1.7 | 40 | 1.6669 | 0.1748 | 1.6757 | 24/40 | 0.268 | 0.244 |
| 3.0 | 40 | 2.8664 | 0.2816 | 2.8387 | 26/40 | 0.081 | 0.0052 |
| 4.0 | 40 | 3.5398 | 0.2404 | 3.5943 | 40/40 | 2e-12 (reported 0.000000) | ~0 |
| 8.0 | 40 | 3.3027 | 0.1953 | 3.3664 | 40/40 | 2e-12 (reported 0.000000) | ~0 |

**beta=1.7**: no significant bias, consistent with the N=20 result --
recovery is genuinely unbiased on average in this regime.

**beta=3.0**: strengthened toward significance relative to the N=20
result (t-test p dropped from 0.039 to 0.0052), though the sign test
still doesn't reach significance on its own (p=0.081). Real bias remains
plausible but not as unambiguous as beta=4/beta=8.

**beta=4.0**: now confirmed with overwhelming statistical force (40/40
realizations below true, vs. 20/20 at the smaller sample size --
the same qualitative conclusion, now on twice the evidence).

**beta=8.0 -- the new, unexpected finding**: the ensemble mean (3.303)
is *lower* than beta=4.0's ensemble mean (3.540), despite the true value
being twice as large. This is not merely "the flattening trend
continues" -- it is a reversal. Recovered beta does not monotonically
increase with true beta across this range; it appears to peak somewhere
around true beta~4 and decline for a steeper input. **No mechanism for
this reversal has been confirmed.** A plausible contributing factor,
not yet tested, is the periodogram-fit method's known sensitivity to
noise in the sparsest low-frequency bins (see the direct periodogram
inspection documented earlier in this project's history) -- a steeper
input concentrates even more of the dynamic range into those few,
noisiest bins, which could plausibly destabilize the fit further rather
than simply saturating it. This is a hypothesis, not a finding, and
would need its own dedicated check (e.g. inspecting individual beta=8
periodogram fits the way the beta=4 case was inspected) before being
stated as an explanation anywhere in the paper.

## Scripts
- `regenerate_ensemble_realizations.py` (in `Appendix_A/reproduction_2026-07-20/`) -- parameterized generator, used for all four top-up/fresh runs
- `fit_ensemble_N40_all_betas.py` (same directory) -- fitting + statistics, all four beta values
- `plot_beta_recovery_calibration_squares_only.py` (this directory) -- the plot

## Data quality note
One realization (beta=4.0, originally index 28 in the N=20-topped-up-to-N=40
set) required 15 retry attempts under the standard reseed scheme and never
escaped a frozen state; a fresh, differently-structured seed (not just a
further increment of the same seed family) resolved it on the first
attempt. All 160 realizations across the four beta values passed the
n_unique(tau_param)>=1000 quality threshold before fitting. Full detail
in `logs/` under `Appendix_A/reproduction_2026-07-20/`.

## Outputs
- `beta_recovery_calibration_squares_only.pdf`
- `beta_recovery_calibration_squares_only.png`

## Draft caption
```latex
Recovered vs. true PSD power-law index $\beta$, mean $\pm$ std across
an ensemble of 40 independent \texttt{colorednoise} realizations per
true $\beta$ (1.7, 3.0, 4.0, 8.0), each processed through the full
inference pipeline (Poisson counts $\to$ AR(1)/HMC inference $\to$
periodogram fit). $\beta=1.7$ recovers the true value with no
significant bias (one-sample $t$-test $p=0.24$). $\beta=3.0$ shows
weaker, marginal evidence of bias ($p=0.005$ by $t$-test; the sign
test alone, $p=0.08$, does not reach significance). $\beta=4.0$ shows a
statistically overwhelming bias: all 40 realizations underestimate
$\beta$ ($p<10^{-11}$). $\beta=8.0$ likewise shows all 40 realizations
below the true value, but with a mean (3.30) \emph{below} $\beta=4.0$'s
recovered mean (3.54) despite the true value being twice as large --
recovered $\beta$ does not increase monotonically with true $\beta$ in
this regime. Dashed line: perfect recovery.
```

## Status
Generated 2026-07-25. Not yet inserted into the paper.
