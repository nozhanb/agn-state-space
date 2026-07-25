# beta3_4_N20_ensemble

## Question
For beta=3 and beta=4 (both above AR(1)'s structural PSD slope ceiling
of 2), the single-realization recovered beta in Table A.1 misses the
true value (2.475 vs true=3.00; 3.857 vs true=4.00). Is this a genuine,
persistent bias -- as the "structural ceiling" explanation in the paper
claims -- or could it be an unlucky single noise realization, the same
way beta=1.7's single-realization miss turned out to be periodogram
sampling noise that vanished once averaged over 20 independent
realizations (see `beta1.7_N20_realization_medians.npy` and the
Discussion paragraph in `tex/july_16_2026.tex`)?

## Motivation
Directly proposed by Nozhan after a detailed check of the periodogram
fitting mechanism (`plots/appendix_a/generate_table_A1.py` inspection,
2026-07-25) showed that a single-draw fitted beta is not actually bounded
by the AR(1) model's asymptotic slope-2 ceiling -- sparse, noisy
low-frequency periodogram bins (n=2 raw points in the lowest bin) can
pull a single fit well past 2 in either direction. That raised a real
question: is the *median* recovered beta in Table A.1 (which comes from
~2000 such per-draw fits *within one realization*) itself just an
unlucky single draw of the underlying data, or does it reflect something
that persists regardless of which specific noise realization you get?
The only way to answer this rigorously is the same test already run for
beta=1.7: generate many independent realizations, run each through the
full pipeline, and see whether the *realization-to-realization* spread
brackets the true value or not.

This connects to the original review comments from Johannes that
motivated the beta=1.7 N=20 test in the first place:
> 12. "Please make several simulations (maybe 20)."
> 13. "Maybe you can add a plot with beta as the y-axis, with the true
>     value as a horizontal dashed line, and the inferred value as an
>     error bar."

This directory extends that same test to beta=3 and beta=4.

## Method
Identical to the beta=1.7 N=20 ensemble
(`Appendix_A/reproduction_2026-07-20/regenerate_beta1.7_N20_shortchain.py`
/ `fit_beta1.7_N20_ensemble.py`), for full comparability:

1. **Generation** (`regenerate_beta3_N20_shortchain.py`,
   `regenerate_beta4_N20_shortchain.py`): 20 independent `colorednoise`
   realizations per beta, T=1000, shift=7 (same SNR level as Table A.1's
   beta=3/4 rows), each run through the corrected, wider-prior AR(1)/HMC
   model (`ar1_hmc_v2_widerprior.py` -- the same model behind the
   published shift=7 SNR-sweep numbers), short chain (warmup=1000,
   samples=2000). Frozen-chain detection (`n_unique(tau_param) > 1`, not
   a std threshold -- see `CORRECTION_LOG_2026-07-21.md` for why)
   built in from the start, with automatic reseed-retry (up to 8
   attempts), since beta=3/4 at shift=7 were known from the earlier SNR
   sweep to occasionally hit this.

2. **Quality re-check and fix** (`fix_low_ess_beta34_N20.py`): after
   generation, a direct scan of all 40 saved realizations' `n_unique_tau`
   found 3 that passed the `n_unique > 1` (not literally frozen) check
   but showed a *milder* version of the same pathology -- severely
   autocorrelated chains with only 10-474 unique values out of 2000
   samples (vs. ~1960-1998 for every healthy realization). This is the
   same low-ESS problem originally found for beta=1.7's short chain
   (see `BAYESIAN_DIAGNOSTICS_LEARNINGS.md`), and the `n_unique > 1`
   check alone doesn't catch it. The 3 flagged realizations
   (beta=3/realization_01, beta=4/realization_08, beta=4/realization_14)
   were re-run with a stricter threshold (`n_unique >= 1000`) and fresh
   seeds until each passed; all three succeeded within 1-4 attempts.
   Final check confirmed 0/40 realizations remain flagged.

3. **Fitting** (`fit_beta34_N20_ensemble.py`): for each of the 20 clean
   realizations per beta, thin the 2000 posterior draws by 40 (same
   thinning as beta=1.7, matching the measured integrated
   autocorrelation time), fit each thinned draw (periodogram -> 9-bin
   -> multi-start `curve_fit`, identical method to Table A.1), take the
   *median* of that within-realization distribution as the
   realization's point estimate. Collect the 20 per-realization medians
   into the across-realization ensemble.

## Parameters
- beta_true: 3.0, 4.0
- T=1000, shift_term=7 (SNR proxy, matches Table A.1)
- Model: `ar1_hmc_v2_widerprior.py`, warmup=1000, samples=2000
- N=20 realizations per beta
- Thinning: 40 (50 draws fit per realization)

## Scripts (run in this order)
1. `regenerate_beta3_N20_shortchain.py`
2. `regenerate_beta4_N20_shortchain.py`
3. `fix_low_ess_beta34_N20.py`
4. `fit_beta34_N20_ensemble.py`

Run with `/opt/anaconda3/envs/pub_one/bin/python3 <script>.py`. Raw
per-realization npz files (flux_predicted, tau_param, etc.) are large
(~8MB-80MB each) and left in place at
`Appendix_A/reproduction_2026-07-20/beta{3,4}_N20_shortchain_data/`
rather than copied here; this directory has the generating scripts, the
final per-realization medians, and the logs, which is what's needed to
reproduce or interpret the result without re-running the (~40-70 min)
inference step.

## Results

### Generation
Both beta=3 and beta=4 completed 20/20 realizations. Total wall time:
beta=4 ~32.6 min, beta=3 ~42.7 min (run concurrently, ~43 min wall time
overall). Frozen-chain retries were common but always resolved within
the 8-attempt budget (typically 1-5 attempts). 3 of 40 realizations
needed the additional low-ESS fix pass (all resolved within 1-4 further
attempts at the stricter threshold).

### Per-realization medians
See `beta3_N20_realization_medians.npy`, `beta4_N20_realization_medians.npy`
(20 values each) and `fit_beta34_ensemble.log` for the full per-realization
detail.

| | beta_true=3.0 | beta_true=4.0 |
|---|---|---|
| ensemble mean | 2.8346 | 3.5336 |
| ensemble median | 2.7458 | 3.6217 |
| ensemble std | 0.3256 | 0.2632 |
| 5-95% range | [2.443, 3.360] | [3.183, 3.867] |
| realizations below true value | 13/20 | **20/20** |
| sign test p-value (below-rate vs. 50/50) | 0.263 | **0.000002** |
| one-sample t-test p-value (mean vs. true) | 0.039 | **<0.0000005** |

## Outcome -- read this carefully, it is NOT the same story as beta=1.7

**beta=4: statistically unambiguous, confirmed real bias.** All 20 of 20
independent realizations underestimate beta. If the true underlying
process were unbiased (median beta = 4 with realization noise
scattering symmetrically around it), the probability of getting 20/20
below by chance is ~1 in a million (sign test p=0.000002). This is not
noise -- this is a real, persistent, structural effect that survives
averaging over independent realizations. It **confirms and strengthens**
the "AR(1) structural PSD ceiling" explanation already in the paper,
now backed by N=20 evidence instead of a single realization.

**beta=3: weaker, more ambiguous evidence.** 13/20 realizations fall
below the true value -- not extreme enough for the sign test to reject
"no bias" (p=0.263), but the one-sample t-test (which uses the size of
each deviation, not just its direction) is significant at p=0.039. This
suggests some real bias is plausible but is much less clear-cut than
beta=4's case, and is not a confirmed result the way beta=4's is.

**This means the original hypothesis being tested here -- "maybe
beta=3/4's miss is just an unlucky sample, like beta=1.7 was" -- is
essentially ruled out for beta=4, and left genuinely unresolved for
beta=3.** Unlike beta=1.7 (where the ensemble mean landed almost exactly
on the true value, 1.71 vs 1.70), beta=4's ensemble mean (3.53) is still
clearly below its true value (4.00) even after averaging over 20
independent realizations -- averaging reduced the *apparent* miss
somewhat compared to the single-realization estimate (3.857) in the
sense that it moved further away, not closer, actually -- see caveat
below.

**Caveat on "did averaging help":** naively, the single-realization
point estimate for beta=4 (3.857) is numerically *closer* to the true
value (4.00) than the 20-realization ensemble mean (3.534) is. This
does NOT mean the single-realization number was more "correct" --
compare their uncertainties: the single-realization estimate's own
[5%,95%] interval ([3.727, 3.925]) is artificially tight (it only
captures within-chain draw-to-draw variation, not real-world
run-to-run variability), so its apparent closeness to 4.00 is not
statistically meaningful the way the ensemble's much more honest,
wider, and now null-hypothesis-tested uncertainty is. The ensemble
result is the more trustworthy one specifically because it is testing
and rejecting a real alternative hypothesis (no bias), not because its
point estimate is numerically closer to 4.

## What this means for the paper
The "structural ceiling, cannot be recovered by construction" claim
(Discussion, Abstract, Appendix A -- all edited earlier this session)
is now on *firmer* ground for beta=4 specifically, with rigorous N=20
statistical backing rather than a single-realization argument. For
beta=3, the claim is directionally supported but should not be
overstated as equally confirmed -- the evidence is real but weaker.
Whether/how to reflect this beta=3-vs-beta=4 asymmetry in the paper's
wording (which currently treats them together) is an open decision for
Nozhan to make; this directory documents the evidence needed to make
it, but does not itself edit the paper.

## Outputs
- `beta3_N20_realization_medians.npy`, `beta4_N20_realization_medians.npy`
- Feeds `plots/appendix_a/plot_beta_recovery_calibration.py` (adds the
  N=20 ensemble markers for beta=3/4 to the existing calibration plot)

## Related files
- `plots/appendix_a/beta_recovery_calibration.pdf` / `.png` -- the
  updated plot, all three beta values, both uncertainty types
- `plots/appendix_a/README_beta_recovery_calibration.md` -- the plot's
  own documentation (written before this ensemble existed for beta=3/4;
  now partially superseded by this README for the beta=3/4 specifics)
- `Appendix_A/reproduction_2026-07-20/BAYESIAN_DIAGNOSTICS_LEARNINGS.md`
  -- background on the low-ESS/frozen-chain diagnostic methodology reused
  here
