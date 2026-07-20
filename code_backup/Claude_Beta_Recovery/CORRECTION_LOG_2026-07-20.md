# Correction Log — Claude_Beta_Recovery/

**Date:** 2026-07-20
**Author:** Nozhan Balafkan (investigation with Claude Code)

---

## Bug Found: Single fixed initial guess in the power-law+constant curve_fit

### Where

Two places, same root cause:

1. **Appendix A** (colorednoise β-recovery validation): the reproduction
   script `Appendix_A/reproduction_2026-07-20/psd_beta_recovery_per_draw_binned_fit.py`
   (written 2026-07-20 while addressing Johannes's review comments 10/11).
2. **Main text** (N_H/K PSD recovery, sec. 3.4.2): `_bin_and_fit_psd` in
   `nh_phi_psd_beta_recovery_methodB_v2.py`, which produced the cache
   `results/cache_sample_psds_Nh21_Tau2_Phi-3_n1000_seed42_v2.npz` behind
   `psd_beta_recovery_Nh21_Tau2_Phi-3_methodB_v2.png` and
   `nh_extreme_comparison_Nh21_Tau2_Phi-3_v2.png` (Figs. 5 and 6 of the paper).

### What the bug is

Both scripts call `scipy.optimize.curve_fit` on the bounded 3-parameter model
`log10(a * f^-b + c)` with a single, hardcoded initial guess for the power-law
index (`p0` beta-component = 2.5 in Appendix A, 1.5 in the main-text script),
used unconditionally for every fit, regardless of the true/expected beta for
that case. This is a nonlinear, bounded, 3-parameter fit with real parameter
degeneracy between `a` and `c`; a fixed initial guess can land the optimizer
in a poor local minimum when the true beta is far from that guess.

### Confirmed real, but severity depends heavily on the dataset

**Appendix A (colorednoise, true beta = 1.7 / 3.0 / 4.0):** the bug is
significant here. For beta=4.0, refitting the ensemble-averaged binned PSD
with different initial guesses gave beta estimates ranging from 2.86 to 3.92
depending on p0 alone, with chi2 varying by ~9x between local optima
(chi2=307.9 at the old default p0=2.5, vs. chi2=39.1 from a better start).
Switching to a multi-start fit (try b0 in [0.5,1.5,2.5,3.5,4.5,6.0], keep the
lowest-chi2 result) changed the actual paper result for beta=4.0 from
mean=3.775 to median=3.825 (5%-95% CI: [3.77, 4.09]) -- a real, non-negligible
shift that changed whether the true value falls inside the reported interval
(it now does; it did not before the fix). See
`Appendix_A/reproduction_2026-07-20/psd_beta_recovery_per_draw_binned_fit.py`
and `results_table_A1.txt` for the corrected numbers now used in
`tex/july_16_2026.tex` Table A.1.

**Main text (N_H PSD recovery, Nh21_Tau2_Phi-3, n=1000 posterior samples):**
the same bug is present in the code (`_bin_and_fit_psd`'s fixed
`p0=[1e-3, 1.5, 1e-6]`, no multi-start), but re-running all 1000 samples with
the same multi-start fix (b0 in [0.3,0.7,1.0,1.5,2.0,2.5,3.0,4.0,5.0]) shows
it does **not** materially change the results:

| Statistic         | Old (fixed p0=1.5) | New (multi-start) |
|--------------------|---------------------|---------------------|
| Mean beta          | 1.392               | 1.394               |
| Median beta        | 1.245               | 1.245               |
| Max beta           | 4.480               | 4.485               |
| n(beta >= 4)        | 17                  | 22                  |

Only 9 of 1000 samples shifted by more than 0.05 (all already in the
beta~4 tail; largest single shift was 0.267). The three specific samples
named in the paper text/figure caption (sec. 3.4.2, fig:nh_extreme_comparison)
are essentially unchanged:

| Sample | Cached beta (old) | Multi-start beta (new) |
|--------|--------------------|--------------------------|
| 382    | 4.000              | 4.000                    |
| 219    | 4.477              | 4.485                    |
| 944    | 4.480              | 4.479                    |

**Why the two datasets respond so differently:** in the main-text dataset,
the true/typical beta values cluster around 1-2 (median 1.245), close to the
fixed guess of 1.5, so the optimizer rarely lands in a bad local minimum --
only the small subset of samples whose true fit already sits near the
beta~4 tail (far from p0=1.5) are at any real risk, and even for those the
shift is modest. In Appendix A, the *entire* validation is built around a
single true beta far from the fixed guess (beta=4.0 vs. p0=2.5), so the whole
result was vulnerable, not just a tail.

### Conclusion

**No change needed to the main-text N_H/K PSD recovery results, Figs. 5/6, or
the associated discussion in sec. 3.4.2 (including the beta=4.48 examples).**
This was checked directly against the actual cached data on 2026-07-20 and
found robust; see the reproduction numbers above. This entry exists so that
if the question "did we check whether the optimizer bug affects the main
text results too" comes up again, the answer and its evidence are already
here rather than needing to be re-derived.

The Appendix A fix (multi-start, per-draw binned fit) is a separate, already
-applied correction; see
`Appendix_A/reproduction_2026-07-20/psd_beta_recovery_per_draw_binned_fit.py`
and the commit `c3ec96d` in this repo's git history for the full change.

### If this needs to be re-verified or extended to other combinations

The check script used to produce the table above is not saved as a permanent
file (it was a one-off diagnostic); the method is: load
`results/cache_sample_psds_<combo>_n1000_seed42_v2.npz`, use `nh_freq_nz` and
`nh_all_psds` to re-bin each sample's periodogram with the same 7-bin
(`NBINS=7`) log-frequency scheme as `_bin_and_fit_psd` in
`nh_phi_psd_beta_recovery_methodB_v2.py`, and refit with several b0 starting
values (spanning the plausible beta range for that combination), keeping the
lowest-chi2 result per sample. Recreate this for any other N_H/tau/phi
combination before trusting its individual extreme-beta examples, if that
combination's true beta is far from 1.5.
