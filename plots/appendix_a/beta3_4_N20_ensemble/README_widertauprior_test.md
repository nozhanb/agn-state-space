# Widened tau-prior test (beta=4 vs beta=8)

## Question
The N=40 ensemble test found beta=4 and beta=8 converge to statistically
indistinguishable tau/mean/var posteriors under the standard prior
(tau ~ LogUniform(2,80)), both pinned at the boundary (~79.5). Does this
simply reflect both wanting *more room than 80* -- possibly by very
different amounts, invisible because both hit the same wall -- such
that widening the prior would reveal beta=8 pushing further than
beta=4, and would change the recovered beta?

## Method
Same model/pipeline, tau prior widened to LogUniform(2, 500)
(`ar1_hmc_v2_widertauprior.py`). beta=4.0 and beta=8.0 only, N=10
realisations each (a quick check, not a full statistical ensemble),
capped at 3 retry attempts per realisation (reduced from the usual 15 --
Nozhan's explicit instruction: don't burn hours chasing frozen chains on
an exploratory check; skip and move on).

## Result

**tau moved dramatically, but identically for both beta values:**

| | n healthy / 10 | tau mean | tau std |
|---|---|---|---|
| beta=4.0 | 7 | 467.83 | 9.24 |
| beta=8.0 | 6 | 462.48 | 23.03 |

Both converge to ~460-470 -- statistically indistinguishable from each
other, same as they were statistically indistinguishable at ~79.5 under
the old prior. Widening the prior did not reveal beta=8 wanting *more*
tau than beta=4; both moved together to the same new location.

**Recovered beta barely changed at all, despite tau moving ~6x:**

| | Old prior (tau<=80) | New prior (tau<=500) |
|---|---|---|
| beta=4.0 recovered | mean=3.54, median=3.59 (N=40) | mean=3.56, median=3.57 (n=7) |
| beta=8.0 recovered | mean=3.30, median=3.37 (N=40) | mean=3.33, median=3.31 (n=6) |

The beta=8-below-beta=4 gap is unchanged (0.24 before, 0.23 now).

## Conclusion

**This rules out the tau-prior boundary as the explanation** -- both for
the beta>2 bias itself and specifically for why beta=8 recovers lower
than beta=4. If the boundary were hiding a real difference in how much
tau each beta value "wants," we would expect (a) different tau values
between beta=4 and beta=8 once given room, and/or (b) a change in
recovered beta once tau moved. Neither happened. tau moving from ~80 to
~465 left the periodogram-fit-based recovered beta essentially
unchanged, meaning the recovered beta is not sensitive to the specific
value of tau once tau is "large enough" -- physically sensible, since
once the AR(1) bend frequency ($f_c = 1/(2\pi\tau)$) is already below
the lowest frequency in the observed periodogram, pushing it even lower
does not change the PSD shape within the observed frequency range very
much; the observed periodogram is already deep in the same asymptotic
regime either way.

**What this means for the open question:** the beta=8-vs-beta=4
non-monotonic reversal is not explained by tau, mean, or var (all
statistically indistinguishable between the two cases, under both the
narrow and the wide prior). Whatever is actually driving it must be
downstream of the model's own posterior summary -- most plausibly in
how the periodogram-fit interacts with the fine-grained shape of the
reconstructed light curve (not its summary statistics), consistent with
the earlier finding that a single-draw fit is highly sensitive to noise
in the sparsest, lowest-frequency periodogram bins. This has not been
tested directly and remains an open question, explicitly out of scope
for this paper.

## Data quality note
4/10 (beta=4.0) and 4/10 (beta=8.0) realisations exhausted the reduced
3-attempt retry budget without reaching a healthy n_unique(tau)>=1000 and
were excluded rather than retried further, per instruction. This is a
substantially higher exclusion rate than the standard-prior runs (which
used a 15-attempt budget and excluded ~0-1 per 40), consistent with the
wider prior being genuinely harder to sample, as anticipated when this
test was designed.

## Scripts
- `ar1_hmc_v2_widertauprior.py`, `regenerate_ensemble_realizations.py --model widertauprior --max-retries 3` (in `Appendix_A/reproduction_2026-07-20/`)
- `fit_widertauprior_test.py` (same directory)

## Follow-up (2026-07-26): does bend-frequency position explain the elevated recovered beta?

### Question
Nozhan's observation: at tau~460-500, the AR(1) bend frequency
f_bend = 1/(2*pi*tau) should sit around 3-4e-4 Hz, while the periodogram's
lowest resolvable frequency is f_min = 1/T = 1e-3 Hz (T=1000). If
f_bend < f_min, every one of the 9 fitting bins sits on the declining
("slope") side of the bend, none straddle it. Does the earlier
hypothesis -- that noisy, sparse bins straddling the bend destabilise
the fit -- still make sense once there's no bend-straddling happening
at all?

### Calculation (exact, not order-of-magnitude)
Using the actual 9 log-spaced bin centers from `bin_periodogram()`
(T=1000: centers at 0.00159, 0.00345, 0.00751, 0.01632, 0.03549,
0.07718, 0.16783, 0.36497 Hz) and the analytic local log-log slope of
an AR(1)/Lorentzian PSD, `local_slope(f) = -2*(f/f_bend)^2 / (1+(f/f_bend)^2)`:

| tau | f_bend (Hz) | bin 1 local slope | bin 2 local slope | bins 3-8 |
|---|---|---|---|---|
| 80  | 1.989e-3 | -0.78 (bin 1 is BELOW the bend -- flat side) | -1.50 (transitional) | -1.87 to -2.00 |
| 460 | 3.460e-4 | -1.91 (already near-asymptotic) | -1.98 | -2.00 (all) |

**Confirmed: Nozhan's arithmetic is correct, and it goes further than
"no bins straddle the bend" at tau~460.** At tau=80, there is real,
substantial curvature spanning nearly the full 0-to-2 slope range across
the 9 bins (bin 1 is literally on the flat side). At tau~460, the *true*
underlying PSD is, for all practical purposes, already a clean, unbent
power law at slope ~2 across the entire observed band -- there is
essentially no curvature left for a bend-position effect to act on.

### Conclusion: bend position is not the explanation
Since the recovered beta is essentially unchanged between tau=80 (real
curvature present) and tau~460 (no curvature left, true shape already
slope~2), bend-frequency position/bin-straddling cannot be the
mechanism producing the elevated (~3.3-3.6) recovered beta. If it were,
removing the curvature entirely (tau~460 case) should have pulled the
fit closer to the true shape's slope (~2), not left it unchanged. This
sharpens (not just repeats) the tau-invariance conclusion above: it's
not just that recovered beta doesn't depend on tau, it's that recovered
beta doesn't depend on tau even though the *true underlying curve being
fit* changes from "meaningfully bent" to "already unbent" between the
two cases.

### Follow-up question: is it "noise", and does 40/40 unanimity rule that out?
Nozhan's pushback: if the explanation is periodogram sampling noise
(chi-squared(2) scatter, worst in the sparsest bins), why does averaging
over 40 independent realizations not wash it out? The whole point of
running N=40 (as opposed to one realization) was specifically to
distinguish "unlucky single draw" from "real bias" -- and it worked
exactly as intended for beta=1.7 (24/40 below true, close to a 50/50
split, consistent with genuine random scatter around an unbiased
center). beta=4 and beta=8 showing 40/40 in the *same direction* is not
what random, zero-mean noise looks like, regardless of how large that
noise is per realization.

**This is correct, and it means "periodogram sampling noise" (as in,
literal per-draw random scatter) is not, by itself, an adequate
explanation.** The more defensible version of the claim is not "random
noise that happens to survive averaging" (incoherent -- random noise is
exactly what averaging over independent realizations is supposed to
cancel) but **a systematic bias in the periodogram-fitting *method*
itself**, which would appear consistently across every realization
*because it is a property of the estimator, not of any one dataset's
luck*. Fitting a power law via (weighted) least-squares to a
chi-squared(2)-distributed periodogram is a known, studied source of
systematic bias in the X-ray timing literature (this is part of why
Vaughan 2003 and Barret & Vaughan 2012 -- both already cited in this
paper -- discuss binning and fitting methodology at all; the underlying
chi-squared(2) sampling distribution is not symmetric, and naive
log-power vs. log-frequency least-squares fitting of it is a recognised
source of bias). If that bias happens to be directionally worse for
steep true slopes specifically, it would explain all three observed
features at once:
- **40/40 unanimity** (beta=4, beta=8): a property of the method,
  recurring every time, not random per-realization luck.
- **tau-invariance**: the bias lives in the curve-fitting step applied
  to the periodogram, not in whatever generated the periodogram (AR(1)
  reconstruction at tau=80 or tau=460) -- so changing tau doesn't touch
  it.
- **beta=1.7 showing no significant bias** (24/40, ~50/50): a
  shallower true slope, plausibly outside or at the edge of the regime
  where this kind of fitting bias is severe.

**Status of this hypothesis: plausible and consistent with all
available evidence, but NOT verified.** It has not been checked against
the actual bias formulas in the literature (Vaughan 2003 / Barret &
Vaughan 2012 discuss this problem but a direct quantitative comparison
has not been done here), and no alternative/bias-corrected estimator has
been tried to see if the effect shrinks or disappears. This remains a
hypothesis for a future, dedicated investigation, not a confirmed
mechanism -- consistent with how this whole beta=8 reversal is scoped
(open question, beyond the scope of the current paper).

## Status
Run and analysed 2026-07-25/26. Confirms this is a genuine open question
rather than an artifact of the tau prior's original boundary. Bend-
frequency/local-slope calculation and the "systematic bias vs. random
noise" distinction added 2026-07-26, in response to Nozhan's direct
challenge to the periodogram-noise explanation -- corrected an imprecise
earlier framing ("just noise") that didn't actually account for the
40/40 unanimity across independent realizations.
