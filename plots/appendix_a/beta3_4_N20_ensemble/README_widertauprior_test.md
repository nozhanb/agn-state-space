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

## Follow-up 2 (2026-07-26): what exactly is the "fitting bias", concretely?

Nozhan asked for the mechanism to be spelled out precisely rather than
gestured at, and for the Vaughan2003 / BarretVaughan2012 citations to be
checked rather than just invoked by reputation (both are already cited
elsewhere in this paper -- Vaughan2003 for the log-binning scheme,
BarretVaughan2012 for the periodogram's chi-squared(2) sampling
distribution -- but citing them for a *specific bias claim* needed to
actually be checked, not assumed).

### What a web search of the literature confirms
Vaughan's own work does document a real, quantified bias in fitting the
*log* of a periodogram, not just the general "periodograms are noisy"
fact. The mechanism: a periodogram value at a given frequency is
distributed as (true PSD) x Exponential(1) (equivalent to chi-squared(2)/2
-- exactly the BarretVaughan2012 fact already used in this paper).
Because log() is concave, E[log(X)] != log(E[X]) (Jensen's inequality) --
the expectation of the log is systematically *below* the log of the
expectation. For a single (unbinned, N=1) exponential(1) draw, this gap
has an exact closed form tied to the Euler-Mascheroni constant gamma
(~0.5772): in base-10 logs the bias works out to a constant offset of
about **-0.2507**. This specific constant shows up in the literature
(traced via web search, not confirmed against the primary PDF text
directly) as a correction applied to remove exactly this bias when
fitting log-periodograms.

### Why this could plausibly steepen (not just offset) the fitted slope
A single, frequency-independent bias would only shift the fitted
intercept (the `a` parameter), not the slope (`beta`) -- so on its own,
this wouldn't explain anything. The reason it's a serious candidate
mechanism for *our specific setup* is that **the bias magnitude
shrinks as more raw points are averaged into a bin, and our 9 bins are
wildly unequal in point count**: 2 raw points in the lowest-frequency
bin vs. ~270 in the highest (see the "Coarser binning" discussion in
the Appendix A text, `tex/july_16_2026.tex`). Averaging N iid
Exponential(1) draws gives a Gamma(N,1)/N-scaled distribution, and the
log-bias for that average is `psi(N) - log(N)` (digamma function) --
large and negative for small N (bin 1, N=2), shrinking toward ~0 for
large N (bin 8, N=270). So the low-frequency bins are pulled down (in
log-power) systematically more than the high-frequency bins. If the
low-frequency end of the fit is pulled down relative to the
(near-unbiased) high-frequency end, that widens the apparent decline
from low to high frequency -- i.e. it could steepen the fitted slope,
in the same direction as what we actually observe (fitted beta > true
beta, consistently, for beta>2).

**This directional argument was reasoned out in this conversation, not
verified analytically or numerically at the time it was proposed.**
Whether it actually produces a slope bias of the observed size (and in
the observed direction) for our exact 9-bin, T=1000 setup needed to be
tested directly -- see the next section.

### The direct test
Isolate the fitting-methodology bias from everything else (no AR(1), no
HMC, no Poisson counts, no colorednoise generation): for a *known* true
PSD `P(f) = a*f^-beta + c`, draw a synthetic periodogram value at each
of the 500 raw frequencies (T=1000) as `P(f) * Exponential(1)` -- exact,
not approximate, given the BarretVaughan2012 sampling distribution --
bin with the exact same 9-bin scheme, fit with the exact same
multi-start curve_fit used everywhere else in this project, repeat 2000
times per true beta (2, 3, 4, 8), and check whether the mean/median
fitted beta comes out biased away from the true value.

Bin point counts (fixed by the frequency grid, independent of beta):
`[2, 2, 6, 12, 26, 57, 124, 270]` -- confirming the intended asymmetry
(2 points in the lowest bin, 270 in the highest) that motivated the
hypothesis.

### Result: the hypothesis does not hold up quantitatively

| true beta | fitted mean | fitted std | bias (mean-true) | sign-test p | t-test p |
|---|---|---|---|---|---|
| 2 | 1.9823 | 0.2865 | -0.018 | 8.5e-06 | 5.8e-03 |
| 3 | 2.9582 | 0.2473 | -0.042 | 5.3e-08 | 6.3e-14 |
| 4 | 3.9499 | 0.2405 | -0.050 | 9.6e-11 | 2.9e-20 |
| 8 | 7.9647 | 0.2766 | -0.035 | 0.737 (n.s.) | 1.4e-08 |

See `periodogram_fit_bias_test.pdf`/`.png` (all four points sit almost
exactly on the 1:1 line -- visually obvious that this is a much smaller
effect than the real ensemble's).

**There is a real, statistically detectable bias from the log-fitting
method itself, and it is in the same direction as the real pipeline's
bias (fitted beta slightly below true, consistently)** -- so the
mechanism proposed above is not wrong in kind. **But it is far too
small to be the explanation.** The pure-statistics bias here is a few
percent at most (-0.02 to -0.05 in absolute beta, out of true values of
2-8) and does not grow with true beta -- beta=8's bias (-0.035) is not
meaningfully larger than beta=2's (-0.018). Compare this to what the
real AR(1)/HMC pipeline actually shows: beta=4 recovers at 3.54 (a
bias of -0.46, ~11.5% miss) and beta=8 recovers at 3.30 (a bias of
-4.70, a ~59% miss). The pure-statistics effect is roughly 5-15x too
small to explain beta=4's miss and roughly 100x too small to explain
beta=8's, and critically, it does not reproduce the beta=8-worse-than-
beta=4 pattern at all (if anything, beta=8's pure-statistics bias is
smaller than beta=4's and beta=3's here).

**Conclusion: the periodogram log-fitting bias (Jensen's inequality /
unequal bin degrees of freedom) is real but ruled out as the dominant
mechanism.** It's a genuine, small, correctly-signed contaminant, not
the explanation for the large beta>2 bias or for the beta=8-vs-beta=4
reversal. Whatever is actually driving the large effect must be
something specific to the AR(1) reconstruction step itself (the actual
`flux_predicted` trajectory produced by the HMC posterior, going
through the `scan` recursion and Poisson-count fitting) -- not a
property of fitting a periodogram to an already-known, ideal power law.
This was not anticipated going in; the "pure fitting bias" hypothesis
looked, on paper, like it could plausibly explain effects of the
observed size, and it does not. The mechanism remains an open question,
now with one more concrete candidate explanation ruled out with a
direct, decisive test rather than left as an untested guess.

### Scripts and outputs
All in this directory (also present in `Appendix_A/reproduction_2026-07-20/`,
the original working copy):
- `test_periodogram_fit_bias.py`
- `periodogram_bias_test_beta{2,3,4,8}.npy` -- raw fitted-beta arrays, 2000 draws each
- `periodogram_fit_bias_test.pdf` / `.png`

## Status
Run and analysed 2026-07-25/26. Confirms this is a genuine open question
rather than an artifact of the tau prior's original boundary. Bend-
frequency/local-slope calculation and the "systematic bias vs. random
noise" distinction added 2026-07-26, in response to Nozhan's direct
challenge to the periodogram-noise explanation -- corrected an imprecise
earlier framing ("just noise") that didn't actually account for the
40/40 unanimity across independent realizations. The concrete
log-transform bias mechanism and the direct isolation test were added
the same day, in response to Nozhan asking for the mechanism to be
spelled out precisely and the literature citations checked rather than
just invoked. The direct isolation test (`test_periodogram_fit_bias.py`)
was then actually run, also 2026-07-26: it rules out the periodogram
log-fitting bias as the dominant mechanism (real, correctly-signed, but
5-100x too small to explain the observed effect, and does not reproduce
the beta=8-worse-than-beta=4 pattern). This closes off a second concrete
candidate explanation with a direct test rather than leaving it as an
untested hypothesis; the beta=8 reversal remains an open question.

**Not included in the paper (decided 2026-07-29):** the prior-boundary
test (Follow-up above) stayed in Appendix A of `july_16_2026.tex`, but
the periodogram-fitting-bias test (Follow-up 2, this section) was
written into Appendix A and then deliberately removed -- Nozhan's call
that it's a side investigation, not central to the paper's actual goal,
and that stating plainly "we do not have a confirmed explanation; this
is beyond the scope of this work" is sufficient without walking through
a second ruled-out mechanism. The test was still run in full and the
result is real (see table above); this file, `test_periodogram_fit_bias.py`,
and `periodogram_fit_bias_test.pdf`/`.png` are the record of that work
for future reference, kept here rather than in the paper.
