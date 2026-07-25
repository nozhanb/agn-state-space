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

## Status
Run and analysed 2026-07-25/26. Confirms this is a genuine open question
rather than an artifact of the tau prior's original boundary.
