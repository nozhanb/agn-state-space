# beta3_4_snr_trend

## Question
For colorednoise inputs with beta > 2 (beta=3, beta=4), the AR(1)/HMC fit
pins tau at the upper edge of its prior instead of converging to an
interior value. Does increasing the signal-to-noise ratio (source count
rate) relax this pinning, as would be expected if the degeneracy were
simply a matter of insufficient information in the data?

## Motivation
This directly follows up on the Appendix A colorednoise validation
(`app:recovery` in the paper), which found tau pinned at its prior's
upper edge (80) for beta=3 and beta=4 with otherwise clean convergence
diagnostics (large ESS, r_hat ~= 1). To determine whether this is a
fixable information deficit or a structural property of fitting a
short-memory (AR(1)) model to a long-memory (colorednoise) process, we
swept the SNR (via a count-rate `shift_term` proxy) at shift=1,4,7 for
both beta=3 and beta=4 and tracked how the recovered tau moved relative
to the prior's edge.

## Parameters
- beta_true: 3.0, 4.0
- shift_term (SNR proxy): 1.0, 4.0, 7.0
- tau prior: LogUniform(2, 80) -- edge referenced in the right panel is 80
- Model: `ar1_hmc_v2_widerprior.py` (widened mean_param prior,
  `Uniform(-10, 20)`, needed to avoid a prior-support failure at higher
  shift values)
- shift=10 and shift=13 were also attempted but could not be reliably
  sampled (persistent frozen-chain pathology even after retries and the
  widened prior) -- excluded from this plot; see `SCRIPT_LOG.md` in
  `code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/`
  for the full debugging record.

## Script
`plot_beta3_4_snr_trend.py`

Run with:
```bash
python plot_beta3_4_snr_trend.py
```

## Data
| File | Source | Description |
|------|--------|-------------|
| `data/beta_3.0_shift_1.0.npz` | `code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/beta3_4_snr_sweep_v2_data/` | Posterior samples, beta=3 input, shift=1 |
| `data/beta_3.0_shift_4.0.npz` | same | Posterior samples, beta=3 input, shift=4 |
| `data/beta_3.0_shift_7.0.npz` | same | Posterior samples, beta=3 input, shift=7 |
| `data/beta_4.0_shift_1.0.npz` | same | Posterior samples, beta=4 input, shift=1 |
| `data/beta_4.0_shift_4.0.npz` | same | Posterior samples, beta=4 input, shift=4 |
| `data/beta_4.0_shift_7.0.npz` | same | Posterior samples, beta=4 input, shift=7 |

## Outputs
- `beta3_4_snr_trend.pdf`
- `beta3_4_snr_trend.png`

## Outcome
Higher signal-to-noise ratio pushes tau **closer** to the prior's upper
edge, not away from it, for both beta=3 and beta=4. Distance from the
edge (80 - tau_mean) shrinks monotonically with increasing shift_term:
4.32 -> 1.55 -> 0.92 (beta=3) and 2.64 -> 0.74 -> 0.28 (beta=4) across
shift=1,4,7. This is the opposite of what an information-deficit
explanation would predict, and instead supports a structural
explanation: because AR(1)'s power spectrum cannot exceed an asymptotic
slope of 2 for any tau, and `colorednoise` has no true decorrelation
timescale to recover, additional signal only sharpens the model's
confidence that an arbitrarily long tau is required -- it does not
resolve the underlying short-memory-vs-long-memory mismatch.

## LaTeX caption
```latex
Posterior mean of $\tau$ (left) and its distance from the prior's upper
edge, $80-\hat\tau$ (right), as a function of signal-to-noise ratio
(shift\_term proxy for source count rate), for $\beta=3$ and $\beta=4$
\texttt{colorednoise} inputs. Higher SNR pushes $\tau$ \emph{closer} to
the prior edge rather than away from it, indicating the $\tau$-pinning
behaviour described in Appendix~\ref{app:recovery} is not resolved by
additional information and instead reflects a structural mismatch
between the short-memory AR(1) model and the long-memory
\texttt{colorednoise} process. Signal-to-noise levels beyond shift=7
could not be reliably sampled with the current implementation.
```
