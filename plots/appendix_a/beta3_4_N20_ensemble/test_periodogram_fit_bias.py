"""
Direct, isolated test of whether the periodogram-fitting *method* itself
(binning + multi-start curve_fit on log-power vs log-frequency) is
systematically biased for steep power-law PSDs -- with NO AR(1), NO
HMC, NO Poisson counts, NO colorednoise generation involved at all.

Mechanism being tested: a periodogram value at a given frequency is
distributed as (true PSD) * Exponential(1) (Barret & Vaughan 2012).
log() is concave, so E[log(X)] < log(E[X]) (Jensen's inequality) -- the
size of this gap shrinks as more raw points are averaged into a bin.
Our 9-bin scheme has wildly unequal point counts (2 in the lowest bin,
~270 in the highest), so if this log-bias is real, the low-frequency
end of the fit is pulled down (in log-power) more than the
high-frequency end -- which could steepen the fitted slope. This
script tests that directly: draw synthetic periodogram values for a
KNOWN true beta, bin and fit exactly as everywhere else in this
project, repeat many times, and see whether recovered beta is biased.

See README_widertauprior_test.md, "Follow-up 2", for the full reasoning
and the web-search-derived context (Vaughan 2003 / Barret & Vaughan
2012) that motivated this test.

NOTE (2026-07-29): this test was run to completion and the result is
real (see README's "Result" and "Status" sections), but it was
deliberately NOT included in the paper (july_16_2026.tex, Appendix A) --
Nozhan's call that it's a side investigation not central to the paper's
goal. Kept here as the record of completed work, for reference only.
"""
import numpy
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

T = 1000
N_BINS = 9
N_REPEATS = 2000
BETAS_TRUE = [2.0, 3.0, 4.0, 8.0]
A_TRUE = 1.0
C_TRUE = 1e-6  # small, near-negligible constant floor -- keeps the true
                # shape close to a pure power law without triggering
                # log(0) edge cases

LOWER_BOUNDS = [1e-10, 1e-10, 1e-10]
UPPER_BOUNDS = [1e3, 10.0, 1e3]
B0_STARTS = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0]

rng = numpy.random.default_rng(20260726)

# Exact same raw frequency grid as scipy.signal.periodogram(series, fs=1,
# nfft=1000) with the f=0 point excluded (periodogram_nonzero() elsewhere
# in this project) -- 500 frequencies from 1/1000 to 0.5.
raw_freq = numpy.fft.rfftfreq(T, d=1.0)[1:]
assert len(raw_freq) == 500 and abs(raw_freq[0] - 0.001) < 1e-12 and abs(raw_freq[-1] - 0.5) < 1e-12


def power_law_func(freq, a, b, c):
    return numpy.log10(a * freq ** (-b) + c)


def bin_periodogram(freq, psd, n_bins):
    bins = numpy.logspace(numpy.log10(freq.min()), numpy.log10(freq.max()), n_bins)
    centers, means, stds, npts = [], [], [], []
    for i in range(len(bins) - 1):
        idx = (freq >= bins[i]) & (freq < bins[i + 1])
        if numpy.sum(idx) > 0:
            mean_val = numpy.mean(psd[idx])
            std_val = numpy.std(psd[idx])
            means.append(mean_val)
            stds.append(std_val / (mean_val * numpy.log(10)) if std_val > 0 else 1e-3)
            centers.append((bins[i + 1] + bins[i]) / 2.0)
            npts.append(int(numpy.sum(idx)))
    return numpy.array(centers), numpy.array(means), numpy.array(stds), npts


def fit_one(centers, means, stds):
    log_means = numpy.log10(means)
    best_popt, best_chi2 = None, numpy.inf
    for b0 in B0_STARTS:
        try:
            popt, _ = curve_fit(
                power_law_func, centers, log_means,
                p0=[1e-3, b0, 1e-3], bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                sigma=stds, absolute_sigma=True, maxfev=10000,
            )
            chi2 = numpy.sum(((power_law_func(centers, *popt) - log_means) / stds) ** 2)
            if chi2 < best_chi2:
                best_popt, best_chi2 = popt, chi2
        except RuntimeError:
            continue
    return best_popt


# Report the bin point-count structure once (same for every true beta,
# since it only depends on the frequency grid / binning, not the PSD).
_, _, _, npts_report = bin_periodogram(raw_freq, numpy.ones_like(raw_freq), N_BINS)
print(f"Bin point counts (fixed, independent of beta): {npts_report}")
print(f"Total repeats per beta: {N_REPEATS}\n")

results = {}
for beta_true in BETAS_TRUE:
    true_psd = A_TRUE * raw_freq ** (-beta_true) + C_TRUE
    fitted_betas = []
    n_failed = 0
    for rep in range(N_REPEATS):
        periodogram_sample = true_psd * rng.exponential(1.0, size=len(raw_freq))
        centers, means, stds, _ = bin_periodogram(raw_freq, periodogram_sample, N_BINS)
        popt = fit_one(centers, means, stds)
        if popt is not None:
            fitted_betas.append(popt[1])
        else:
            n_failed += 1
    fitted_betas = numpy.array(fitted_betas)
    results[beta_true] = fitted_betas

    mean_b, median_b, std_b = fitted_betas.mean(), numpy.median(fitted_betas), fitted_betas.std()
    n_below = int((fitted_betas < beta_true).sum())
    binom = stats.binomtest(n_below, len(fitted_betas), 0.5)
    t_stat, t_p = stats.ttest_1samp(fitted_betas, beta_true)

    print(f"beta_true={beta_true}: n_fits={len(fitted_betas)} (failed={n_failed})")
    print(f"  mean={mean_b:.4f}  median={median_b:.4f}  std={std_b:.4f}")
    print(f"  {n_below}/{len(fitted_betas)} below true  sign_p={binom.pvalue:.3e}  ttest_p={t_p:.3e}")
    print(f"  bias (mean - true) = {mean_b - beta_true:+.4f}\n")

    numpy.save(f"periodogram_bias_test_beta{beta_true:g}.npy", fitted_betas)

# Plot: recovered vs true, same style as the real ensemble calibration plot
fig, ax = plt.subplots(figsize=(6.5, 6.5))
lims = [1.2, 9.0]
ax.plot(lims, lims, linestyle="--", color="grey", linewidth=1.5, label="perfect recovery (1:1)")
means = [results[b].mean() for b in BETAS_TRUE]
stds = [results[b].std() for b in BETAS_TRUE]
ax.errorbar(BETAS_TRUE, means, yerr=stds, fmt="D", markersize=9, capsize=5,
            color="#55A868", ecolor="#55A868", linewidth=2,
            label=f"pure periodogram-fit bias test (N={N_REPEATS} each)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel(r"true $\beta$ (pure power law, no AR(1)/HMC)", fontsize=15)
ax.set_ylabel(r"fitted $\beta$ (periodogram fit only)", fontsize=15)
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=13, length=7, width=1.3)
ax.set_aspect("equal")
fig.tight_layout()
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    path = f"periodogram_fit_bias_test.{ext}"
    fig.savefig(path, bbox_inches="tight", **kw)
    print(f"Saved -> {path}")

print("\nDone.")
