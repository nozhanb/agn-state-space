"""
Step 2 for the beta=3/4 N=20 realization-ensemble test -- identical
method to fit_beta1.7_N20_ensemble.py, applied to both beta=3.0 and
beta=4.0. For each of the 20 short-chain realizations per beta: thin the
2000 posterior draws by 40 (same thinning as beta=1.7, for consistency),
fit each thinned draw (periodogram -> 9-bin -> multi-start curve_fit),
take the MEDIAN of that within-realization distribution as the
realization's point estimate. Collect the 20 medians into the
across-realization ensemble and summarize it.

Run AFTER fix_low_ess_beta34_N20.py has replaced the 3 flagged
low-n_unique realizations (beta=3/realization_01, beta=4/realization_08,
beta=4/realization_14).
"""
import numpy
from scipy import signal, stats
from scipy.optimize import curve_fit

TIME_STEP = 1000
N_REALIZATIONS = 20
N_BINS = 9
THIN = 40
LOWER_BOUNDS = [1e-10, 1e-10, 1e-10]
UPPER_BOUNDS = [1e3, 10.0, 1e3]
B0_STARTS = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0]


def power_law_func(freq, a, b, c):
    return numpy.log10(a * freq ** (-b) + c)


def periodogram_nonzero(series, nfft):
    freq, psd = signal.periodogram(series, fs=1, nfft=nfft)
    valid = freq > 0
    return freq[valid], psd[valid]


def bin_periodogram(freq, psd, n_bins):
    bins = numpy.logspace(numpy.log10(freq.min()), numpy.log10(freq.max()), n_bins)
    centers, means, stds = [], [], []
    for i in range(len(bins) - 1):
        idx = (freq >= bins[i]) & (freq < bins[i + 1])
        if numpy.sum(idx) > 0:
            mean_val = numpy.mean(psd[idx])
            std_val = numpy.std(psd[idx])
            means.append(mean_val)
            stds.append(std_val / (mean_val * numpy.log(10)) if std_val > 0 else 1e-3)
            centers.append((bins[i + 1] + bins[i]) / 2.0)
    return numpy.array(centers), numpy.array(means), numpy.array(stds)


def fit_one_draw(series, nfft, n_bins):
    freq, psd = periodogram_nonzero(series, nfft)
    centers, means, stds = bin_periodogram(freq, psd, n_bins)
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
    if best_popt is None:
        raise RuntimeError("all multi-start fits failed")
    return best_popt


for beta_true, data_dir, out_name in [
    (3.0, "beta3_N20_shortchain_data", "beta3_N20_realization_medians.npy"),
    (4.0, "beta4_N20_shortchain_data", "beta4_N20_realization_medians.npy"),
]:
    print(f"\n########## beta_true={beta_true} ##########", flush=True)
    realization_medians = []

    for i in range(N_REALIZATIONS):
        data = numpy.load(f"{data_dir}/realization_{i:02d}.npz", allow_pickle=True)
        posterior_sample = data["flux_predicted"].astype("float64")
        posterior_sample = posterior_sample - posterior_sample.mean()
        thinned = posterior_sample[:, ::THIN]
        n_draws = thinned.shape[1]

        b_list = []
        for j in range(n_draws):
            try:
                popt = fit_one_draw(thinned[:, j], TIME_STEP, N_BINS)
                b_list.append(popt[1])
            except RuntimeError:
                continue
        b_array = numpy.array(b_list)
        median_b = numpy.median(b_array)
        realization_medians.append(median_b)
        print(f"realization {i}: n_thinned_draws={n_draws}  median_beta={median_b:.4f}  "
              f"within_std={b_array.std():.4f}", flush=True)

    realization_medians = numpy.array(realization_medians)
    numpy.save(out_name, realization_medians)

    ens_mean = realization_medians.mean()
    ens_median = numpy.median(realization_medians)
    ens_std = realization_medians.std()
    ens_skew = stats.skew(realization_medians)
    p5, p95 = numpy.percentile(realization_medians, [5, 95])

    print(f"\n=== Across-realization ensemble, beta_true={beta_true} (n={N_REALIZATIONS}) ===", flush=True)
    print(f"mean={ens_mean:.4f}  median={ens_median:.4f}  std={ens_std:.4f}  skew={ens_skew:.3f}", flush=True)
    print(f"5-95% range=[{p5:.4f},{p95:.4f}]  true_in_2std=["
          f"{ens_mean-2*ens_std:.4f},{ens_mean+2*ens_std:.4f}]->"
          f"{ens_mean-2*ens_std <= beta_true <= ens_mean+2*ens_std}", flush=True)

print("\nDone.", flush=True)
