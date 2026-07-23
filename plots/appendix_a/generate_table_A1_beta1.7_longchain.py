"""
Table A.1, beta=1.7 row only: redo using the 20,000-sample long chain,
thinned by 50 (400 draws), instead of the 2000-sample short chain.

WHY
---
generate_table_A1.py's beta=1.7 result from the short chain
(Appendix_A_..._v2scan.npz, NUM_WARMUP=1000/NUM_SAMPLES=2000) came out
implausibly tight (p5=1.586, median=1.591, p95=1.595 -- a 0.009-wide
interval). This chain is already known from this session's earlier ESS
diagnostics to have ESS~=41 for tau_param due to severe autocorrelation
(see BAYESIAN_DIAGNOSTICS_LEARNINGS.md) -- the ~2000 "draws" are mostly
near-duplicates of a much smaller number of effectively independent
samples, which understates the true within-run posterior uncertainty on
the per-draw beta fit. The long chain + thin-by-50 fix (derived earlier
this session via direct ACF computation, tau_int ~= 45-50) is the already-
established remedy for this specific beta=1.7 mixing problem; reusing it
here for Table A.1 for consistency rather than accepting the short
chain's understated spread.

Run with: /opt/anaconda3/envs/pub_one/bin/python3 generate_table_A1_beta1.7_longchain.py
"""
import os
import time
import numpy
from scipy import signal
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = SCRIPT_DIR

TIME_STEP = 1000
N_BINS = 9
THIN = 50

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


path = os.path.join(
    DATA_DIR,
    "Appendix_A_Inference_OutPut_red_noise_psd_recovery_beta_1.7_and_1000_time_step_and_7_unit_shift_v2scan_longchain.npz",
)
data = numpy.load(path, allow_pickle=True)
posterior_sample = data["flux_predicted"].astype("float64")
posterior_sample = posterior_sample - posterior_sample.mean()

n_total = posterior_sample.shape[1]
thinned_idx = numpy.arange(0, n_total, THIN)
print(f"n_total_draws={n_total}  thin={THIN}  n_thinned={len(thinned_idx)}")

t0 = time.time()
b_list = []
n_failed = 0
for i in thinned_idx:
    try:
        _, b_fit, _ = fit_one_draw(posterior_sample[:, i], TIME_STEP, N_BINS)
        b_list.append(b_fit)
    except RuntimeError:
        n_failed += 1
dt = time.time() - t0

b_array = numpy.array(b_list)
p5, median_b, p95 = numpy.percentile(b_array, [5, 50, 95])

print(f"n_fitted={len(b_array)}  n_failed_fits={n_failed}  time={dt:.1f}s ({dt/len(thinned_idx)*1000:.2f} ms/draw)")
print(f"beta=1.7 (long chain, thin={THIN}) row: p5={p5:.3f}  median={median_b:.3f}  p95={p95:.3f}  (true=1.7)")

with open(os.path.join(OUT_DIR, "results_table_A1_beta1.7_longchain.txt"), "w") as f:
    f.write("beta_true  p5  median  p95  n_draws_used  thin\n")
    f.write(f"1.7  {p5:.3f}  {median_b:.3f}  {p95:.3f}  {len(b_array)}  {THIN}\n")
print(f"Saved: {os.path.join(OUT_DIR, 'results_table_A1_beta1.7_longchain.txt')}")
