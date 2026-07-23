"""
Regenerate Table A.1 (tab:appendix_params in tex/july_16_2026.tex) from the
corrected AR(1)/HMC model's T=1000 output.

WHY THIS SCRIPT EXISTS
-----------------------
The currently published Table A.1 traces back (via
psd_beta_recovery_per_draw_binned_fit.py's DATA_DIR, an external path) to
Appendix_A_Inference_OutPut_red_noise_psd_recovery_beta_<B>_and_1000_time_step_and_7_unit_shift_v2.npz,
generated 2025-03 by colored_noise_light_curve_generation_with_one_beta_and_multiple_time_lengths.py,
which imports its model directly from simple_HMC.py:

    from simple_HMC import model
    ...
    y_prev = numpyro.sample("y_prev", dist.Normal(mean, 1))

That fixed-unit-variance initialization is the exact AR(1) bug found and
fixed in this session (see CORRECTION_LOG_2026-07-21.md) -- the correct
stationary variance is sigma/sqrt(1-alpha**2), not 1. So Table A.1's
currently published numbers are built on the buggy model, not the
corrected ar1_hmc_v2.py used everywhere else in this session's Appendix A
work (including plot_light_curves.py and plot_psd_beta4_recovery.py in
this same directory).

METHOD
------
Identical per-draw fitting method to
code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/psd_beta_recovery_per_draw_binned_fit.py
(periodogram -> 9 log-frequency bins -> multi-start bounded curve_fit,
kept for the same reasons documented there), applied here to the
corrected model's T=1000, beta={1.7,3.0,4.0} posteriors in data/.

Run with: /opt/anaconda3/envs/pub_one/bin/python3 generate_table_A1.py
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
SHIFT_VALUE = 7
BETAS_TRUE = [1.7, 3.0, 4.0]
N_BINS = 9

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
    return best_popt  # a, b (beta), c


def main():
    results = {}
    for beta_true in BETAS_TRUE:
        path = os.path.join(
            DATA_DIR,
            f"Appendix_A_Inference_OutPut_red_noise_psd_recovery_beta_{beta_true}_and_{TIME_STEP}_time_step_and_{SHIFT_VALUE}_unit_shift_v2scan.npz",
        )
        data = numpy.load(path, allow_pickle=True)
        posterior_sample = data["flux_predicted"].astype("float64")
        posterior_sample = posterior_sample - posterior_sample.mean()

        n_draws = posterior_sample.shape[1]
        t0 = time.time()
        b_list = []
        n_failed = 0
        for i in range(n_draws):
            try:
                _, b_fit, _ = fit_one_draw(posterior_sample[:, i], TIME_STEP, N_BINS)
                b_list.append(b_fit)
            except RuntimeError:
                n_failed += 1
        dt = time.time() - t0

        b_array = numpy.array(b_list)
        p5, median_b, p95 = numpy.percentile(b_array, [5, 50, 95])
        results[beta_true] = (p5, median_b, p95)

        print(f"beta_true={beta_true}  n_draws={n_draws}  n_failed_fits={n_failed}  "
              f"time={dt:.1f}s ({dt/n_draws*1000:.2f} ms/draw)")
        print(f"  new Table A.1 row: p5={p5:.3f}  median={median_b:.3f}  p95={p95:.3f}  (true={beta_true})")

    with open(os.path.join(OUT_DIR, "results_table_A1_corrected.txt"), "w") as f:
        f.write("beta_true  p5  median  p95\n")
        for beta_true, (p5, median_b, p95) in results.items():
            f.write(f"{beta_true}  {p5:.3f}  {median_b:.3f}  {p95:.3f}\n")
    print(f"\nSaved: {os.path.join(OUT_DIR, 'results_table_A1_corrected.txt')}")


if __name__ == "__main__":
    main()
