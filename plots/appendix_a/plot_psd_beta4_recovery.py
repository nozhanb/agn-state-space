"""
Appendix A, Fig. A.2: PSD recovery example for a single beta=4 colorednoise
realisation. Reuses the exact periodogram / 9-bin / multi-start curve_fit
methodology already established in
code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/psd_beta_recovery_per_draw_binned_fit.py
(see that script's docstring for why multi-start is necessary and why the
binning follows Vaughan et al. 2003), applied here to ONE representative
posterior draw (draw index 0) purely for illustration -- the beta value
plotted here is expected to differ somewhat from Table A.1's median, which
is computed across the full distribution of per-draw fits, not this one
draw (this is stated explicitly in the figure's LaTeX caption).

Uses the T=1000, beta=4 HMC posterior produced by the corrected AR(1)/HMC
model (ar1_hmc_v2.py) -- the same "latest results" data used for
plot_light_curves.py (Fig. A.1) in this directory.
"""
import os
import numpy
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = SCRIPT_DIR

BETA_TRUE = 4.0
TIME_STEP = 1000
N_BINS = 9
DRAW_INDEX = 0

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
    return best_popt, centers, means, stds, freq, psd


path = os.path.join(
    DATA_DIR,
    f"Appendix_A_Inference_OutPut_red_noise_psd_recovery_beta_{BETA_TRUE}_and_1000_time_step_and_7_unit_shift_v2scan.npz",
)
d = numpy.load(path, allow_pickle=True)

generated_flux = d["generated_flux"].astype("float64")
generated_flux = generated_flux - generated_flux.mean()
posterior_sample = d["flux_predicted"].astype("float64")
posterior_sample = posterior_sample - posterior_sample.mean()

draw = posterior_sample[:, DRAW_INDEX]
popt, centers, means, stds, plot_freq, plot_psd = fit_one_draw(draw, TIME_STEP, N_BINS)
yerr = means * numpy.log(10) * stds

input_freq, input_psd = periodogram_nonzero(generated_flux, TIME_STEP)

print(f"beta={BETA_TRUE} draw={DRAW_INDEX}: fitted a={popt[0]:.4g}  beta={popt[1]:.3f}  c={popt[2]:.4g}")

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(plot_freq, plot_psd, "ok", markersize=4, alpha=0.5, label="Simulated PSD (one posterior draw)")
ax.errorbar(centers, means, yerr=yerr, fmt="o", markersize=7, color="#FFA500",
            ecolor="grey", elinewidth=2.5, capsize=4, label="Binned PSD")
ax.plot(plot_freq, 10 ** power_law_func(plot_freq, *popt), "r--", linewidth=2.5,
        label=fr"Fit ($\beta$ = {popt[1]:.2f})")
ax.plot(input_freq, input_psd, "-", color="steelblue", linewidth=2, alpha=0.8,
        label="Input PSD (true colorednoise, pre-AR(1))")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency [Hz]", fontsize=15)
ax.set_ylabel("Power Spectral Density", fontsize=15)
ax.minorticks_on()
ax.tick_params(which="major", length=9, width=1.7, labelsize=13)
ax.tick_params(which="minor", length=6, width=1.2)
ax.legend(fontsize=13)
fig.tight_layout()

base_name = "app_A_fig_beta_4_v2"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    out_path = os.path.join(OUT_DIR, f"{base_name}.{ext}")
    fig.savefig(out_path, bbox_inches="tight", **kw)
    print(f"Saved -> {out_path}")
