"""
nh_extreme_comparison.py
=========================
Two-row × three-column comparison of the three extreme NH samples.

Row 1: Time series  (one panel per index)
Row 2: PSD log-log  (one panel per index)

Each column = one draw index: 382, 219, 944.
All panels share the same grey halo, true spectrum reference, and colour scheme.
"""

import matplotlib
matplotlib.use("Agg")
import numpy
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.linewidth":    1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size":  6,
    "ytick.major.size":  6,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "axes.labelweight":  "bold",
})

# ── Config ────────────────────────────────────────────────────────────────────
NH, TAU, PHI  = 21, 2, -3
NBINS         = 7
N_DRAW        = 1000
RANDOM_SEED   = 42
N_GREY        = 50
RESULTS_DIR   = "results"
INDICES       = [382, 219, 944]

INFERENCE_FILE = (
    "inference_data/"
    "synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_"
    f"stochastic_Nh_{NH}_and_Tau_{TAU}_and_phi_{PHI}_steps_100_iter_6000_"
    "spectrum_500_redshift_0108.npz"
)
SIM_FILE = (
    "simulation_data/"
    "synthetic_count_NH_and_phi_spec_visualisation_SDSSJ0932+0405_multiple_"
    f"simulated_stochastic_Nh_{NH}_and_Tau_{TAU}_and_phi_{PHI}.npz"
)
CACHE_SAMPLES = f"results/cache_sample_psds_Nh{NH}_Tau{TAU}_Phi{PHI}_n1000_seed42.npz"
CACHE_TRUE    = f"results/cache_true_fits_Nh{NH}_Tau{TAU}_Phi{PHI}.npz"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
params    = numpy.load(INFERENCE_FILE, allow_pickle=True)
sim       = numpy.load(SIM_FILE,       allow_pickle=True)
nh_y      = numpy.swapaxes(params["nh_y"], 0, 1).astype("float64")
true_nh   = sim["nh_process"].astype("float64")
T         = nh_y.shape[0]
steps     = numpy.arange(T)

rng       = numpy.random.default_rng(RANDOM_SEED)
draw_idx  = rng.choice(nh_y.shape[1], size=N_DRAW, replace=False)
nh_y_draw = nh_y[:, draw_idx]
nh_mean   = nh_y.mean(axis=1)

cache_s        = numpy.load(CACHE_SAMPLES)
nh_freq_nz     = cache_s["nh_freq_nz"]
nh_all_psds    = cache_s["nh_all_psds"]
nh_bin_centers = cache_s["nh_bin_centers"]
nh_binned_means= cache_s["nh_binned_means"]
nh_betas       = cache_s["nh_betas"]

cache_t = numpy.load(CACHE_TRUE)
true_bin_centers = cache_t["nh_bin_centers"]
true_psd_means   = cache_t["nh_psd_means"]
true_popt        = cache_t["nh_popt"]
true_beta        = float(cache_t["nh_beta"])

bin_p5,  bin_p50, bin_p95 = [
    numpy.nanpercentile(nh_binned_means, q, axis=0) for q in (5, 50, 95)]

# ── Helpers ───────────────────────────────────────────────────────────────────
def power_law_func(freq, a, b, c):
    return numpy.log10(a * freq ** (-b) + c)

def fit_popt(bin_centers, bin_vals, beta_init):
    try:
        popt, _ = curve_fit(
            power_law_func, bin_centers, numpy.log10(bin_vals),
            p0=[1e-3, beta_init, 1e-6],
            bounds=([1e-12, 1e-6, 1e-12], [1e6, 10.0, 1e6]),
            maxfev=20000,
        )
        return popt
    except Exception:
        return None

# Grey trace indices (same for every panel, excluding any of the three targets)
rng2 = numpy.random.default_rng(99)
grey_pool = [i for i in range(N_DRAW) if i not in INDICES]
grey_idx  = rng2.choice(grey_pool, size=N_GREY, replace=False)

# True fit curve (dense)
freq_dense = numpy.linspace(nh_freq_nz.min(), nh_freq_nz.max(), 500)
fit_true_curve = 10 ** power_law_func(nh_freq_nz, *true_popt)

# ── Figure: 2 rows × 3 cols ───────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

for col, draw_index in enumerate(INDICES):
    beta       = float(nh_betas[draw_index])
    traj       = nh_y_draw[:, draw_index]
    raw_psd    = nh_all_psds[draw_index]
    binned_psd = nh_binned_means[draw_index]
    popt_ext   = fit_popt(nh_bin_centers, binned_psd, beta)

    ax_ts  = fig.add_subplot(gs[0, col])   # time series on top
    ax_psd = fig.add_subplot(gs[1, col])   # PSDs on bottom

    # ── Row 1: Time series ────────────────────────────────────────────────────
    ax_ts.plot(steps, nh_y_draw[:, grey_idx],
               color="silver", alpha=0.35, linewidth=0.6, zorder=1)
    ax_ts.plot(steps, nh_mean,
               color="black", linewidth=2.0, zorder=3,
               label="Posterior mean")
    ax_ts.plot(steps, true_nh,
               color="crimson", linewidth=1.5, linestyle="--", zorder=4,
               label="True")
    ax_ts.plot(steps, traj,
               color="darkorange", linewidth=2.2, zorder=5,
               label=rf"Sample {draw_index}  ($\beta={beta:.2f}$)")

    ax_ts.set_xlabel("Time step", fontsize=11)
    if col == 0:
        ax_ts.set_ylabel(r"log $N_H$", fontsize=11)
    ax_ts.legend(fontsize=8, loc="best")

    # ── Row 2: PSD ────────────────────────────────────────────────────────────
    ax_psd.fill_between(nh_bin_centers, bin_p5, bin_p95,
                        color="grey", alpha=0.55, zorder=1,
                        label="5%–95% binned PSDs")
    ax_psd.plot(cache_t["nh_freq_nz"],
                cache_t["nh_psd_nz"],
                ".", color="#DD2222", markersize=3, alpha=0.5, zorder=2,
                label="True periodogram")
    ax_psd.plot(true_bin_centers, true_psd_means,
                "s", color="crimson", markersize=7, zorder=3,
                label=rf"True binned  ($\beta={true_beta:.2f}$)")
    ax_psd.plot(nh_freq_nz, fit_true_curve,
                "--", color="black", linewidth=2.0, zorder=4,
                label="Fit (true)")
    ax_psd.plot(nh_freq_nz, raw_psd,
                ".", color="darkorange", markersize=3, alpha=0.6, zorder=5)
    ax_psd.plot(nh_bin_centers, binned_psd,
                "o", color="darkorange", markersize=7, zorder=6,
                markeredgecolor="saddlebrown", markeredgewidth=0.8,
                label=rf"Sample {draw_index} binned")
    if popt_ext is not None:
        fit_ext_curve = 10 ** power_law_func(nh_freq_nz, *popt_ext)
        ax_psd.plot(nh_freq_nz, fit_ext_curve,
                    "-", color="darkorange", linewidth=2.0, zorder=6,
                    label=rf"Fit ($\beta={beta:.2f}$)")

    ax_psd.set_xscale("log")
    ax_psd.set_yscale("log")
    ax_psd.minorticks_on()
    ax_psd.tick_params(which="major", length=8, width=1.5)
    ax_psd.tick_params(which="minor", length=5, width=1.0)
    ax_psd.set_xlabel("Frequency [Hz]", fontsize=11)
    if col == 0:
        ax_psd.set_ylabel("PSD", fontsize=11)
    ax_psd.legend(fontsize=8, loc="lower left")

# ── Save ──────────────────────────────────────────────────────────────────────
base = f"{RESULTS_DIR}/nh_extreme_comparison_Nh{NH}_Tau{TAU}_Phi{PHI}"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    fig.savefig(f"{base}.{ext}", bbox_inches="tight", **kw)
    print(f"Saved → {base}.{ext}")
plt.close(fig)
print("Done.")
