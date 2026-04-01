"""
nh_extreme_sample_diagnostic.py
================================
Diagnostic plot for the NH posterior sample with β ≈ 4.0 (index 382 in the
1000-draw sub-sample).

Figure layout  —  1 row × 3 panels
  Panel 1 (left)  : NH time-series
      - 50 random "normal" posterior samples (grey, thin)
      - the extreme-β sample trajectory (orange, thick)
      - posterior mean (black)
      - true generating trajectory (crimson dashed)

  Panel 2 (middle): Periodograms
      - 5th–95th percentile halo of all 1000 binned PSDs (grey fill)
      - true binned PSD (dark-green squares) + true power-law fit (black dashed)
      - extreme sample raw periodogram (orange dots)
      - extreme sample binned PSD (orange squares) + power-law fit (orange dashed)

  Panel 3 (right) : β histogram
      - all 1000 posterior sample betas (steelblue)
      - vertical orange line at the extreme β ≈ 4.0
      - vertical crimson dashed line at the true β
      - vertical black line at the median β
"""

import matplotlib
matplotlib.use("Agg")
import numpy
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Style ────────────────────────────────────────────────────────────────────
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

# ── Configuration ─────────────────────────────────────────────────────────────
NH  = 21
TAU = 2
PHI = -3

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

CACHE_SAMPLES = (
    f"results/cache_sample_psds_Nh{NH}_Tau{TAU}_Phi{PHI}_n1000_seed42.npz"
)
CACHE_TRUE    = f"results/cache_true_fits_Nh{NH}_Tau{TAU}_Phi{PHI}.npz"

NBINS       = 7
N_DRAW      = 1000
RANDOM_SEED = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_GREY_TRACES = 50   # normal posterior traces to show in panel 1

# ── Load data ─────────────────────────────────────────────────────────────────
params = numpy.load(INFERENCE_FILE, allow_pickle=True)
sim    = numpy.load(SIM_FILE,       allow_pickle=True)

nh_y   = numpy.swapaxes(params["nh_y"], 0, 1).astype("float64")   # (100, 6000)
true_nh = sim["nh_process"].astype("float64")                      # (100,)

T         = nh_y.shape[0]
n_samples = nh_y.shape[1]
steps     = numpy.arange(T)

rng      = numpy.random.default_rng(RANDOM_SEED)
draw_idx = rng.choice(n_samples, size=N_DRAW, replace=False)
nh_y_draw = nh_y[:, draw_idx]       # (100, 1000)
nh_mean   = nh_y.mean(axis=1)       # (100,)  — over all 6000

# ── Load cached PSD results (same as main script) ─────────────────────────────
cache_s = numpy.load(CACHE_SAMPLES)
nh_freq_nz     = cache_s["nh_freq_nz"]
nh_all_psds    = cache_s["nh_all_psds"]       # (1000, n_freq)
nh_bin_centers = cache_s["nh_bin_centers"]
nh_binned_means= cache_s["nh_binned_means"]   # (1000, n_bins)
nh_betas       = cache_s["nh_betas"]          # (1000,)

cache_t = numpy.load(CACHE_TRUE)
true_nh_res = dict(
    freq_nz    = cache_t["nh_freq_nz"],
    psd_nz     = cache_t["nh_psd_nz"],
    bin_centers= cache_t["nh_bin_centers"],
    psd_means  = cache_t["nh_psd_means"],
    popt       = cache_t["nh_popt"],
    beta       = float(cache_t["nh_beta"]),
)
true_nh_beta_polyfit = float(cache_t["nh_beta_polyfit"])

# ── Identify extreme sample ───────────────────────────────────────────────────
extreme_idx  = int(numpy.argmin(numpy.abs(nh_betas - 4.0)))
extreme_beta = float(nh_betas[extreme_idx])
extreme_traj = nh_y_draw[:, extreme_idx]          # (100,)
extreme_raw  = nh_all_psds[extreme_idx]            # raw periodogram
extreme_bins = nh_binned_means[extreme_idx]        # binned PSD

print(f"Extreme sample: draw index={extreme_idx},  β={extreme_beta:.3f}")
print(f"Trajectory mean={extreme_traj.mean():.3f},  std={extreme_traj.std():.3f}")
print(f"True trajectory: mean={true_nh.mean():.3f}, std={true_nh.std():.3f}")

# ── Power-law fit to extreme sample binned PSD ────────────────────────────────
def power_law_func(freq, a, b, c):
    return numpy.log10(a * freq ** (-b) + c)

try:
    popt_ext, _ = curve_fit(
        power_law_func,
        nh_bin_centers,
        numpy.log10(extreme_bins),
        p0=[1e-3, extreme_beta, 1e-6],
        bounds=([1e-12, 1e-6, 1e-12], [1e6, 10.0, 1e6]),
        maxfev=20000,
    )
    fit_extreme = 10 ** power_law_func(nh_freq_nz, *popt_ext)
except Exception:
    fit_extreme = None

# ── Percentile halos ──────────────────────────────────────────────────────────
bin_p5,  bin_p50, bin_p95 = [
    numpy.nanpercentile(nh_binned_means, q, axis=0) for q in (5, 50, 95)]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 4.8))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

ax_ts   = fig.add_subplot(gs[0, 0])
ax_psd  = fig.add_subplot(gs[0, 1])
ax_beta = fig.add_subplot(gs[0, 2])

# ── Panel 1: Time series ──────────────────────────────────────────────────────
# Pick N_GREY_TRACES random indices that are NOT the extreme sample
rng2     = numpy.random.default_rng(99)
grey_idx = rng2.choice(
    [i for i in range(N_DRAW) if i != extreme_idx],
    size=N_GREY_TRACES, replace=False
)

ax_ts.plot(steps, nh_y_draw[:, grey_idx],
           color="silver", alpha=0.35, linewidth=0.6, zorder=1)
ax_ts.plot(steps, nh_mean,
           color="black", linewidth=2.0, zorder=3, label="Posterior mean")
ax_ts.plot(steps, true_nh,
           color="crimson", linewidth=1.5, linestyle="--", zorder=4,
           label="True trajectory")
ax_ts.plot(steps, extreme_traj,
           color="darkorange", linewidth=2.2, zorder=5,
           label=rf"Extreme sample ($\beta={extreme_beta:.2f}$)")

ax_ts.set_xlabel("Time step", fontsize=12)
ax_ts.set_ylabel(r"log $N_H$", fontsize=12)
ax_ts.legend(fontsize=9, loc="best")

# ── Panel 2: Periodograms ─────────────────────────────────────────────────────
# Grey halo — binned PSD
ax_psd.fill_between(nh_bin_centers, bin_p5, bin_p95,
                    color="grey", alpha=0.55, zorder=1,
                    label="5%–95% binned PSDs (posterior)")

# True spectrum
ax_psd.plot(true_nh_res["freq_nz"], true_nh_res["psd_nz"],
            ".", color="#DD2222", markersize=3, alpha=0.5, zorder=2,
            label="True periodogram")
ax_psd.plot(true_nh_res["bin_centers"], true_nh_res["psd_means"],
            "s", color="crimson", markersize=7, zorder=3,
            label=rf"True binned PSD ($\beta={true_nh_res['beta']:.2f}$)")
fit_true = 10 ** power_law_func(nh_freq_nz, *true_nh_res["popt"])
ax_psd.plot(nh_freq_nz, fit_true,
            "--", color="black", linewidth=2.0, zorder=4,
            label="Power-law fit (true)")

# Extreme sample
ax_psd.plot(nh_freq_nz, extreme_raw,
            ".", color="darkorange", markersize=3, alpha=0.6, zorder=5,
            label=rf"Extreme periodogram ($\beta={extreme_beta:.2f}$)")
ax_psd.plot(nh_bin_centers, extreme_bins,
            "o", color="darkorange", markersize=7, zorder=6,
            markeredgecolor="saddlebrown", markeredgewidth=0.8)
if fit_extreme is not None:
    ax_psd.plot(nh_freq_nz, fit_extreme,
                "-", color="darkorange", linewidth=2.0, zorder=6,
                label="Power-law fit (extreme)")

ax_psd.set_xscale("log")
ax_psd.set_yscale("log")
ax_psd.minorticks_on()
ax_psd.tick_params(which="major", length=8, width=1.5)
ax_psd.tick_params(which="minor", length=5, width=1.0)
ax_psd.set_xlabel("Frequency [Hz]", fontsize=12)
ax_psd.set_ylabel("PSD", fontsize=12)
ax_psd.legend(fontsize=8, loc="lower left")

# ── Panel 3: β histogram ──────────────────────────────────────────────────────
b5, b50, b95 = numpy.percentile(nh_betas, [5, 50, 95])

ax_beta.hist(nh_betas, bins=60, color="steelblue", alpha=0.55,
             density=True, label="Posterior sample betas")
ax_beta.axvspan(b5, b95, alpha=0.18, color="steelblue",
                label=f"5%–95% CI  [{b5:.2f}, {b95:.2f}]")
ax_beta.axvline(b50, color="black", linewidth=2.0, linestyle="-",
                label=rf"Posterior median  $\beta={b50:.2f}$")
ax_beta.axvline(true_nh_beta_polyfit, color="crimson", linewidth=2.0,
                linestyle="--",
                label=rf"True $\beta={true_nh_beta_polyfit:.2f}$")
ax_beta.axvline(extreme_beta, color="darkorange", linewidth=2.5,
                linestyle="-",
                label=rf"Extreme sample  $\beta={extreme_beta:.2f}$")

ax_beta.set_xlabel(r"$\beta$", fontsize=13)
ax_beta.set_ylabel("Density", fontsize=12)
ax_beta.legend(fontsize=9)

# ── Save ──────────────────────────────────────────────────────────────────────
base = f"results/nh_extreme_sample_diagnostic_Nh{NH}_Tau{TAU}_Phi{PHI}"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    fig.savefig(f"{base}.{ext}", bbox_inches="tight", **kw)
    print(f"Saved → {base}.{ext}")

plt.close(fig)
print("Done.")
