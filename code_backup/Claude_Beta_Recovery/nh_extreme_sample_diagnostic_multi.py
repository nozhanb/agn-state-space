"""
nh_extreme_sample_diagnostic_multi.py
======================================
Generates the 1-row × 3-panel diagnostic plot for any specified draw index.
Produces individual plots for indices 382, 219, and 944.

Panel layout (same as nh_extreme_sample_diagnostic.py):
  Left   : NH time-series  (grey normal samples, orange extreme, black mean, crimson true)
  Middle : PSD              (grey halo, true red squares + black dashed fit,
                             orange extreme dots + circles + orange fit)
  Right  : β histogram      (steelblue, median black, true crimson dashed,
                             orange line at this sample's β)
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

# ── Configuration ─────────────────────────────────────────────────────────────
NH, TAU, PHI    = 21, 2, -3
NBINS           = 7
N_DRAW          = 1000
RANDOM_SEED     = 42
RESULTS_DIR     = "results"
N_GREY_TRACES   = 50

INFERENCE_FILE = (
    "inference_data/"
    "synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_"
    f"stochastic_Nh_{NH}_and_Tau_{TAU}_and_phi_{PHI}_steps_100_iter_6000_"
    "spectrum_500_redshift_0108.npz"
)
CACHE_SAMPLES = f"results/cache_sample_psds_Nh{NH}_Tau{TAU}_Phi{PHI}_n1000_seed42.npz"
CACHE_TRUE    = f"results/cache_true_fits_Nh{NH}_Tau{TAU}_Phi{PHI}.npz"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data once ─────────────────────────────────────────────────────────────
params    = numpy.load(INFERENCE_FILE, allow_pickle=True)
sim_file  = (
    "simulation_data/"
    "synthetic_count_NH_and_phi_spec_visualisation_SDSSJ0932+0405_multiple_"
    f"simulated_stochastic_Nh_{NH}_and_Tau_{TAU}_and_phi_{PHI}.npz"
)
sim       = numpy.load(sim_file, allow_pickle=True)

nh_y      = numpy.swapaxes(params["nh_y"], 0, 1).astype("float64")  # (100, 6000)
true_nh   = sim["nh_process"].astype("float64")                      # (100,)
T         = nh_y.shape[0]
steps     = numpy.arange(T)

rng       = numpy.random.default_rng(RANDOM_SEED)
draw_idx  = rng.choice(nh_y.shape[1], size=N_DRAW, replace=False)
nh_y_draw = nh_y[:, draw_idx]   # (100, 1000)
nh_mean   = nh_y.mean(axis=1)   # (100,) over all 6000

# Caches
cache_s        = numpy.load(CACHE_SAMPLES)
nh_freq_nz     = cache_s["nh_freq_nz"]
nh_all_psds    = cache_s["nh_all_psds"]       # (1000, n_freq)
nh_bin_centers = cache_s["nh_bin_centers"]
nh_binned_means= cache_s["nh_binned_means"]   # (1000, n_bins)
nh_betas       = cache_s["nh_betas"]          # (1000,)

cache_t = numpy.load(CACHE_TRUE)
true_nh_res = dict(
    freq_nz     = cache_t["nh_freq_nz"],
    psd_nz      = cache_t["nh_psd_nz"],
    bin_centers = cache_t["nh_bin_centers"],
    psd_means   = cache_t["nh_psd_means"],
    popt        = cache_t["nh_popt"],
    beta        = float(cache_t["nh_beta"]),
)
true_nh_beta_polyfit = float(cache_t["nh_beta_polyfit"])

# Percentile halos
bin_p5,  bin_p50, bin_p95 = [
    numpy.nanpercentile(nh_binned_means, q, axis=0) for q in (5, 50, 95)]

# ── Helpers ────────────────────────────────────────────────────────────────────
def power_law_func(freq, a, b, c):
    return numpy.log10(a * freq ** (-b) + c)

def power_law_linear(freq, a, b, c):
    return a * freq ** (-b) + c

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

# ── Per-sample plot function ───────────────────────────────────────────────────
def make_diagnostic_plot(draw_index):
    beta        = float(nh_betas[draw_index])
    traj        = nh_y_draw[:, draw_index]
    raw_psd     = nh_all_psds[draw_index]
    binned_psd  = nh_binned_means[draw_index]

    popt_ext = fit_popt(nh_bin_centers, binned_psd, beta)
    freq_dense = numpy.linspace(nh_freq_nz.min(), nh_freq_nz.max(), 500)

    print(f"\n── draw_index={draw_index},  β={beta:.4f} ──")
    sentinel = round(0.3 / numpy.log(10), 4)
    # recompute sigmas for diagnostics
    bins_edges = numpy.logspace(
        numpy.log10(nh_freq_nz.min()), numpy.log10(nh_freq_nz.max()), NBINS)
    freq_raw, psd_raw = signal.periodogram(traj, fs=1, nfft=T)
    freq_nz_raw = freq_raw[freq_raw > 0]
    psd_nz_raw  = psd_raw[freq_raw > 0]
    for i in range(len(bins_edges)-1):
        idx = ((freq_nz_raw >= bins_edges[i]) & (freq_nz_raw <= bins_edges[i+1])
               if i == len(bins_edges)-2
               else (freq_nz_raw >= bins_edges[i]) & (freq_nz_raw < bins_edges[i+1]))
        n = idx.sum()
        if n < 1: continue
        m     = numpy.mean(psd_nz_raw[idx])
        s_lin = numpy.std(psd_nz_raw[idx]) if n > 1 else 0.3 * m
        s_log = max(s_lin / (m * numpy.log(10)), 1e-4)
        flag  = " <-- sentinel" if round(s_log,4)==sentinel else ""
        print(f"  Bin {i+1}: n={n:2d}  PSD={m:.4e}  sigma={s_log:.4f}  "
              f"weight={1/s_log**2:.1f}{flag}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 4.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax_ts   = fig.add_subplot(gs[0, 0])
    ax_psd  = fig.add_subplot(gs[0, 1])
    ax_beta = fig.add_subplot(gs[0, 2])

    # Panel 1: Time series
    rng2     = numpy.random.default_rng(99)
    grey_idx = rng2.choice(
        [i for i in range(N_DRAW) if i != draw_index],
        size=N_GREY_TRACES, replace=False)
    ax_ts.plot(steps, nh_y_draw[:, grey_idx],
               color="silver", alpha=0.35, linewidth=0.6, zorder=1)
    ax_ts.plot(steps, nh_mean,
               color="black", linewidth=2.0, zorder=3, label="Posterior mean")
    ax_ts.plot(steps, true_nh,
               color="crimson", linewidth=1.5, linestyle="--", zorder=4,
               label="True trajectory")
    ax_ts.plot(steps, traj,
               color="darkorange", linewidth=2.2, zorder=5,
               label=rf"Sample {draw_index}  ($\beta={beta:.2f}$)")
    ax_ts.set_xlabel("Time step", fontsize=12)
    ax_ts.set_ylabel(r"log $N_H$", fontsize=12)
    ax_ts.legend(fontsize=9, loc="best")

    # Panel 2: PSD
    ax_psd.fill_between(nh_bin_centers, bin_p5, bin_p95,
                        color="grey", alpha=0.55, zorder=1,
                        label="5%–95% binned PSDs (posterior)")
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
    ax_psd.plot(nh_freq_nz, raw_psd,
                ".", color="darkorange", markersize=3, alpha=0.6, zorder=5,
                label=rf"Sample {draw_index} periodogram")
    ax_psd.plot(nh_bin_centers, binned_psd,
                "o", color="darkorange", markersize=7, zorder=6,
                markeredgecolor="saddlebrown", markeredgewidth=0.8)
    if popt_ext is not None:
        fit_ext = 10 ** power_law_func(nh_freq_nz, *popt_ext)
        ax_psd.plot(nh_freq_nz, fit_ext,
                    "-", color="darkorange", linewidth=2.0, zorder=6,
                    label=rf"Power-law fit ($\beta={beta:.2f}$)")
    ax_psd.set_xscale("log")
    ax_psd.set_yscale("log")
    ax_psd.minorticks_on()
    ax_psd.tick_params(which="major", length=8, width=1.5)
    ax_psd.tick_params(which="minor", length=5, width=1.0)
    ax_psd.set_xlabel("Frequency [Hz]", fontsize=12)
    ax_psd.set_ylabel("PSD", fontsize=12)
    ax_psd.legend(fontsize=8, loc="lower left")

    # Panel 3: β histogram
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
    ax_beta.axvline(beta, color="darkorange", linewidth=2.5, linestyle="-",
                    label=rf"Sample {draw_index}  $\beta={beta:.2f}$")
    ax_beta.set_xlabel(r"$\beta$", fontsize=13)
    ax_beta.set_ylabel("Density", fontsize=12)
    ax_beta.legend(fontsize=9)

    # Save
    base = f"{RESULTS_DIR}/nh_extreme_diagnostic_Nh{NH}_Tau{TAU}_Phi{PHI}_idx{draw_index}"
    for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight", **kw)
        print(f"  Saved → {base}.{ext}")
    plt.close(fig)


# ── Run for all three indices ──────────────────────────────────────────────────
for idx in [382, 219, 944]:
    make_diagnostic_plot(idx)

print("\nDone.")
