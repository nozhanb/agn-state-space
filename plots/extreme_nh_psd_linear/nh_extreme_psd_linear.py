"""
nh_extreme_psd_linear.py
=========================
Single PSD panel comparing the extreme-β NH sample (β≈4) against the true
spectrum, with BOTH axes in LINEAR scale so the slope difference is visible.

Keeps:
  - Red squares  + black dashed line  : true binned PSD + power-law fit (β≈1)
  - Orange circles + orange solid line: extreme sample binned PSD + fit (β≈4)
"""

import matplotlib
matplotlib.use("Agg")
import numpy
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "axes.linewidth":    1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size":  6,
    "ytick.major.size":  6,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "axes.labelweight":  "bold",
})

# ── Config ────────────────────────────────────────────────────────────────────
NH, TAU, PHI = 21, 2, -3
NBINS        = 7
RANDOM_SEED  = 42
N_DRAW       = 1000
RESULTS_DIR  = "results"

INFERENCE_FILE = (
    "inference_data/"
    "synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_"
    f"stochastic_Nh_{NH}_and_Tau_{TAU}_and_phi_{PHI}_steps_100_iter_6000_"
    "spectrum_500_redshift_0108.npz"
)
CACHE_SAMPLES = f"results/cache_sample_psds_Nh{NH}_Tau{TAU}_Phi{PHI}_n1000_seed42.npz"
CACHE_TRUE    = f"results/cache_true_fits_Nh{NH}_Tau{TAU}_Phi{PHI}.npz"

# ── Load caches ───────────────────────────────────────────────────────────────
cache_s = numpy.load(CACHE_SAMPLES)
nh_freq_nz     = cache_s["nh_freq_nz"]
nh_binned_means= cache_s["nh_binned_means"]   # (1000, n_bins)
nh_bin_centers = cache_s["nh_bin_centers"]
nh_betas       = cache_s["nh_betas"]

cache_t = numpy.load(CACHE_TRUE)
true_bin_centers = cache_t["nh_bin_centers"]
true_psd_means   = cache_t["nh_psd_means"]
true_popt        = cache_t["nh_popt"]
true_beta        = float(cache_t["nh_beta"])

# ── Identify extreme sample ───────────────────────────────────────────────────
extreme_idx  = int(numpy.argmin(numpy.abs(nh_betas - 4.0)))
extreme_beta = float(nh_betas[extreme_idx])
extreme_bins = nh_binned_means[extreme_idx]

print(f"Extreme sample index={extreme_idx},  β={extreme_beta:.3f}")
print(f"True spectrum        β={true_beta:.3f}")

# ── Power-law model ───────────────────────────────────────────────────────────
def power_law_func(freq, a, b, c):
    return numpy.log10(a * freq ** (-b) + c)

def power_law_linear(freq, a, b, c):
    return a * freq ** (-b) + c

# Fit to extreme bins
try:
    popt_ext, _ = curve_fit(
        power_law_func,
        nh_bin_centers,
        numpy.log10(extreme_bins),
        p0=[1e-3, extreme_beta, 1e-6],
        bounds=([1e-12, 1e-6, 1e-12], [1e6, 10.0, 1e6]),
        maxfev=20000,
    )
except Exception as e:
    print(f"Fit failed: {e}")
    popt_ext = None

# Dense frequency grid for smooth fit curves
freq_dense = numpy.linspace(nh_freq_nz.min(), nh_freq_nz.max(), 500)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# True binned PSD — red squares
ax.plot(true_bin_centers, true_psd_means,
        "s", color="crimson", markersize=8, zorder=3,
        label=rf"True binned PSD  ($\beta={true_beta:.2f}$)")

# True power-law fit — black dashed
fit_true_dense = power_law_linear(freq_dense, *true_popt)
ax.plot(freq_dense, fit_true_dense,
        "--", color="black", linewidth=2.0, zorder=4,
        label=rf"Power-law fit, true  ($\beta={true_beta:.2f}$)")

# Extreme binned PSD — orange circles
ax.plot(nh_bin_centers, extreme_bins,
        "o", color="darkorange", markersize=8, zorder=5,
        markeredgecolor="saddlebrown", markeredgewidth=0.8,
        label=rf"Extreme binned PSD  ($\beta={extreme_beta:.2f}$)")

# Extreme power-law fit — orange solid line
if popt_ext is not None:
    fit_ext_dense = power_law_linear(freq_dense, *popt_ext)
    ax.plot(freq_dense, fit_ext_dense,
            "-", color="darkorange", linewidth=2.0, zorder=6,
            label=rf"Power-law fit, extreme  ($\beta={extreme_beta:.2f}$)")

ax.set_xscale("log")
ax.minorticks_on()
ax.tick_params(which="major", length=8, width=1.5)
ax.tick_params(which="minor", length=5, width=1.0)
ax.set_xlabel("Frequency [Hz]", fontsize=13)
ax.set_ylabel("PSD  (linear scale)", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(bottom=0)

fig.tight_layout()

base = f"results/nh_extreme_psd_linear_Nh{NH}_Tau{TAU}_Phi{PHI}"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    fig.savefig(f"{base}.{ext}", bbox_inches="tight", **kw)
    print(f"Saved → {base}.{ext}")

plt.close(fig)
print("Done.")
