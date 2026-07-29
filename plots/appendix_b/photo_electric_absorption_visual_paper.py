"""
photo_electric_absorption_visual_paper.py
==========================================
Publication-quality observed-flux plot.

Key message: intrinsic absorption is negligible for N_H <= 10^19 cm^-2.
Curves for N_H = 10^10 and 10^15 are indistinguishable from 10^19,
while N_H = 10^21 shows clear soft-X-ray suppression.

No Milky Way foreground included — intrinsic source absorption only.

Data files (relative to this directory)
----------------------------------------
  ./data/photo_electric_sigma_redshift_0108.npz   Wisconsin cross-sections at z=0.0108
  ./data/fake_count.npz                           energy grid

Output
------
  ./observed_flux_NH_comparison.pdf
  ./observed_flux_NH_comparison.png  (300 dpi)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — no plt.show() hang
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = "./data"
RESULTS_DIR = "."
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
data_sigma = np.load(f"{DATA_DIR}/photo_electric_sigma_redshift_0108.npz")
data_obs   = np.load(f"{DATA_DIR}/fake_count.npz")

sigma   = data_sigma["sigma"].astype("float64")   # rest-frame cross-section (z=0)
energy  = data_obs["energy"].astype("float64")    # energy grid (keV)

# ── Model parameters ───────────────────────────────────────────────────────
GAMMA  = 2.0    # photon index (Γ)
LOG_K  = -4.0   # log10 normalisation K
# No redshift: we are computing the intrinsic absorbed spectrum at the source.
# Photons are at their rest-frame energies, so z=0 and sigma (not sigma_redshift).

# ── Helper functions ───────────────────────────────────────────────────────
def power_law(gamma, K, energy):
    """Intrinsic (rest-frame) power-law photon flux."""
    return K * energy ** (-gamma)

def absorption_factor(nh, sigma):
    """Photoelectric absorption transmission: exp(-N_H * sigma(E))."""
    return np.exp(-float(nh) * sigma)

# ── Base power-law (no absorption) ────────────────────────────────────────
flux_pl = power_law(GAMMA, 10**LOG_K, energy)

# ── N_H values ─────────────────────────────────────────────────────────────
# 10^21: clearly absorbed (contrast case, above threshold)
# 10^19: at the detection threshold
# 10^15, 10^10: well below threshold — should overlap with 10^19
nh_values = [1e21, 1e19, 1e15, 1e10]
nh_labels = [
    r"$N_H = 10^{21}$ cm$^{-2}$",
    r"$N_H = 10^{19}$ cm$^{-2}$  (threshold)",
    r"$N_H = 10^{15}$ cm$^{-2}$",
    r"$N_H = 10^{10}$ cm$^{-2}$",
]
colors  = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
lstyles = ["-",       "-",       "--",      "-."]
lwidths = [2.5,       2.2,       2.0,       2.0]

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))

for nh, label, color, ls, lw in zip(
    nh_values, nh_labels, colors, lstyles, lwidths
):
    trans = absorption_factor(nh, sigma)
    flux  = flux_pl * trans
    ax.plot(energy, flux, color=color, lw=lw, ls=ls, label=label, zorder=3)

# ── Axes ───────────────────────────────────────────────────────────────────
# No title, per project plot-style convention (see plots/appendix_a/plot_light_curves.py).
ax.set_yscale("log")
ax.set_xlim(energy[0], energy[-1])
ax.set_xlabel(r"Energy (keV)", fontsize=15)
ax.set_ylabel(r"$F_E$ (photons keV$^{-1}$ cm$^{-2}$ s$^{-1}$)", fontsize=15)

ax.legend(loc="lower right", framealpha=0.92, edgecolor="gray", fontsize=13)
ax.minorticks_on()
ax.tick_params(which="major", labelsize=13, length=9, width=1.5)
ax.tick_params(which="minor", length=5, width=1.2)

plt.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────
pdf_path = f"{RESULTS_DIR}/observed_flux_NH_comparison.pdf"
png_path = f"{RESULTS_DIR}/observed_flux_NH_comparison.png"
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved:\n  {pdf_path}\n  {png_path}")
plt.close()
