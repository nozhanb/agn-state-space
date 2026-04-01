"""
tau_posterior_paper.py
======================
Two-panel (1 column × 2 rows) publication-quality figure showing prior vs
posterior for both timescale parameters for:

    N_H = 22,  tau = 38,  phi = -4

Plotted in tau space.
    Prior : Log-Uniform on tau in [2, 80]  →  p(tau) = 1 / (tau * ln(80/2))
    Posterior : KDE of actual 6000 HMC samples (converted from ln-tau space)

Top panel   : tau_NH
Bottom panel: tau_phi

No title — for paper use.

Output
------
  results/tau_posterior_Nh22_Tau38_phi-4_paper.pdf
  results/tau_posterior_Nh22_Tau38_phi-4_paper.png
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(
    BASE, "remote",
    "synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_"
    "stochastic_Nh_22_and_Tau_38_and_phi_-4_steps_100_iter_6000_spectrum_500"
    "_redshift_0108.npz"
)
OUT_DIR  = os.path.join(BASE, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── True injected value ────────────────────────────────────────────────────
TAU_TRUE = 38.0

# ── Prior bounds: log-uniform on tau in [2, 80] ───────────────────────────
TAU_LO   = 2.0
TAU_HI   = 80.0
LN_RANGE = math.log(TAU_HI / TAU_LO)   # ln(80/2) = ln(40)

def log_uniform_pdf(x):
    """p(tau) = 1 / (tau * ln(80/2)) for tau in [TAU_LO, TAU_HI]."""
    y = np.zeros_like(x, dtype=float)
    mask = (x >= TAU_LO) & (x <= TAU_HI)
    y[mask] = 1.0 / (x[mask] * LN_RANGE)
    return y

# ── Load actual posterior samples (stored as ln tau in the npz) ───────────
npz       = np.load(NPZ_PATH, allow_pickle=True)
tau_nh    = np.exp(npz["nh_tau"].ravel())    # convert ln(tau) → tau
tau_phi   = np.exp(npz["phi_tau"].ravel())   # convert ln(tau) → tau

# ── Credible intervals from CSV summary (consistent with paper table) ─────
# The CSV was generated from the authoritative inference run; the npz loaded
# above is from a separate run with a different random seed.  Both are
# converged (r_hat≈1) but give slightly different Monte Carlo percentiles.
# We use the CSV values for CI markers so the plot matches the table exactly.
CSV_PATH = os.path.join(BASE, "summary",
                        "summary_Nh_22_and_Tau_38_and_phi_-4.csv")
_df = pd.read_csv(CSV_PATH, index_col=0)

def tau_ci_from_csv(param):
    r = _df.loc[param]
    return (math.exp(float(r["5.0%"])),
            math.exp(float(r["median"])),
            math.exp(float(r["95.0%"])))

ci_nh_lo,  ci_nh_med,  ci_nh_hi  = tau_ci_from_csv("nh_log_tau")
ci_phi_lo, ci_phi_med, ci_phi_hi = tau_ci_from_csv("phi_log_tau")

# ── x-grid for curves ──────────────────────────────────────────────────────
x = np.linspace(0.5, 100.0, 1200)

# ── KDE of posterior in tau space ─────────────────────────────────────────
kde_nh  = gaussian_kde(tau_nh,  bw_method="scott")
kde_phi = gaussian_kde(tau_phi, bw_method="scott")

# ── Style kwargs ───────────────────────────────────────────────────────────
prior_kw = dict(color="steelblue",  lw=2.2, zorder=3)
post_kw  = dict(color="darkorange", lw=2.2, zorder=4)
fill_kw  = dict(color="darkorange", alpha=0.22, zorder=2)
band_kw  = dict(color="darkorange", alpha=0.10, zorder=1)
med_kw   = dict(color="darkorange", ls="--", lw=1.6, zorder=5)
true_kw  = dict(color="crimson",    ls=":",  lw=2.2, zorder=6)

# ── Figure: 2 rows × 1 column ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(6.0, 8.0), sharex=True)

def draw_panel(ax, samples, kde, ci_lo, ci_med, ci_hi, xlabel, subscript):
    prior_y = log_uniform_pdf(x)
    post_y  = kde(x)

    ax.plot(x, prior_y,
            label=r"Prior: Uniform$(\ln 2,\,\ln 80)$", **prior_kw)
    ax.plot(x, post_y,
            label="Posterior (KDE)", **post_kw)
    ax.fill_between(x, post_y, **fill_kw)
    ax.axvspan(ci_lo, ci_hi,
               label=fr"5%–95% CI $[{ci_lo:.1f},\,{ci_hi:.1f}]$",
               **band_kw)
    ax.axvline(ci_med,
               label=fr"Post. median $= {ci_med:.1f}$", **med_kw)
    ax.axvline(TAU_TRUE,
               label=fr"True $\tau_{{{subscript}}} = {TAU_TRUE:.0f}$",
               **true_kw)

    ax.set_ylabel("Density", fontsize=12)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85,
              edgecolor="lightgray")

# ── Top panel: tau_NH ──────────────────────────────────────────────────────
draw_panel(axes[0], tau_nh,  kde_nh,
           ci_nh_lo,  ci_nh_med,  ci_nh_hi,
           r"$\tau_{N_H}$ (time steps)", r"N_H")

# ── Bottom panel: tau_phi ─────────────────────────────────────────────────
draw_panel(axes[1], tau_phi, kde_phi,
           ci_phi_lo, ci_phi_med, ci_phi_hi,
           r"$\tau_{\phi}$ (time steps)", r"\phi")

axes[0].set_xlim(0.5, 100.0)

# ── Save ───────────────────────────────────────────────────────────────────
fig.tight_layout()
tag = "Nh22_Tau38_phi-4"
for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"tau_posterior_{tag}_paper.{ext}")
    dpi = 300 if ext == "png" else None
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
