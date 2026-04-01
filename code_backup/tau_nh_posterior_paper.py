"""
tau_nh_posterior_paper.py
=========================
Single-panel publication-quality figure showing the prior and posterior
distribution of the NH timescale tau_NH (plotted in linear tau-space) for:

    N_H = 22,  tau = 38,  phi = -4

The model samples nh_log_tau ~ Uniform(log2, log80), which in linear tau-space
is a log-uniform prior: p(tau) = 1 / (tau * ln(80/2)).
The posterior is approximated as Log-Normal: since nh_log_tau ~ N(mu, sig),
tau ~ LogNormal with the same mu/sig parameters.

Output
------
  results/tau_nh_posterior_Nh22_Tau38_phi-4_paper.pdf
  results/tau_nh_posterior_Nh22_Tau38_phi-4_paper.png
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "summary",
                        "summary_Nh_22_and_Tau_38_and_phi_-4.csv")
OUT_DIR  = os.path.join(BASE, "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── True injected values ───────────────────────────────────────────────────
TAU_TRUE = 38

# ── Prior bounds: Uniform(log(2), log(80)) → log-uniform on tau ────────────
TAU_LO = 2.0
TAU_HI = 80.0
LOG_RANGE = math.log(TAU_HI / TAU_LO)   # ln(40)

# ── Load posterior summary (nh_log_tau is in log-space) ────────────────────
df      = pd.read_csv(CSV_PATH, index_col=0)
row     = df.loc["nh_log_tau"]
log_mu  = float(row["mean"])     # mean of log(tau)
log_sig = float(row["std"])      # std  of log(tau)
log_p5  = float(row["5.0%"])
log_p95 = float(row["95.0%"])

# Convert CI bounds back to tau-space
tau_med = math.exp(log_mu)
tau_p5  = math.exp(log_p5)
tau_p95 = math.exp(log_p95)

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.2))

x = np.linspace(0.2, 110.0, 1200)

# Prior: log-uniform p(tau) = 1 / (tau * ln(80/2))
prior_y = np.where((x >= TAU_LO) & (x <= TAU_HI),
                   1.0 / (x * LOG_RANGE), 0.0)
ax.plot(x, prior_y,
        color="steelblue", lw=2.2, zorder=3,
        label=r"Prior: Log-Uniform$(2,\,80)$")

# Posterior: Log-Normal  (tau ~ LogNormal since log_tau ~ Normal)
post_y = lognorm.pdf(x, s=log_sig, scale=math.exp(log_mu))
ax.plot(x, post_y,
        color="darkorange", lw=2.2, zorder=4,
        label=fr"Posterior $\approx$ LogN($\mu$={log_mu:.2f}, $\sigma$={log_sig:.2f})")
ax.fill_between(x, post_y,
                color="darkorange", alpha=0.22, zorder=2)

# 5–95% credible interval band (in tau-space)
ax.axvspan(tau_p5, tau_p95,
           color="darkorange", alpha=0.10, zorder=1,
           label=fr"5%–95% CI $[{tau_p5:.1f},\,{tau_p95:.1f}]$")

# Posterior median
ax.axvline(tau_med,
           color="darkorange", ls="--", lw=1.6, zorder=5,
           label=fr"Post. median $= {tau_med:.1f}$")

# True value
ax.axvline(TAU_TRUE,
           color="crimson", ls=":", lw=2.2, zorder=6,
           label=fr"True $\tau_{{N_H}} = {TAU_TRUE}$")

# ── Cosmetics ──────────────────────────────────────────────────────────────
ax.set_xlabel(r"$\tau_{N_H}$ (days)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_xlim(0.2, 110.0)
ax.set_ylim(bottom=0.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
ax.legend(fontsize=9, loc="upper right", framealpha=0.85, edgecolor="lightgray")
fig.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────
tag = "Nh22_Tau38_phi-4"
for ext in ("pdf", "png"):
    out = os.path.join(OUT_DIR, f"tau_nh_posterior_{tag}_paper.{ext}")
    dpi = 300 if ext == "png" else None
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
