"""
Plot tau vs average source count for beta=3.0 and beta=4.0, using only
the reliable count levels tested (three count levels per beta -- higher
counts could not be reliably sampled; see SCRIPT_LOG.md in
code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/).
"""
import os
import numpy
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = SCRIPT_DIR

SHIFTS = [1.0, 4.0, 7.0]  # internal identifiers for the 3 reliable count levels tested
BETAS = [3.0, 4.0]
COLORS = {3.0: "#4C72B0", 4.0: "#C44E52"}

fig, ax = plt.subplots(figsize=(7, 5.5))

for beta_true in BETAS:
    counts, taus, tau_stds = [], [], []
    for shift_term in SHIFTS:
        d = numpy.load(os.path.join(DATA_DIR, f"beta_{beta_true}_shift_{shift_term}.npz"), allow_pickle=True)
        tau = d["tau_param"].astype("float64")
        counts.append(d["generated_count"].mean())
        taus.append(tau.mean())
        tau_stds.append(tau.std())

    ax.errorbar(counts, taus, yerr=tau_stds, marker="o", markersize=8, capsize=4,
                color=COLORS[beta_true], label=fr"$\beta$={beta_true}", linewidth=2)

ax.axhline(80, color="purple", linestyle="-.", alpha=0.6, label="prior upper edge (80)")
ax.set_xscale("log")
ax.set_xlabel("average source count", fontsize=15)
ax.set_ylabel(r"$\tau$ posterior mean $\pm$ std", fontsize=15)
ax.legend(fontsize=13)
ax.tick_params(labelsize=13, length=7, width=1.3)

fig.tight_layout()

base_name = "beta3_4_snr_trend"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    path = os.path.join(OUT_DIR, f"{base_name}.{ext}")
    fig.savefig(path, bbox_inches="tight", **kw)
    print(f"Saved -> {path}")
