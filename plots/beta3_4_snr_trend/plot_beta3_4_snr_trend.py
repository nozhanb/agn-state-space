"""
Plot tau (and distance from the prior's upper edge, 80) vs shift_term
(SNR proxy) for beta=3.0 and beta=4.0, using only the reliable SNR levels
(shift=1,4,7 -- shift=10,13 could not be reliably sampled; see
SCRIPT_LOG.md in code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/).
"""
import os
import numpy
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = SCRIPT_DIR

SHIFTS = [1.0, 4.0, 7.0]
BETAS = [3.0, 4.0]
COLORS = {3.0: "#4C72B0", 4.0: "#C44E52"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for beta_true in BETAS:
    taus, tau_stds, dists = [], [], []
    for shift_term in SHIFTS:
        d = numpy.load(os.path.join(DATA_DIR, f"beta_{beta_true}_shift_{shift_term}.npz"), allow_pickle=True)
        tau = d["tau_param"].astype("float64")
        taus.append(tau.mean())
        tau_stds.append(tau.std())
        dists.append(80 - tau.mean())

    axes[0].errorbar(SHIFTS, taus, yerr=tau_stds, marker="o", markersize=8, capsize=4,
                      color=COLORS[beta_true], label=fr"$\beta$={beta_true}", linewidth=2)
    axes[1].plot(SHIFTS, dists, marker="o", markersize=8, color=COLORS[beta_true],
                 label=fr"$\beta$={beta_true}", linewidth=2)

axes[0].axhline(80, color="purple", linestyle="-.", alpha=0.6, label="prior upper edge (80)")
axes[0].set_xlabel("shift_term (SNR proxy)", fontsize=13)
axes[0].set_ylabel(r"$\tau$ posterior mean $\pm$ std", fontsize=13)
axes[0].legend(fontsize=11)
axes[0].tick_params(labelsize=11)

axes[1].set_xlabel("shift_term (SNR proxy)", fontsize=13)
axes[1].set_ylabel("distance from prior edge (80 - tau_mean)", fontsize=13)
axes[1].legend(fontsize=11)
axes[1].tick_params(labelsize=11)
axes[1].axhline(0, color="grey", linestyle=":", alpha=0.5)

fig.tight_layout()

base_name = "beta3_4_snr_trend"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    path = os.path.join(OUT_DIR, f"{base_name}.{ext}")
    fig.savefig(path, bbox_inches="tight", **kw)
    print(f"Saved -> {path}")
