"""
Appendix A, Fig. A.1: visualise the three true (pre-AR(1), pre-Poisson)
colorednoise light curves used throughout the Appendix A beta-recovery
validation (beta = 1.7, 3, 4; T = 1000).

Uses the exact `generated_flux` arrays saved alongside the T=1000 HMC
posterior runs produced by the corrected AR(1)/HMC model (ar1_hmc_v2.py,
see code_backup/Claude_Beta_Recovery/CORRECTION_LOG_2026-07-21.md) -- so
this is the same input signal actually used for the current Appendix A
inference, not a freshly redrawn one.
"""
import os
import numpy
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = SCRIPT_DIR

BETAS = [1.7, 3.0, 4.0]
COLORS = {1.7: "black", 3.0: "#C44E52", 4.0: "#4C9A2A"}

fig, ax = plt.subplots(figsize=(8, 4.5))

for beta_true in BETAS:
    path = os.path.join(
        DATA_DIR,
        f"Appendix_A_Inference_OutPut_red_noise_psd_recovery_beta_{beta_true}_and_1000_time_step_and_7_unit_shift_v2scan.npz",
    )
    d = numpy.load(path, allow_pickle=True)
    flux = d["generated_flux"].astype("float64")
    flux = flux - flux.mean()
    ax.plot(numpy.arange(len(flux)), flux, color=COLORS[beta_true], linewidth=1.2,
             label=fr"$\beta$={beta_true:g}")

ax.set_xlabel("Time", fontsize=13)
ax.set_ylabel("Flux", fontsize=13)
ax.legend(fontsize=11)
ax.tick_params(labelsize=11)
fig.tight_layout()

base_name = "app_A_light_curves"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    path = os.path.join(OUT_DIR, f"{base_name}.{ext}")
    fig.savefig(path, bbox_inches="tight", **kw)
    print(f"Saved -> {path}")
