"""
Recovered vs. true PSD power-law index (beta), across the three
colorednoise validation cases in Table A.1 (beta_true = 1.7, 3.0, 4.0).

WHY THIS PLOT EXISTS
---------------------
This is the plot Johannes originally requested in his Appendix A review
comments (see CORRECTION_LOG_2026-07-23.md and this session's discussion
for the full list of his 14 comments):

  12. "Please make several simulations (maybe 20)."
  13. "Maybe you can add a plot with beta as the y-axis, with the true
      value as a horizontal dashed line, and the inferred value as an
      error bar."
  14. "Please also vary the 'signal-to-noise' ratios (normalisations).
      This can be the x-axis on the plot."

We built a related plot for comment 14 already (the beta=3/4 SNR sweep,
see plots/beta3_4_snr_trend/), and this script is the literal
beta-recovery plot from comments 12-13, using true beta on the x-axis
(one panel covering all three cases with a single 1:1 reference line)
rather than a separate horizontal line per case.

TWO MARKERS PER BETA -- READ THIS BEFORE CHANGING ANYTHING
-------------------------------------------------------------
There are two fundamentally different kinds of uncertainty shown here,
for all three beta values:

1. "single realization" (blue circles): the [5%,95%] spread of per-draw
   beta fits *within one HMC run* -- i.e. how much the reconstructed
   light curve varies draw-to-draw within a single posterior. This is
   what's read from results_table_A1_beta1.7_longchain.txt (beta=1.7)
   and results_table_A1_corrected.txt (beta=3.0, beta=4.0) -- the exact
   files behind the published Table A.1 (tab:appendix_params in
   tex/july_16_2026.tex).

2. "N=20 realization ensemble" (orange squares): mean +/- std of the
   recovered-beta *point estimate* across 20 independent colorednoise
   realizations per beta, each run through the full real pipeline
   (Poisson counts -> AR(1)/HMC -> periodogram fit). This is run-to-run
   scatter, i.e. what you'd see if you generated the data over again
   with a new random seed -- a completely different source of
   uncertainty than (1), and what Johannes's comment 12 was actually
   asking about. Sources:
     data/beta1.7_N20_realization_medians.npy
     ../../code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/beta3_N20_realization_medians.npy
     ../../code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20/beta4_N20_realization_medians.npy
   Full provenance and results for all three in
   plots/appendix_a/beta3_4_N20_ensemble/README.md.

Do not average or merge the single-realization and ensemble markers for
a given beta -- they answer different questions ("how much does one
posterior vary" vs. "how much would my answer change with different
data") and conflating them would misstate the uncertainty either way.

IMPORTANT -- the beta=3/4 ensemble result is NOT the same story as
beta=1.7's. For beta=1.7, averaging over realizations resolved the
single-realization miss (it was periodogram noise). For beta=3/4, a
sign test on the 20 realization medians shows beta=4's miss is a
real, statistically unambiguous bias (20/20 realizations below the
true value, p=0.000002) -- NOT noise that averages out. beta=3 is a
weaker, more ambiguous case (13/20 below, sign test p=0.26, but a
one-sample t-test against the true value is still significant,
p=0.039). See beta3_4_N20_ensemble/README.md for the full statistical
analysis. Do not caption this plot as "confirms recovery is unbiased
on average" for beta=3/4 -- it does not.
"""
import os
import numpy
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = SCRIPT_DIR
REPRO_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "code_backup", "Claude_Beta_Recovery",
    "Appendix_A", "reproduction_2026-07-20",
)


def read_results(path):
    rows = {}
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.split()
            if not parts:
                continue
            beta_true = float(parts[0])
            p5, median, p95 = float(parts[1]), float(parts[2]), float(parts[3])
            rows[beta_true] = (p5, median, p95)
    return rows

results_corrected = read_results(os.path.join(SCRIPT_DIR, "results_table_A1_corrected.txt"))
results_beta17 = read_results(os.path.join(SCRIPT_DIR, "results_table_A1_beta1.7_longchain.txt"))

beta_true_vals = [1.7, 3.0, 4.0]
source = {1.7: results_beta17, 3.0: results_corrected, 4.0: results_corrected}

medians, err_lo, err_hi = [], [], []
for bt in beta_true_vals:
    p5, median, p95 = source[bt][bt]
    medians.append(median)
    err_lo.append(median - p5)
    err_hi.append(p95 - median)

# N=20 realization ensembles, all three beta values
ensemble_files = {
    1.7: os.path.join(DATA_DIR, "beta1.7_N20_realization_medians.npy"),
    3.0: os.path.join(REPRO_DIR, "beta3_N20_realization_medians.npy"),
    4.0: os.path.join(REPRO_DIR, "beta4_N20_realization_medians.npy"),
}
ensemble_mean, ensemble_std = {}, {}
for bt, path in ensemble_files.items():
    arr = numpy.load(path)
    ensemble_mean[bt] = arr.mean()
    ensemble_std[bt] = arr.std()

fig, ax = plt.subplots(figsize=(6, 6))

lims = [1.2, 4.4]
ax.plot(lims, lims, linestyle="--", color="grey", linewidth=1.5, label="perfect recovery (1:1)")

# Thin connectors so each beta's two markers (same true x, two different
# uncertainty estimates) read as clearly linked rather than as separate
# candidate x-positions
for bt, med in zip(beta_true_vals, medians):
    ax.plot([bt, bt], [med, ensemble_mean[bt]], color="grey", linewidth=1, zorder=1)

ax.errorbar(beta_true_vals, medians, yerr=[err_lo, err_hi], fmt="o", markersize=9,
            capsize=5, color="#4C72B0", ecolor="#4C72B0", linewidth=2,
            label=r"single realization (median, 5-95% CI)")

ax.errorbar(beta_true_vals, [ensemble_mean[bt] for bt in beta_true_vals],
            yerr=[ensemble_std[bt] for bt in beta_true_vals], fmt="s", markersize=9,
            capsize=5, color="#DD8452", ecolor="#DD8452", linewidth=2,
            label=r"N=20 realization ensemble (mean $\pm$ std)")

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel(r"true $\beta$", fontsize=13)
ax.set_ylabel(r"recovered $\beta$", fontsize=13)
ax.legend(fontsize=11, loc="upper left")
ax.tick_params(labelsize=11)
ax.set_aspect("equal")

fig.tight_layout()

base_name = "beta_recovery_calibration"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    path = os.path.join(OUT_DIR, f"{base_name}.{ext}")
    fig.savefig(path, bbox_inches="tight", **kw)
    print(f"Saved -> {path}")

print()
print("Single-realization values:")
for bt, med, lo, hi in zip(beta_true_vals, medians, err_lo, err_hi):
    print(f"  true={bt}  recovered={med:.3f}  [-{lo:.3f}, +{hi:.3f}]")
print()
print("N=20 ensemble values:")
for bt in beta_true_vals:
    print(f"  true={bt}  mean={ensemble_mean[bt]:.4f}  std={ensemble_std[bt]:.4f}")
