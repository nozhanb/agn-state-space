"""
Recovered vs. true PSD power-law index (beta) -- N=40 realization-
ensemble means only, all four beta values tested (1.7, 3.0, 4.0, 8.0).

This REPLACES the two-marker version (plot_beta_recovery_calibration.py,
kept for the record but no longer the version intended for the paper).
The single-realization (Table A.1) markers were dropped deliberately --
see beta3_4_N20_ensemble/README.md and the git commit that made this
change for the full reasoning. Short version: the single-realization
points are a Bayesian credible interval (percentile spread of per-draw
fits within one posterior), while the ensemble points are a frequentist-
style interval (spread of independent point estimates across repeated,
separately-generated datasets). These are not the same kind of interval
and are not statistically comparable on one axis -- showing them
together invited exactly the "why doesn't the blue point fall inside
the red interval" question it cannot answer, because it isn't supposed
to always be inside it (see the base-rate check in the git history: an
individual draw falling outside a +/-1 std band ~30% of the time is
expected, not anomalous, but the two-marker plot did not make that
distinction available to a reader without a lot of extra explanation).
Dropping to ensemble-only removes the ambiguity and keeps the plot to
a single, well-defined statistical quantity throughout.

N=40 (not 20): beta=1.7/3.0/4.0 were topped up from their original N=20
ensembles with 20 further realizations each (same model/config,
different seeds); beta=8.0 is fresh N=40. See
regenerate_ensemble_realizations.py and fit_ensemble_N40_all_betas.py.
"""
import os
import numpy
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR
REPRO_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "code_backup", "Claude_Beta_Recovery",
    "Appendix_A", "reproduction_2026-07-20",
)

beta_true_vals = [1.7, 3.0, 4.0, 8.0]
ensemble_filenames = {
    1.7: "beta1.7_N40_realization_medians.npy",
    3.0: "beta3_N40_realization_medians.npy",
    4.0: "beta4_N40_realization_medians.npy",
    8.0: "beta8_N40_realization_medians.npy",
}
ensemble_files = {bt: os.path.join(REPRO_DIR, fname) for bt, fname in ensemble_filenames.items()}

ensemble_mean, ensemble_std, stats_summary = {}, {}, {}
for bt, path in ensemble_files.items():
    arr = numpy.load(path)
    ensemble_mean[bt] = arr.mean()
    ensemble_std[bt] = arr.std()
    n_below = int((arr < bt).sum())
    binom = stats.binomtest(n_below, len(arr), 0.5)
    t_stat, t_p = stats.ttest_1samp(arr, bt)
    stats_summary[bt] = (n_below, len(arr), binom.pvalue, t_p)

fig, ax = plt.subplots(figsize=(6.5, 6.5))

lims = [1.2, 9.0]
ax.plot(lims, lims, linestyle="--", color="grey", linewidth=1.5, label="perfect recovery (1:1)")

ax.errorbar(beta_true_vals, [ensemble_mean[bt] for bt in beta_true_vals],
            yerr=[ensemble_std[bt] for bt in beta_true_vals], fmt="s", markersize=9,
            capsize=5, color="#C44E52", ecolor="#C44E52", linewidth=2,
            label=r"N=40 realization ensemble (mean $\pm$ std)")

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel(r"true $\beta$", fontsize=15)
ax.set_ylabel(r"recovered $\beta$", fontsize=15)
ax.legend(fontsize=13, loc="upper left")
ax.tick_params(labelsize=13, length=7, width=1.3)
ax.set_aspect("equal")

fig.tight_layout()

base_name = "beta_recovery_calibration_squares_only"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    path = os.path.join(OUT_DIR, f"{base_name}.{ext}")
    fig.savefig(path, bbox_inches="tight", **kw)
    print(f"Saved -> {path}")

print()
print("N=40 ensemble values and significance tests:")
for bt in beta_true_vals:
    n_below, n, p_sign, p_t = stats_summary[bt]
    print(f"  true={bt}  mean={ensemble_mean[bt]:.4f}  std={ensemble_std[bt]:.4f}  "
          f"{n_below}/{n} below true  sign_p={p_sign:.6f}  ttest_p={p_t:.6f}")
