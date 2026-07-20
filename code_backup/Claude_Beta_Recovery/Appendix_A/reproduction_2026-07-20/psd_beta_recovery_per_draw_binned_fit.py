"""
Appendix A beta-recovery: per-draw, binned power-law+constant fit.

WHY THIS SCRIPT EXISTS
-----------------------
During the review response for july_16_2026.tex, Johannes's comments 10/11 on
Appendix A ("Recovery of non-periodic power spectra") pointed out that beta
should be recovered by fitting each posterior draw's periodogram separately,
giving a *distribution* of beta values (mean + spread), not a single fit to
an ensemble-mean trajectory.

Investigating the existing code turned up two different, already-written
pipelines in code_backup/Claude_Beta_Recovery/Appendix_A/PSD_beta_recovery.py,
and neither is right on its own:

  (a) The ACTIVE (uncommented) code in that file averages the posterior
      draws FIRST (`posterior_sample_4.mean(1)`), then bins and fits once.
      This is what currently produces Fig. A.2. It gives one beta estimate
      with a curve_fit covariance, not a draw-to-draw distribution.

  (b) A COMMENTED-OUT block in the same file loops over every posterior
      draw and fits each one individually, producing exactly the
      `mean +/- 1.96*std` format that Table A.1 uses. This looked like the
      likely source of Table A.1 -- but block (b) does NOT bin the
      periodogram before fitting, which contradicts the appendix's own text
      (it explicitly describes binning into 9 log-frequency segments before
      fitting, to deal with the periodogram's chi-squared sampling noise).
      Empirically, running block (b) as-is on the actual T=1000, shift=7
      posterior data for beta = 1.7/3.0/4.0 does NOT reproduce the published
      Table A.1 numbers (checked 2026-07-20; off by up to ~1.7 in the mean
      for beta=4, and the reported spread is ~8x tighter than published).
      So block (b), taken literally, is not the source of the published
      table either -- the original provenance is not recoverable from the
      code currently on disk.

RESOLUTION (this script)
-------------------------
Rather than keep chasing an unreproducible historical number, this script
implements the method exactly as described in the paper text (both sec.
3.4.2 and Appendix A): for EACH posterior draw, periodogram -> bin into 9
log-frequency segments -> curve_fit the power-law+constant model to the
binned result -> collect beta. Then report mean +/- 1.96*std across draws,
matching Table A.1's format. This is internally consistent with everything
already written in the paper, is fully reproducible from this one script,
and is expected to yield numbers close to (but not necessarily identical
to) the currently published Table A.1.

INPUT DATA
----------
Same combination already described in Fig. A1's caption ("each running for
1000 time steps") and used to build the existing (soon-to-be-replaced)
Table A.1: posterior inference output at time_step=1000, shift_value=7, for
beta_true in {1.7, 3.0, 4.0}. Files:
  Appendix_A_Inference_OutPut_red_noise_psd_recovery_beta_<B>_and_1000_time_step_and_7_unit_shift_v2.npz
Each contains:
  generated_flux   -- the TRUE colorednoise-generated flux (pre-AR1, pre-Poisson)
  flux_predicted   -- posterior draws of the AR(1)-modeled flux, shape (T, n_draws)

OUTPUTS (all written into this same reproduction_2026-07-20/ directory --
nothing outside this directory is read for writing, and nothing existing
anywhere in the repo is modified or overwritten)
  results_table_A1.txt        -- new Table A.1 numbers (mean, +/-2sigma) per beta
  results_table_A1.npz        -- per-draw beta/a/c arrays for all three beta, plus
                                  the input (true) and binned PSD arrays used for the figures
  app_A_fig_beta_1_7_v2.pdf/png,
  app_A_fig_beta_3_0_v2.pdf/png,
  app_A_fig_beta_4_0_v2.pdf/png -- one PSD panel per beta (1.7, 3.0, 4.0), each with
                                  (i) the correct per-draw-binned fit and (ii) an overlay
                                  of the true input colorednoise PSD (reviewer item 1).
                                  Only the beta=4 panel is currently wired into
                                  july_16_2026.tex; 1.7 and 3.0 added 2026-07-20 so all
                                  three can be compared side by side (see chat: why does
                                  beta=1.7 not bracket the true value at 2sigma -- answer:
                                  single-realization periodogram sampling scatter, same
                                  bias is present in the raw pre-AR(1) input signal too).

Run with: /opt/anaconda3/envs/pub_one/bin/python3 psd_beta_recovery_per_draw_binned_fit.py
"""

import time
import numpy
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

DATA_DIR = "/Users/home/Documents/science/project/johannes/publication/paper_one/data"
OUT_DIR = "/Users/home/Documents/Claude/project/agn_sbi/code_backup/Claude_Beta_Recovery/Appendix_A/reproduction_2026-07-20"

TIME_STEP = 1000
SHIFT_VALUE = 7
BETAS_TRUE = [1.7, 3.0, 4.0]
N_BINS = 9  # matches the 9-segment binning already described in the paper text

LOWER_BOUNDS = [1e-10, 1e-10, 1e-10]
UPPER_BOUNDS = [1e3, 10.0, 1e3]
# Multi-start initial guesses for beta. A single fixed p0=2.5 (used throughout
# the original pipeline and in an earlier version of this script) was found
# 2026-07-20 to land the bounded 3-parameter fit in a poor local optimum for
# steep beta (b=4 converged to beta~2.9-3.9 depending on p0, with chi2 varying
# by ~9x between local optima). Trying several starting points and keeping the
# lowest-chi2 result removes this dependence on a lucky initial guess.
B0_STARTS = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0]


def power_law_func(freq, a, b, c):
    return numpy.log10(a * freq ** (-b) + c)


def periodogram_nonzero(series, nfft):
    freq, psd = signal.periodogram(series, fs=1, nfft=nfft)
    valid = freq > 0
    return freq[valid], psd[valid]


def bin_periodogram(freq, psd, n_bins):
    """9-segment log-frequency binning, same recipe as the paper's existing
    (pre-review) binning code: mean/std of PSD per bin, std propagated to
    log10 space for use as curve_fit sigma."""
    bins = numpy.logspace(numpy.log10(freq.min()), numpy.log10(freq.max()), n_bins)
    centers, means, stds = [], [], []
    for i in range(len(bins) - 1):
        idx = (freq >= bins[i]) & (freq < bins[i + 1])
        if numpy.sum(idx) > 0:
            mean_val = numpy.mean(psd[idx])
            std_val = numpy.std(psd[idx])
            means.append(mean_val)
            stds.append(std_val / (mean_val * numpy.log(10)) if std_val > 0 else 1e-3)
            centers.append((bins[i + 1] + bins[i]) / 2.0)
    return numpy.array(centers), numpy.array(means), numpy.array(stds)


def fit_one_draw(series, nfft, n_bins):
    """Multi-start bounded fit: try every b0 in B0_STARTS, keep the lowest-chi2
    solution. See the B0_STARTS comment above for why this is necessary."""
    freq, psd = periodogram_nonzero(series, nfft)
    centers, means, stds = bin_periodogram(freq, psd, n_bins)
    log_means = numpy.log10(means)
    best_popt, best_chi2 = None, numpy.inf
    for b0 in B0_STARTS:
        try:
            popt, _ = curve_fit(
                power_law_func, centers, log_means,
                p0=[1e-3, b0, 1e-3], bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                sigma=stds, absolute_sigma=True, maxfev=10000,
            )
            chi2 = numpy.sum(((power_law_func(centers, *popt) - log_means) / stds) ** 2)
            if chi2 < best_chi2:
                best_popt, best_chi2 = popt, chi2
        except RuntimeError:
            continue
    if best_popt is None:
        raise RuntimeError("all multi-start fits failed")
    return best_popt  # a, b (beta), c


def main():
    results = {}
    save_dict = {}

    for beta_true in BETAS_TRUE:
        path = (
            f"{DATA_DIR}/Appendix_A_Inference_OutPut_red_noise_psd_recovery_"
            f"beta_{beta_true}_and_{TIME_STEP}_time_step_and_{SHIFT_VALUE}_unit_shift_v2.npz"
        )
        data = numpy.load(path, allow_pickle=True)
        generated_flux = data["generated_flux"].astype("float64")
        generated_flux = generated_flux - generated_flux.mean()
        posterior_sample = data["flux_predicted"].astype("float64")
        posterior_sample = posterior_sample - posterior_sample.mean()

        n_draws = posterior_sample.shape[1]
        t0 = time.time()
        a_list, b_list, c_list = [], [], []
        n_failed = 0
        for i in range(n_draws):
            try:
                a_fit, b_fit, c_fit = fit_one_draw(posterior_sample[:, i], TIME_STEP, N_BINS)
                a_list.append(a_fit)
                b_list.append(b_fit)
                c_list.append(c_fit)
            except RuntimeError:
                n_failed += 1
        dt = time.time() - t0

        a_array = numpy.array(a_list)
        b_array = numpy.array(b_list)
        c_array = numpy.array(c_list)

        # Switched from mean +/- 1.96*std to the 5%/50%/95% percentiles of the
        # per-draw beta distribution (2026-07-20). mean+/-std assumes a
        # roughly Gaussian, symmetric spread; the beta=4.0 per-draw
        # distribution has real skew (skew=1.94, mean=3.854 vs median=3.825),
        # so mean+/-std both mischaracterizes the shape and is inconsistent
        # with the [5%-95%] credible-interval language already used
        # elsewhere in this paper (sec. 3.4.2) and the median+asymmetric-error
        # convention used in the paper's other results tables.
        p5, median_b, p95 = numpy.percentile(b_array, [5, 50, 95])
        mean_b = b_array.mean()

        results[beta_true] = (p5, median_b, p95)

        print(f"beta_true={beta_true}  n_draws={n_draws}  n_failed_fits={n_failed}  "
              f"time={dt:.1f}s ({dt/n_draws*1000:.2f} ms/draw)")
        print(f"  new Table A.1 row: p5={p5:.3f}  median={median_b:.3f}  "
              f"p95={p95:.3f}  (mean={mean_b:.3f}, true={beta_true})")

        # input (true, pre-AR1) PSD for the Fig. A.2 overlay (reviewer item 1)
        input_freq, input_psd = periodogram_nonzero(generated_flux, TIME_STEP)

        save_dict[f"beta_{beta_true}_a_array"] = a_array
        save_dict[f"beta_{beta_true}_b_array"] = b_array
        save_dict[f"beta_{beta_true}_c_array"] = c_array
        save_dict[f"beta_{beta_true}_input_freq"] = input_freq
        save_dict[f"beta_{beta_true}_input_psd"] = input_psd

        # Keep the binned PSD + fit of one representative draw (draw 0) purely
        # for plotting each beta's PSD panel (black/orange points + fit line);
        # the table numbers above already come from the full per-draw
        # distribution across all n_draws draws, not this one draw.
        plot_freq, plot_psd = periodogram_nonzero(posterior_sample[:, 0], TIME_STEP)
        plot_centers, plot_means, plot_stds = bin_periodogram(plot_freq, plot_psd, N_BINS)
        plot_popt = numpy.array([a_array[0], b_array[0], c_array[0]])
        plot_yerr = plot_means * numpy.log(10) * plot_stds
        print(f"\nbeta={beta_true} draw-0 binned points (for figure-caption accuracy check):")
        for fc, m, e in zip(plot_centers, plot_means, plot_yerr):
            print(f"  freq={fc:.4f}  psd_mean={m:.4e}  abs_err={e:.4e}  rel_err={e/m:.4f}")
        save_dict[f"beta_{beta_true}_plot_freq"] = plot_freq
        save_dict[f"beta_{beta_true}_plot_psd"] = plot_psd
        save_dict[f"beta_{beta_true}_plot_centers"] = plot_centers
        save_dict[f"beta_{beta_true}_plot_means"] = plot_means
        save_dict[f"beta_{beta_true}_plot_yerr"] = plot_yerr
        save_dict[f"beta_{beta_true}_plot_popt"] = plot_popt

    numpy.savez(f"{OUT_DIR}/results_table_A1.npz", **save_dict)

    with open(f"{OUT_DIR}/results_table_A1.txt", "w") as f:
        f.write("beta_true  p5  median  p95\n")
        for beta_true, (p5, median_b, p95) in results.items():
            f.write(f"{beta_true}  {p5:.3f}  {median_b:.3f}  {p95:.3f}\n")

    # --- Fig. A.2-style panel, one per beta ---
    # Font/tick sizes bumped up from the original (13pt labels, 10pt legend,
    # default tick labels) per user request 2026-07-20 for readability.
    for beta_true in BETAS_TRUE:
        tag = str(beta_true).replace(".", "_")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(save_dict[f"beta_{beta_true}_plot_freq"], save_dict[f"beta_{beta_true}_plot_psd"],
                "ok", markersize=4, alpha=0.5, label="Simulated PSD (one posterior draw)")
        ax.errorbar(save_dict[f"beta_{beta_true}_plot_centers"], save_dict[f"beta_{beta_true}_plot_means"],
                    yerr=save_dict[f"beta_{beta_true}_plot_yerr"],
                    fmt="o", markersize=7, color="#FFA500", ecolor="grey", elinewidth=2.5, capsize=4, label="Binned PSD")
        plot_popt = save_dict[f"beta_{beta_true}_plot_popt"]
        plot_freq = save_dict[f"beta_{beta_true}_plot_freq"]
        ax.plot(plot_freq, 10 ** power_law_func(plot_freq, *plot_popt), "r--", linewidth=2.5,
                label=fr"Fit ($\beta$ = {plot_popt[1]:.2f})")

        ax.plot(save_dict[f"beta_{beta_true}_input_freq"], save_dict[f"beta_{beta_true}_input_psd"],
                "-", color="steelblue", linewidth=2, alpha=0.8,
                label="Input PSD (true colorednoise, pre-AR(1))")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency [Hz]", fontsize=18)
        ax.set_ylabel("Power Spectral Density", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=15, length=7, width=1.3)
        ax.tick_params(axis="both", which="minor", length=4, width=1.0)
        ax.legend(fontsize=13)
        ax.set_title(fr"$\beta_{{\rm true}}$ = {beta_true}", fontsize=16)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/app_A_fig_beta_{tag}_v2.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{OUT_DIR}/app_A_fig_beta_{tag}_v2.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {OUT_DIR}/app_A_fig_beta_{tag}_v2.{{pdf,png}}")


if __name__ == "__main__":
    main()
