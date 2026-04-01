"""
nh_phi_psd_beta_recovery_methodB.py
====================================

Method B version of nh_phi_psd_beta_recovery.py.

Key differences from the original (Method A):
  - No detrending: raw log-N_H and log-K values used throughout
  - No analytical AR(1) PSD overlay
  - Method B: periodogram computed for every single posterior sample
    (6000 draws), not just the posterior mean
  - Grey halo = 5th–95th percentile of all 6000 sample periodograms
  - True periodogram + power-law fit to true = reference line
  - Beta distribution shown as a histogram (row 3)
  - Layout: 3 rows × 2 columns

Parameter combination:
    NH = 21   (mean log N_H = 21)
    Tau = 2   (AR(1) timescale τ = 2 → φ = exp(-1/τ) ≈ 0.607)
    Phi = -3  (mean log K = -3)
"""

import matplotlib
matplotlib.use("Agg")
import numpy
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams["axes.linewidth"]    = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["xtick.major.size"]  = 6
plt.rcParams["ytick.major.size"]  = 6
plt.rcParams["xtick.labelsize"]   = 12
plt.rcParams["ytick.labelsize"]   = 12
plt.rcParams["axes.labelweight"]  = "bold"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NH  = 21
TAU = 2
PHI = -3

INFERENCE_FILE = (
    "inference_data/"
    "synthetic_count_spec_visualisation_SDSSJ0932+0405_multiple_simulated_"
    f"stochastic_Nh_{NH}_and_Tau_{TAU}_and_phi_{PHI}_steps_100_iter_6000_"
    "spectrum_500_redshift_0108.npz"
)
SIM_FILE = (
    "simulation_data/"
    "synthetic_count_NH_and_phi_spec_visualisation_SDSSJ0932+0405_multiple_"
    f"simulated_stochastic_Nh_{NH}_and_Tau_{TAU}_and_phi_{PHI}.npz"
)

NBINS        = 7
N_DRAW       = 1000   # number of posterior samples to use for Method B (randomly selected)
RANDOM_SEED  = 42     # for reproducibility
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Method B — NH={NH}, Tau={TAU}, Phi={PHI}")
print("=" * 60)

params = numpy.load(INFERENCE_FILE, allow_pickle=True)
sim    = numpy.load(SIM_FILE,       allow_pickle=True)

# Posterior samples: (6000, 100) → swap → (100, 6000)
nh_y  = numpy.swapaxes(params["nh_y"],  0, 1).astype("float64")   # (100, 6000)
phi_y = numpy.swapaxes(params["phi_y"], 0, 1).astype("float64")   # (100, 6000)

T         = nh_y.shape[0]    # 100
n_samples = nh_y.shape[1]    # 6000

# True (generating) time series — used as-is, no detrending
true_nh  = sim["nh_process"].astype("float64")    # (100,)
true_phi = sim["phi_process"].astype("float64")   # (100,)

# Randomly subsample N_DRAW columns from the 6000 posterior samples
rng        = numpy.random.default_rng(RANDOM_SEED)
draw_idx   = rng.choice(n_samples, size=N_DRAW, replace=False)
nh_y_draw  = nh_y[:,  draw_idx]    # (100, N_DRAW)
phi_y_draw = phi_y[:, draw_idx]    # (100, N_DRAW)

# Posterior mean — computed over ALL 6000 samples, no detrending
nh_mean  = nh_y.mean(axis=1)    # (100,)
phi_mean = phi_y.mean(axis=1)   # (100,)

print(f"T={T}, total samples={n_samples}, using {N_DRAW} random draws")
print(f"True NH:  mean={true_nh.mean():.3f}, std={true_nh.std():.3f}")
print(f"True Phi: mean={true_phi.mean():.3f}, std={true_phi.std():.3f}")


# ---------------------------------------------------------------------------
# Power-law + constant model (same as original)
# ---------------------------------------------------------------------------
def power_law_func(freq, a, b, c):
    """log10( a * freq^{-b} + c )"""
    return numpy.log10(a * freq ** (-b) + c)


# ---------------------------------------------------------------------------
# Low-level helper: Vaughan binning + curve_fit on a pre-computed periodogram
# Called both by fit_psd_beta_binned (true spectrum) and by the per-sample loop
# ---------------------------------------------------------------------------
def _bin_and_fit_psd(freq_nz, psd_nz, nbins=NBINS):
    """
    Apply Vaughan log-frequency binning and fit a power-law + constant model
    to a pre-computed periodogram.

    Parameters
    ----------
    freq_nz : (n_freq,)   frequencies > 0
    psd_nz  : (n_freq,)   corresponding PSD values

    Returns
    -------
    (bin_centers, bin_means, popt) if the fit succeeded, else None.
    bin_centers : (n_bins,)
    bin_means   : (n_bins,)  mean PSD in each bin
    popt        : (3,)       [a, beta, c] power-law parameters
    """
    bins = numpy.logspace(
        numpy.log10(freq_nz.min()),
        numpy.log10(freq_nz.max()),
        nbins,
    )
    bin_centers, psd_means, psd_stds = [], [], []
    for i in range(len(bins) - 1):
        idx = ((freq_nz >= bins[i]) & (freq_nz <= bins[i + 1])
               if i == len(bins) - 2
               else (freq_nz >= bins[i]) & (freq_nz < bins[i + 1]))
        n = idx.sum()
        if n < 1:
            continue
        m     = numpy.mean(psd_nz[idx])
        s_lin = numpy.std(psd_nz[idx]) if n > 1 else 0.3 * m
        s_log = max(s_lin / (m * numpy.log(10)), 1e-4)
        bin_centers.append((bins[i + 1] + bins[i]) / 2.0)
        psd_means.append(m)
        psd_stds.append(s_log)

    bin_centers = numpy.array(bin_centers)
    psd_means   = numpy.array(psd_means)
    psd_stds    = numpy.array(psd_stds)

    if len(bin_centers) < 3:
        return None

    p0     = [1e-3, 1.5, 1e-6]
    bounds = ([1e-12, 1e-6, 1e-12], [1e6, 10.0, 1e6])

    def _try(bc, pm, ps):
        popt, pcov = curve_fit(
            power_law_func, bc, numpy.log10(pm),
            p0=p0, bounds=bounds,
            sigma=ps, absolute_sigma=True, maxfev=20000,
        )
        return popt, pcov

    try:
        popt, pcov = _try(bin_centers, psd_means, psd_stds)
        if numpy.sqrt(numpy.diag(pcov))[1] > 100:
            raise RuntimeError("Degenerate")
    except RuntimeError:
        sentinel = round(0.3 / numpy.log(10), 4)
        keep = numpy.array([round(s, 4) != sentinel for s in psd_stds])
        if keep.sum() < 3:
            keep = numpy.ones(len(bin_centers), dtype=bool)
        try:
            popt, pcov = _try(bin_centers[keep], psd_means[keep], psd_stds[keep])
        except RuntimeError:
            return None

    return bin_centers, psd_means, popt


# ---------------------------------------------------------------------------
# Vaughan-style binning + curve_fit for a SINGLE time series
# (used for the true spectrum — wraps _bin_and_fit_psd and adds print output)
# ---------------------------------------------------------------------------
def fit_psd_beta_binned(time_series, T, label, nbins=NBINS):
    """
    Compute periodogram of time_series, then call _bin_and_fit_psd.
    Returns dict with all quantities needed for plotting, or None.
    """
    freq, psd = signal.periodogram(time_series, fs=1, nfft=T)
    valid   = freq > 0
    freq_nz = freq[valid]
    psd_nz  = psd[valid]

    result = _bin_and_fit_psd(freq_nz, psd_nz, nbins)
    if result is None:
        print(f"  {label}: too few bins / fit failed.")
        return None

    bin_centers, psd_means, popt = result
    beta_fit = popt[1]
    # Rerun with pcov to get uncertainty (curve_fit doesn't return pcov from _bin_and_fit_psd)
    try:
        _, pcov = curve_fit(
            power_law_func, bin_centers, numpy.log10(psd_means),
            p0=list(popt), maxfev=20000,
        )
        beta_err = float(numpy.sqrt(numpy.diag(pcov))[1])
    except Exception:
        beta_err = numpy.nan

    print(f"  {label}: β = {beta_fit:.3f} ± {beta_err:.3f}  "
          f"(95% CI: [{beta_fit - 2*beta_err:.3f}, {beta_fit + 2*beta_err:.3f}])")

    return dict(
        freq_nz=freq_nz, psd_nz=psd_nz,
        bin_centers=bin_centers, psd_means=psd_means,
        popt=popt, beta=beta_fit, beta_err=beta_err,
        beta_ci_lo=beta_fit - 2 * beta_err,
        beta_ci_hi=beta_fit + 2 * beta_err,
    )


# ---------------------------------------------------------------------------
# Method B: periodogram + Vaughan-binned beta for EVERY posterior sample
# ---------------------------------------------------------------------------
def compute_sample_psds_and_betas(y_samples, T, label):
    """
    For each posterior sample (column of y_samples):
      1. Compute the periodogram (no detrending)
      2. Apply Vaughan log-frequency binning  (same as true spectrum)
      3. Fit power-law + constant with curve_fit → beta

    The binned PSD values are stored per sample so the grey halo in the PSD
    panel reflects the actual spread of the *binned* PSDs (i.e., the same
    quantity from which beta is derived), not the raw periodogram scatter.

    Returns
    -------
    freq_nz         : (n_freq,)            raw frequency axis (f > 0)
    all_psds        : (n_s, n_freq)        raw periodograms (for background halo)
    bin_centers     : (n_bins,)            common bin centres (same for all samples)
    all_binned_means: (n_s, n_bins)        binned PSD per sample (nan if fit failed)
    betas           : (n_valid,)           beta from binned fit (failed fits excluded)
    """
    n_s = y_samples.shape[1]
    print(f"\nComputing {n_s} periodograms + binned fits [{label}] ...")

    # Frequency axis (same for every sample)
    freq_template, _ = signal.periodogram(y_samples[:, 0], fs=1, nfft=T)
    valid   = freq_template > 0
    freq_nz = freq_template[valid]
    n_freq  = freq_nz.size

    # Pre-derive the bin centres from the shared frequency axis
    edges = numpy.logspace(numpy.log10(freq_nz.min()), numpy.log10(freq_nz.max()), NBINS)
    bin_centers = numpy.array([(edges[i + 1] + edges[i]) / 2.0
                                for i in range(len(edges) - 1)])
    n_bins = len(bin_centers)

    all_psds         = numpy.zeros((n_s, n_freq))
    all_binned_means = numpy.full((n_s, n_bins), numpy.nan)
    betas_raw        = numpy.full(n_s, numpy.nan)
    n_failed         = 0

    for s in range(n_s):
        _, psd = signal.periodogram(y_samples[:, s], fs=1, nfft=T)
        p = psd[valid]
        all_psds[s] = p

        result = _bin_and_fit_psd(freq_nz, p)
        if result is not None:
            bc, bm, popt = result
            # bc may have fewer entries than bin_centers if some bins were empty;
            # map by nearest match to the common bin_centers array
            for j, c in enumerate(bc):
                k = numpy.argmin(numpy.abs(bin_centers - c))
                all_binned_means[s, k] = bm[j]
            betas_raw[s] = popt[1]
        else:
            n_failed += 1

    betas = betas_raw[~numpy.isnan(betas_raw)]
    b5, b50, b95 = numpy.percentile(betas, [5, 50, 95])
    print(f"  β (binned+fit): median={b50:.3f},  5–95% CI=[{b5:.3f}, {b95:.3f}]  "
          f"({n_failed}/{n_s} fits failed)")

    return freq_nz, all_psds, bin_centers, all_binned_means, betas


# ---------------------------------------------------------------------------
# Cache paths — save expensive intermediate results so the plot can be
# regenerated without rerunning the full analysis.
# ---------------------------------------------------------------------------
CACHE_TRUE = os.path.join(RESULTS_DIR,
    f"cache_true_fits_Nh{NH}_Tau{TAU}_Phi{PHI}.npz")
CACHE_SAMPLES = os.path.join(RESULTS_DIR,
    f"cache_sample_psds_Nh{NH}_Tau{TAU}_Phi{PHI}_n{N_DRAW}_seed{RANDOM_SEED}.npz")


# ---------------------------------------------------------------------------
# Step A – True spectrum fits (Vaughan binning + polyfit)
# ---------------------------------------------------------------------------
def _polyfit_beta(ts, T):
    freq, psd = signal.periodogram(ts, fs=1, nfft=T)
    valid = freq > 0
    slope, _ = numpy.polyfit(numpy.log10(freq[valid]),
                              numpy.log10(psd[valid] + 1e-30), 1)
    return -slope


if os.path.exists(CACHE_TRUE):
    print(f"\nLoading cached true fits from {CACHE_TRUE}")
    cache = numpy.load(CACHE_TRUE, allow_pickle=True)

    def _restore_res(prefix, cache):
        return dict(
            freq_nz    = cache[f"{prefix}_freq_nz"],
            psd_nz     = cache[f"{prefix}_psd_nz"],
            bin_centers= cache[f"{prefix}_bin_centers"],
            psd_means  = cache[f"{prefix}_psd_means"],
            psd_stds   = cache[f"{prefix}_psd_stds"],
            popt       = cache[f"{prefix}_popt"],
            beta       = float(cache[f"{prefix}_beta"]),
            beta_err   = float(cache[f"{prefix}_beta_err"]),
            beta_ci_lo = float(cache[f"{prefix}_beta_ci_lo"]),
            beta_ci_hi = float(cache[f"{prefix}_beta_ci_hi"]),
        )

    true_nh_res  = _restore_res("nh",  cache)
    true_phi_res = _restore_res("phi", cache)
    true_nh_beta_polyfit  = float(cache["nh_beta_polyfit"])
    true_phi_beta_polyfit = float(cache["phi_beta_polyfit"])
    print(f"  NH  true β (polyfit): {true_nh_beta_polyfit:.3f}")
    print(f"  Phi true β (polyfit): {true_phi_beta_polyfit:.3f}")

else:
    print("\n--- True spectrum fits (Vaughan binning) ---")
    true_nh_res  = fit_psd_beta_binned(true_nh,  T, "NH  [true]")
    true_phi_res = fit_psd_beta_binned(true_phi, T, "Phi [true]")

    true_nh_beta_polyfit  = _polyfit_beta(true_nh,  T)
    true_phi_beta_polyfit = _polyfit_beta(true_phi, T)
    print(f"\n  True NH  β (polyfit): {true_nh_beta_polyfit:.3f}")
    print(f"  True Phi β (polyfit): {true_phi_beta_polyfit:.3f}")

    # Save
    def _save_res(prefix, res):
        return {f"{prefix}_{k}": v for k, v in res.items()}

    numpy.savez(
        CACHE_TRUE,
        **_save_res("nh",  true_nh_res),
        **_save_res("phi", true_phi_res),
        nh_beta_polyfit  = true_nh_beta_polyfit,
        phi_beta_polyfit = true_phi_beta_polyfit,
    )
    print(f"  Saved → {CACHE_TRUE}")


# ---------------------------------------------------------------------------
# Step B – Method B: 1000 randomly selected posterior samples
# ---------------------------------------------------------------------------
if os.path.exists(CACHE_SAMPLES):
    print(f"\nLoading cached sample PSDs from {CACHE_SAMPLES}")
    cache_s = numpy.load(CACHE_SAMPLES)
    nh_freq_nz        = cache_s["nh_freq_nz"]
    nh_all_psds       = cache_s["nh_all_psds"]
    nh_bin_centers    = cache_s["nh_bin_centers"]
    nh_binned_means   = cache_s["nh_binned_means"]
    nh_betas          = cache_s["nh_betas"]
    phi_freq_nz       = cache_s["phi_freq_nz"]
    phi_all_psds      = cache_s["phi_all_psds"]
    phi_bin_centers   = cache_s["phi_bin_centers"]
    phi_binned_means  = cache_s["phi_binned_means"]
    phi_betas         = cache_s["phi_betas"]
    print(f"  Loaded {nh_all_psds.shape[0]} NH samples, {phi_all_psds.shape[0]} Phi samples.")

else:
    (nh_freq_nz,  nh_all_psds,  nh_bin_centers,
     nh_binned_means,  nh_betas)  = compute_sample_psds_and_betas(nh_y_draw,  T, "NH")
    (phi_freq_nz, phi_all_psds, phi_bin_centers,
     phi_binned_means, phi_betas) = compute_sample_psds_and_betas(phi_y_draw, T, "Phi")

    numpy.savez(
        CACHE_SAMPLES,
        nh_freq_nz      = nh_freq_nz,
        nh_all_psds     = nh_all_psds,
        nh_bin_centers  = nh_bin_centers,
        nh_binned_means = nh_binned_means,
        nh_betas        = nh_betas,
        phi_freq_nz     = phi_freq_nz,
        phi_all_psds    = phi_all_psds,
        phi_bin_centers = phi_bin_centers,
        phi_binned_means= phi_binned_means,
        phi_betas       = phi_betas,
    )
    print(f"  Saved → {CACHE_SAMPLES}")


# Percentile halos — from the BINNED PSDs (same quantity used for betas)
nh_bin_p5,  nh_bin_p50,  nh_bin_p95  = [
    numpy.nanpercentile(nh_binned_means,  q, axis=0) for q in (5, 50, 95)]
phi_bin_p5, phi_bin_p50, phi_bin_p95 = [
    numpy.nanpercentile(phi_binned_means, q, axis=0) for q in (5, 50, 95)]

# Also keep raw periodogram percentiles for the faint background halo
nh_raw_p5,  nh_raw_p95  = [numpy.percentile(nh_all_psds,  q, axis=0) for q in (5, 95)]
phi_raw_p5, phi_raw_p95 = [numpy.percentile(phi_all_psds, q, axis=0) for q in (5, 95)]

# ---------------------------------------------------------------------------
# Extreme-beta sample — for overlay on row-2 PSD panels
#   NH : sample whose beta is closest to 4.0  (user observed values up to ~4.5)
#   Phi: sample with the maximum beta          (phi betas are very tightly clustered,
#        so there is no natural ~4 outlier; show the overall maximum instead)
# Index alignment is valid because 0/1000 fits failed, so betas[i] == sample i.
# ---------------------------------------------------------------------------
extreme_nh_idx   = int(numpy.argmin(numpy.abs(nh_betas  - 4.0)))
extreme_nh_beta  = float(nh_betas[extreme_nh_idx])
extreme_nh_raw   = nh_all_psds[extreme_nh_idx]        # (n_freq,)
extreme_nh_bins  = nh_binned_means[extreme_nh_idx]     # (n_bins,)

extreme_phi_idx  = int(numpy.argmax(phi_betas))
extreme_phi_beta = float(phi_betas[extreme_phi_idx])
extreme_phi_raw  = phi_all_psds[extreme_phi_idx]
extreme_phi_bins = phi_binned_means[extreme_phi_idx]

print(f"\nExtreme NH  sample: index={extreme_nh_idx},  β={extreme_nh_beta:.3f}")
print(f"Extreme Phi sample: index={extreme_phi_idx}, β={extreme_phi_beta:.3f}")


# ---------------------------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------------------------
txt_path = os.path.join(RESULTS_DIR, f"beta_recovery_Nh{NH}_Tau{TAU}_Phi{PHI}_methodB.txt")
with open(txt_path, "w") as fout:
    fout.write(f"PSD β recovery (Method B) — NH={NH}, Tau={TAU}, Phi={PHI}\n")
    fout.write("=" * 55 + "\n")
    fout.write(f"T = {T},  n_samples = {n_samples}\n\n")
    for label, betas, true_beta in [
        ("NH  process", nh_betas,  true_nh_beta_polyfit),
        ("Phi process", phi_betas, true_phi_beta_polyfit),
    ]:
        b5, b50, b95 = numpy.percentile(betas, [5, 50, 95])
        fout.write(f"{label}\n")
        fout.write(f"  True β (polyfit)    = {true_beta:.3f}\n")
        fout.write(f"  Posterior β median  = {b50:.3f}\n")
        fout.write(f"  Posterior β 5–95%   = [{b5:.3f}, {b95:.3f}]\n")
        fout.write(f"  Posterior β mean    = {betas.mean():.3f} ± {betas.std():.3f}\n\n")
print(f"\nNumerical results saved → {txt_path}")


# ---------------------------------------------------------------------------
# Final plot  (3 rows × 2 columns)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 17))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.25, wspace=0.35)

ax_nh_ts   = fig.add_subplot(gs[0, 0])
ax_phi_ts  = fig.add_subplot(gs[0, 1])
ax_nh_psd  = fig.add_subplot(gs[1, 0])
ax_phi_psd = fig.add_subplot(gs[1, 1])
ax_nh_beta = fig.add_subplot(gs[2, 0])
ax_phi_beta= fig.add_subplot(gs[2, 1])

steps = numpy.arange(T)


# ── Row 1: time series (original values, no detrend) ──────────────────────

def _plot_ts(ax, y_samples, post_mean, true_vals, ylabel, title):
    # 200 posterior sample traces in light grey
    ax.plot(steps, y_samples[:, :200], color="silver", alpha=0.4, linewidth=0.5)
    ax.plot(steps, post_mean, color="black",   linewidth=2.0, label="Posterior mean")
    ax.plot(steps, true_vals, color="crimson", linewidth=1.5, linestyle="--", label="True")
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel(ylabel,      fontsize=12)
    ax.legend(fontsize=11, loc="lower left")

_plot_ts(ax_nh_ts, nh_y_draw, nh_mean, true_nh,
         ylabel=r"log $N_H$",
         title=rf"NH process   ($N_H = 10^{{{NH}}}$ cm$^{{-2}}$, $\tau={TAU}$)")
_plot_ts(ax_phi_ts, phi_y_draw, phi_mean, true_phi,
         ylabel=r"log $K$",
         title=rf"Phi process   ($\log K={PHI}$, $\tau={TAU}$)")


# ── Row 2: PSD — grey halo + true spectrum fit ────────────────────────────

def _plot_psd_methodB(ax, freq_nz,
                      raw_p5, raw_p95,
                      bin_centers, bin_p5, bin_p50, bin_p95,
                      true_res, label, fit_color,
                      dot_color="#DD2222",
                      extreme_raw=None, extreme_bins=None, extreme_beta=None):
    """
    Two-layer halo:
      • faint background  = 5th–95th of raw (unbinned) periodograms
      • main grey band    = 5th–95th of *binned* PSDs at bin centres
                           (same quantity from which beta is derived)
    True periodogram + binned true PSD + power-law fit = reference.

    Optional extreme sample overlay (magenta):
      extreme_raw   : (n_freq,)  raw periodogram of the selected sample
      extreme_bins  : (n_bins,)  binned PSD of the selected sample
      extreme_beta  : float      beta value for the legend label
    """
    # --- main binned-PSD halo (at bin centres)
    ax.fill_between(bin_centers, bin_p5, bin_p95,
                    color="grey", alpha=0.55, zorder=1,
                    label="5%–95% binned PSDs (posterior)")

    # --- True spectrum
    if true_res is not None:
        ax.plot(true_res["freq_nz"], true_res["psd_nz"],
                ".", color=dot_color, markersize=4, alpha=0.6, zorder=2,
                label="True periodogram")
        ax.plot(true_res["bin_centers"], true_res["psd_means"],
                "s", color=fit_color, markersize=7, zorder=3,
                label=rf"True binned PSD  ($\beta={true_res['beta']:.2f}$)")
        fit_curve = 10 ** power_law_func(freq_nz, *true_res["popt"])
        ax.plot(freq_nz, fit_curve, "--", color="black",
                linewidth=2.5, zorder=4, label="Power-law fit (true)")

    # --- orange circles on top so they remain visible over the binned PSD squares
    ax.plot(bin_centers, bin_p50, "o", color="darkorange",
            markersize=5, linewidth=0, zorder=5,
            label="Binned PSD median (posterior)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.tick_params(which="major", length=8,  width=1.5)
    ax.tick_params(which="minor", length=5,  width=1.0)
    ax.set_xlabel("Frequency [Hz]", fontsize=12)
    ax.set_ylabel("PSD", fontsize=12)
    ax.legend(fontsize=11, loc="lower left")

_plot_psd_methodB(ax_nh_psd,
                  nh_freq_nz, nh_raw_p5, nh_raw_p95,
                  nh_bin_centers, nh_bin_p5, nh_bin_p50, nh_bin_p95,
                  true_nh_res, "NH process PSD", "crimson")
_plot_psd_methodB(ax_phi_psd,
                  phi_freq_nz, phi_raw_p5, phi_raw_p95,
                  phi_bin_centers, phi_bin_p5, phi_bin_p50, phi_bin_p95,
                  true_phi_res, "Phi process PSD", "darkgreen",
                  dot_color="#3A9E3A")


# ── Row 3: β distributions ────────────────────────────────────────────────

def _plot_beta_dist(ax, betas, true_beta_polyfit, label, hist_color):
    """
    Histogram of β values from all 6000 posterior sample periodograms.
    Vertical line marks the β recovered from the TRUE time series.
    """
    b5, b50, b95 = numpy.percentile(betas, [5, 50, 95])

    ax.hist(betas, bins=60, color=hist_color, alpha=0.55,
            density=True, label="Posterior sample betas")

    # 5–95% credible span
    ax.axvspan(b5, b95, alpha=0.18, color=hist_color,
               label=f"5%–95% CI  [{b5:.2f}, {b95:.2f}]")

    # Posterior median
    ax.axvline(b50, color="black", linewidth=2.0, linestyle="-",
               label=rf"Posterior median  $\beta={b50:.2f}$")

    # True beta (same polyfit method)
    ax.axvline(true_beta_polyfit, color="crimson", linewidth=2.0, linestyle="--",
               label=rf"True $\beta={true_beta_polyfit:.2f}$")

    ax.set_xlabel(r"$\beta$", fontsize=13)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)

_plot_beta_dist(ax_nh_beta,  nh_betas,  true_nh_beta_polyfit,
                r"NH  —  $\beta$ distribution (Method B)", "steelblue")
_plot_beta_dist(ax_phi_beta, phi_betas, true_phi_beta_polyfit,
                r"Phi —  $\beta$ distribution (Method B)", "seagreen")


# ── Save PDF and PNG (no suptitle) ───────────────────────────────────────
base_name = f"psd_beta_recovery_Nh{NH}_Tau{TAU}_Phi{PHI}_methodB_paper"
for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
    path = os.path.join(RESULTS_DIR, f"{base_name}.{ext}")
    fig.savefig(path, bbox_inches="tight", **kw)
    print(f"Plot saved → {path}")
plt.close(fig)
print("\nDone.")
