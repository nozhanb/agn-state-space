"""
visualise_results.py
====================
Visualise posterior inference results against the ground-truth synthetic data.

Produces two diagnostic figures for a given inference run:

**Figure 1 — Trace comparison (3 panels)**
    Top-to-bottom: posterior NH(t) trajectories vs ground truth, posterior
    φ(t) trajectories vs ground truth, and posterior predictive total counts
    vs observed total counts.  Black traces = posterior samples; red = truth.

**Figure 2 — Posterior histograms (2 panels)**
    Marginal posteriors for the long-run mean parameters μ_NH (``nH_shift``)
    and μ_φ (``phi_shift``), with vertical lines marking the true values,
    posterior medians, and 2.5/97.5 percentiles.

Usage
-----
Run from the repository root::

    python src/visualise_results.py

Edit the ``CONFIGURATION`` block below to select the inference output file and
adjust which realisation was used as the "observed" data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION — edit here to select the dataset and display options
# ---------------------------------------------------------------------------

# Must match the labels used in run_inference.py and generate_synthetic_data.py
NH_LABEL:  int = 24
TAU_LABEL: int = 50
PHI_LABEL: int = -3

# Simulation realisation index used as the "observed" spectrum
SIM_INDEX: int = 500

# Number of posterior samples to overlay (set lower for faster plotting)
N_SAMPLES_TO_PLOT: int = 500

# ---------------------------------------------------------------------------
# MATPLOTLIB STYLE
# ---------------------------------------------------------------------------
plt.rcParams['axes.linewidth']    = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['xtick.major.size']  = 6
plt.rcParams['ytick.major.size']  = 6
plt.rcParams['xtick.minor.size']  = 4
plt.rcParams['ytick.minor.size']  = 4
plt.rcParams['xtick.labelsize']   = 12
plt.rcParams['ytick.labelsize']   = 12
plt.rcParams['axes.labelweight']  = 'bold'

# ---------------------------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------------------------
_SRC_DIR    = Path(__file__).parent
_REPO_DIR   = _SRC_DIR.parent
_DATA_DIR   = _REPO_DIR / "data"
_INFER_DIR  = _REPO_DIR / "output" / "inference"
_PLOT_DIR   = _REPO_DIR / "output" / "plots"


# ---------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------------------------

def plot_traces(
    steps: np.ndarray,
    nh_y: np.ndarray,
    phi_y: np.ndarray,
    total_count: np.ndarray,
    observed_nh: np.ndarray,
    observed_phi: np.ndarray,
    observed_count: np.ndarray,
    n_samples: int = N_SAMPLES_TO_PLOT,
    save_path: Path = None,
) -> plt.Figure:
    """Plot posterior AR(1) trajectories and predictive counts vs ground truth.

    Creates a 3-panel figure (shared x-axis) showing:

    - **Top**: log₁₀(NH(t)) — posterior trajectories (black) and truth (red).
    - **Middle**: log₁₀(φ(t)) — posterior trajectories (black) and truth (red).
    - **Bottom**: Total counts per time step — posterior predictive (black)
      and observed (red).

    Parameters
    ----------
    steps : np.ndarray, shape (T,)
        Integer time indices.
    nh_y : np.ndarray, shape (T, S)
        Posterior log₁₀(NH(t)) trajectories (S = number of posterior samples).
    phi_y : np.ndarray, shape (T, S)
        Posterior log₁₀(φ(t)) trajectories.
    total_count : np.ndarray, shape (T, S)
        Posterior predictive total counts per time step.
    observed_nh : np.ndarray, shape (T,)
        Ground-truth log₁₀(NH(t)) time series.
    observed_phi : np.ndarray, shape (T,)
        Ground-truth log₁₀(φ(t)) time series.
    observed_count : np.ndarray, shape (T, N_chan)
        Observed count spectrum; summed over channels for comparison.
    n_samples : int, optional
        Number of posterior samples to overlay (default: ``N_SAMPLES_TO_PLOT``).
    save_path : Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(13, 8))

    ax[0].plot(steps, nh_y[:, :n_samples],        "k", alpha=0.7)
    ax[0].plot(steps, observed_nh,                 "r", alpha=1.0)

    ax[1].plot(steps, phi_y[:, :n_samples],        "k", alpha=0.7)
    ax[1].plot(steps, observed_phi,                "r", alpha=1.0)

    ax[2].plot(steps, total_count[:, :n_samples],  "k", alpha=0.7)
    ax[2].plot(steps, observed_count.sum(axis=-1), "r", alpha=1.0)

    ax[2].set_xlabel(r"Time",    fontsize=15)
    ax[0].set_ylabel(r"$N_H$",   fontsize=15)
    ax[1].set_ylabel(r"$\phi$",  fontsize=15)
    ax[2].set_ylabel(r"$Count$", fontsize=15)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Trace figure saved → {save_path}")

    return fig


def plot_posterior_histograms(
    nh_shift_samples: np.ndarray,
    phi_shift_samples: np.ndarray,
    observed_nh: np.ndarray,
    observed_phi: np.ndarray,
    observed_count: np.ndarray,
    save_path: Path = None,
) -> plt.Figure:
    """Plot marginal posteriors for the NH and φ long-run mean parameters.

    Creates a 2-panel histogram figure with reference lines at:

    - The ground-truth median (red solid).
    - The posterior median (magenta dashed).
    - The 2.5 and 97.5 percentiles of the posterior (black dashed).

    Also annotates the panel with the ground-truth NH mean, φ mean, and
    total observed count.

    Parameters
    ----------
    nh_shift_samples : np.ndarray, shape (S,)
        Posterior samples of μ_NH (``nH_shift`` in log₁₀ space).
    phi_shift_samples : np.ndarray, shape (S,)
        Posterior samples of μ_φ (``phi_shift`` in log₁₀ space).
    observed_nh : np.ndarray, shape (T,)
        Ground-truth log₁₀(NH(t)) time series.
    observed_phi : np.ndarray, shape (T,)
        Ground-truth log₁₀(φ(t)) time series.
    observed_count : np.ndarray, shape (T, N_chan)
        Observed count spectrum.
    save_path : Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig : plt.Figure
    """
    thickness = 2.5

    nh_5,  nh_95  = np.percentile(nh_shift_samples,  [2.5, 97.5], method="nearest")
    phi_5, phi_95 = np.percentile(phi_shift_samples, [2.5, 97.5], method="nearest")

    print(f"NH  posterior 95% CI : [{nh_5:.3f}, {nh_95:.3f}]")
    print(f"phi posterior 95% CI : [{phi_5:.3f}, {phi_95:.3f}]")

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # ── NH histogram ──────────────────────────────────────────────────────────
    nh_counts, _, _ = ax[0].hist(nh_shift_samples, bins=30,
                                 label=r"posterior $\mu_{N_H}$ samples", alpha=0.7)
    nh_max = float(np.max(nh_counts))

    ax[0].vlines(np.median(observed_nh), 0, nh_max,
                 colors="r",   linewidth=thickness, label="ground-truth median")
    ax[0].vlines(np.median(nh_shift_samples), 0, nh_max,
                 colors="m",   linewidth=thickness, linestyle="dashed", label="posterior median")
    ax[0].vlines([nh_5, nh_95], 0, nh_max,
                 colors="k",   linewidth=thickness, linestyle="dashed", label=r"95% CI")

    ax[0].set_ylim(0, nh_max * 1.01)
    ax[0].set_xlabel(r"$\mu_{N_H}$")
    ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax[0].legend()

    # Annotation with ground-truth statistics
    ax[0].annotate(
        r"$N_H$ mean = {:.2f}".format(observed_nh.mean()),
        xy=(0.025, 0.70), xycoords="axes fraction", ha="left", va="bottom",
    )
    ax[0].annotate(
        r"$\phi$ mean = {:.2f}".format(observed_phi.mean()),
        xy=(0.025, 0.65), xycoords="axes fraction", ha="left", va="bottom",
    )
    ax[0].annotate(
        "Count = {}".format(observed_count.sum()),
        xy=(0.025, 0.60), xycoords="axes fraction", ha="left", va="bottom",
    )

    # ── φ histogram ───────────────────────────────────────────────────────────
    phi_counts, _, _ = ax[1].hist(phi_shift_samples, bins=30,
                                  label=r"posterior $\mu_\phi$ samples", alpha=0.7)
    phi_max = float(np.max(phi_counts))

    ax[1].vlines(np.median(observed_phi), 0, phi_max,
                 colors="r",   linewidth=thickness, label="ground-truth median")
    ax[1].vlines(np.median(phi_shift_samples), 0, phi_max,
                 colors="m",   linewidth=thickness, linestyle="dashed", label="posterior median")
    ax[1].vlines([phi_5, phi_95], 0, phi_max,
                 colors="k",   linewidth=thickness, linestyle="dashed", label=r"95% CI")

    ax[1].set_ylim(0, phi_max * 1.01)
    ax[1].set_xlabel(r"$\mu_\phi$", fontsize=15)
    ax[1].legend()

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Histogram figure saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """Load inference results and synthetic data, then produce diagnostic plots."""
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"Nh_{NH_LABEL}_and_Tau_{TAU_LABEL}_and_phi_{PHI_LABEL}"

    # ── Load inference output ────────────────────────────────────────────────
    infer_file = (
        _INFER_DIR /
        f"synthetic_count_spec_visualisation_SDSSJ0932+0405_"
        f"multiple_simulated_stochastic_{tag}_"
        f"steps_100_iter_6000_spectrum_{SIM_INDEX}_redshift_0108.npz"
    )
    print(f"Loading inference file: {infer_file}")
    parameters = np.load(infer_file, allow_pickle=True)

    # ── Load ground-truth synthetic data ─────────────────────────────────────
    data_file = (
        _DATA_DIR /
        f"synthetic_count_NH_and_phi_spec_visualisation_"
        f"SDSSJ0932+0405_multiple_simulated_stochastic_{tag}.npz"
    )
    print(f"Loading synthetic data: {data_file}")
    object_data = np.load(data_file)

    # ── Extract arrays ───────────────────────────────────────────────────────
    observed_nh    = object_data["nh_process"]                              # (T,)
    observed_phi   = object_data["phi_process"]                             # (T,)
    observed_count = object_data["all_simulated_count"][SIM_INDEX, :, :]   # (T, N_chan)

    nh_shift_samples  = parameters["nH_shift"]                             # (S,)
    phi_shift_samples = parameters["phi_shift"]                            # (S,)
    total_count       = parameters["posterior_predictive_total"]            # (S, T)

    # Swap axes to (T, S) for plotting
    nh_y        = np.swapaxes(parameters["nh_y"],  0, 1)  # (T, S)
    phi_y       = np.swapaxes(parameters["phi_y"], 0, 1)  # (T, S)
    total_count = np.swapaxes(total_count,          0, 1)  # (T, S)

    steps = np.arange(observed_nh.shape[0])

    print(f"\nGround-truth shapes — NH: {observed_nh.shape}  φ: {observed_phi.shape}  count: {observed_count.shape}")
    print(f"Posterior shapes    — NH_y: {nh_y.shape}  phi_y: {phi_y.shape}  total_count: {total_count.shape}")

    # ── Figure 1: trace comparison ───────────────────────────────────────────
    plot_traces(
        steps, nh_y, phi_y, total_count,
        observed_nh, observed_phi, observed_count,
        n_samples=N_SAMPLES_TO_PLOT,
        save_path=_PLOT_DIR / f"traces_{tag}.png",
    )

    # ── Figure 2: posterior histograms ───────────────────────────────────────
    plot_posterior_histograms(
        nh_shift_samples, phi_shift_samples,
        observed_nh, observed_phi, observed_count,
        save_path=_PLOT_DIR / f"histograms_{tag}.png",
    )

    plt.show()


if __name__ == "__main__":
    main()
