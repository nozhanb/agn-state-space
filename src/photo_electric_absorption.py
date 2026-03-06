"""
photo_electric_absorption.py
============================
Computes X-ray photoelectric absorption cross-sections following the
piecewise polynomial approximation of Morrison & McCammon (1983).

The cross-section σ(E) is given by:

    σ(E) = (c₀ + c₁·E + c₂·E²) · E⁻³ × 10⁻²⁴  [cm²/H atom]

where c₀, c₁, c₂ are tabulated coefficients that depend on the energy band.
This model underlies the WABS and ZWABS absorption components in XSPEC/CIAO,
which account for interstellar and intrinsic X-ray absorption respectively.

References
----------
Morrison, R. & McCammon, D. (1983), ApJ, 270, 119.
XSPEC WABS model: https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelWabs.html

Usage
-----
Run as a script to pre-compute and save cross-sections for a given source::

    python photo_electric_absorption.py

This saves ``photo_electric_sigma_<tag>.npz`` into the ``data/`` directory.
Edit ``REDSHIFT`` and ``OUTPUT_TAG`` at the bottom of this file for a new source.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Morrison & McCammon (1983) coefficient table.
# Each row: [E_lo (keV), E_hi (keV), c0, c1, c2]
#
# Note: energies above 10 keV are not covered by the original table.
# The final band (8.331 keV → ∞) extends the 8.331–10.0 keV coefficients
# to higher energies, following common practice in X-ray spectral modelling.
# ---------------------------------------------------------------------------
_MM83_BANDS = np.array([
    [0.030, 0.100,   17.3,  608.1, -2150.0],
    [0.100, 0.284,   34.6,  267.9,  -476.1],
    [0.284, 0.400,   78.1,   18.8,     4.3],
    [0.400, 0.532,   71.4,   66.8,   -51.4],
    [0.532, 0.707,   95.5,  145.8,   -61.1],
    [0.707, 0.867,  308.9, -380.6,   294.0],
    [0.867, 1.303,  120.6,  169.3,   -47.7],
    [1.303, 1.840,  141.3,  146.8,   -31.5],
    [1.840, 2.471,  202.7,  104.7,   -17.0],
    [2.471, 3.210,  342.7,   18.7,     0.0],
    [3.210, 4.038,  352.2,   18.7,     0.0],
    [4.038, 7.111,  433.9,   -2.4,     0.75],
    [7.111, 8.331,  629.0,   30.9,     0.0],
    [8.331, np.inf, 701.2,   25.2,     0.0],  # extended beyond 10 keV
])


def photo_electric_absorption(energy: np.ndarray) -> np.ndarray:
    """Compute X-ray photoelectric absorption cross-sections.

    Evaluates the piecewise polynomial approximation from Morrison &
    McCammon (1983) for the photoelectric absorption cross-section σ(E):

        σ(E) = (c₀ + c₁·E + c₂·E²) · E⁻³ × 10⁻²⁴  [cm²/H atom]

    where the coefficients (c₀, c₁, c₂) are tabulated per energy band
    (see ``_MM83_BANDS`` above). This function underlies the WABS (galactic
    absorption) and ZWABS (intrinsic absorption) spectral components used
    in the count simulator.

    Parameters
    ----------
    energy : np.ndarray
        Array of photon energies in keV. Typical range for X-ray
        spectroscopy: 0.03–11 keV. Energies below 0.03 keV are outside
        the Morrison & McCammon table and will return σ = 0.

    Returns
    -------
    sigma : np.ndarray
        Photoelectric absorption cross-section in cm² per hydrogen atom,
        with the same shape as ``energy``.

    Examples
    --------
    >>> import numpy as np
    >>> energy = np.linspace(0.3, 10.0, 5)
    >>> sigma = photo_electric_absorption(energy)
    >>> sigma.shape
    (5,)
    >>> sigma.min() > 0
    True

    References
    ----------
    Morrison, R. & McCammon, D. (1983), ApJ, 270, 119.
    """
    energy = np.asarray(energy, dtype=np.float64)
    sigma  = np.zeros_like(energy)

    for e_lo, e_hi, c0, c1, c2 in _MM83_BANDS:
        mask = (energy >= e_lo) & (energy < e_hi)
        E = energy[mask]
        sigma[mask] = (c0 + c1 * E + c2 * E**2) * E**-3 * 1e-24

    return sigma


# ---------------------------------------------------------------------------
# Script entry point: pre-compute and save cross-sections for a given source.
#
# Loads the instrument energy grid from data/fake_count.npz, computes σ(E)
# at both rest-frame and redshifted energies, and saves the results.
#
# Edit REDSHIFT and OUTPUT_TAG below for a different source.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    REDSHIFT   = 0.0108   # redshift of SDSSJ0932+0405
    OUTPUT_TAG = "0108"   # appended to the output filename

    # Resolve data directory relative to this script, regardless of where
    # the script is called from.
    data_dir = Path(__file__).parent.parent / "data"

    data  = np.load(data_dir / "fake_count.npz")
    e_mid = data["energy"].astype("float32")

    sigma          = photo_electric_absorption(e_mid)
    sigma_redshift = photo_electric_absorption((1.0 + REDSHIFT) * e_mid)

    out_path = data_dir / f"photo_electric_sigma_redshift_{OUTPUT_TAG}.npz"
    np.savez(out_path, sigma=sigma, sigma_redshift=sigma_redshift)

    print(f"Cross-sections saved  : {out_path}")
    print(f"Energy grid           : {e_mid.shape[0]} channels, "
          f"{e_mid.min():.3f}–{e_mid.max():.3f} keV")
    print(f"σ range (rest-frame)  : {sigma.min():.3e} – {sigma.max():.3e} cm²/H")
    print(f"σ range (redshifted)  : {sigma_redshift.min():.3e} – "
          f"{sigma_redshift.max():.3e} cm²/H")
