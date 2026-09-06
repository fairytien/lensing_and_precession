"""Plot the secular precessional phase accumulation over the (omega, theta) plane.

Reproduces the appendix figure "secular phase RP" of the lensing-vs-precession
paper.  The left panel shows the precessional contribution to the GW phase
accumulated across the inspiral band,

    Delta(phi_p + 2 delta_Phi) = (phi_p + 2 delta_Phi) evaluated f_min -> f_cut,

as a function of the dimensionless precession frequency omega_tilde and
amplitude theta_tilde.  The right panel shows the fractional residual of the
quadratic-order approximation
Delta(phi_p + 2 delta_Phi) ~ C * omega_tilde * theta_tilde**2, whose constant
C is evaluated in the small-amplitude limit so that the residual map is an
independent test of where the approximation holds.

The orientation is fixed to System 1 (Taman_faceon), for which the oscillatory
contributions are suppressed and the phase accumulation is almost entirely
secular.

Usage:
    python -m scripts.analysis.plot_secular_phase_omega_theta
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.Classes import Precessing
from modules.cosmology import apply_z
from modules.default_params import SOLMASS2SEC, RP_params_0, loc_params
from modules.filenames import secular_phase_omega_theta_figure_filename
from modules.plot_utils import (
    LBL_OMEGA,
    LBL_SECULAR_PHASE,
    LBL_SECULAR_PHASE_RESIDUAL,
    LBL_THETA,
    apply_physics_paper_style,
    format_colorbar_ticks,
    save_figure,
    set_square_axes,
)
from modules.waveform import set_orientation


# ============================================================================
# Fixed Parameters
# ============================================================================

ETA = 0.25
F_MIN = 20.0
DELTA_F = 0.05
GAMMA_P = 0.0
ORIENTATION_TAG = "Taman_faceon"
PHASE_TICK_STEP = 2.5  # colorbar tick spacing for the phase panel [rad]

# Small-amplitude probe used to evaluate the leading-order constant C.
THETA_PROBE = 0.5
OMEGA_PROBE = 1.0


# ============================================================================
# Phase Grid
# ============================================================================


def _build_rp_params(mcz_src_msun, z, theta_tilde, omega_tilde):
    """Build a redshifted System 1 RP parameter dict for one grid point."""
    base = RP_params_0.copy()
    base["mcz"] = mcz_src_msun * SOLMASS2SEC
    base["eta"] = ETA
    base["theta_tilde"] = theta_tilde
    base["omega_tilde"] = omega_tilde
    base["gamma_P"] = GAMMA_P

    (params,) = set_orientation(loc_params["Taman"]["faceon"], base)
    return apply_z(params, z)


def accumulated_phase(mcz_src_msun, z, theta_tilde, omega_tilde):
    """Return Delta(phi_p + 2 delta_Phi) [rad] accumulated over the band.

    The endpoint difference is what cancels the arbitrary branch offset that
    np.unwrap introduces inside phase_phi_P.
    """
    inst = Precessing(_build_rp_params(mcz_src_msun, z, theta_tilde, omega_tilde))
    f_arr = np.arange(F_MIN, inst.f_cut(), DELTA_F)
    phase = inst.phase_phi_P(f_arr) + 2.0 * inst.phase_delta_phi(f_arr)
    return float(phase[-1] - phase[0])


def compute_secular_phase_grid(mcz_src_msun, z, omega_arr, theta_arr):
    """Return Delta(phi_p + 2 delta_Phi) [rad] on the (omega, theta) grid.

    The returned array has shape (n_theta, n_omega) to match the meshgrid
    convention used by the contour panels.
    """
    delta_phase = np.empty((theta_arr.size, omega_arr.size))
    for i, theta_tilde in enumerate(theta_arr):
        for j, omega_tilde in enumerate(omega_arr):
            delta_phase[i, j] = accumulated_phase(
                mcz_src_msun, z, theta_tilde, omega_tilde
            )

    return delta_phase


def leading_order_constant(mcz_src_msun, z):
    """Return C in Delta(phi_p + 2 delta_Phi) = C * omega * theta**2.

    The relation is the leading term of an expansion in theta_LJ, so C is
    defined by the small-amplitude limit rather than by an average over the
    grid: averaging would let the large-theta region, where the expansion is
    known to fail, set the constant used to judge where it holds.  The ratio
    is independent of omega and flat in theta well below THETA_PROBE, while
    much smaller amplitudes lose accuracy to the phase integration itself.
    """
    delta_phase = accumulated_phase(mcz_src_msun, z, THETA_PROBE, OMEGA_PROBE)
    return delta_phase / (OMEGA_PROBE * THETA_PROBE**2)


# ============================================================================
# Plotting
# ============================================================================


def create_figure(
    output_path: str | None,
    mcz_src_msun: float,
    z: float,
    omega_max: float,
    theta_max: float,
    n_omega: int,
    n_theta: int,
    constant: float | None,
    levels_count: int,
    cmap: str,
    residual_vmax: float,
) -> None:
    omega_arr = np.linspace(0.0, omega_max, n_omega)
    theta_arr = np.linspace(0.0, theta_max, n_theta)
    omega_grid, theta_grid = np.meshgrid(omega_arr, theta_arr)

    delta_phase = compute_secular_phase_grid(mcz_src_msun, z, omega_arr, theta_arr)
    predictor = omega_grid * theta_grid**2

    if constant is None:
        constant = leading_order_constant(mcz_src_msun, z)
        print(f"Leading-order constant C = {constant:.4g}")
    else:
        print(f"Supplied constant C = {constant:.4g}")

    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.ma.masked_invalid(delta_phase / (constant * predictor) - 1.0)

    if output_path is None:
        output_path = secular_phase_omega_theta_figure_filename(
            fig_dir="figures/secular_phase",
            mcz_msun=mcz_src_msun,
            omega_min=float(omega_arr[0]),
            omega_max=float(omega_arr[-1]),
            theta_min=float(theta_arr[0]),
            theta_max=float(theta_arr[-1]),
            orientation_tag=ORIENTATION_TAG,
            z=z,
        )

    apply_physics_paper_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.3),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="compressed",
    )
    axes = axes[0]

    # A handful of points at small omega and large theta accumulate slightly
    # negative phase, so the scale is anchored at zero and extended downward.
    # Round the top up to a whole tick step so the ticks land on the boundaries.
    phase_vmin = 0.0
    phase_vmax = float(np.ceil(np.max(delta_phase) / PHASE_TICK_STEP) * PHASE_TICK_STEP)
    phase_cf = axes[0].contourf(
        omega_grid,
        theta_grid,
        delta_phase,
        levels=np.linspace(phase_vmin, phase_vmax, levels_count),
        cmap=cmap,
        extend="min",
    )
    residual_cf = axes[1].contourf(
        omega_grid,
        theta_grid,
        residual,
        levels=np.linspace(-residual_vmax, residual_vmax, levels_count),
        cmap=cmap,
        extend="both",
    )

    for ax in axes:
        ax.set_xlabel(LBL_OMEGA)
        ax.set_ylabel(LBL_THETA)
        # A colorbar separates the panels, so both keep their own tick labels.
        ax.tick_params(axis="y", which="both", labelleft=True)
        set_square_axes(ax)

    # Pin both sets of ticks to the color limits: a locator is free to overshoot
    # them, and with extend enabled the stray tick renders inside the arrow.
    phase_cbar = fig.colorbar(phase_cf, ax=axes[0])
    phase_cbar.set_label(LBL_SECULAR_PHASE)
    format_colorbar_ticks(
        phase_cbar,
        phase_vmin,
        phase_vmax,
        n_ticks=int(round(phase_vmax / PHASE_TICK_STEP)) + 1,
        use_locator=False,
        decimals=1,
    )

    residual_cbar = fig.colorbar(residual_cf, ax=axes[1])
    residual_cbar.set_label(LBL_SECULAR_PHASE_RESIDUAL)
    format_colorbar_ticks(
        residual_cbar,
        -residual_vmax,
        residual_vmax,
        n_ticks=5,
        use_locator=False,
        decimals=2,
    )

    save_figure(fig, output_path)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output figure path. Defaults to figures/secular_phase/"
            "secular_phase_RP_z<z>_mcz<mcz>_omega<range>_theta<range>_Taman_faceon.pdf."
        ),
    )
    parser.add_argument(
        "--mcz", type=float, default=15.0, help="Source-frame chirp mass [Msun]"
    )
    parser.add_argument("--z", type=float, default=1.0, help="Source redshift")
    parser.add_argument("--omega-max", type=float, default=6.0)
    parser.add_argument("--theta-max", type=float, default=15.0)
    parser.add_argument("--n-omega", type=int, default=61)
    parser.add_argument("--n-theta", type=int, default=76)
    parser.add_argument(
        "--constant",
        type=float,
        default=None,
        help="Fix C instead of evaluating the small-amplitude limit.",
    )
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument(
        "--residual-vmax",
        type=float,
        default=0.5,
        help="Symmetric colorbar limit for the fractional residual panel.",
    )
    args = parser.parse_args()

    create_figure(
        output_path=args.output,
        mcz_src_msun=args.mcz,
        z=args.z,
        omega_max=args.omega_max,
        theta_max=args.theta_max,
        n_omega=args.n_omega,
        n_theta=args.n_theta,
        constant=args.constant,
        levels_count=args.levels,
        cmap=args.cmap,
        residual_vmax=args.residual_vmax,
    )


if __name__ == "__main__":
    main()
