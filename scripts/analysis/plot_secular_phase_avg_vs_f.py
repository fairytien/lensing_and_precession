"""Plot the precession-cycle-averaged secular phase ⟨φ_p + 2δΦ⟩ vs frequency.

Layout mirrors Figure 4 of the RP paper (fig: phi_p plus 2 deltaphi p) but
shows the precession-cycle average instead of the raw values.  The γ_P row
is omitted (the secular average is independent of γ_P).

Columns: Systems 1 (face-on), 2 (edge-on), 3 (random).
Row 0:   Vary θ̃ at fixed Ω̃ = 2, γ_P = 0.
Row 1:   Vary Ω̃ at fixed θ̃ = 4, γ_P = 0.

Purpose: investigate the secular phase accumulation δΦ ∝ Ω̃ θ̃² derived in
         shenanigans/secular_phase.tex.

Usage:
    python -m scripts.analysis.plot_secular_phase_avg_vs_f [--save]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.Classes import Precessing
from modules.cosmology import apply_z
from modules.default_params import SOLMASS2SEC, RP_params_0, loc_params
from modules.plot_utils import apply_physics_paper_style, LBL_OMEGA, LBL_THETA, LBL_F
from modules.waveform import set_orientation


# ============================================================================
# Parameters matching paper Figure 4
# ============================================================================

MCZ_SRC_MSUN = 10.0  # source-frame chirp mass [Msun]
ETA = 0.25
Z = 0.3
F_MIN = 20.0
DELTA_F = 0.05  # fine grid for accurate cycle detection

THETA_VALS = [1, 4, 8]
OMEGA_VALS = [1, 2, 3]
FIXED_THETA = 4
FIXED_OMEGA = 2

SYSTEM_NAMES = ["faceon", "edgeon", "random"]
SYSTEM_LABELS = ["System 1 (face-on)", "System 2 (edge-on)", "System 3 (random)"]

LINE_STYLES = ["-", "--", ":"]
LINE_COLORS = ["black", "black", "black"]


# ============================================================================
# Cycle-Boundary Averaging
# ============================================================================


def precession_cycle_average(f_arr, phase_arr, phi_LJ_arr):
    """Average *phase_arr* over each complete precession cycle.

    A cycle boundary is defined where φ_LJ crosses a multiple of 2π.
    Returns (f_mid, phase_avg) arrays with one point per complete cycle.
    """
    # phi_LJ is monotonically increasing; find where it crosses 2πk
    cycle_number = phi_LJ_arr / (2 * np.pi)
    # Integer crossings: floor(cycle_number) changes value
    floor_cn = np.floor(cycle_number).astype(int)
    # Find indices where a new cycle begins
    boundary_mask = np.diff(floor_cn) > 0
    boundary_indices = np.where(boundary_mask)[0] + 1  # index of first point in new cycle

    if len(boundary_indices) < 2:
        # Fewer than 2 boundaries → no complete cycles to average
        return np.array([]), np.array([])

    # Average within each pair of consecutive boundaries
    n_cycles = len(boundary_indices) - 1
    f_mid = np.empty(n_cycles)
    phase_avg = np.empty(n_cycles)

    for i in range(n_cycles):
        i_start = boundary_indices[i]
        i_end = boundary_indices[i + 1]
        f_mid[i] = 0.5 * (f_arr[i_start] + f_arr[i_end - 1])
        phase_avg[i] = np.mean(phase_arr[i_start:i_end])

    return f_mid, phase_avg


# ============================================================================
# Parameter Setup
# ============================================================================


def _build_rp_params(orientation_name, theta_tilde, omega_tilde, gamma_P=0.0):
    """Build a redshifted RP parameter dict for the given system and RP params."""
    base = RP_params_0.copy()
    base["mcz"] = MCZ_SRC_MSUN * SOLMASS2SEC
    base["eta"] = ETA
    base["theta_tilde"] = theta_tilde
    base["omega_tilde"] = omega_tilde
    base["gamma_P"] = gamma_P

    orient = loc_params["Taman"][orientation_name]
    (params,) = set_orientation(orient, base)
    params = apply_z(params, Z)
    return params


# ============================================================================
# Plotting
# ============================================================================


def make_figure(save: bool = False):
    apply_physics_paper_style()

    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(18, 10), sharex="col"
    )
    fig.subplots_adjust(wspace=0.30, hspace=0.12)

    for col, (orient_name, sys_label) in enumerate(
        zip(SYSTEM_NAMES, SYSTEM_LABELS)
    ):
        # ---- Row 0: vary θ̃ at fixed Ω̃ ----
        for i, theta in enumerate(THETA_VALS):
            params = _build_rp_params(orient_name, theta, FIXED_OMEGA)
            inst = Precessing(params)
            f_cut = inst.f_cut()
            f_arr = np.arange(F_MIN, f_cut, DELTA_F)

            raw_phase = inst.phase_phi_P(f_arr) + 2 * inst.phase_delta_phi(f_arr)
            phi_LJ = inst.phi_LJ(f_arr)

            f_mid, phase_avg = precession_cycle_average(f_arr, raw_phase, phi_LJ)

            label = f"{LBL_THETA} = {theta}"
            axes[0, col].plot(
                f_mid, phase_avg,
                ls=LINE_STYLES[i], color=LINE_COLORS[i], label=label,
            )

        # ---- Row 1: vary Ω̃ at fixed θ̃ ----
        for i, omega in enumerate(OMEGA_VALS):
            params = _build_rp_params(orient_name, FIXED_THETA, omega)
            inst = Precessing(params)
            f_cut = inst.f_cut()
            f_arr = np.arange(F_MIN, f_cut, DELTA_F)

            raw_phase = inst.phase_phi_P(f_arr) + 2 * inst.phase_delta_phi(f_arr)
            phi_LJ = inst.phi_LJ(f_arr)

            f_mid, phase_avg = precession_cycle_average(f_arr, raw_phase, phi_LJ)

            label = f"{LBL_OMEGA} = {omega}"
            axes[1, col].plot(
                f_mid, phase_avg,
                ls=LINE_STYLES[i], color=LINE_COLORS[i], label=label,
            )

        # Column title
        axes[0, col].set_title(sys_label, fontsize=13)

    # Axis labels and legends
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            ax.legend(fontsize=10, loc="best")
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if row == 1:
                ax.set_xlabel(LBL_F, fontsize=13)

    # Shared y-label
    fig.text(
        0.04, 0.5,
        r"$\langle\phi_p + 2\delta\Phi\rangle\;\;[\mathrm{rad}]$",
        va="center", rotation="vertical", fontsize=16,
    )

    fig.suptitle(
        rf"Precession-cycle average of $\phi_p + 2\delta\Phi$"
        rf"  ($\mathcal{{M}} = {MCZ_SRC_MSUN:.0f}\,\mathrm{{M}}_\odot$,"
        rf"  $z = {Z}$)",
        fontsize=14, y=0.98,
    )

    if save:
        out_dir = os.path.join(REPO_ROOT, "figures", "secular_phase")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"secular_phase_avg_mcz{MCZ_SRC_MSUN:.0f}_z{Z}.pdf"
        )
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out_path}")

    plt.show()
    return fig, axes


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot ⟨φ_p + 2δΦ⟩ (precession-cycle average) vs frequency."
    )
    parser.add_argument(
        "--save", action="store_true", help="Save figure to figures/secular_phase/"
    )
    args = parser.parse_args()
    make_figure(save=args.save)


if __name__ == "__main__":
    main()
