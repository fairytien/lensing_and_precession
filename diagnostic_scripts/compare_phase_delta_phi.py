#!/usr/bin/env python3
"""
Compare phase_delta_phi BEFORE vs AFTER adding the correction term in integrand_delta_phi.

- BEFORE: integrate the "base" integrand (no correction term), with face-on branch
- AFTER:  use the class implementation (Classes_v2.Precessing.phase_delta_phi),
          which integrates base + correction term and handles special cases.

We provide frequency-sweep plots for face-on (cos i_JN = 1) and edge-on (cos i_JN = 0),
plus an optional difference plot (after - before).

Usage examples:
  python scripts/compare_phase_delta_phi.py
  python scripts/compare_phase_delta_phi.py --no-show --save
  python scripts/compare_phase_delta_phi.py --save --outdir ./my_plots
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Ensure we can import from modules
HERE = os.path.dirname(__file__)
MODULES_DIR = os.path.abspath(os.path.join(HERE, "..", "modules"))
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from Classes_v2 import Precessing, SOLMASS2SEC, GIGAPC2SEC, NEAR_ZERO_THRESHOLD  # type: ignore


def make_precessing_params(theta_S, phi_S, theta_J, phi_J):
    """Construct a minimal valid parameter set for Precessing."""
    params = {
        "theta_S": float(theta_S),
        "phi_S": float(phi_S),
        "theta_J": float(theta_J),
        "phi_J": float(phi_J),
        "mcz": 30.0 * SOLMASS2SEC,
        "dist": 1.0 * GIGAPC2SEC,
        "eta": 0.25,
        "t_c": 0.0,
        "phi_c": 0.0,
        "theta_tilde": 0.3,  # precession amplitude
        "omega_tilde": 0.1,  # precession frequency
        "gamma_P": 0.0,
    }
    return params


def integrand_components(pre: Precessing, f: float):
    """Compute common components used by both BEFORE and AFTER formulas."""
    LdotN = pre.LdotN(f)
    cos_i_JN, sin_i_JN, *_ = pre.precession_angles()
    theta_LJ = pre.theta_LJ(f)
    phi_LJ = pre.phi_LJ(f)
    f_dot = pre.f_dot(f)
    Omega_LJ = (
        1000.0
        * pre.omega_tilde
        * (f / pre.f_cut()) ** (5.0 / 3.0)
        / (pre.total_mass() / SOLMASS2SEC)
    )
    return LdotN, cos_i_JN, sin_i_JN, theta_LJ, phi_LJ, f_dot, Omega_LJ


def integrand_before(pre: Precessing, f: float):
    """
    BEFORE: No correction term.
    Matches the structure of current implementation but omits the extra correction.
    """
    if pre.theta_tilde == 0:
        return 0.0

    LdotN, cos_i_JN, sin_i_JN, theta_LJ, phi_LJ, f_dot, Omega_LJ = integrand_components(
        pre, f
    )

    # Face-on / face-off branch
    face_on = np.abs(1.0 - np.abs(cos_i_JN)) < NEAR_ZERO_THRESHOLD
    if face_on:
        return -Omega_LJ * np.cos(theta_LJ) / f_dot

    # Generic (non face-on)
    base = (
        (LdotN / (1.0 - LdotN**2))
        * Omega_LJ
        * np.sin(theta_LJ)
        * (np.cos(theta_LJ) * sin_i_JN * np.sin(phi_LJ) - np.sin(theta_LJ) * cos_i_JN)
        / f_dot
    )
    return base


def phase_before(pre: Precessing, f_grid: np.ndarray) -> np.ndarray:
    """
    Compute phase_delta_phi BEFORE by integrating integrand_before over frequency.
    Uses cumulative trapezoid integration to match the d/d f formulation.
    """
    integrand_vals = np.array([integrand_before(pre, float(fi)) for fi in f_grid])
    phase_vals = cumulative_trapezoid(integrand_vals, f_grid, initial=0.0)
    return phase_vals


def phase_after(pre: Precessing, f_grid: np.ndarray) -> np.ndarray:
    """
    Compute phase_delta_phi AFTER using the class method (which includes the correction term).
    """
    return pre.phase_delta_phi(f_grid)


def sweep_frequency_and_plot(
    show: bool = True, save: bool = False, outdir: str = "figures/phase_compare"
):
    os.makedirs(outdir, exist_ok=True)

    # Frequency grid
    f = np.linspace(20.0, 300.0, 600)

    # Geometry: choose theta_S = 0 so cos_i_JN = cos(theta_J)
    theta_S = 0.0
    phi_S = 0.0
    phi_J = 0.0

    # Face-on: cos_i_JN = 1 => theta_J = 0
    params_on = make_precessing_params(theta_S, phi_S, 0.0, phi_J)
    pre_on = Precessing(params_on)

    # Edge-on: cos_i_JN = 0 => theta_J = pi/2
    params_edge = make_precessing_params(theta_S, phi_S, np.pi / 2.0, phi_J)
    pre_edge = Precessing(params_edge)

    # Evaluate phases
    ph_b_on = phase_before(pre_on, f)
    ph_a_on = phase_after(pre_on, f)
    ph_b_ed = phase_before(pre_edge, f)
    ph_a_ed = phase_after(pre_edge, f)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Face-on phase
    axs[0, 0].plot(f, ph_b_on, label="before", alpha=0.85)
    axs[0, 0].plot(f, ph_a_on, label="after", alpha=0.85)
    axs[0, 0].set_title("phase_delta_phi vs f (face-on: cos i_JN = 1)")
    axs[0, 0].set_ylabel("phase_delta_phi")
    axs[0, 0].legend()

    # Face-on difference
    axs[0, 1].plot(f, ph_a_on - ph_b_on, color="C3", alpha=0.9)
    axs[0, 1].set_title("after - before (face-on)")

    # Edge-on phase
    axs[1, 0].plot(f, ph_b_ed, label="before", alpha=0.85)
    axs[1, 0].plot(f, ph_a_ed, label="after", alpha=0.85)
    axs[1, 0].set_title("phase_delta_phi vs f (edge-on: cos i_JN = 0)")
    axs[1, 0].set_xlabel("f (Hz)")
    axs[1, 0].set_ylabel("phase_delta_phi")
    axs[1, 0].legend()

    # Edge-on difference
    axs[1, 1].plot(f, ph_a_ed - ph_b_ed, color="C3", alpha=0.9)
    axs[1, 1].set_title("after - before (edge-on)")
    axs[1, 1].set_xlabel("f (Hz)")

    plt.tight_layout()
    outpath = os.path.join(outdir, "phase_delta_phi_face_on_edge_on.png")
    if save:
        fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare phase_delta_phi BEFORE vs AFTER the correction term for face-on and edge-on."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display figures interactively."
    )
    parser.add_argument("--save", action="store_true", help="Save figures to disk.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures/phase_compare",
        help="Output directory for saved figures.",
    )
    args = parser.parse_args()

    show = not args.no_show
    save = args.save
    outdir = args.outdir

    sweep_frequency_and_plot(show=show, save=save, outdir=outdir)
