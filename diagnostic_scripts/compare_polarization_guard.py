#!/usr/bin/env python3
"""
Compare polarization calculations WITH and WITHOUT the asymptotic guard
to demonstrate Python's floating-point behavior near tan(psi) singularities.

This script shows:
1. WITHOUT guard: raw algebraic formulas (what Python does naturally)
2. WITH guard: asymptotic limit handling when |tan_psi| is extreme

Usage:
  python scripts/compare_polarization_with_without_guard.py
  python scripts/compare_polarization_with_without_guard.py --no-show --save
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import from modules
HERE = os.path.dirname(__file__)
MODULES_DIR = os.path.abspath(os.path.join(HERE, "..", "modules"))
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from Classes_v2 import Precessing, SOLMASS2SEC, GIGAPC2SEC, NEAR_ZERO_THRESHOLD  # type: ignore


def make_precessing_params(
    theta_S, phi_S, theta_J, phi_J, theta_tilde=4.0, omega_tilde=3.0
):
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
        "theta_tilde": float(theta_tilde),
        "omega_tilde": float(omega_tilde),
        "gamma_P": 0.0,
    }
    return params


def polarization_without_guard(pre: Precessing, f: float):
    """
    Compute polarization WITHOUT asymptotic guard - raw algebraic formulas.
    This shows what Python naturally does with floating-point arithmetic.
    """
    cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = pre.precession_angles()

    # C amplitude
    C_amp = np.sqrt(
        0.25 * (1 + np.cos(pre.theta_S) ** 2) ** 2 * (np.cos(2 * pre.phi_S) ** 2)
        + (np.cos(pre.theta_S) ** 2 * np.sin(2 * pre.phi_S) ** 2)
    )

    # alpha
    sin_alpha = np.cos(pre.theta_S) * np.sin(2 * pre.phi_S) / C_amp
    cos_alpha = (1 + np.cos(pre.theta_S) ** 2) * np.cos(2 * pre.phi_S) / (2 * C_amp)

    # tan_psi
    theta_LJ = pre.theta_LJ(f)
    phi_LJ = pre.phi_LJ(f)

    num_psi = (
        np.sin(theta_LJ)
        * (np.cos(phi_LJ) * sin_o_XH + np.sin(phi_LJ) * cos_i_JN * cos_o_XH)
        - np.cos(theta_LJ) * sin_i_JN * cos_o_XH
    )
    den_psi = (
        np.sin(theta_LJ)
        * (np.cos(phi_LJ) * cos_o_XH - np.sin(phi_LJ) * cos_i_JN * sin_o_XH)
        + np.cos(theta_LJ) * sin_i_JN * sin_o_XH
    )

    # Face-on special case
    face_on = np.abs(1 - np.abs(cos_i_JN)) < NEAR_ZERO_THRESHOLD
    if face_on:
        o_XH = np.arctan2(sin_o_XH, cos_o_XH)
        tan_psi = np.tan(o_XH + np.sign(cos_i_JN) * phi_LJ)
    else:
        tan_psi = num_psi / den_psi

    # Raw algebraic formulas WITHOUT guard
    T = tan_psi
    sin_2pa = (2 * cos_alpha * T + sin_alpha * (1 - T**2)) / (1 + T**2)
    cos_2pa = (cos_alpha * (1 - T**2) - 2 * sin_alpha * T) / (1 + T**2)

    return C_amp, sin_2pa, cos_2pa, tan_psi, den_psi


def polarization_with_guard(pre: Precessing, f: float):
    """
    Compute polarization WITH asymptotic guard for numerical stability.
    Uses asymptotic limits when |tan_psi| is extreme or den_psi is tiny.
    """
    cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = pre.precession_angles()

    # C amplitude
    C_amp = np.sqrt(
        0.25 * (1 + np.cos(pre.theta_S) ** 2) ** 2 * (np.cos(2 * pre.phi_S) ** 2)
        + (np.cos(pre.theta_S) ** 2 * np.sin(2 * pre.phi_S) ** 2)
    )

    # alpha
    sin_alpha = np.cos(pre.theta_S) * np.sin(2 * pre.phi_S) / C_amp
    cos_alpha = (1 + np.cos(pre.theta_S) ** 2) * np.cos(2 * pre.phi_S) / (2 * C_amp)

    # tan_psi
    theta_LJ = pre.theta_LJ(f)
    phi_LJ = pre.phi_LJ(f)

    num_psi = (
        np.sin(theta_LJ)
        * (np.cos(phi_LJ) * sin_o_XH + np.sin(phi_LJ) * cos_i_JN * cos_o_XH)
        - np.cos(theta_LJ) * sin_i_JN * cos_o_XH
    )
    den_psi = (
        np.sin(theta_LJ)
        * (np.cos(phi_LJ) * cos_o_XH - np.sin(phi_LJ) * cos_i_JN * sin_o_XH)
        + np.cos(theta_LJ) * sin_i_JN * sin_o_XH
    )

    # Face-on special case
    face_on = np.abs(1 - np.abs(cos_i_JN)) < NEAR_ZERO_THRESHOLD
    if face_on:
        o_XH = np.arctan2(sin_o_XH, cos_o_XH)
        tan_psi = np.tan(o_XH + np.sign(cos_i_JN) * phi_LJ)
    else:
        tan_psi = num_psi / den_psi

    # Algebraic formulas WITH asymptotic guard
    T = tan_psi

    # Detect extreme cases
    den_small = np.abs(den_psi) < 1e-12
    T_bad = ~np.isfinite(T) | (np.abs(T) > 1e12) | den_small

    # Algebraic forms (stable when |T| not extreme)
    sin_2pa_alg = (2 * cos_alpha * T + sin_alpha * (1 - T**2)) / (1 + T**2)
    cos_2pa_alg = (cos_alpha * (1 - T**2) - 2 * sin_alpha * T) / (1 + T**2)

    # Asymptotic fallback: as T -> ±∞, sin(2ψ+α) -> -sin α, cos(2ψ+α) -> -cos α
    sin_2pa = np.where(T_bad, -sin_alpha, sin_2pa_alg)
    cos_2pa = np.where(T_bad, -cos_alpha, cos_2pa_alg)

    return C_amp, sin_2pa, cos_2pa, tan_psi, den_psi


def sweep_and_plot(
    show: bool = True,
    save: bool = False,
    outdir: str = "figures/polarization_guard_compare",
    theta_tilde: float = 4.0,
    omega_tilde: float = 3.0,
    f: float = 50.0,
):
    os.makedirs(outdir, exist_ok=True)

    # Configuration that produces tan(psi) singularities
    # Use theta_S = pi/2, phi_S = 0, sweep theta_J to hit configurations where den_psi ~ 0
    theta_S = np.pi / 2.0
    phi_S = 0.0
    phi_J = 0.0

    # Sweep around where singularities occur
    thetas = np.linspace(0.0, np.pi, 500)

    sin_without = []
    cos_without = []
    tan_vals_without = []
    den_vals_without = []

    sin_with = []
    cos_with = []
    tan_vals_with = []
    den_vals_with = []

    for theta_J in thetas:
        params = make_precessing_params(
            theta_S,
            phi_S,
            theta_J,
            phi_J,
            theta_tilde=theta_tilde,
            omega_tilde=omega_tilde,
        )
        pre = Precessing(params)

        # Without guard
        _, s_wo, c_wo, t_wo, d_wo = polarization_without_guard(pre, f)
        sin_without.append(s_wo)
        cos_without.append(c_wo)
        tan_vals_without.append(t_wo)
        den_vals_without.append(d_wo)

        # With guard
        _, s_w, c_w, t_w, d_w = polarization_with_guard(pre, f)
        sin_with.append(s_w)
        cos_with.append(c_w)
        tan_vals_with.append(t_w)
        den_vals_with.append(d_w)

    sin_without = np.array(sin_without)
    cos_without = np.array(cos_without)
    tan_vals_without = np.array(tan_vals_without)
    den_vals_without = np.array(den_vals_without)

    sin_with = np.array(sin_with)
    cos_with = np.array(cos_with)
    tan_vals_with = np.array(tan_vals_with)
    den_vals_with = np.array(den_vals_with)

    # Plot
    fig = plt.figure(figsize=(14, 10))

    # tan(psi) and den_psi
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(np.degrees(thetas), tan_vals_without, alpha=0.7, label="tan(psi)")
    ax1.set_ylabel("tan(psi)")
    ax1.set_title("tan(psi) vs theta_J")
    ax1.legend()
    ax1.set_ylim(-50, 50)

    ax2 = plt.subplot(3, 2, 2)
    ax2.semilogy(np.degrees(thetas), np.abs(den_vals_without), alpha=0.7)
    ax2.axhline(1e-12, color="r", linestyle="--", label="guard threshold")
    ax2.set_ylabel("|den_psi|")
    ax2.set_title("Denominator magnitude (log scale)")
    ax2.legend()

    # sin(2psi+alpha) comparison
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(np.degrees(thetas), sin_without, alpha=0.7, label="without guard")
    ax3.plot(
        np.degrees(thetas), sin_with, alpha=0.7, label="with guard", linestyle="--"
    )
    ax3.set_ylabel("sin(2psi+alpha)")
    ax3.set_title("sin(2psi+alpha) comparison")
    ax3.legend()
    ax3.set_ylim(-1.5, 1.5)

    # cos(2psi+alpha) comparison
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(np.degrees(thetas), cos_without, alpha=0.7, label="without guard")
    ax4.plot(
        np.degrees(thetas), cos_with, alpha=0.7, label="with guard", linestyle="--"
    )
    ax4.set_ylabel("cos(2psi+alpha)")
    ax4.set_title("cos(2psi+alpha) comparison")
    ax4.legend()
    ax4.set_ylim(-1.5, 1.5)

    # Differences
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(np.degrees(thetas), sin_with - sin_without, alpha=0.7)
    ax5.set_xlabel("theta_J (deg)")
    ax5.set_ylabel("difference")
    ax5.set_title("sin(2psi+alpha): with_guard - without_guard")
    ax5.axhline(0, color="k", linestyle=":", alpha=0.3)

    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(np.degrees(thetas), cos_with - cos_without, alpha=0.7)
    ax6.set_xlabel("theta_J (deg)")
    ax6.set_ylabel("difference")
    ax6.set_title("cos(2psi+alpha): with_guard - without_guard")
    ax6.axhline(0, color="k", linestyle=":", alpha=0.3)

    plt.tight_layout()

    if save:
        fname = f"polarization_guard_comparison_thetaT_{theta_tilde:g}_f_{int(f)}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare polarization WITH and WITHOUT asymptotic guard."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display figures interactively."
    )
    parser.add_argument("--save", action="store_true", help="Save figures to disk.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures/polarization_guard_compare",
        help="Output directory for saved figures.",
    )
    parser.add_argument(
        "--theta-tilde",
        type=float,
        default=4.0,
        help="Precession amplitude theta_tilde (radians).",
    )
    parser.add_argument(
        "--omega-tilde",
        type=float,
        default=3.0,
        help="Precession frequency scale omega_tilde.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=50.0,
        help="Analysis frequency in Hz.",
    )
    args = parser.parse_args()

    show = not args.no_show
    save = args.save
    outdir = args.outdir

    sweep_and_plot(
        show=show,
        save=save,
        outdir=outdir,
        theta_tilde=args.theta_tilde,
        omega_tilde=args.omega_tilde,
        f=args.freq,
    )
