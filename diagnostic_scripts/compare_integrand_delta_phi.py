#!/usr/bin/env python3
"""
Compare integrand_delta_phi BEFORE vs AFTER adding the correction term
for face-on (cos_i_JN = 1) and edge-on (cos_i_JN = 0) configurations.

- BEFORE: base expression without the correction term, with face-on branch
- AFTER:  base + correction (matches current Classes_v2 implementation), with face-on branch

Produces frequency-sweep plots for both configurations.

Usage examples:
  python scripts/compare_integrand_delta_phi.py
  python scripts/compare_integrand_delta_phi.py --no-show --save
  python scripts/compare_integrand_delta_phi.py --save --outdir ./my_plots
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


def integrand_after(pre: Precessing, f: float):
    """
    AFTER: base + correction term (matches current Classes_v2 implementation).
    """
    if pre.theta_tilde == 0:
        return 0.0

    LdotN, cos_i_JN, sin_i_JN, theta_LJ, phi_LJ, f_dot, Omega_LJ = integrand_components(
        pre, f
    )

    # Face-on / face-off branch (same as BEFORE)
    face_on = np.abs(1.0 - np.abs(cos_i_JN)) < NEAR_ZERO_THRESHOLD
    if face_on:
        return -Omega_LJ * np.cos(theta_LJ) / f_dot

    # Generic (non face-on): base + correction
    base = (
        (LdotN / (1.0 - LdotN**2))
        * Omega_LJ
        * np.sin(theta_LJ)
        * (np.cos(theta_LJ) * sin_i_JN * np.sin(phi_LJ) - np.sin(theta_LJ) * cos_i_JN)
        / f_dot
    )
    corr = (LdotN / (1.0 - LdotN**2)) * (
        -(theta_LJ / (3.0 * f)) * np.cos(phi_LJ) * sin_i_JN
    )
    return base + corr


def sweep_frequency_and_plot(
    show: bool = True, save: bool = False, outdir: str = "figures/integrand_compare"
):
    """Sweep frequency and compare BEFORE vs AFTER for face-on and edge-on configurations."""
    os.makedirs(outdir, exist_ok=True)

    # Frequency grid
    f = np.linspace(20.0, 300.0, 400)

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

    def eval_arrays(pre: Precessing):
        yb, ya = [], []
        for fi in f:
            yb.append(integrand_before(pre, float(fi)))
            ya.append(integrand_after(pre, float(fi)))
        return np.array(yb), np.array(ya)

    yb_on, ya_on = eval_arrays(pre_on)
    yb_edge, ya_edge = eval_arrays(pre_edge)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(f, yb_on, label="before", alpha=0.85)
    axs[0].plot(f, ya_on, label="after", alpha=0.85)
    axs[0].set_title("integrand_delta_phi vs f (face-on: cos i_JN = 1)")
    axs[0].set_ylabel("integrand")
    axs[0].legend()

    axs[1].plot(f, yb_edge, label="before", alpha=0.85)
    axs[1].plot(f, ya_edge, label="after", alpha=0.85)
    axs[1].set_title("integrand_delta_phi vs f (edge-on: cos i_JN = 0)")
    axs[1].set_xlabel("f (Hz)")
    axs[1].set_ylabel("integrand")
    axs[1].legend()

    plt.tight_layout()
    outpath = os.path.join(outdir, "integrand_delta_phi_face_on_edge_on.png")
    if save:
        fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def angle_sweep_and_plot(
    show: bool = True, save: bool = False, outdir: str = "figures/integrand_compare"
):
    """
    Sweep theta_J around face-on and edge-on at a fixed frequency and compare BEFORE vs AFTER.
    """
    os.makedirs(outdir, exist_ok=True)

    # Fixed frequency for angle sweeps
    f = 50.0  # Hz

    # Geometry baseline: set theta_S = 0 so cos_i_JN ~= cos(theta_J)
    theta_S = 0.0
    phi_S = 0.0
    phi_J = 0.0

    # Face-on region: small angles near 0
    thetas_face_on = np.linspace(0.0, 0.2, 300)  # ~0 to ~11.5 degrees
    # Edge-on region: around pi/2
    thetas_edge_on = np.linspace(np.pi / 2 - 0.2, np.pi / 2 + 0.2, 300)

    def eval_arrays(thetas):
        yb, ya = [], []
        for theta_J in thetas:
            params = make_precessing_params(theta_S, phi_S, float(theta_J), phi_J)
            pre = Precessing(params)
            yb.append(integrand_before(pre, f))
            ya.append(integrand_after(pre, f))
        return np.array(yb), np.array(ya)

    yb_on, ya_on = eval_arrays(thetas_face_on)
    yb_ed, ya_ed = eval_arrays(thetas_edge_on)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=False)

    axs[0].plot(np.degrees(thetas_face_on), yb_on, label="before", alpha=0.85)
    axs[0].plot(np.degrees(thetas_face_on), ya_on, label="after", alpha=0.85)
    axs[0].set_title("integrand vs theta_J (face-on region)")
    axs[0].set_xlabel("theta_J (deg)")
    axs[0].set_ylabel("integrand")
    axs[0].legend()

    axs[1].plot(np.degrees(thetas_edge_on), yb_ed, label="before", alpha=0.85)
    axs[1].plot(np.degrees(thetas_edge_on), ya_ed, label="after", alpha=0.85)
    axs[1].set_title("integrand vs theta_J (edge-on region)")
    axs[1].set_xlabel("theta_J (deg)")
    axs[1].set_ylabel("integrand")
    axs[1].legend()

    plt.tight_layout()
    outpath = os.path.join(outdir, "integrand_delta_phi_angle_sweep.png")
    if save:
        fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare integrand_delta_phi BEFORE vs AFTER the correction term for face-on and edge-on."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display figures interactively."
    )
    parser.add_argument("--save", action="store_true", help="Save figures to disk.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures/integrand_compare",
        help="Output directory for saved figures.",
    )
    args = parser.parse_args()

    show = not args.no_show
    save = args.save
    outdir = args.outdir

    sweep_frequency_and_plot(show=show, save=save, outdir=outdir)
    angle_sweep_and_plot(show=show, save=save, outdir=outdir)
