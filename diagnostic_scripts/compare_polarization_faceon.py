#!/usr/bin/env python3
"""
Compare polarization components sin(2psi+alpha) and cos(2psi+alpha)
BEFORE (original Classes_v2 logic) vs AFTER (hybrid logic now in Classes_v2/Classes_v3)
for a sweep near face-on and face-off. Produces simple plots.

Usage (optional):
  python scripts/compare_polarization_faceon.py

This script is self-contained and does not modify repo state.
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

from Classes_v2 import Precessing, SOLMASS2SEC, GIGAPC2SEC  # type: ignore


def polarization_before(pre: Precessing, f: float):
    """Compute (C_amp, sin_2pa, cos_2pa) using the original v2 logic (BEFORE change).

    This reproduces the pre-hybrid branch selection:
    - tan_psi uses exact-equality check on (phi_S == phi_J and theta_S == theta_J)
    - no tolerant face-on handling
    - sin/cos(2psi+alpha) via algebraic identities
    """
    # Unpack for readability
    theta_S, phi_S = pre.theta_S, pre.phi_S
    theta_J, phi_J = pre.theta_J, pre.phi_J

    # Angles between frames (precession_angles, matching v2 code)
    if phi_J == phi_S:
        if theta_J == theta_S:
            cos_i_JN = 1
        else:
            cos_i_JN = np.cos(theta_J - theta_S)
    else:
        cos_i_JN = np.sin(theta_J) * np.sin(theta_S) * np.cos(phi_J - phi_S) + np.cos(
            theta_J
        ) * np.cos(theta_S)
    sin_i_JN = np.sqrt(1 - cos_i_JN**2.0)

    if np.abs(sin_i_JN) < 1e-10:  # use same threshold as v2 constants
        cos_o_XH = 1
        sin_o_XH = 0
    else:
        cos_o_XH = (
            np.cos(theta_S) * np.sin(theta_J) * np.cos(phi_J - phi_S)
            - np.sin(theta_S) * np.cos(theta_J)
        ) / (sin_i_JN)
        sin_o_XH = (np.sin(theta_J) * np.sin(phi_J - phi_S)) / (sin_i_JN)

    # C amplitude (beam pattern)
    C_amp = np.sqrt(
        0.25 * (1 + (np.cos(theta_S)) ** 2) ** 2 * ((np.cos(2 * phi_S)) ** 2)
        + ((np.cos(theta_S)) ** 2 * (np.sin(2 * phi_S)) ** 2)
    )

    # alpha
    sin_alpha = np.cos(theta_S) * np.sin(2 * phi_S) / C_amp
    cos_alpha = (1 + np.cos(theta_S) ** 2) * np.cos(2 * phi_S) / (2 * C_amp)

    # tan(psi)
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

    if phi_S == phi_J:
        if theta_S == theta_J:
            tan_psi = np.tan(phi_LJ)
        else:
            tan_psi = num_psi / den_psi
    else:
        tan_psi = num_psi / den_psi

    # Algebraic identities for sin/cos(2psi+alpha)
    sin_2pa = (2 * cos_alpha * tan_psi + sin_alpha * (1 - (tan_psi) ** 2)) / (
        1 + (tan_psi) ** 2
    )
    cos_2pa = (cos_alpha * (1 - (tan_psi) ** 2) - 2 * sin_alpha * tan_psi) / (
        1 + (tan_psi) ** 2
    )

    return C_amp, sin_2pa, cos_2pa


def polarization_after(pre: Precessing, f: float):
    """Compute (C_amp, sin_2pa, cos_2pa) using the CURRENT hybrid logic in the class."""
    C_amp, sin_2pa, cos_2pa = pre.polarization_amplitude_and_phase(f)
    return C_amp, sin_2pa, cos_2pa


def make_precessing_params(theta_S, phi_S, theta_J, phi_J):
    # Reasonable defaults (units follow the module conventions: seconds)
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
        "theta_tilde": 0.3,
        "omega_tilde": 0.1,
        "gamma_P": 0.0,
    }
    return params


def sweep_and_plot(
    show: bool = True, save: bool = False, outdir: str = "figures/polarization_compare"
):
    # Fixed frequency near LIGO band
    f = 50.0  # Hz

    # Choose a face-on configuration baseline (theta_S = 0); sweep theta_J around 0
    theta_S = 0.0
    phi_S = 0.0
    phi_J = 0.0

    # Sweep near face-on (theta_J ~ 0)
    thetas_face_on = np.linspace(0.0, 0.05, 200)  # radians (~0 to ~2.9 deg)

    sin_before = []
    cos_before = []
    sin_after = []
    cos_after = []

    for theta_J in thetas_face_on:
        params = make_precessing_params(theta_S, phi_S, theta_J, phi_J)
        pre = Precessing(params)
        _, s_b, c_b = polarization_before(pre, f)
        _, s_a, c_a = polarization_after(pre, f)
        sin_before.append(s_b)
        cos_before.append(c_b)
        sin_after.append(s_a)
        cos_after.append(c_a)

    # Sweep near face-off (theta_J ~ pi)
    thetas_face_off = np.linspace(np.pi - 0.05, np.pi, 200)

    sin_before_off = []
    cos_before_off = []
    sin_after_off = []
    cos_after_off = []

    for theta_J in thetas_face_off:
        params = make_precessing_params(theta_S, phi_S, theta_J, phi_J)
        pre = Precessing(params)
        _, s_b, c_b = polarization_before(pre, f)
        _, s_a, c_a = polarization_after(pre, f)
        sin_before_off.append(s_b)
        cos_before_off.append(c_b)
        sin_after_off.append(s_a)
        cos_after_off.append(c_a)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    axs[0, 0].plot(np.degrees(thetas_face_on), sin_before, label="before", alpha=0.8)
    axs[0, 0].plot(np.degrees(thetas_face_on), sin_after, label="after", alpha=0.8)
    axs[0, 0].set_title("sin(2psi+alpha) near face-on")
    axs[0, 0].set_xlabel("theta_J (deg)")
    axs[0, 0].set_ylabel("sin(2psi+alpha)")
    axs[0, 0].legend()

    axs[0, 1].plot(np.degrees(thetas_face_on), cos_before, label="before", alpha=0.8)
    axs[0, 1].plot(np.degrees(thetas_face_on), cos_after, label="after", alpha=0.8)
    axs[0, 1].set_title("cos(2psi+alpha) near face-on")
    axs[0, 1].set_xlabel("theta_J (deg)")
    axs[0, 1].set_ylabel("cos(2psi+alpha)")
    axs[0, 1].legend()

    axs[1, 0].plot(
        np.degrees(thetas_face_off), sin_before_off, label="before", alpha=0.8
    )
    axs[1, 0].plot(np.degrees(thetas_face_off), sin_after_off, label="after", alpha=0.8)
    axs[1, 0].set_title("sin(2psi+alpha) near face-off")
    axs[1, 0].set_xlabel("theta_J (deg)")
    axs[1, 0].set_ylabel("sin(2psi+alpha)")
    axs[1, 0].legend()

    axs[1, 1].plot(
        np.degrees(thetas_face_off), cos_before_off, label="before", alpha=0.8
    )
    axs[1, 1].plot(np.degrees(thetas_face_off), cos_after_off, label="after", alpha=0.8)
    axs[1, 1].set_title("cos(2psi+alpha) near face-off")
    axs[1, 1].set_xlabel("theta_J (deg)")
    axs[1, 1].set_ylabel("cos(2psi+alpha)")
    axs[1, 1].legend()

    plt.tight_layout()
    if save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "angle_sweep_face_on_off.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def frequency_sweep_and_plot(
    show: bool = True, save: bool = False, outdir: str = "figures/polarization_compare"
):
    """
    Fix geometry near face-on and face-off, sweep frequency, and compare before vs after.
    """
    # Frequency grid (covers detector band where model is used)
    f = np.linspace(20.0, 300.0, 400)

    # Geometry: face-on and face-off baselines
    theta_S = 0.0
    phi_S = 0.0
    phi_J = 0.0
    theta_J_face_on = 0.0  # include exact face-on to trigger special-case branches
    theta_J_face_off = np.pi  # exact face-off

    # Helper to compute arrays
    def eval_arrays(theta_J_val):
        params = make_precessing_params(theta_S, phi_S, theta_J_val, phi_J)
        pre = Precessing(params)
        sin_b, cos_b, sin_a, cos_a = [], [], [], []
        for fi in f:
            _, s_b, c_b = polarization_before(pre, float(fi))
            _, s_a, c_a = polarization_after(pre, float(fi))
            sin_b.append(s_b)
            cos_b.append(c_b)
            sin_a.append(s_a)
            cos_a.append(c_a)
        return np.array(sin_b), np.array(cos_b), np.array(sin_a), np.array(cos_a)

    sin_b_on, cos_b_on, sin_a_on, cos_a_on = eval_arrays(theta_J_face_on)
    sin_b_off, cos_b_off, sin_a_off, cos_a_off = eval_arrays(theta_J_face_off)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axs[0, 0].plot(f, sin_b_on, label="before", alpha=0.8)
    axs[0, 0].plot(f, sin_a_on, label="after", alpha=0.8)
    axs[0, 0].set_title("sin(2psi+alpha) vs f (face-on)")
    axs[0, 0].set_ylabel("sin(2psi+alpha)")
    axs[0, 0].legend()

    axs[0, 1].plot(f, cos_b_on, label="before", alpha=0.8)
    axs[0, 1].plot(f, cos_a_on, label="after", alpha=0.8)
    axs[0, 1].set_title("cos(2psi+alpha) vs f (face-on)")
    axs[0, 1].set_ylabel("cos(2psi+alpha)")
    axs[0, 1].legend()

    axs[1, 0].plot(f, sin_b_off, label="before", alpha=0.8)
    axs[1, 0].plot(f, sin_a_off, label="after", alpha=0.8)
    axs[1, 0].set_title("sin(2psi+alpha) vs f (face-off)")
    axs[1, 0].set_xlabel("f (Hz)")
    axs[1, 0].set_ylabel("sin(2psi+alpha)")
    axs[1, 0].legend()

    axs[1, 1].plot(f, cos_b_off, label="before", alpha=0.8)
    axs[1, 1].plot(f, cos_a_off, label="after", alpha=0.8)
    axs[1, 1].set_title("cos(2psi+alpha) vs f (face-off)")
    axs[1, 1].set_xlabel("f (Hz)")
    axs[1, 1].set_ylabel("cos(2psi+alpha)")
    axs[1, 1].legend()

    plt.tight_layout()
    if save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "frequency_sweep_face_on_off.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def angle_zoom_plot(
    show: bool = True, save: bool = False, outdir: str = "figures/polarization_compare"
):
    """
    Narrow zoom near exact face-on to magnify behavior right at theta_J -> 0.
    """
    f = 50.0  # Hz

    # Geometry baseline at face-on
    theta_S = 0.0
    phi_S = 0.0
    phi_J = 0.0

    # Very small angles up to ~0.29 degrees
    thetas_zoom = np.linspace(0.0, 0.005, 500)  # radians

    sin_before = []
    cos_before = []
    sin_after = []
    cos_after = []

    for theta_J in thetas_zoom:
        params = make_precessing_params(theta_S, phi_S, theta_J, phi_J)
        pre = Precessing(params)
        _, s_b, c_b = polarization_before(pre, f)
        _, s_a, c_a = polarization_after(pre, f)
        sin_before.append(s_b)
        cos_before.append(c_b)
        sin_after.append(s_a)
        cos_after.append(c_a)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axs[0].plot(np.degrees(thetas_zoom), sin_before, label="before", alpha=0.8)
    axs[0].plot(np.degrees(thetas_zoom), sin_after, label="after", alpha=0.8)
    axs[0].set_title("sin(2psi+alpha) narrow zoom (face-on)")
    axs[0].set_xlabel("theta_J (deg)")
    axs[0].set_ylabel("sin(2psi+alpha)")
    axs[0].legend()

    axs[1].plot(np.degrees(thetas_zoom), cos_before, label="before", alpha=0.8)
    axs[1].plot(np.degrees(thetas_zoom), cos_after, label="after", alpha=0.8)
    axs[1].set_title("cos(2psi+alpha) narrow zoom (face-on)")
    axs[1].set_xlabel("theta_J (deg)")
    axs[1].set_ylabel("cos(2psi+alpha)")
    axs[1].legend()

    plt.tight_layout()
    if save:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "angle_zoom_face_on.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare polarization before vs after near face-on/face-off."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display figures interactively."
    )
    parser.add_argument("--save", action="store_true", help="Save figures to disk.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures/polarization_compare",
        help="Output directory for saved figures.",
    )
    args = parser.parse_args()

    show = not args.no_show
    save = args.save
    outdir = args.outdir

    sweep_and_plot(show=show, save=save, outdir=outdir)
    frequency_sweep_and_plot(show=show, save=save, outdir=outdir)
    angle_zoom_plot(show=show, save=save, outdir=outdir)
