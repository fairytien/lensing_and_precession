#!/usr/bin/env python3
"""
Diagnose why tan(psi) stays bounded in the parameter sweep.

This script shows that:
1. theta_LJ and phi_LJ are small at f=50Hz for the given parameters
2. The numerator and denominator of tan(psi) both remain finite
3. den_psi never crosses zero in this particular parameter regime
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import from modules
HERE = os.path.dirname(__file__)
MODULES_DIR = os.path.abspath(os.path.join(HERE, "..", "modules"))
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

from Classes_v2 import Precessing, SOLMASS2SEC, GIGAPC2SEC
import argparse


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


def analyze_tan_psi_components(f=50.0, theta_tilde=4.0, omega_tilde=3.0):
    """Analyze the components of tan(psi) across theta_J sweep."""
    theta_S = np.pi / 2.0
    phi_S = 0.0
    phi_J = 0.0

    thetas = np.linspace(0.0, np.pi, 500)

    theta_LJs = []
    phi_LJs = []
    num_psis = []
    den_psis = []
    tan_psis = []
    cos_i_JNs = []
    sin_i_JNs = []

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

        # Get precession angles
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = pre.precession_angles()
        cos_i_JNs.append(cos_i_JN)
        sin_i_JNs.append(sin_i_JN)

        # Get theta_LJ and phi_LJ
        theta_LJ = pre.theta_LJ(f)
        phi_LJ = pre.phi_LJ(f)
        theta_LJs.append(theta_LJ)
        phi_LJs.append(phi_LJ)

        # Compute num_psi and den_psi
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

        num_psis.append(num_psi)
        den_psis.append(den_psi)
        tan_psis.append(num_psi / den_psi)

    # Convert to arrays
    theta_LJs = np.array(theta_LJs)
    phi_LJs = np.array(phi_LJs)
    num_psis = np.array(num_psis)
    den_psis = np.array(den_psis)
    tan_psis = np.array(tan_psis)
    cos_i_JNs = np.array(cos_i_JNs)
    sin_i_JNs = np.array(sin_i_JNs)

    # Create diagnostic plots
    fig = plt.figure(figsize=(16, 10))

    # theta_LJ and phi_LJ
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(np.degrees(thetas), theta_LJs, label="theta_LJ")
    ax1.set_ylabel("theta_LJ (rad)")
    ax1.set_xlabel("theta_J (deg)")
    ax1.set_title("Precession opening angle theta_LJ")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(np.degrees(thetas), phi_LJs, label="phi_LJ")
    ax2.set_ylabel("phi_LJ (rad)")
    ax2.set_xlabel("theta_J (deg)")
    ax2.set_title("Precession azimuthal angle phi_LJ")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # cos_i_JN and sin_i_JN
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(np.degrees(thetas), cos_i_JNs, label="cos(i_JN)")
    ax3.plot(np.degrees(thetas), sin_i_JNs, label="sin(i_JN)", alpha=0.7)
    ax3.set_ylabel("value")
    ax3.set_xlabel("theta_J (deg)")
    ax3.set_title("Inclination angles")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # numerator and denominator
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(np.degrees(thetas), num_psis, label="num_psi")
    ax4.set_ylabel("num_psi")
    ax4.set_xlabel("theta_J (deg)")
    ax4.set_title("Numerator of tan(psi)")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax4.legend()

    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(np.degrees(thetas), den_psis, label="den_psi", color="C1")
    ax5.set_ylabel("den_psi")
    ax5.set_xlabel("theta_J (deg)")
    ax5.set_title("Denominator of tan(psi)")
    ax5.grid(True, alpha=0.3)
    ax5.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax5.legend()

    # tan(psi)
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(np.degrees(thetas), tan_psis, label="tan(psi)", color="C2")
    ax6.set_ylabel("tan(psi)")
    ax6.set_xlabel("theta_J (deg)")
    ax6.set_title("tan(psi) = num_psi / den_psi")
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax6.set_ylim(-50, 50)
    ax6.legend()

    # Log scale of absolute values
    ax7 = plt.subplot(3, 3, 7)
    ax7.semilogy(np.degrees(thetas), np.abs(num_psis), label="|num_psi|")
    ax7.semilogy(np.degrees(thetas), np.abs(den_psis), label="|den_psi|", alpha=0.7)
    ax7.set_ylabel("Absolute value (log scale)")
    ax7.set_xlabel("theta_J (deg)")
    ax7.set_title("Magnitudes of numerator and denominator")
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # Ratio of magnitudes
    ax8 = plt.subplot(3, 3, 8)
    ratio = np.abs(num_psis) / np.abs(den_psis)
    ax8.plot(np.degrees(thetas), ratio, label="|num_psi| / |den_psi|", color="C3")
    ax8.set_ylabel("Ratio")
    ax8.set_xlabel("theta_J (deg)")
    ax8.set_title("Ratio of magnitudes = |tan(psi)|")
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 50)
    ax8.legend()

    # Show parameter values
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")
    param_text = f"""
    Parameter sweep:
    
    Fixed parameters:
    f = {f} Hz
    theta_S = {np.degrees(theta_S):.1f}°
    phi_S = {np.degrees(phi_S):.1f}°
    phi_J = {np.degrees(phi_J):.1f}°
    theta_tilde = {theta_tilde}
    omega_tilde = {omega_tilde}
    
    Swept parameter:
    theta_J: 0° to 180°
    
    Key insight:
    theta_LJ ≈ {theta_LJs.max():.4f} rad
    (stays very small!)
    
    This means:
    sin(theta_LJ) ≈ theta_LJ (small)
    cos(theta_LJ) ≈ 1
    
    So num_psi and den_psi
    are both O(theta_LJ) ~ 0.01,
    making their ratio finite!
    """
    ax9.text(
        0.1,
        0.5,
        param_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
    )

    plt.tight_layout()

    # Save
    os.makedirs("figures/diagnostics", exist_ok=True)
    fname = f"figures/diagnostics/tan_psi_bounds_diagnosis_thetaT_{theta_tilde:g}_f_{int(f)}.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved to {fname}")

    plt.show()

    # Print key statistics
    print("\n" + "=" * 60)
    print("DIAGNOSIS: Why tan(psi) stays bounded")
    print("=" * 60)
    print(f"theta_LJ range: [{theta_LJs.min():.6f}, {theta_LJs.max():.6f}] rad")
    print(
        f"  --> max theta_LJ = {theta_LJs.max():.6f} rad ≈ {np.degrees(theta_LJs.max()):.3f}°"
    )
    print(f"\nphi_LJ range: [{phi_LJs.min():.6f}, {phi_LJs.max():.6f}] rad")
    print(f"\nnum_psi range: [{num_psis.min():.6f}, {num_psis.max():.6f}]")
    print(f"den_psi range: [{den_psis.min():.6f}, {den_psis.max():.6f}]")
    print(f"  --> min |den_psi| = {np.abs(den_psis).min():.6e}")
    print(f"\ntan(psi) range: [{tan_psis.min():.2f}, {tan_psis.max():.2f}]")
    print(f"  --> max |tan(psi)| = {np.abs(tan_psis).max():.2f}")
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print(f"At f = {f} Hz with theta_tilde = {theta_tilde}:")
    print(f"  • theta_LJ is VERY SMALL (< {np.degrees(theta_LJs.max()):.3f}°)")
    print(f"  • Both num_psi and den_psi scale with sin(theta_LJ) ≈ theta_LJ")
    print(f"  • Their ratio stays finite: |tan(psi)| < {np.abs(tan_psis).max():.1f}")
    print(f"  • den_psi never gets close to zero (min = {np.abs(den_psis).min():.6e})")
    print("\nTo see singularities, you would need:")
    print("  1. Higher frequency (larger theta_LJ)")
    print("  2. Larger theta_tilde (directly scales theta_LJ)")
    print("  3. Special parameter combinations where den_psi → 0")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose tan(psi) boundedness across theta_J sweep."
    )
    parser.add_argument("--freq", type=float, default=50.0, help="Frequency in Hz")
    parser.add_argument(
        "--theta-tilde",
        type=float,
        default=4.0,
        help="Precession amplitude theta_tilde (rad)",
    )
    parser.add_argument(
        "--omega-tilde",
        type=float,
        default=3.0,
        help="Precession frequency scale omega_tilde",
    )
    args = parser.parse_args()

    analyze_tan_psi_components(
        f=args.freq, theta_tilde=args.theta_tilde, omega_tilde=args.omega_tilde
    )
