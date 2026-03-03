import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# Ensure project root is importable when script is launched from scripts/utils
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import cosmology


def _format_float(value: float, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser(
        description="Plot luminosity distance vs redshift using modules/cosmology.py"
    )
    parser.add_argument(
        "--zmin",
        type=float,
        default=1e-8,
        help="Minimum redshift (default: 1e-8)",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=20.0,
        help="Maximum redshift (default: 20)",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=2000,
        help="Number of redshift samples (default: 2000)",
    )
    parser.add_argument(
        "--zzoom_max",
        type=float,
        default=1.0,
        help="Upper bound of zoom-panel redshift range (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/dist_vs_z.pdf",
        help="Output path for plot image",
    )
    args = parser.parse_args()

    if args.zmin < 0:
        raise ValueError("zmin must be >= 0.")
    if args.zmax <= args.zmin:
        raise ValueError("zmax must be greater than zmin.")
    if args.npoints < 2:
        raise ValueError("npoints must be at least 2.")
    if args.zzoom_max <= args.zmin:
        raise ValueError("zzoom_max must be greater than zmin.")

    z = np.linspace(args.zmin, args.zmax, args.npoints)
    dl_gpc = cosmology.z_to_DL(z)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), constrained_layout=True)
    ax_full, ax_zoom = axes

    ax_full.plot(z, dl_gpc, lw=2.8, color="C0")
    ax_full.set_xlim(args.zmin, args.zmax)
    ax_full.set_ylim(bottom=0)
    ax_full.set_xlabel("Redshift $z$")
    ax_full.set_ylabel("Luminosity Distance $D_L$ [Gpc]")
    ax_full.set_title("Full Range", pad=10)
    ax_full.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_full.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_full.grid(which="major", alpha=0.35, linestyle="-")
    ax_full.grid(which="minor", alpha=0.18, linestyle=":")

    z_zoom_max = min(args.zzoom_max, args.zmax)
    zoom_mask = (z >= args.zmin) & (z <= z_zoom_max)
    ax_zoom.plot(z[zoom_mask], dl_gpc[zoom_mask], lw=2.8, color="C1")
    ax_zoom.set_xlim(args.zmin, z_zoom_max)
    ax_zoom.set_ylim(bottom=0)
    ax_zoom.set_xlabel("Redshift $z$")
    ax_zoom.set_ylabel("Luminosity Distance $D_L$ [Gpc]")
    ax_zoom.set_title(f"Zoom: $z \leq {z_zoom_max:g}$", pad=10)
    ax_zoom.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_zoom.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_zoom.grid(which="major", alpha=0.35, linestyle="-")
    ax_zoom.grid(which="minor", alpha=0.18, linestyle=":")

    fig.suptitle("Luminosity Distance vs Redshift", fontsize=17)

    backend = "astropy" if cosmology._USE_ASTROPY else "scipy fallback"
    setup_text = (
        "Planck 2018 Flat $\\Lambda$CDM\n"
        f"$H_0={_format_float(cosmology.H0, 1)}$ km s$^{{-1}}$ Mpc$^{{-1}}$\n"
        f"$\\Omega_m={_format_float(cosmology.OM0)}$, "
        f"$\\Omega_\\Lambda={_format_float(cosmology.OL0)}$\n"
        f"backend: {backend}"
    )
    ax_full.text(
        0.03,
        0.97,
        setup_text,
        transform=ax_full.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.3", alpha=0.92
        ),
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
