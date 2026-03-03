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
        description=(
            "Example: luminosity distance vs redshift with a single-panel style broken x-axis"
        )
    )
    parser.add_argument("--zmin", type=float, default=1e-8)
    parser.add_argument("--zmax", type=float, default=20.0)
    parser.add_argument("--npoints", type=int, default=3000)
    parser.add_argument(
        "--xbreak_left_max",
        type=float,
        default=2.0,
        help="Upper bound of left x segment",
    )
    parser.add_argument(
        "--xbreak_right_min",
        type=float,
        default=8.0,
        help="Lower bound of right x segment",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/dist_vs_z_broken_axis.pdf",
    )
    args = parser.parse_args()

    if args.zmin < 0:
        raise ValueError("zmin must be >= 0")
    if args.zmax <= args.zmin:
        raise ValueError("zmax must be greater than zmin")
    if args.npoints < 2:
        raise ValueError("npoints must be at least 2")
    if not (args.zmin < args.xbreak_left_max < args.xbreak_right_min < args.zmax):
        raise ValueError(
            "Require zmin < xbreak_left_max < xbreak_right_min < zmax for broken axis"
        )

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
        }
    )

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(10.8, 5.6),
        gridspec_kw={"width_ratios": [1.8, 1.2], "wspace": 0.05},
        constrained_layout=True,
    )

    left_mask = z <= args.xbreak_left_max
    right_mask = z >= args.xbreak_right_min

    ax_left.plot(z[left_mask], dl_gpc[left_mask], color="C0", lw=2.8)
    ax_right.plot(z[right_mask], dl_gpc[right_mask], color="C0", lw=2.8)

    ax_left.set_xlim(args.zmin, args.xbreak_left_max)
    ax_right.set_xlim(args.xbreak_right_min, args.zmax)
    ax_left.set_ylim(bottom=0)

    ax_left.set_ylabel("Luminosity Distance $D_L$ [Gpc]")
    ax_left.set_xlabel("Redshift $z$")
    ax_right.set_xlabel("Redshift $z$")
    fig.suptitle("Luminosity Distance vs Redshift (Broken $x$-Axis)", fontsize=17)

    for ax in (ax_left, ax_right):
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which="major", alpha=0.35, linestyle="-")
        ax.grid(which="minor", alpha=0.18, linestyle=":")

    # Hide touching spines to create single-panel style with an axis break
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.tick_right()
    ax_right.tick_params(labelright=False)

    # Draw diagonal break marks
    d = 0.013
    kwargs_left = dict(transform=ax_left.transAxes, color="k", clip_on=False, lw=1.2)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs_left)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_left)

    kwargs_right = dict(transform=ax_right.transAxes, color="k", clip_on=False, lw=1.2)
    ax_right.plot((-d, +d), (-d, +d), **kwargs_right)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_right)

    backend = "astropy" if cosmology._USE_ASTROPY else "scipy fallback"
    setup_text = (
        "Planck 2018 Flat $\\Lambda$CDM\n"
        f"$H_0={_format_float(cosmology.H0, 1)}$ km s$^{{-1}}$ Mpc$^{{-1}}$\n"
        f"$\\Omega_m={_format_float(cosmology.OM0)}$, "
        f"$\\Omega_\\Lambda={_format_float(cosmology.OL0)}$\n"
        f"backend: {backend}"
    )
    ax_left.text(
        0.03,
        0.97,
        setup_text,
        transform=ax_left.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.3", alpha=0.92
        ),
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
