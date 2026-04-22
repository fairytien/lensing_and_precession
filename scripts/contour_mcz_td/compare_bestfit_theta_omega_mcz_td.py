"""Compare best-fit precession parameters across Systems 1/2/3.

Creates a 2x3 panel figure for face-on, edge-on, and random systems:
- Top row: best-matching omega_tilde on (td, mcz) grid
- Bottom row: best-matching theta_tilde on (td, mcz) grid

Each row uses a shared color scale across all 3 systems.
N_lensed = 1/2/3 contours are overlaid on every panel.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is importable when running this file directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.plot_utils import apply_physics_paper_style
from modules.waveform import number_of_lens_cycles


DEFAULT_PATHS = [
    "data/mismatch_I0p5_z1_mcz5-45_td20-70_Taman_faceon/best_match/"
    "best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-6x61_theta0-15x151_"
    "gamma0-2pix51_Taman_faceon.h5",
    "data/mismatch_I0p5_z1e-08_mcz10-90_td20-70_Taman_edgeon/best_match/"
    "best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-6x61_theta0-15x151_"
    "gamma0-2pix51_Taman_edgeon.h5",
    "data/mismatch_I0p5_z1_mcz5-45_td20-70_Taman_random/best_match/"
    "best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-6x61_theta0-15x151_"
    "gamma0-2pix51_Taman_random.h5",
]

DEFAULT_LABELS = ["System 1 (face-on)", "System 2 (edge-on)", "System 3 (random)"]

DEFAULT_OUTPUT = "figures/contour_mcz_td/" "compare_bestfit_omega_theta_systems123.pdf"


def _load_best_match(path: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        mcz = np.asarray(h5["mcz"], dtype=float)
        td = np.asarray(h5["td"], dtype=float)
        omega_best = np.asarray(h5["omega_best"], dtype=float)
        theta_best = np.asarray(h5["theta_best"], dtype=float)
        z = float(h5.attrs.get("z", h5.attrs.get("source_param_z", 0.0)))
        I = float(h5.attrs.get("I", np.nan))

    td_ms = td * 1e3
    TD, MCZ = np.meshgrid(td, mcz)

    # mcz axis is source-frame in these files; convert to detector-frame for f_cut.
    nlens = number_of_lens_cycles(MCZ * (1 + z), TD)

    return {
        "mcz": mcz,
        "td_ms": td_ms,
        "omega_best": omega_best,
        "theta_best": theta_best,
        "nlens": nlens,
        "z": z,
        "I": I,
    }


def _validate_inputs(paths: List[str], labels: List[str]) -> None:
    if len(paths) != 3:
        raise ValueError(f"Expected exactly 3 --paths, got {len(paths)}")
    if len(labels) != 3:
        raise ValueError(f"Expected exactly 3 --labels, got {len(labels)}")

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))


def create_figure(
    paths: List[str],
    labels: List[str],
    output_path: str,
    levels_count: int,
    dpi: int,
    cmap: str,
) -> None:
    _validate_inputs(paths, labels)

    datasets = [_load_best_match(p) for p in paths]

    omega_min = min(float(np.nanmin(d["omega_best"])) for d in datasets)
    omega_max = max(float(np.nanmax(d["omega_best"])) for d in datasets)
    theta_min = min(float(np.nanmin(d["theta_best"])) for d in datasets)
    theta_max = max(float(np.nanmax(d["theta_best"])) for d in datasets)

    omega_levels = np.linspace(omega_min, omega_max, levels_count)
    theta_levels = np.linspace(theta_min, theta_max, levels_count)

    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=10)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharex=True, sharey=True)

    omega_cf = None
    theta_cf = None

    for i, (d, label) in enumerate(zip(datasets, labels)):
        ax_top = axes[0, i]
        ax_bottom = axes[1, i]

        omega_cf = ax_top.contourf(
            d["td_ms"], d["mcz"], d["omega_best"], levels=omega_levels, cmap=cmap
        )
        theta_cf = ax_bottom.contourf(
            d["td_ms"], d["mcz"], d["theta_best"], levels=theta_levels, cmap=cmap
        )

        for ax in (ax_top, ax_bottom):
            for n, ls in [(1, ":"), (2, "--"), (3, "-")]:
                cs = ax.contour(
                    d["td_ms"],
                    d["mcz"],
                    d["nlens"],
                    levels=[n],
                    colors=["white"],
                    linestyles=[ls],
                    linewidths=1.2,
                )
                ax.clabel(cs, fmt=rf"$N_\mathrm{{lensed}}={n}$", fontsize=8)

            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect(1)
            ax.tick_params(direction="in", top=True, right=True)

        ax_top.set_title(label)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mathcal{M}_{\mathrm{s}}\,[\mathrm{M}_\odot]$")

    # Row labels
    axes[0, 0].text(
        -0.27,
        0.5,
        r"best $\tilde{\Omega}$",
        transform=axes[0, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )
    axes[1, 0].text(
        -0.27,
        0.5,
        r"best $\tilde{\theta}$",
        transform=axes[1, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )

    # Right-side row colorbars
    fig.canvas.draw()
    top_right = axes[0, 2].get_position()
    bot_right = axes[1, 2].get_position()

    cax_omega = fig.add_axes(
        [top_right.x1 + 0.012, top_right.y0, 0.016, top_right.height]
    )
    cax_theta = fig.add_axes(
        [bot_right.x1 + 0.012, bot_right.y0, 0.016, bot_right.height]
    )

    cbar_omega = fig.colorbar(omega_cf, cax=cax_omega)
    cbar_theta = fig.colorbar(theta_cf, cax=cax_theta)
    cbar_omega.set_label(r"$\tilde{\Omega}_{\mathrm{best}}$")
    cbar_theta.set_label(r"$\tilde{\theta}_{\mathrm{best}}$")

    # Global title
    z_vals = [d["z"] for d in datasets]
    i_vals = [d["I"] for d in datasets]
    z_txt = ", ".join(sorted({f"{z:g}" for z in z_vals}))
    i_txt = ", ".join(sorted({f"{iv:g}" for iv in i_vals if np.isfinite(iv)}))
    fig.suptitle(
        "Best-matching precession parameters across Systems 1/2/3\n"
        + rf"($I={i_txt}$, $z={z_txt}$)",
        fontsize=13,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.86,
        top=0.90,
        bottom=0.11,
        wspace=0.03,
        hspace=0.08,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs=3,
        default=DEFAULT_PATHS,
        help="Three best_match HDF5 paths for System 1/2/3",
    )
    parser.add_argument(
        "--labels",
        nargs=3,
        default=DEFAULT_LABELS,
        help="Three panel labels for System 1/2/3",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output figure path",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=60,
        help="Number of contour levels per row",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI",
    )
    parser.add_argument(
        "--cmap",
        default="jet",
        help="Matplotlib colormap",
    )
    args = parser.parse_args()

    create_figure(
        paths=args.paths,
        labels=args.labels,
        output_path=args.output,
        levels_count=args.levels,
        dpi=args.dpi,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
