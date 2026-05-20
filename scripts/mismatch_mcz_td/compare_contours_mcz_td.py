"""Generate the finalized paper-style 2x2 contour comparison figure.

This script reproduces the latest figure style used in this project:
- 2x2 shared-axis layout
- global/shared color scale across all panels
- in-panel label boxes: Non-Precessing, System 1/2/3
- N_lensed=1/2/3 overlays on all panels
- peaks (magenta) and troughs (white) overlays on panel 1 only
- overlay curves scaled from z_from to z_to (defaults: 1e-8 -> 1)
- colorbar aligned to right-column axis bounds (excluding bottom xlabel height)
- colorbar tick labels formatted to fixed decimals (default: 2)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import h5py
import numpy as np

# Ensure repository root is importable when running this file directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.bank_io import read_best_match_mcz_td_contour_data
from modules.cosmology import source_mass_redshift_scale
from modules.filenames import compare_mcz_td_figure_filename
from modules.lens_cycle_extrema import find_mcz_peaks, find_mcz_troughs
from modules.plot_utils import (
    COLORBAR_PAD,
    COLORBAR_WIDTH,
    add_overlay_legend,
    apply_physics_paper_style,
    format_colorbar_ticks,
    save_figure,
    set_square_axes,
)
from scripts.utils.compare_contours import compute_color_scale, load_generic_dataset
from scripts.utils.plot_cycles_and_extrema import (
    make_fixed_mcz_overlay_legend_handles,
    plot_cycle_lines,
)

DEFAULT_PATHS = [
    "data/contour_mcz_td/contour_L_NP_I0.5_z1_mcz5-45Msun_td20-70ms_min_mismatch_Taman_edgeon.h5",
    "data/mismatch_I0p5_z1_mcz5-45_td20-70_Taman_faceon/best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_faceon.h5",
    "data/mismatch_I0p5_z1e-08_mcz10-90_td20-70_Taman_edgeon/best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5",
    "data/mismatch_I0p5_z1_mcz5-45_td20-70_Taman_random/best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_random.h5",
]

DEFAULT_PANEL_LABELS = [
    "Non-Precessing",
    "System 1",
    "System 2",
    "System 3",
]

# Physics-style math labels: variables italic, identifiers/units upright roman.
X_AXIS_LABEL = r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$"
Y_AXIS_LABEL = r"$\mathcal{M}_{\mathrm{s}}\,[\mathrm{M}_\odot]$"
COLORBAR_LABEL = (
    r"$\min_{\tilde{\Omega},\,\tilde{\theta},\,\gamma_{\mathrm{P}}}\,"
    r"\epsilon\left(\tilde{\mathit{h}}_{\mathrm{L}},\,\tilde{\mathit{h}}_{\mathrm{P}}\right)$"
)


def _validate_paths(paths: List[str]) -> List[str]:
    if len(paths) != 4:
        raise ValueError(f"Expected exactly 4 paths, got {len(paths)}")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    return paths


def _load_panel(path: str) -> tuple:
    """Load one panel's (X, Y, Z, meta) for a td_mcz grid.

    Tries the best_match schema (mcz/td/epsilon_min) first via
    read_best_match_mcz_td_contour_data; falls back to load_generic_dataset
    for legacy contour_mcz_td or pickle files.  meta is a dict with
    'I', 'z', and 'orientation_tag' when available, otherwise None.
    """
    if os.path.splitext(path)[1].lower() == ".h5":
        with h5py.File(path, "r") as _h5:
            is_best_match = all(k in _h5 for k in ("mcz", "td", "epsilon_min"))
        if is_best_match:
            ds = read_best_match_mcz_td_contour_data(path, "epsilon_min")
            X, Y = np.meshgrid(ds["td"] * 1e3, ds["mcz"])
            meta = {
                "I": float(ds["I"]),
                "z": ds["z"],
                "orientation_tag": str(ds["orientation_tag"]),
            }
            return X, Y, np.asarray(ds["values"], dtype=float), meta
    X, Y, Z, _, _, data_type = load_generic_dataset(path)
    if data_type != "td_mcz":
        raise ValueError(f"Expected a td_mcz dataset, got '{data_type}' for {path}")
    return X, Y, Z, None


def create_figure(
    paths: List[str],
    panel_labels: List[str],
    output_path: str | None,
    fig_dir: str,
    decimals: int,
    levels_count: int,
    dpi: int,
    eta: float,
    f_min: float,
    z_from: float,
    z_to: float,
    cbar_n_ticks: int,
    cmap: str,
) -> None:
    if len(panel_labels) != 4:
        raise ValueError("Expected exactly 4 panel labels")

    panels = [_load_panel(p) for p in paths]
    xs = [p[0] for p in panels]
    ys = [p[1] for p in panels]
    eps = [p[2] for p in panels]
    metas = [p[3] for p in panels]

    if output_path is None:
        meta = next((m for m in metas if m is not None), None)
        if meta is None:
            raise ValueError(
                "Cannot auto-derive output path: no panel has best_match metadata. "
                "Pass --output explicitly."
            )
        orientation_tags = [m["orientation_tag"] for m in metas if m is not None]
        output_path = compare_mcz_td_figure_filename(
            fig_dir,
            I=meta["I"],
            z=meta["z"],
            orientation_tags=orientation_tags,
        )

    eps_masked, global_min, global_max = compute_color_scale(eps, "auto")
    overlay_mcz_scale = source_mass_redshift_scale(z_from, z_to)

    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=11)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.8), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    levels = np.linspace(global_min, global_max, levels_count)

    for i, ax in enumerate(axes):
        cf = ax.contourf(xs[i], ys[i], eps_masked[i], levels=levels, cmap=cmap)

        td_arr_ms = xs[i][0, :]
        td_arr = td_arr_ms / 1e3
        mcz_min = float(np.nanmin(ys[i]))
        mcz_max = float(np.nanmax(ys[i]))

        # N_lensed=1/2/3 overlays on all panels.
        plot_cycle_lines(
            td_arr,
            td_arr_ms,
            eta=eta,
            f_min=f_min,
            mcz_scale=overlay_mcz_scale,
            ax=ax,
        )

        # Peaks/troughs only on panel 1 (Non-Precessing).
        if i == 0:
            mcz_min_unscaled = mcz_min / overlay_mcz_scale
            mcz_max_unscaled = mcz_max / overlay_mcz_scale

            td_peak, mcz_peak = find_mcz_peaks(
                td_arr,
                eta=eta,
                mcz_min=mcz_min_unscaled,
                mcz_max=mcz_max_unscaled,
            )
            if td_peak.size > 0:
                ax.scatter(
                    td_peak * 1e3,
                    mcz_peak * overlay_mcz_scale,
                    c="magenta",
                    marker=".",
                    s=7,
                    alpha=0.9,
                    zorder=6,
                )

            td_trough, mcz_trough = find_mcz_troughs(
                td_arr,
                eta=eta,
                mcz_min=mcz_min_unscaled,
                mcz_max=mcz_max_unscaled,
            )
            if td_trough.size > 0:
                ax.scatter(
                    td_trough * 1e3,
                    mcz_trough * overlay_mcz_scale,
                    c="white",
                    marker=".",
                    s=7,
                    alpha=0.9,
                    zorder=6,
                )

        row, col = divmod(i, 2)
        ax.set_xlabel(X_AXIS_LABEL if row == 1 else "")
        ax.set_ylabel(Y_AXIS_LABEL if col == 0 else "")

        # In-panel label boxes (final style, non-bold text).
        ax.text(
            0.03,
            0.97,
            panel_labels[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="black",
            fontsize=11,
            fontweight="normal",
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.75,
                "pad": 3.0,
            },
            zorder=10,
        )

        set_square_axes(ax)

    # Leave fixed margin for manual colorbar and bottom legend.
    fig.subplots_adjust(
        left=0.09,
        right=0.84,
        top=0.90,
        bottom=0.12,
        wspace=0.01,
        hspace=0.08,
    )

    # Colorbar aligned to right-column axes bounds (excludes xlabel extent).
    fig.canvas.draw()
    pos_top_right = axes[1].get_position()
    pos_bottom_right = axes[3].get_position()

    x_right = max(pos_top_right.x1, pos_bottom_right.x1)
    y0 = pos_bottom_right.y0
    y1 = pos_top_right.y1

    cax = fig.add_axes([x_right + COLORBAR_PAD, y0, COLORBAR_WIDTH, y1 - y0])
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label(COLORBAR_LABEL)
    format_colorbar_ticks(
        cbar, global_min, global_max,
        use_locator=True, nbins=cbar_n_ticks, decimals=decimals,
    )

    overlay_handles = make_fixed_mcz_overlay_legend_handles(
        cycle_n_list=[1, 2, 3],
        include_peaks=True,
        include_troughs=True,
    )
    add_overlay_legend(fig, overlay_handles, ncol=5)

    save_figure(fig, output_path, dpi=dpi)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the finalized paper-style 2x2 contour comparison figure."
    )
    parser.add_argument(
        "--paths",
        nargs=4,
        default=DEFAULT_PATHS,
        help="Exactly 4 input paths in panel order.",
    )
    parser.add_argument(
        "--panel-labels",
        nargs=4,
        default=DEFAULT_PANEL_LABELS,
        help="Exactly 4 in-panel box labels.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fig-dir", type=str, default="figures/contour_mcz_td")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--levels", type=int, default=160)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--overlay-z-from", type=float, default=1e-8)
    parser.add_argument("--overlay-z-to", type=float, default=1.0)
    parser.add_argument("--cbar-n-ticks", type=int, default=12)
    parser.add_argument("--cmap", type=str, default="jet")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = _validate_paths(args.paths)
    create_figure(
        paths=paths,
        panel_labels=args.panel_labels,
        output_path=args.output,
        fig_dir=args.fig_dir,
        decimals=args.decimals,
        levels_count=args.levels,
        dpi=args.dpi,
        eta=args.eta,
        f_min=args.f_min,
        z_from=args.overlay_z_from,
        z_to=args.overlay_z_to,
        cbar_n_ticks=args.cbar_n_ticks,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
