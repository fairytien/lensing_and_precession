"""6-panel comparison: Lensing vs NP (top row) and Lensing vs RP (bottom row).

All panels share a single colorbar saturated at --vmax (default 0.3).
Cycle and extrema overlays are controlled by the standard overlay flags.

Usage example::

    python -m scripts.mismatch_I_td.compare_NP_RP_I_td \\
      --np_paths \\
        data/contour_I_td/results_np_z1_mcz5_I0p1-0p9_td20-70_Taman_edgeon/best_match/np_z1_mcz5_*.h5 \\
        data/contour_I_td/results_np_z1_mcz15_I0p1-0p9_td20-70_Taman_edgeon/best_match/np_z1_mcz15_*.h5 \\
        data/contour_I_td/results_np_z1_mcz25_I0p1-0p9_td20-70_Taman_edgeon/best_match/np_z1_mcz25_*.h5 \\
      --rp_paths \\
        data/mismatch_z1_mcz5_I0p1-0p9_td20-70_Taman_edgeon/best_match/best_match_z1_mcz5_*.h5 \\
        data/mismatch_z1_mcz15_I0p1-0p9_td20-70_Taman_edgeon/best_match/best_match_z1_mcz15_*.h5 \\
        data/mismatch_z1_mcz25_I0p1-0p9_td20-70_Taman_edgeon/best_match/best_match_z1_mcz25_*.h5 \\
      --overlay-cycles --overlay-peaks --overlay-troughs
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.bank_io import read_best_match_I_td_contour_data
from modules.cli_utils import add_cycle_extrema_overlay_args
from modules.cosmology import mcz_src_to_det
from modules.plot_utils import (
    add_colorbar_axes,
    add_overlay_legend,
    apply_physics_paper_style,
    format_colorbar_ticks,
    save_figure,
    set_square_axes,
)
from scripts.utils.plot_cycles_and_extrema import (
    draw_fixed_mcz_overlays,
    make_fixed_mcz_overlay_legend_handles,
)

X_AXIS_LABEL = r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$"
Y_AXIS_LABEL = r"$I$"
COLORBAR_LABEL = (
    r"$\epsilon\left(\tilde{h}_{\mathrm{s}},\,\tilde{h}_{\mathrm{t}}\right)$"
)
ROW_LABELS = ["NP", "RP"]


def _load_row(paths: Sequence[str]) -> list[dict]:
    datasets = []
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing input file: {path}")
        dataset = read_best_match_I_td_contour_data(path, "epsilon_min")
        datasets.append(dataset)
    return sorted(datasets, key=lambda d: float(d["mcz"]))


def _detector_mcz(mcz_src: float, z: float | None) -> float:
    if z is None:
        return float(mcz_src)
    return float(mcz_src_to_det(float(mcz_src), float(z)))


def _add_mass_box(ax, mcz_source_msun: float) -> None:
    legend = ax.legend(
        [Line2D([], [], linestyle="none")],
        [
            rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {mcz_source_msun:g}\,\mathrm{{M}}_\odot$"
        ],
        loc="upper left",
        frameon=True,
        handlelength=0,
        handletextpad=0.0,
        borderpad=0.35,
        fontsize=10.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(0.4)


def create_figure(
    np_paths: Sequence[str],
    rp_paths: Sequence[str],
    output_path: str | None,
    levels_count: int,
    vmax: float,
    cmap: str,
    f_min: float,
    eta: float,
    overlay_cycles: bool,
    overlay_peaks: bool,
    overlay_troughs: bool,
) -> None:
    np_datasets = _load_row(np_paths)
    rp_datasets = _load_row(rp_paths)

    if len(np_datasets) != 3 or len(rp_datasets) != 3:
        raise ValueError("Expected exactly 3 NP paths and 3 RP paths.")

    all_datasets = np_datasets + rp_datasets
    all_masked = [
        np.ma.masked_invalid(np.asarray(d["values"], dtype=float)) for d in all_datasets
    ]

    global_min = min(float(m.min()) for m in all_masked if m.count() > 0)
    levels = np.linspace(global_min, vmax, levels_count)

    z_value = np_datasets[0]["z"]
    orientation_tag = str(np_datasets[0]["orientation_tag"])
    mcz_values = [float(d["mcz"]) for d in np_datasets]

    if output_path is None:
        mcz_token = "_".join(str(int(v)) if v == int(v) else str(v) for v in mcz_values)
        z_token = f"{z_value:g}" if z_value is not None else "noredshift"
        stem = f"compare_LensingvsNP_RP_{orientation_tag}_z{z_token}_mcz{mcz_token}"
        output_path = os.path.join("figures/contour_I_td", f"{stem}.pdf")

    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=11)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 7.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="compressed",
    )

    xticks = np.arange(20.0, 70.1, 10.0)
    contour_set = None

    row_data = [np_datasets, rp_datasets]
    row_masked = [all_masked[:3], all_masked[3:]]

    for row_idx, (datasets, masked_row) in enumerate(zip(row_data, row_masked)):
        for col_idx, (dataset, masked) in enumerate(zip(datasets, masked_row)):
            ax = axes[row_idx][col_idx]

            td_ms = np.asarray(dataset["td"], dtype=float) * 1e3
            i_arr = np.asarray(dataset["I"], dtype=float)
            td_grid, i_grid = np.meshgrid(td_ms, i_arr)

            contour_set = ax.contourf(
                td_grid,
                i_grid,
                masked,
                levels=levels,
                cmap=cmap,
                extend="max",
            )

            mcz_src = float(dataset["mcz"])
            mcz_det = _detector_mcz(mcz_src, z_value)
            td_min_ms = float(td_ms.min())
            td_max_ms = float(td_ms.max())

            draw_fixed_mcz_overlays(
                ax,
                mcz_det,
                td_min_ms,
                td_max_ms,
                overlay_cycles=overlay_cycles,
                overlay_peaks=overlay_peaks,
                overlay_troughs=overlay_troughs,
                eta=eta,
                f_min=f_min,
            )

            set_square_axes(ax)

        # NP/RP row label as a descriptive text box on the leftmost panel
        axes[row_idx][0].text(
            0.97,
            0.97,
            ROW_LABELS[row_idx],
            transform=axes[row_idx][0].transAxes,
            ha="right",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                alpha=0.5,
                boxstyle="round,pad=0.3",
            ),
        )
        axes[row_idx][0].set_ylabel(r"$I$", fontsize=13)

    # Column headers (chirp mass) and bottom row x-axis labels
    for col_idx in range(3):
        axes[0][col_idx].set_title(
            rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {mcz_values[col_idx]:g}\,\mathrm{{M}}_\odot$",
            fontsize=12,
        )
        ax = axes[1][col_idx]
        ax.set_xlabel(X_AXIS_LABEL)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(t):d}" for t in xticks])

    axes[0][0].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    axes[0][0].yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    # Add overlay legend before freezing the layout so the engine reserves space for it.
    handles = make_fixed_mcz_overlay_legend_handles(
        cycle_n_list=[1, 2, 3] if overlay_cycles else None,
        include_peaks=overlay_peaks,
        include_troughs=overlay_troughs,
    )
    add_overlay_legend(
        fig,
        handles,
        loc="outside lower center",
        bbox_to_anchor=None,
    )

    cax = add_colorbar_axes(fig, axes)
    colorbar = fig.colorbar(contour_set, cax=cax, extend="max")
    colorbar.set_label(COLORBAR_LABEL)
    format_colorbar_ticks(
        colorbar,
        0.0,
        vmax,
        use_locator=False,
        n_ticks=int(round(vmax / 0.05)) + 1,
    )

    save_figure(fig, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--np_paths",
        nargs="+",
        required=True,
        help="3 aggregated NP best-match HDF5 files (sorted by mcz).",
    )
    parser.add_argument(
        "--rp_paths",
        nargs="+",
        required=True,
        help="3 aggregated RP best-match HDF5 files (sorted by mcz).",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--levels", type=int, default=160)
    parser.add_argument(
        "--vmax", type=float, default=0.3, help="Colorbar maximum (default 0.3)."
    )
    parser.add_argument("--cmap", type=str, default="jet")
    add_cycle_extrema_overlay_args(parser, include_show_legend=False)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    create_figure(
        np_paths=args.np_paths,
        rp_paths=args.rp_paths,
        output_path=args.output,
        levels_count=args.levels,
        vmax=args.vmax,
        cmap=args.cmap,
        f_min=args.f_min,
        eta=args.eta,
        overlay_cycles=args.overlay_cycles,
        overlay_peaks=args.overlay_peaks,
        overlay_troughs=args.overlay_troughs,
    )


if __name__ == "__main__":
    main()
