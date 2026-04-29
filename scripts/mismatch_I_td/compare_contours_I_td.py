"""Create a 3-panel publication-style td-I mismatch comparison for Taman system 2.

This script expects three aggregated best-match HDF5 files produced by
``python -m scripts.mismatch_I_td.aggregate_best_match``. For the intended
non-precessing comparison, those inputs should come from the existing I_td
pipeline run with a degenerate one-template bank:

- omega = 0 with omega_pts = 1
- theta = 0 with theta_pts = 1
- gamma_P = 0 with gamma_pts = 1

The three panels are sorted by source-frame chirp mass and rendered with a
shared color scale. Each panel overlays:

- visible N_lensed = 1, 2, 3 lines
- visible peak lines in mismatch
- visible trough lines in mismatch
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
from modules.cosmology import mcz_src_to_det
from modules.plot_utils import apply_physics_paper_style
from scripts.utils.plot_cycles_and_extrema_mcz import (
    fixed_mcz_cycle_positions_ms,
    fixed_mcz_peak_positions_ms,
    fixed_mcz_trough_positions_ms,
)

DEFAULT_OUTPUT = "figures/contour_I_td/" "compare_LensingvsNP_sys2_z1_mcz5_15_25.pdf"

X_AXIS_LABEL = r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$"
Y_AXIS_LABEL = r"$I$"
COLORBAR_LABEL = (
    r"$\epsilon\left(\tilde{h}_{\mathrm{L}},\,\tilde{h}_{\mathrm{NP}}\right)$"
)

_CYCLE_STYLES = {1: "-", 2: "--", 3: ":"}


def _validate_paths(paths: Sequence[str]) -> list[str]:
    if len(paths) != 3:
        raise ValueError(f"Expected exactly 3 input paths, got {len(paths)}")
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    return list(paths)


def _sort_datasets(paths: Sequence[str]) -> list[dict]:
    datasets = [
        read_best_match_I_td_contour_data(path, "epsilon_min") for path in paths
    ]
    return sorted(datasets, key=lambda dataset: float(dataset["mcz"]))


def _validate_shared_metadata(datasets: Sequence[dict]) -> None:
    reference = datasets[0]
    td_ref = np.asarray(reference["td"], dtype=float)
    i_ref = np.asarray(reference["I"], dtype=float)
    orientation_ref = str(reference["orientation_tag"])
    z_ref = reference["z"]

    for dataset in datasets[1:]:
        if not np.allclose(
            np.asarray(dataset["td"], dtype=float), td_ref, atol=0.0, rtol=0.0
        ):
            raise ValueError("All inputs must share the same td grid.")
        if not np.allclose(
            np.asarray(dataset["I"], dtype=float), i_ref, atol=0.0, rtol=0.0
        ):
            raise ValueError("All inputs must share the same I grid.")
        if str(dataset["orientation_tag"]) != orientation_ref:
            raise ValueError("All inputs must share the same orientation_tag.")

        z_val = dataset["z"]
        if z_ref is None and z_val is None:
            continue
        if z_ref is None or z_val is None:
            raise ValueError("All inputs must either define z or omit it.")
        if not np.isclose(float(z_ref), float(z_val), atol=0.0, rtol=0.0):
            raise ValueError("All inputs must share the same redshift z.")


def _detector_mcz(mcz_source_msun: float, z: float | None) -> float:
    if z is None:
        return float(mcz_source_msun)
    return float(mcz_src_to_det(float(mcz_source_msun), float(z)))


def _draw_vertical_lines(
    ax,
    td_positions_ms,
    *,
    color,
    ls,
    lw,
    alpha,
    zorder,
) -> None:
    for td_ms in td_positions_ms:
        ax.axvline(
            td_ms,
            color=color,
            ls=ls,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
        )


def _draw_cycle_overlay(ax, mcz_source_msun, z, td_ms_arr, f_min, eta):
    mcz_det = _detector_mcz(mcz_source_msun, z)
    td_min_ms = float(np.min(td_ms_arr))
    td_max_ms = float(np.max(td_ms_arr))
    cycle_positions = fixed_mcz_cycle_positions_ms(
        mcz_det,
        td_min_ms,
        td_max_ms,
        eta=eta,
        f_min=f_min,
        cycle_counts=tuple(_CYCLE_STYLES),
    )
    for n_cycles, td_ms in cycle_positions.items():
        _draw_vertical_lines(
            ax,
            [td_ms],
            color="black",
            ls=_CYCLE_STYLES[n_cycles],
            lw=1.0,
            alpha=0.9,
            zorder=6,
        )


def _draw_extrema_overlay(ax, mcz_source_msun, z, td_ms_arr, eta):
    mcz_det = _detector_mcz(mcz_source_msun, z)
    td_min_ms = float(np.min(td_ms_arr))
    td_max_ms = float(np.max(td_ms_arr))
    _draw_vertical_lines(
        ax,
        fixed_mcz_peak_positions_ms(mcz_det, td_min_ms, td_max_ms, eta=eta),
        color="magenta",
        ls=":",
        lw=1.0,
        alpha=0.9,
        zorder=6,
    )
    _draw_vertical_lines(
        ax,
        fixed_mcz_trough_positions_ms(mcz_det, td_min_ms, td_max_ms, eta=eta),
        color="white",
        ls=":",
        lw=1.0,
        alpha=0.9,
        zorder=6,
    )


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


def _add_overlay_legend(fig) -> None:
    overlay_handles = [
        Line2D([0], [0], color="black", lw=1, ls="-", label=r"$N_{\mathrm{lensed}}=1$"),
        Line2D(
            [0], [0], color="black", lw=1, ls="--", label=r"$N_{\mathrm{lensed}}=2$"
        ),
        Line2D([0], [0], color="black", lw=1, ls=":", label=r"$N_{\mathrm{lensed}}=3$"),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor="magenta",
            markeredgecolor="magenta",
            label="peak",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            label="trough",
        ),
    ]
    overlay_legend = fig.legend(
        handles=overlay_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=True,
        fontsize=11,
    )
    overlay_legend.get_frame().set_alpha(0.35)


def create_figure(
    paths: Sequence[str],
    output_path: str,
    levels_count: int,
    dpi: int,
    cmap: str,
    f_min: float,
    eta: float,
) -> None:
    paths = _validate_paths(paths)
    datasets = _sort_datasets(paths)
    _validate_shared_metadata(datasets)

    masked_values = [
        np.ma.masked_invalid(np.asarray(dataset["values"], dtype=float))
        for dataset in datasets
    ]
    global_min = min(
        float(masked.min()) for masked in masked_values if masked.count() > 0
    )
    global_max = max(
        float(masked.max()) for masked in masked_values if masked.count() > 0
    )

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("Input datasets do not contain any finite mismatch values.")

    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=11)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), sharex=True, sharey=True)
    levels = np.linspace(global_min, global_max, levels_count)
    contour_set = None
    # Ensure 20 ms is always included
    xticks = np.arange(20.0, 70.1, 10.0)

    z_value = datasets[0]["z"]

    for index, (ax, dataset, masked) in enumerate(zip(axes, datasets, masked_values)):
        td_ms = np.asarray(dataset["td"], dtype=float) * 1e3
        i_arr = np.asarray(dataset["I"], dtype=float)
        td_grid, i_grid = np.meshgrid(td_ms, i_arr)

        contour_set = ax.contourf(
            td_grid,
            i_grid,
            masked,
            levels=levels,
            cmap=cmap,
        )

        mcz_source_msun = float(dataset["mcz"])

        _draw_extrema_overlay(ax, mcz_source_msun, z_value, td_ms, eta)
        _draw_cycle_overlay(
            ax,
            mcz_source_msun,
            z_value,
            td_ms,
            f_min,
            eta,
        )
        _add_mass_box(ax, mcz_source_msun)

        ax.set_xlabel(X_AXIS_LABEL)
        ax.set_xticks(xticks)
        xtick_labels = [f"{int(tick):d}" for tick in xticks]
        ax.set_xticklabels(xtick_labels)
        ax.tick_params(direction="in", top=True, right=True)
        # Use fixed locator for y-ticks for clarity, let matplotlib handle labels
        major_locator = mticker.MultipleLocator(0.2)
        minor_locator = mticker.MultipleLocator(0.1)
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        if index == 0:
            ax.tick_params(axis="y", which="both", labelleft=True)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

    axes[0].set_ylabel(Y_AXIS_LABEL)
    axes[0].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    axes[0].yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    # No figure title per user request

    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.22, top=0.88, wspace=0.04)

    fig.canvas.draw()
    right_pos = axes[-1].get_position()
    left_pos = axes[0].get_position()
    # Slimmer colorbar (was 0.022, now 0.016)
    cax = fig.add_axes([right_pos.x1 + 0.018, left_pos.y0, 0.016, left_pos.height])
    colorbar = fig.colorbar(contour_set, cax=cax)
    colorbar.set_label(COLORBAR_LABEL)
    tick_locator = mticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
    colorbar_ticks = tick_locator.tick_values(global_min, global_max)
    colorbar_ticks = colorbar_ticks[
        (colorbar_ticks >= global_min - 1e-12) & (colorbar_ticks <= global_max + 1e-12)
    ]
    colorbar_ticks = np.unique(
        np.concatenate(([global_min], colorbar_ticks, [global_max]))
    )
    colorbar.set_ticks(colorbar_ticks)
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    _add_overlay_legend(fig)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs=3,
        required=True,
        help="Three aggregated I_td best-match HDF5 files. Panels are sorted by source chirp mass.",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--levels", type=int, default=160)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--eta", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    create_figure(
        paths=args.paths,
        output_path=args.output,
        levels_count=args.levels,
        dpi=args.dpi,
        cmap=args.cmap,
        f_min=args.f_min,
        eta=args.eta,
    )


if __name__ == "__main__":
    main()
