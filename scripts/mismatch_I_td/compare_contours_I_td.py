"""Create a publication-style td-I mismatch comparison for one Taman orientation.

This script expects one or more aggregated best-match HDF5 files produced by
``python -m scripts.mismatch_I_td.aggregate_best_match`` (RP templates) or
the legacy single-template NP runs in ``data/contour_I_td/``. The template
family is auto-detected per file and validated to match across panels:

- NP: degenerate bank (no ``omega_best``/``theta_best``/``gamma_best``
  datasets, or ``template_family='NP'`` attribute).
- RP: full ``omega x theta x gamma`` bank with per-cell best-fit datasets.

The panels are sorted by source-frame chirp mass and rendered with a
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

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.bank_io import read_best_match_I_td_contour_data
from modules.cli_utils import add_cycle_extrema_overlay_args
from modules.filenames import compare_I_td_figure_filename
from modules.cosmology import mcz_src_to_det
from modules.plot_utils import apply_physics_paper_style
from scripts.utils.plot_cycles_and_extrema import (
    draw_fixed_mcz_cycle_overlay,
    draw_fixed_mcz_extrema_overlay,
    make_fixed_mcz_overlay_legend_handles,
)

X_AXIS_LABEL = r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$"
Y_AXIS_LABEL = r"$I$"
COLORBAR_LABEL_TEMPLATE = (
    r"$\epsilon\left(\tilde{{h}}_{{\mathrm{{L}}}},\,"
    r"\tilde{{h}}_{{\mathrm{{{family}}}}}\right)$"
)
TEMPLATE_FAMILIES = ("NP", "RP")


def _validate_paths(paths: Sequence[str]) -> list[str]:
    if len(paths) < 1:
        raise ValueError("Expected at least 1 input path")
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    return list(paths)


def _detect_template_family(path: str) -> str:
    with h5py.File(path, "r") as h5:
        raw = h5.attrs.get("template_family")
        if raw is not None:
            tag = raw.decode() if isinstance(raw, bytes) else str(raw)
            tag = tag.strip().upper()
            if tag in TEMPLATE_FAMILIES:
                return tag
        bank_keys = ("omega_best", "theta_best", "gamma_best")
        return "RP" if all(k in h5 for k in bank_keys) else "NP"


def _sort_datasets(paths: Sequence[str]) -> list[dict]:
    datasets = []
    for path in paths:
        dataset = read_best_match_I_td_contour_data(path, "epsilon_min")
        dataset["template_family"] = _detect_template_family(path)
        datasets.append(dataset)
    return sorted(datasets, key=lambda dataset: float(dataset["mcz"]))


def _validate_shared_metadata(datasets: Sequence[dict]) -> None:
    reference = datasets[0]
    td_ref = np.asarray(reference["td"], dtype=float)
    i_ref = np.asarray(reference["I"], dtype=float)
    orientation_ref = str(reference["orientation_tag"])
    z_ref = reference["z"]
    family_ref = str(reference["template_family"])

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
        if str(dataset["template_family"]) != family_ref:
            raise ValueError(
                "All inputs must share the same template family "
                f"(got {family_ref!r} and {dataset['template_family']!r})."
            )

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


def _add_overlay_legend(
    fig,
    *,
    include_cycles: bool,
    include_peaks: bool,
    include_troughs: bool,
) -> None:
    handles = make_fixed_mcz_overlay_legend_handles(
        cycle_n_list=[1, 2, 3] if include_cycles else None,
        include_peaks=include_peaks,
        include_troughs=include_troughs,
    )
    if not handles:
        return
    overlay_legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(handles),
        frameon=True,
        fontsize=11,
    )
    overlay_legend.get_frame().set_alpha(0.35)


def create_figure(
    paths: Sequence[str],
    output_path: str | None,
    levels_count: int,
    dpi: int,
    cmap: str,
    f_min: float,
    eta: float,
    template_family: str | None,
    overlay_cycles: bool,
    overlay_peaks: bool,
    overlay_troughs: bool,
) -> None:
    paths = _validate_paths(paths)
    datasets = _sort_datasets(paths)
    if template_family is not None:
        family = template_family.strip().upper()
        if family not in TEMPLATE_FAMILIES:
            raise ValueError(
                f"--template_family must be one of {TEMPLATE_FAMILIES}, got {template_family!r}."
            )
        for dataset in datasets:
            dataset["template_family"] = family
    _validate_shared_metadata(datasets)
    family = str(datasets[0]["template_family"])
    if output_path is None:
        output_path = compare_I_td_figure_filename(
            fig_dir="figures/contour_I_td",
            template_family=family,
            mcz_values=[float(dataset["mcz"]) for dataset in datasets],
            orientation_tag=str(datasets[0]["orientation_tag"]),
            z=datasets[0]["z"],
        )
    colorbar_label = COLORBAR_LABEL_TEMPLATE.format(family=family)

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

    ncols = len(datasets)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(4.8, 4.0 * ncols + 0.6), 4.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
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

        mcz_det = _detector_mcz(mcz_source_msun, z_value)
        td_min_ms = float(td_ms.min())
        td_max_ms = float(td_ms.max())
        if overlay_peaks or overlay_troughs:
            draw_fixed_mcz_extrema_overlay(
                ax,
                mcz_det,
                td_min_ms,
                td_max_ms,
                eta=eta,
                plot_peaks=overlay_peaks,
                plot_troughs=overlay_troughs,
            )
        if overlay_cycles:
            draw_fixed_mcz_cycle_overlay(
                ax,
                mcz_det,
                td_min_ms,
                td_max_ms,
                eta=eta,
                f_min=f_min,
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

    # No figure title per user request

    fig.subplots_adjust(
        left=0.11 if ncols > 1 else 0.14,
        right=0.84 if ncols > 1 else 0.82,
        bottom=0.22,
        top=0.88,
        wspace=0.04,
    )

    fig.canvas.draw()
    right_pos = axes[-1].get_position()
    left_pos = axes[0].get_position()
    # Slimmer colorbar (was 0.022, now 0.016)
    cax = fig.add_axes([right_pos.x1 + 0.018, left_pos.y0, 0.016, left_pos.height])
    colorbar = fig.colorbar(contour_set, cax=cax)
    colorbar.set_label(colorbar_label)
    colorbar.set_ticks(np.linspace(global_min, global_max, 8))
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    _add_overlay_legend(
        fig,
        include_cycles=overlay_cycles,
        include_peaks=overlay_peaks,
        include_troughs=overlay_troughs,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help=(
            "One or more aggregated I_td best-match HDF5 files. "
            "Panels are sorted by source chirp mass."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output figure path. Defaults to "
            "figures/contour_I_td/compare_Lensingvs<family>_Taman_<orientation>_z<z>_mcz<mass-list>.pdf."
        ),
    )
    parser.add_argument("--levels", type=int, default=160)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--cmap", type=str, default="jet")
    add_cycle_extrema_overlay_args(
        parser,
        include_show_legend=False,
    )
    parser.add_argument(
        "--template_family",
        type=str,
        choices=list(TEMPLATE_FAMILIES),
        default=None,
        help=(
            "Override auto-detected template family used in the colorbar "
            "label and default output filename."
        ),
    )
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
        template_family=args.template_family,
        overlay_cycles=args.overlay_cycles,
        overlay_peaks=args.overlay_peaks,
        overlay_troughs=args.overlay_troughs,
    )


if __name__ == "__main__":
    main()
