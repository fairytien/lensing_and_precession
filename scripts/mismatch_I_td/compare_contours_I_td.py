"""Create a publication-style td-I mismatch comparison figure.

This script expects one or more aggregated best-match HDF5 files produced by
``python -m scripts.mismatch_I_td.aggregate_best_match``. The template family
is taken from explicit ``template_family`` metadata on current pipeline
outputs. RP files that predate that attribute are still accepted when they
contain the standard ``omega_best``/``theta_best``/``gamma_best`` datasets.
Legacy ``best_match_np_*`` files are not supported.

When all inputs share the same chirp mass but differ in orientation, panels
are sorted by system number (1=face-on, 2=edge-on, 3=random) and each panel
shows "System N" in its legend box. Otherwise panels are sorted by
source-frame chirp mass and labelled with the chirp mass value.

Each panel overlays:

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
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.bank_io import read_best_match_I_td_data
from modules.cli_utils import add_cycle_extrema_overlay_args
from modules.filenames import (
    compare_I_td_figure_filename,
    compare_systems_I_td_figure_filename,
)
from modules.cosmology import mcz_src_to_det
from modules.default_params import ORIENTATION_TO_SYSTEM
from modules.plot_utils import (
    LBL_I,
    LBL_TD,
    add_colorbar_axes,
    add_overlay_legend,
    apply_physics_paper_style,
    configure_I_axis,
    format_colorbar_ticks,
    save_figure,
    set_square_axes,
)
from scripts.utils.plot_cycles_and_extrema import (
    draw_fixed_mcz_overlays,
    make_fixed_mcz_overlay_legend_handles,
)

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
            raise ValueError(f"Unsupported template_family={tag!r} in {path}.")
        bank_keys = ("omega_best", "theta_best", "gamma_best")
        if all(k in h5 for k in bank_keys):
            return "RP"
    raise ValueError(
        "Missing template_family metadata for NP compare input. "
        f"Legacy NP best-match files are not supported: {path}"
    )


def _load_datasets(paths: Sequence[str]) -> list[dict]:
    datasets = []
    for path in paths:
        dataset = read_best_match_I_td_data(path, "epsilon_min")
        dataset["template_family"] = _detect_template_family(path)
        datasets.append(dataset)
    return datasets


def _is_multi_orientation(datasets: Sequence[dict]) -> bool:
    tags = {str(d["orientation_tag"]) for d in datasets}
    return len(tags) > 1


def _sort_datasets(datasets: list[dict]) -> list[dict]:
    if _is_multi_orientation(datasets):
        return sorted(
            datasets,
            key=lambda d: ORIENTATION_TO_SYSTEM.get(str(d["orientation_tag"]), 99),
        )
    return sorted(datasets, key=lambda d: float(d["mcz"]))


def _validate_shared_metadata(datasets: Sequence[dict]) -> None:
    reference = datasets[0]
    td_ref = np.asarray(reference["td"], dtype=float)
    i_ref = np.asarray(reference["I"], dtype=float)
    z_ref = reference["z"]
    family_ref = str(reference["template_family"])
    multi_orient = _is_multi_orientation(datasets)

    for dataset in datasets[1:]:
        if not np.allclose(
            np.asarray(dataset["td"], dtype=float), td_ref, atol=0.0, rtol=0.0
        ):
            raise ValueError("All inputs must share the same td grid.")
        if not np.allclose(
            np.asarray(dataset["I"], dtype=float), i_ref, atol=0.0, rtol=0.0
        ):
            raise ValueError("All inputs must share the same I grid.")
        if not multi_orient:
            if str(dataset["orientation_tag"]) != str(reference["orientation_tag"]):
                raise ValueError("All inputs must share the same orientation_tag.")
        else:
            if not np.isclose(
                float(dataset["mcz"]), float(reference["mcz"]), atol=0.0, rtol=0.0
            ):
                raise ValueError(
                    "Multi-orientation inputs must share the same chirp mass "
                    f"(got {reference['mcz']} and {dataset['mcz']})."
                )
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


def _add_panel_box(ax, label: str) -> None:
    legend = ax.legend(
        [Line2D([], [], linestyle="none")],
        [label],
        loc="upper left",
        frameon=True,
        handlelength=0,
        handletextpad=0.0,
        borderpad=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(0.75)
    legend.set_zorder(10)


def _panel_label(dataset: dict, multi_orient: bool) -> str:
    if multi_orient:
        tag = str(dataset["orientation_tag"])
        system_num = ORIENTATION_TO_SYSTEM.get(tag)
        return f"System {system_num}" if system_num is not None else tag
    mcz = float(dataset["mcz"])
    return rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {mcz:g}\,\mathrm{{M}}_\odot$"


def create_figure(
    paths: Sequence[str],
    output_path: str | None,
    levels_count: int,
    cmap: str,
    f_min: float,
    eta: float,
    template_family: str | None,
    overlay_cycles: bool,
    overlay_peaks: bool,
    overlay_troughs: bool,
    cbar_n_ticks: int = 6,
    decimals: int = 2,
) -> None:
    paths = _validate_paths(paths)
    datasets = _load_datasets(paths)
    if template_family is not None:
        family = template_family.strip().upper()
        if family not in TEMPLATE_FAMILIES:
            raise ValueError(
                f"--template_family must be one of {TEMPLATE_FAMILIES}, got {template_family!r}."
            )
        for dataset in datasets:
            dataset["template_family"] = family
    datasets = _sort_datasets(datasets)
    _validate_shared_metadata(datasets)
    family = str(datasets[0]["template_family"])
    multi_orient = _is_multi_orientation(datasets)
    if output_path is None:
        if multi_orient:
            output_path = compare_systems_I_td_figure_filename(
                fig_dir="figures/contour_I_td",
                template_family=family,
                mcz_msun=float(datasets[0]["mcz"]),
                orientation_tags=[str(d["orientation_tag"]) for d in datasets],
                z=datasets[0]["z"],
            )
        else:
            output_path = compare_I_td_figure_filename(
                fig_dir="figures/contour_I_td",
                template_family=family,
                mcz_values=[float(dataset["mcz"]) for dataset in datasets],
                orientation_tag=str(datasets[0]["orientation_tag"]),
                z=datasets[0]["z"],
            )
    family_display = "P" if family == "RP" else family
    colorbar_label = COLORBAR_LABEL_TEMPLATE.format(family=family_display)

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

    # Pad the colorbar limits to nice even numbers so the top and bottom ticks
    # align exactly with the boundaries.
    vmin = 0.0
    if global_max <= 0.1:
        vmax = float(np.ceil(global_max / 0.01) * 0.01)
    elif global_max <= 0.25:
        vmax = float(np.ceil(global_max / 0.02) * 0.02)
    else:
        vmax = float(np.ceil(global_max / 0.05) * 0.05)

    apply_physics_paper_style()

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
    levels = np.linspace(vmin, vmax, levels_count)
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
        _add_panel_box(ax, _panel_label(dataset, multi_orient))

        ax.set_xlabel(LBL_TD)
        ax.set_xticks(xticks)
        xtick_labels = [f"{int(tick):d}" for tick in xticks]
        ax.set_xticklabels(xtick_labels)
        set_square_axes(ax)
        if index == 0:
            ax.tick_params(axis="y", which="both", labelleft=True)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)

    axes[0].set_ylabel(LBL_I)
    configure_I_axis(axes[0])

    # No figure title per user request

    fig.subplots_adjust(
        left=0.11 if ncols > 1 else 0.14,
        right=0.84 if ncols > 1 else 0.82,
        bottom=0.22,
        top=0.88,
        wspace=0.04,
    )

    cax = add_colorbar_axes(fig, axes)
    colorbar = fig.colorbar(contour_set, cax=cax)
    colorbar.set_label(colorbar_label)
    format_colorbar_ticks(
        colorbar,
        vmin,
        vmax,
        nbins=cbar_n_ticks,
        decimals=decimals,
    )

    overlay_handles = make_fixed_mcz_overlay_legend_handles(
        cycle_n_list=[1, 2, 3] if overlay_cycles else None,
        include_peaks=overlay_peaks,
        include_troughs=overlay_troughs,
    )
    add_overlay_legend(fig, overlay_handles)

    save_figure(fig, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help=(
            "One or more aggregated I_td best-match HDF5 files. "
            "When all inputs share a chirp mass but differ in orientation, "
            "panels sort by system number and show 'System N' labels. "
            "Otherwise panels sort by source chirp mass."
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
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--cbar-n-ticks", type=int, default=6)
    parser.add_argument("--decimals", type=int, default=2)
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
        cmap=args.cmap,
        f_min=args.f_min,
        eta=args.eta,
        template_family=args.template_family,
        overlay_cycles=args.overlay_cycles,
        overlay_peaks=args.overlay_peaks,
        overlay_troughs=args.overlay_troughs,
        cbar_n_ticks=args.cbar_n_ticks,
        decimals=args.decimals,
    )


if __name__ == "__main__":
    main()
