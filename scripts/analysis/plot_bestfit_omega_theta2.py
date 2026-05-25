"""Plot the best-fit field omega_tilde * theta_tilde^2 on the native best-match grid."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is importable when running this file directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.filenames import bestfit_prec_params_I_td_figure_filename
from modules.plot_utils import (
    add_colorbar_axes,
    apply_physics_paper_style,
    format_colorbar_ticks,
    save_figure,
    set_square_axes,
)

BestMatchData = Dict[str, Any]


def _decode_attr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _load_best_match(path: str) -> BestMatchData:
    with h5py.File(path, "r") as h5:
        td = np.asarray(h5["td"], dtype=float)
        omega_best = np.asarray(h5["omega_best"], dtype=float)
        theta_best = np.asarray(h5["theta_best"], dtype=float)
        z = float(h5.attrs.get("z", h5.attrs.get("source_param_z", 0.0)))
        orientation = _decode_attr_text(h5.attrs.get("orientation_tag", ""))
        axis_order = _decode_attr_text(h5["omega_best"].attrs.get("axis_order", ""))
        axis_order = axis_order.replace(" ", "")

        if axis_order == "mcz,td" or (
            "mcz" in h5 and np.asarray(h5["mcz"]).size == omega_best.shape[0]
        ):
            axis_values = np.asarray(h5["mcz"], dtype=float).reshape(-1)
            axis_kind = "mcz"
            axis_label = r"$\mathcal{M}_{\mathrm{s}}\,[\mathrm{M}_\odot]$"
            I_value = float(h5.attrs.get("I", h5.attrs.get("source_param_I", np.nan)))
            mcz_value = np.nan
        elif axis_order == "I,td" or (
            "I" in h5 and np.asarray(h5["I"]).size == omega_best.shape[0]
        ):
            axis_values = np.asarray(h5["I"], dtype=float).reshape(-1)
            axis_kind = "I"
            axis_label = r"$I$"
            mcz_value = float(np.asarray(h5["mcz"], dtype=float).reshape(-1)[0])
            I_value = np.nan
        else:
            raise ValueError(
                f"Could not infer vertical axis for {path}: omega_best axis_order='{axis_order}'"
            )

    return {
        "axis_kind": axis_kind,
        "axis_values": axis_values,
        "axis_label": axis_label,
        "td_ms": td * 1e3,
        "omega_best": omega_best,
        "theta_best": theta_best,
        "z": z,
        "orientation": orientation,
        "I_value": I_value,
        "mcz_value": mcz_value,
    }


def _resolve_labels(paths: List[str], labels: List[str] | None) -> List[str]:
    if labels is None:
        return [os.path.splitext(os.path.basename(path))[0] for path in paths]
    if len(labels) != len(paths):
        raise ValueError(
            f"Expected --labels to match --paths ({len(paths)}), got {len(labels)}"
        )
    return labels


def _validate_inputs(paths: List[str]) -> None:
    if len(paths) < 1:
        raise ValueError("Expected at least 1 --path")

    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))


def _derived_field(dataset: BestMatchData) -> np.ndarray:
    omega = np.asarray(dataset["omega_best"], dtype=float)
    theta = np.asarray(dataset["theta_best"], dtype=float)
    return omega * theta**2


def _levels_for_arrays(arrays: List[np.ndarray], n: int) -> np.ndarray:
    lo = min(float(np.nanmin(arr)) for arr in arrays)
    hi = max(float(np.nanmax(arr)) for arr in arrays)
    return np.linspace(lo, hi, n)


def _default_output_path(datasets: List[BestMatchData], axis_kind: str) -> str:
    if axis_kind == "I":
        base = bestfit_prec_params_I_td_figure_filename(
            fig_dir="figures/contour_I_td",
            mcz_values=[float(dataset["mcz_value"]) for dataset in datasets],
            I_min=float(np.nanmin(datasets[0]["axis_values"])),
            I_max=float(np.nanmax(datasets[0]["axis_values"])),
            td_min_ms=float(np.nanmin(datasets[0]["td_ms"])),
            td_max_ms=float(np.nanmax(datasets[0]["td_ms"])),
            orientation_tag=str(datasets[0]["orientation"]),
            z=float(datasets[0]["z"]),
        )
        stem, ext = os.path.splitext(base)
        return f"{stem}_omega_theta2{ext or '.pdf'}"
    return "figures/contour_mcz_td/bestfit_prec_params_omega_theta2.pdf"


def create_figure(
    paths: List[str],
    labels: List[str] | None,
    output_path: str | None,
    levels_count: int,
    dpi: int,
    cmap: str,
) -> None:
    _validate_inputs(paths)
    labels = _resolve_labels(paths, labels)
    datasets = [_load_best_match(path) for path in paths]

    axis_kinds = {dataset["axis_kind"] for dataset in datasets}
    if len(axis_kinds) != 1:
        raise ValueError(
            "All inputs must use the same vertical axis; do not mix mcz-td and I-td files."
        )

    axis_kind = str(datasets[0]["axis_kind"])
    axis_label = str(datasets[0]["axis_label"])
    output_path = output_path or _default_output_path(datasets, axis_kind)

    fields = [_derived_field(dataset) for dataset in datasets]
    levels = _levels_for_arrays(fields, levels_count)
    ncols = len(datasets)

    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=10)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.6 * ncols + 1.2, 4.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="compressed",
    )

    contour = None
    for index, (dataset, label, field) in enumerate(zip(datasets, labels, fields)):
        ax = axes[0, index]
        contour = ax.contourf(
            dataset["td_ms"],
            dataset["axis_values"],
            field,
            levels=levels,
            cmap=cmap,
        )
        set_square_axes(ax)
        ax.set_title(label)
        ax.set_xlabel(r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$")

    axes[0, 0].set_ylabel(axis_label)

    if contour is None:
        raise ValueError("No datasets were plotted")

    cax = add_colorbar_axes(fig, axes[0, -1])
    cb = fig.colorbar(
        contour,
        cax=cax,
        label=r"$\tilde{\Omega}_{\mathrm{best}}\,\tilde{\theta}_{\mathrm{best}}^2$",
    )
    format_colorbar_ticks(cb, levels[0], levels[-1])
    save_figure(fig, output_path, dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more best_match HDF5 paths (one panel per path)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional panel labels. Defaults to input file stems.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path. Defaults to an orientation-based filename for I-td inputs.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=60,
        help="Number of contour levels",
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