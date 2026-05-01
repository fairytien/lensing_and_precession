"""Visualize best-fit precession parameters on the (td, y) grid.

Creates a 2xN panel figure for any number of systems:
- Top row: best-matching omega_tilde on the native best-match grid
- Bottom row: best-matching theta_tilde on the native best-match grid

For multi-system runs, each row uses a shared color scale across panels.
N_lensed = 1/2/3 contours are overlaid on every panel.

When --slice-mcz is provided for mcz-td inputs, the script instead creates
two line plots versus time delay at the requested source-frame chirp mass.
When --slice-td-ms is provided, the script instead creates two line plots
versus the native vertical-axis variable at the requested time delay.
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
    "data/mismatch_z1_mcz15_I0p1-0p9_td20-70_Taman_edgeon/best_match/"
    "best_match_z1_mcz15_I0p1-0p9x81_td20-70x51_omega0-6x61_theta0-15x151_"
    "gamma0-2pix51_Taman_edgeon.h5",
]

DEFAULT_LABELS = ["System 2 (edge-on)"]

DEFAULT_OUTPUT = "figures/contour_mcz_td/bestfit_prec_params_sys2.pdf"


def _decode_attr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _load_best_match(path: str) -> Dict[str, np.ndarray]:
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
            mcz = np.asarray(h5["mcz"], dtype=float).reshape(-1)
            axis_values = mcz
            axis_kind = "mcz"
            axis_label = r"$\mathcal{M}_{\mathrm{s}}\,[\mathrm{M}_\odot]$"
            mcz_grid = np.broadcast_to(mcz[:, None], omega_best.shape)
            I_value = float(h5.attrs.get("I", h5.attrs.get("source_param_I", np.nan)))
            mcz_value = np.nan
        elif axis_order == "I,td" or (
            "I" in h5 and np.asarray(h5["I"]).size == omega_best.shape[0]
        ):
            axis_values = np.asarray(h5["I"], dtype=float).reshape(-1)
            axis_kind = "I"
            axis_label = r"$I$"
            mcz_value = float(np.asarray(h5["mcz"], dtype=float).reshape(-1)[0])
            mcz_grid = np.full(omega_best.shape, mcz_value, dtype=float)
            I_value = np.nan
        else:
            raise ValueError(
                f"Could not infer vertical axis for {path}: omega_best axis_order='{axis_order}'"
            )

    td_ms = td * 1e3
    TD, _ = np.meshgrid(td, np.arange(omega_best.shape[0], dtype=float))
    MCZ = np.asarray(mcz_grid, dtype=float)
    nlens = number_of_lens_cycles(MCZ * (1 + z), TD)

    return {
        "axis_kind": axis_kind,
        "axis_values": axis_values,
        "axis_label": axis_label,
        "td_ms": td_ms,
        "omega_best": omega_best,
        "theta_best": theta_best,
        "nlens": nlens,
        "z": z,
        "orientation": orientation,
        "I_value": I_value,
        "mcz_value": mcz_value,
    }


def _validate_inputs(paths: List[str], labels: List[str]) -> None:
    if len(paths) < 1:
        raise ValueError("Expected at least 1 --path")
    if len(labels) != len(paths):
        raise ValueError(
            f"Expected --labels to match --paths ({len(paths)}), got {len(labels)}"
        )

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))


def _select_axis_index(
    axis_values: np.ndarray,
    target: float,
    axis_name: str,
    unit_suffix: str = "",
) -> tuple[int, float]:
    axis = np.asarray(axis_values, dtype=float).reshape(-1)
    if axis.size == 0:
        raise ValueError("Cannot slice an empty axis")

    idx = int(np.nanargmin(np.abs(axis - target)))
    nearest = float(axis[idx])

    unique_axis = np.unique(axis[np.isfinite(axis)])
    if unique_axis.size > 1:
        steps = np.diff(np.sort(unique_axis))
        positive_steps = steps[steps > 0]
        min_step = float(np.min(positive_steps)) if positive_steps.size else 0.0
    else:
        min_step = 0.0

    atol = max(1e-6, 0.51 * min_step)
    if not np.isfinite(nearest) or abs(nearest - target) > atol:
        raise ValueError(
            f"Requested {axis_name} slice {target:g}{unit_suffix} is not on the input grid; "
            f"nearest available value is {nearest:g}{unit_suffix}"
        )

    return idx, nearest


def _is_effectively_exact(
    selected_values: List[float], target: float, atol: float = 1e-6
) -> bool:
    return max(abs(float(val) - target) for val in selected_values) <= atol


def _uniq_sorted_field(datasets: List[Dict[str, np.ndarray]], key: str) -> str:
    return ", ".join(
        sorted({f"{float(d[key]):g}" for d in datasets if np.isfinite(float(d[key]))})
    )


def _z_detail_part(datasets: List[Dict[str, np.ndarray]]) -> str | None:
    z_txt = ", ".join(sorted({f"{d['z']:g}" for d in datasets}))
    return rf"$z={z_txt}$" if z_txt else None


def _extend_z(details: List[str], datasets: List[Dict[str, np.ndarray]]) -> None:
    if p := _z_detail_part(datasets):
        details.append(p)


def _axis_colors(n: int) -> List[str | None]:
    cycle = plt.rcParams.get("axes.prop_cycle")
    colors = cycle.by_key().get("color", []) if cycle is not None else []
    return [colors[i % len(colors)] if colors else None for i in range(n)]


def _plot_line_pair_figure(
    xs: List[np.ndarray],
    omega_ys: List[np.ndarray],
    theta_ys: List[np.ndarray],
    labels: List[str],
    xlabel: str,
    suptitle: str,
    output_path: str,
    dpi: int,
    selection_lines: List[str],
) -> None:
    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=10)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    colors = _axis_colors(len(labels))
    kw = dict(linewidth=2.0, marker="o", markersize=3.4)
    for x, oy, ty, label, color in zip(xs, omega_ys, theta_ys, labels, colors):
        axes[0].plot(x, oy, label=label, color=color, **kw)
        axes[1].plot(x, ty, label=label, color=color, **kw)

    axes[0].set_ylabel(r"$\tilde{\Omega}_{\mathrm{best}}$")
    axes[1].set_ylabel(r"$\tilde{\theta}_{\mathrm{best}}$")
    axes[1].set_xlabel(xlabel)
    for ax in axes:
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.tick_params(direction="in", top=True, right=True)
        ax.margins(x=0.02)
    axes[0].legend(loc="best", frameon=True)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.11, hspace=0.08)
    fig.suptitle(suptitle, fontsize=13)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    for line in selection_lines:
        print(line)
    print(f"Saved figure: {output_path}")


def _slice_line_figure(
    datasets: List[Dict[str, np.ndarray]],
    labels: List[str],
    output_path: str,
    dpi: int,
    *,
    mcz_row: float | None = None,
    td_column_ms: float | None = None,
) -> None:
    if mcz_row is not None:
        idxs: List[int] = []
        selected: List[float] = []
        for d in datasets:
            i, v = _select_axis_index(
                d["axis_values"], mcz_row, axis_name="mcz", unit_suffix=" Msun"
            )
            idxs.append(i)
            selected.append(v)
        xs = [d["td_ms"] for d in datasets]
        omega_ys = [d["omega_best"][r, :] for d, r in zip(datasets, idxs)]
        theta_ys = [d["theta_best"][r, :] for d, r in zip(datasets, idxs)]
        xlabel = r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$"
        rel = "=" if _is_effectively_exact(selected, mcz_row) else r"\approx"
        details = [
            rf"$\mathcal{{M}}_{{\mathrm{{s}}}}{rel} {mcz_row:g}\,\mathrm{{M}}_\odot$"
        ]
        _extend_z(details, datasets)
        if u := _uniq_sorted_field(datasets, "I_value"):
            details.append(rf"$I={u}$")
        if max(abs(v - mcz_row) for v in selected) > 1e-6:
            sel = ", ".join(f"{v:g}" for v in selected)
            details.append(
                rf"$\mathcal{{M}}_{{\mathrm{{s}}}}^{{\mathrm{{selected}}}}={sel}\,\mathrm{{M}}_\odot$"
            )
        head = "Best-matching precession parameters vs time delay"
        logs = [f"Using mcz={v:.8g} Msun for {lab}" for lab, v in zip(labels, selected)]
    else:
        assert td_column_ms is not None
        idxs = []
        selected = []
        for d in datasets:
            i, v = _select_axis_index(
                d["td_ms"], td_column_ms, axis_name="td", unit_suffix=" ms"
            )
            idxs.append(i)
            selected.append(v)
        kind = datasets[0]["axis_kind"]
        xlabel = datasets[0]["axis_label"]
        xs = [d["axis_values"] for d in datasets]
        omega_ys = [d["omega_best"][:, c] for d, c in zip(datasets, idxs)]
        theta_ys = [d["theta_best"][:, c] for d, c in zip(datasets, idxs)]
        rel = "=" if _is_effectively_exact(selected, td_column_ms) else r"\approx"
        details = [rf"$\Delta t_{{\mathrm{{d}}}}{rel} {td_column_ms:g}\,\mathrm{{ms}}$"]
        _extend_z(details, datasets)
        if kind == "I":
            if u := _uniq_sorted_field(datasets, "mcz_value"):
                details.append(
                    rf"$\mathcal{{M}}_{{\mathrm{{s}}}}={u}\,\mathrm{{M}}_\odot$"
                )
            head = "Best-matching precession parameters vs flux ratio"
        else:
            if u := _uniq_sorted_field(datasets, "I_value"):
                details.append(rf"$I={u}$")
            head = "Best-matching precession parameters vs source-frame chirp mass"
        if max(abs(v - td_column_ms) for v in selected) > 1e-6:
            sel = ", ".join(f"{v:g}" for v in selected)
            details.append(
                rf"$\Delta t_{{\mathrm{{d}}}}^{{\mathrm{{selected}}}}={sel}\,\mathrm{{ms}}$"
            )
        logs = [f"Using td={v:.8g} ms for {lab}" for lab, v in zip(labels, selected)]

    suptitle = head + "\n" + ", ".join(details)
    _plot_line_pair_figure(
        xs, omega_ys, theta_ys, labels, xlabel, suptitle, output_path, dpi, logs
    )


def _levels_for_field(
    datasets: List[Dict[str, np.ndarray]], key: str, n: int
) -> np.ndarray:
    lo = min(float(np.nanmin(d[key])) for d in datasets)
    hi = max(float(np.nanmax(d[key])) for d in datasets)
    return np.linspace(lo, hi, n)


def create_figure(
    paths: List[str],
    labels: List[str],
    output_path: str,
    levels_count: int,
    dpi: int,
    cmap: str,
    slice_mcz: float | None = None,
    slice_td_ms: float | None = None,
) -> None:
    _validate_inputs(paths, labels)

    datasets = [_load_best_match(p) for p in paths]
    ncols = len(datasets)
    axis_kinds = {d["axis_kind"] for d in datasets}
    if len(axis_kinds) != 1:
        raise ValueError(
            "All inputs must use the same vertical axis; do not mix mcz-td and I-td files."
        )
    axis_kind = datasets[0]["axis_kind"]
    axis_label = datasets[0]["axis_label"]

    if slice_mcz is not None:
        if axis_kind != "mcz":
            raise ValueError(
                "--slice-mcz is only supported for mcz-td best-match files"
            )
        _slice_line_figure(datasets, labels, output_path, dpi, mcz_row=slice_mcz)
        return

    if slice_td_ms is not None:
        _slice_line_figure(datasets, labels, output_path, dpi, td_column_ms=slice_td_ms)
        return

    omega_levels = _levels_for_field(datasets, "omega_best", levels_count)
    theta_levels = _levels_for_field(datasets, "theta_best", levels_count)

    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=10)

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(4.6 * ncols + 1.2, 8.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    omega_cf = None
    theta_cf = None

    for i, (d, label) in enumerate(zip(datasets, labels)):
        ax_top = axes[0, i]
        ax_bottom = axes[1, i]

        omega_cf = ax_top.contourf(
            d["td_ms"],
            d["axis_values"],
            d["omega_best"],
            levels=omega_levels,
            cmap=cmap,
        )
        theta_cf = ax_bottom.contourf(
            d["td_ms"],
            d["axis_values"],
            d["theta_best"],
            levels=theta_levels,
            cmap=cmap,
        )

        for ax in (ax_top, ax_bottom):
            for n, ls in [(1, "-"), (2, "--"), (3, ":")]:
                cs = ax.contour(
                    d["td_ms"],
                    d["axis_values"],
                    d["nlens"],
                    levels=[n],
                    colors=["black"],
                    linestyles=[ls],
                    linewidths=2.0,
                )
                ax.clabel(cs, fmt=rf"$N_\mathrm{{lensed}}={n}$", fontsize=8)

            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect(1)
            ax.tick_params(direction="in", top=True, right=True)

        ax_top.set_title(label)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(axis_label)

    details: List[str] = []
    if axis_kind == "mcz":
        if u := _uniq_sorted_field(datasets, "I_value"):
            details.append(rf"$I={u}$")
    else:
        if u := _uniq_sorted_field(datasets, "mcz_value"):
            details.append(rf"$\mathcal{{M}}_{{\mathrm{{s}}}}={u}\,\mathrm{{M}}_\odot$")
    _extend_z(details, datasets)
    ori_tags = ", ".join(sorted({d["orientation"] for d in datasets if d["orientation"]}))
    if ori_tags:
        details.append(ori_tags)

    fig.subplots_adjust(
        left=0.12 if ncols == 1 else 0.10,
        right=0.86,
        top=0.90,
        bottom=0.11,
        wspace=0.05,
        hspace=0.08,
    )

    if details:
        fig.suptitle(", ".join(details), fontsize=13)

    fig.canvas.draw()
    ylab_omega = r"$\tilde{\Omega}_{\mathrm{best}}$"
    ylab_theta = r"$\tilde{\theta}_{\mathrm{best}}$"
    for row, cf, ylab in ((0, omega_cf, ylab_omega), (1, theta_cf, ylab_theta)):
        pos = axes[row, -1].get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.016, pos.height])
        fig.colorbar(cf, cax=cax).set_label(ylab)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_PATHS,
        help="One or more best_match HDF5 paths (one panel per path)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="One panel label per input path",
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
    slice_group = parser.add_mutually_exclusive_group()
    slice_group.add_argument(
        "--slice-mcz",
        type=float,
        default=None,
        help=(
            "If set, select the nearest source-frame mcz row and plot best theta/omega "
            "versus time delay instead of the 2D contour grids"
        ),
    )
    slice_group.add_argument(
        "--slice-td-ms",
        type=float,
        default=None,
        help=(
            "If set, select the nearest time-delay column and plot best theta/omega "
            "versus the native vertical-axis variable instead of the 2D contour grids"
        ),
    )
    args = parser.parse_args()

    create_figure(
        paths=args.paths,
        labels=args.labels,
        output_path=args.output,
        levels_count=args.levels,
        dpi=args.dpi,
        cmap=args.cmap,
        slice_mcz=args.slice_mcz,
        slice_td_ms=args.slice_td_ms,
    )


if __name__ == "__main__":
    main()
