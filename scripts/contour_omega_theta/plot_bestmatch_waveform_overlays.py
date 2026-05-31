import argparse
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.waveform_plotting import customize_2x1_axes_ratio
from modules.plot_utils import (
    apply_physics_paper_style,
    save_figure,
    LBL_BRATIO_TS,
    LBL_F,
    LBL_PHASE_TS,
)
from modules.waveform_plotting import plot_best_match_overlay_from_contour

DEFAULT_INPUTS = [
    "data/contour_omega_theta/v3_indiv_contour_mcz10_td30ms_I0.5_thetaS0.785_phiS0.0_thetaJ1.571_phiJ1.571_z1_2026-04-01_10-01-50.pkl",
    "data/contour_omega_theta/v3_indiv_contour_mcz20_td30ms_I0.5_thetaS0.785_phiS0.0_thetaJ1.571_phiJ1.571_z1_2026-04-01_10-04-00.pkl",
]

LINE_STYLES = ["-", "--", ":"]
LINE_COLORS = ["black", "blue", "magenta"]


def load_pickle(path: str) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def infer_mcz_label(data: dict, fallback_name: str) -> str:
    mcz_msun = data.get("mcz_msun", None)
    if mcz_msun is not None and np.isfinite(mcz_msun):
        return f"{float(mcz_msun):.3g}".replace(".", "p")
    match = re.search(r"mcz([0-9]+(?:\.[0-9]+)?)", fallback_name)
    if match:
        return match.group(1).replace(".", "p")
    return "unknown"


def _format_stat(value: float, fmt: str = ".3g") -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    return format(value, fmt)


def _row_parameter_box_text(summary: dict) -> str:
    return (
        rf"$\mathcal{{M}}_{{\mathrm{{s}}}}={_format_stat(summary['mcz_msun'])}\,\mathrm{{M}}_\odot$, "
        rf"$\tilde{{\Omega}}={_format_stat(summary['omega_tilde'])}$, "
        rf"$\tilde{{\theta}}={_format_stat(summary['theta_tilde'])}$, "
        rf"$\gamma_{{\mathrm{{P}}}}={_format_stat(summary['gamma_P'])}$, "
        rf"$\epsilon_{{\mathrm{{RP}}}}={_format_stat(summary['epsilon'], '.2g')}$"
    )


def _place_column_header_boxes(
    fig,
    axes: np.ndarray,
    summaries: list[dict],
    *,
    y_pad: float = 0.004,
    fontsize: int = 20,
) -> None:
    """Place one unboxed metadata header above each system column."""
    fig.canvas.draw()
    ncols = axes.shape[1]
    for col in range(ncols):
        top_pos = axes[0, col].get_position()
        x_center = 0.5 * (top_pos.x0 + top_pos.x1)
        y = top_pos.y1 + y_pad
        fig.text(
            x_center,
            y,
            _row_parameter_box_text(summaries[col]),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_combined(
    input_paths: list[str],
    output_dir: str,
    f_min: float,
    npoints: int,
    output_prefix: str,
) -> str:
    if not input_paths:
        raise ValueError("At least one input pickle is required.")

    for input_path in input_paths:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input pickle not found: {input_path}")

    datasets = [load_pickle(path) for path in input_paths]
    ncols = len(datasets)

    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=11)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(8.6 * ncols, 8.4),
        sharex="col",
        sharey="row",
    )
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.095, top=0.865, wspace=0.07, hspace=0.10
    )

    summaries: list[dict] = []
    for col, data in enumerate(datasets):
        col_axes = axes[:, col]
        summary = plot_best_match_overlay_from_contour(
            data,
            col_axes,
            f_min=f_min,
            npoints=npoints,
            baseline_color=LINE_COLORS[1],
            lensed_color=LINE_COLORS[0],
            np_label="NP",
            rp_color=LINE_COLORS[2],
            rp_linestyle=LINE_STYLES[0],
            rp_label="best RP",
        )
        summaries.append(summary)

        customize_2x1_axes_ratio(col_axes)

        col_axes[1].axhline(0.0, color=LINE_COLORS[0], linestyle=LINE_STYLES[0])

        for ax in col_axes:
            ax.set_xlim(f_min, float(summary["f_cut"]))

        col_axes[0].set_xlabel("")
        col_axes[0].set_ylabel("")
        col_axes[1].set_ylabel("")
        col_axes[0].tick_params(axis="x", labelbottom=False)

        if col > 0:
            for ax in col_axes:
                ax.tick_params(axis="y", labelleft=False)

    for ax in axes[1, :]:
        ax.set_xlabel(LBL_F, fontsize=24, labelpad=2)

    axes[0, 0].set_ylabel(
        LBL_BRATIO_TS,
        fontsize=24,
        labelpad=4,
    )
    axes[1, 0].set_ylabel(
        LBL_PHASE_TS,
        fontsize=24,
        labelpad=4,
    )
    axes[0, 0].yaxis.set_label_coords(-0.105, 0.5)
    axes[1, 0].yaxis.set_label_coords(-0.105, 0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes[0, :]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
        fontsize=20,
    )

    _place_column_header_boxes(fig, axes, summaries)

    mcz_labels = [
        infer_mcz_label(data, os.path.basename(path))
        for data, path in zip(datasets, input_paths)
    ]
    mcz_token = "-".join(mcz_labels)
    out_path = os.path.join(
        output_dir, f"{output_prefix}_mcz{mcz_token}_combined_fracamp.pdf"
    )
    save_figure(fig, out_path)

    for summary in summaries:
        print(
            "Best match:",
            f"omega_tilde={summary['omega_tilde']:.6g},",
            f"theta_tilde={summary['theta_tilde']:.6g},",
            f"gamma_P={summary['gamma_P']:.6g},",
            f"epsilon={summary['epsilon']:.6g}",
        )

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create one combined best-match RP waveform figure for multiple contour pickles, "
            "with shared axis labels, shared legend, and fractional strain-amplitude panels."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="One or more contour pickle paths.",
    )
    parser.add_argument(
        "--output_dir",
        default="figures/waveforms",
        help="Directory for output PDF files.",
    )
    parser.add_argument(
        "--output_prefix",
        default="sys2_bestmatch_waveform_overlays",
        help="Prefix for output filename stems.",
    )
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--npoints", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_combined(
        args.input,
        args.output_dir,
        args.f_min,
        args.npoints,
        args.output_prefix,
    )


if __name__ == "__main__":
    main()
