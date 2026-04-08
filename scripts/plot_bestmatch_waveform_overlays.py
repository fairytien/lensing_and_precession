import argparse
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.plot_utils_v3 import customize_2x1_axes_ratio, set_default_plot_style
from modules.waveform_plotting import plot_best_match_overlay_from_contour


DEFAULT_INPUTS = [
    "data/indiv_contours/v3_indiv_contour_mcz10_td30ms_I0.5_thetaS0.785_phiS0.0_thetaJ1.571_phiJ1.571_z1_2026-04-01_10-01-50.pkl",
    "data/indiv_contours/v3_indiv_contour_mcz20_td30ms_I0.5_thetaS0.785_phiS0.0_thetaJ1.571_phiJ1.571_z1_2026-04-01_10-04-00.pkl",
]

NOTEBOOK_LINE_STYLES = ["-", "--", ":"]
NOTEBOOK_LINE_COLORS = ["magenta", "blue", "blue"]


def _apply_notebook_font_style() -> None:
    """Match the typography used by notebook waveform figures."""
    set_default_plot_style()
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"


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


def _format_token(value: float, decimals: int = 1) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    if not np.isfinite(value):
        return "unknown"
    token = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    token = token.replace("-", "m").replace(".", "p")
    return token if token else "0"


def _row_parameter_box_text(summary: dict) -> str:
    lines = [
        "System 2",
        rf"$\mathcal{{M}}_{{\rm s}} = {_format_stat(summary['mcz_msun'])}\,M_\odot$",
        rf"$\Delta t_d = {_format_stat(summary['td_ms'])}\,\mathrm{{ms}}$",
        rf"$I = {_format_stat(summary['I'])}$",
        rf"$\tilde{{\Omega}} = {_format_stat(summary['omega_tilde'])}$",
        rf"$\tilde{{\theta}} = {_format_stat(summary['theta_tilde'])}$",
        rf"$\gamma_P = {_format_stat(summary['gamma_P'])}$",
        rf"$\epsilon_\min = {_format_stat(summary['epsilon'])}$",
    ]
    return "\n".join(lines)


def _place_right_row_boxes(
    fig,
    axes: np.ndarray,
    summaries: list[dict],
    *,
    box_pad: float = 0.012,
    fontsize: int = 14,
) -> None:
    """Place one metadata box per row to the right of the right-hand panel."""
    fig.canvas.draw()
    nrows = axes.shape[0]
    for row in range(nrows):
        right_pos = axes[row, 1].get_position()
        x = min(right_pos.x1 + box_pad, 0.985)
        y_center = 0.5 * (right_pos.y0 + right_pos.y1)
        fig.text(
            x,
            y_center,
            _row_parameter_box_text(summaries[row]),
            ha="left",
            va="center",
            fontsize=fontsize,
            fontfamily="DejaVu Sans",
            linespacing=1.25,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85,
                "pad": 4.0,
            },
        )


def plot_combined(
    input_paths: list[str],
    output_dir: str,
    f_min: float,
    npoints: int,
    dpi: int,
    output_prefix: str,
) -> str:
    if len(input_paths) < 1:
        raise ValueError("At least one input pickle is required.")

    _apply_notebook_font_style()

    for input_path in input_paths:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input pickle not found: {input_path}")

    datasets = [load_pickle(path) for path in input_paths]
    nrows = len(datasets)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(18, 4.3 * nrows),
        sharex=False,
    )
    if nrows == 1:
        axes = np.atleast_2d(axes)

    fig.subplots_adjust(
        left=0.08, right=0.80, bottom=0.10, top=0.86, wspace=0.22, hspace=0.42
    )

    summaries: list[dict] = []
    for row, data in enumerate(datasets):
        row_axes = axes[row]
        summary = plot_best_match_overlay_from_contour(
            data,
            row_axes,
            f_min=f_min,
            npoints=npoints,
            baseline_color="#000000",
            lensed_color=NOTEBOOK_LINE_COLORS[0],
            rp_color=NOTEBOOK_LINE_COLORS[1],
            rp_linestyle=NOTEBOOK_LINE_STYLES[1],
            rp_label="best RP",
        )
        summaries.append(summary)

        customize_2x1_axes_ratio(row_axes)

        for ax in row_axes:
            ax.set_xlim(left=f_min)

        row_axes[0].set_xlabel("")
        row_axes[1].set_xlabel("")
        row_axes[0].set_ylabel("")
        row_axes[1].set_ylabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for row in range(nrows):
        legend = axes[row, 0].get_legend()
        if legend is not None:
            legend.remove()

    fig.canvas.draw()
    left_pos = axes[0, 0].get_position()
    right_pos = axes[0, 1].get_position()
    legend_x = 0.5 * (left_pos.x0 + right_pos.x1)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(legend_x, 0.95),
        ncol=3,
        frameon=True,
        fontsize=20,
    )

    _place_right_row_boxes(fig, axes, summaries)

    fig.text(
        0.016,
        0.5,
        r"$\left(B/B_{\rm unlensed}\right) - 1$",
        va="center",
        rotation="vertical",
        fontsize=24,
    )
    fig.text(
        0.44,
        0.5,
        r"$\Phi_{\rm L} - \Phi_{\rm RP}$ (rad)",
        va="center",
        rotation="vertical",
        fontsize=24,
    )
    fig.text(0.50, 0.04, "f (Hz)", ha="center", va="center", fontsize=24)

    mcz_labels = [
        infer_mcz_label(data, os.path.basename(path))
        for data, path in zip(datasets, input_paths)
    ]
    mcz_token = "-".join(mcz_labels)
    out_path = os.path.join(
        output_dir, f"{output_prefix}_mcz{mcz_token}_combined_fracamp.pdf"
    )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, dpi=dpi)
    plt.close(fig)

    print(f"Saved: {out_path}")
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
        default="figures/waveforms/paper",
        help="Directory for output PDF files.",
    )
    parser.add_argument(
        "--output_prefix",
        default="bestmatch_waveform_overlays",
        help="Prefix for output filename stems.",
    )
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--npoints", type=int, default=10000)
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_combined(
        args.input,
        args.output_dir,
        args.f_min,
        args.npoints,
        args.dpi,
        args.output_prefix,
    )


if __name__ == "__main__":
    main()
