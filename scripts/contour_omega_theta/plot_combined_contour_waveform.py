"""Combined 3-row × 2-column contour + waveform figure.

Left column: mismatch contour ε_RP(Ω̃, θ̃) for each chirp mass.
Right column: best-match waveform overlay (amplitude top, phase bottom).

Reads pre-computed mismatch cube HDF5 files from the mcz_td pipeline.
"""

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.bank_io import extract_prefixed_params
from modules.cli_utils import add_redshift_arg
from modules.filenames import (
    find_mismatch_mcz_cube_files,
    _canonical_token,
    _canonical_z_token,
)
from modules.plot_utils import (
    LBL_BRATIO_TS,
    LBL_EPS_LP,
    LBL_F,
    LBL_OMEGA,
    LBL_PHASE_TS,
    LBL_THETA,
    apply_physics_paper_style,
    save_figure,
    set_square_axes,
    add_colorbar_axes,
    format_colorbar_ticks,
)
from modules.waveform_plotting import plot_best_match_overlay_from_contour

SHARED_DATA_ROOT = "/work/10000/fairytien33/gw_shared_data"
DEFAULT_RUN_DIR = os.path.join(
    SHARED_DATA_ROOT,
    "mismatch_I0p5_z1_mcz5-45_td20-70_Taman_edgeon",
)
DEFAULT_MCZ_LIST = [5, 15, 25]

LINE_COLORS = ["black", "blue", "magenta"]


# ============================================================================
# HDF5 Cube Readers
# ============================================================================


def _resolve_cube_paths(
    run_dir: str, mcz_list: list, orientation_tag: str, z: float
) -> list:
    """Discover cube paths under run_dir for each requested mcz."""
    paths = []
    for mcz in mcz_list:
        found = find_mismatch_mcz_cube_files(
            run_dir,
            td_min_ms=None,
            td_max_ms=None,
            orientation_tag=orientation_tag,
            z=z,
            mcz_msun=mcz,
        )
        if not found:
            raise FileNotFoundError(
                f"No mismatch cube found for mcz={mcz} in {run_dir}"
            )
        paths.append(found[0])
    return paths


def load_contour_slice(cube_path: str, td_ms: float) -> dict:
    """Load a 2D contour slice from a mismatch cube at a fixed td.

    Returns a dict compatible with plot_best_match_overlay_from_contour().
    """
    with h5py.File(cube_path, "r") as h5:
        td_arr = np.asarray(h5["td"], dtype=np.float64)
        omega_arr = np.asarray(h5["omega"], dtype=np.float64)
        theta_arr = np.asarray(h5["theta"], dtype=np.float64)

        td_s = td_ms / 1e3
        td_idx = int(np.argmin(np.abs(td_arr - td_s)))
        actual_td_ms = td_arr[td_idx] * 1e3

        epsilon_min_grid = np.asarray(h5["epsilon_min_grid"], dtype=np.float64)
        gamma_best_grid = np.asarray(h5["gamma_best_grid"], dtype=np.float64)

        epsilon_2d = epsilon_min_grid[td_idx, :, :]
        gamma_2d = gamma_best_grid[td_idx, :, :]  # γ_P at the best-match template

        omega_matrix, theta_matrix = np.meshgrid(omega_arr, theta_arr)

        mcz_msun = float(np.asarray(h5["mcz"]).flat[0])

        source_params = extract_prefixed_params(h5.attrs, "source_param_")
        template_params = extract_prefixed_params(h5.attrs, "template_param_")

        I_val = float(h5.attrs.get("I", np.nan))

    return {
        "omega_matrix": omega_matrix,
        "theta_matrix": theta_matrix,
        "epsilon_matrix": epsilon_2d,
        "gammaP_min_matrix": gamma_2d,
        "source_params": source_params,
        "template_params": template_params,
        "mcz_msun": mcz_msun,
        "td_ms": actual_td_ms,
        "I": I_val,
    }


# ============================================================================
# Contour Panel
# ============================================================================


def plot_contour_panel(
    ax,
    contour_data: dict,
    *,
    vmin: float,
    vmax: float,
    saturated: bool = True,
    levels: int = 100,
    show_xlabel: bool = False,
    show_ylabel: bool = True,
) -> plt.contour:
    X = contour_data["omega_matrix"]
    Y = contour_data["theta_matrix"]
    Z = contour_data["epsilon_matrix"]

    cf = ax.contourf(
        X,
        Y,
        Z,
        levels=np.linspace(vmin, vmax, levels),
        cmap="jet",
        extend="max" if saturated else "neither",
    )

    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    min_omega = float(X[min_idx])
    min_theta = float(Y[min_idx])
    min_epsilon = float(Z[min_idx])
    gamma_P = float(contour_data["gammaP_min_matrix"][min_idx])
    epsilon_NP = float(Z[0, 0])
    ratio = epsilon_NP / min_epsilon if min_epsilon > 0 else float("nan")

    ax.plot(
        min_omega,
        min_theta,
        "*",
        color="white",
        markersize=16,
        markeredgewidth=0.8,
        markeredgecolor="0.3",
        label=(
            rf"$\epsilon_{{\mathrm{{RP}}}}={min_epsilon:.3g}$, "
            rf"$\epsilon_{{\mathrm{{NP}}}}/\epsilon_{{\mathrm{{RP}}}}={ratio:.2f}$"
            "\n"
            rf"$\tilde{{\theta}}={min_theta:.2f}$, "
            rf"$\tilde{{\Omega}}={min_omega:.2f}$, "
            rf"$\gamma_{{\mathrm{{P}}}}={gamma_P:.2f}$"
        ),
    )
    ax.legend(
        loc="upper right",
        framealpha=0.7,
        edgecolor="none",
        handletextpad=0.3,
    )

    if show_xlabel:
        ax.set_xlabel(LBL_OMEGA)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(LBL_THETA)

    set_square_axes(ax)

    mcz = contour_data["mcz_msun"]
    ax.text(
        0.03,
        0.96,
        rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {mcz:.3g}\,M_{{\odot}}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
    )

    return cf


# ============================================================================
# Waveform Panel
# ============================================================================


def plot_waveform_panel(
    ax_amp,
    ax_phase,
    contour_data: dict,
    *,
    f_min: float = 20.0,
    npoints: int = 10000,
    show_xlabel: bool = False,
    ylabel_right: bool = False,
) -> dict:
    summary = plot_best_match_overlay_from_contour(
        contour_data,
        [ax_amp, ax_phase],
        f_min=f_min,
        npoints=npoints,
        baseline_color=LINE_COLORS[1],
        lensed_color=LINE_COLORS[0],
        np_label="NP",
        rp_color=LINE_COLORS[2],
        rp_linestyle="-",
        rp_label="best RP",
    )

    ax_phase.axhline(0.0, color=LINE_COLORS[0], linestyle="-")

    for ax in (ax_amp, ax_phase):
        ax.set_xlim(f_min, float(summary["f_cut"]))

    ax_amp.tick_params(axis="x", labelbottom=False)

    if show_xlabel:
        ax_phase.set_xlabel(LBL_F)
    else:
        ax_phase.tick_params(axis="x", labelbottom=False)

    ax_amp.set_ylabel(LBL_BRATIO_TS, labelpad=4)
    ax_phase.set_ylabel(LBL_PHASE_TS, labelpad=4)
    if ylabel_right:
        for ax in (ax_amp, ax_phase):
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")

    return summary


# ============================================================================
# Combined Figure
# ============================================================================


def plot_combined(
    contour_datasets: list,
    output_path: str,
    *,
    f_min: float = 20.0,
    npoints: int = 10000,
    colorbar_side: str = "right",
    vmax_cap: float | None = 0.5,
) -> str:
    nrows = len(contour_datasets)

    apply_physics_paper_style(base_font=16, label_font=20, tick_font=16, legend_font=14)

    left_margin = 0.22 if colorbar_side == "left" else 0.07
    col_wspace = 0.18 if colorbar_side == "left" else 0.29
    fig_width = 16.0
    gs_right, gs_top, gs_bottom = 0.88, 0.94, 0.06
    gs_hspace = 0.08
    width_ratios = [1, 1.3]
    # Size figure height so each outer row height = contour column width,
    # which makes set_box_aspect(1) fill the row and match the waveform pair height.
    # Formula: row_height = (gs_top-gs_bottom)*fig_height / (nrows + (nrows-1)*gs_hspace)
    ncols = len(width_ratios)
    _contour_w_in = (
        (gs_right - left_margin)
        * fig_width
        * width_ratios[0]
        / (sum(width_ratios) * (1.0 + col_wspace * (ncols - 1) / ncols))
    )
    fig_height = (
        _contour_w_in * (nrows + (nrows - 1) * gs_hspace) / (gs_top - gs_bottom)
    )
    fig = plt.figure(figsize=(fig_width, fig_height))

    outer_gs = gridspec.GridSpec(
        nrows,
        2,
        figure=fig,
        width_ratios=width_ratios,
        wspace=col_wspace,
        hspace=gs_hspace,
        left=left_margin,
        right=gs_right,
        bottom=gs_bottom,
        top=gs_top,
    )

    vmin = min(float(np.nanmin(d["epsilon_matrix"])) for d in contour_datasets)
    vmax = max(float(np.nanmax(d["epsilon_matrix"])) for d in contour_datasets)
    saturated = vmax_cap is not None and vmax_cap < vmax
    if saturated:
        vmax = vmax_cap

    contour_axes = []
    cf_last = None
    first_wf_ax_amp = None
    all_wf_axes = []
    summaries = []

    for row, data in enumerate(contour_datasets):
        is_last = row == nrows - 1

        ax_contour = fig.add_subplot(outer_gs[row, 0])
        contour_axes.append(ax_contour)
        cf_last = plot_contour_panel(
            ax_contour,
            data,
            vmin=vmin,
            vmax=vmax,
            saturated=saturated,
            show_xlabel=is_last,
        )

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_gs[row, 1],
            height_ratios=[1, 1],
            hspace=0.08,
        )
        ax_amp = fig.add_subplot(inner_gs[0])
        ax_phase = fig.add_subplot(inner_gs[1], sharex=ax_amp)
        if first_wf_ax_amp is None:
            first_wf_ax_amp = ax_amp
        all_wf_axes.extend([ax_amp, ax_phase])

        summary = plot_waveform_panel(
            ax_amp,
            ax_phase,
            data,
            f_min=f_min,
            npoints=npoints,
            show_xlabel=is_last,
            ylabel_right=(colorbar_side == "right"),
        )
        summaries.append(summary)

    fig.align_ylabels(all_wf_axes)

    if colorbar_side == "left":
        fig.canvas.draw()
        positions = [ax.get_position() for ax in contour_axes]
        x0 = min(pos.x0 for pos in positions)
        y0 = min(pos.y0 for pos in positions)
        y1 = max(pos.y1 for pos in positions)
        fig.set_layout_engine("none")
        cax = fig.add_axes([x0 - 0.060, y0, 0.015, y1 - y0])
    else:
        cax = add_colorbar_axes(fig, contour_axes, pad=0.015, width=0.015)
    cbar = fig.colorbar(cf_last, cax=cax)
    if colorbar_side == "left":
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
    cbar.set_label(LBL_EPS_LP)
    format_colorbar_ticks(cbar, vmin, vmax, decimals=2)

    h, l = first_wf_ax_amp.get_legend_handles_labels()
    if h:
        first_wf_ax_amp.legend(
            h,
            l,
            loc="upper left" if colorbar_side == "right" else "upper right",
            ncol=1,
            frameon=True,
            framealpha=0.7,
            edgecolor="none",
        )

    save_figure(fig, output_path)

    for s in summaries:
        print(
            f"mcz={s['mcz_msun']:.3g}: "
            f"omega={s['omega_tilde']:.4f}, "
            f"theta={s['theta_tilde']:.4f}, "
            f"gamma_P={s['gamma_P']:.4f}, "
            f"epsilon_RP={s['epsilon']:.6f}"
        )

    return output_path


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combined 3-row × 2-column figure: mismatch contours (left) and "
            "best-match waveform overlays (right) for multiple chirp masses."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[DEFAULT_RUN_DIR],
        help=(
            "HDF5 cube paths (one per chirp mass), or a single run directory "
            "from which cubes are auto-discovered using --mcz_list, "
            "--orientation_tag, and --z."
        ),
    )
    parser.add_argument(
        "--mcz_list",
        nargs="+",
        type=float,
        default=DEFAULT_MCZ_LIST,
        help="Chirp masses (Msun) to plot as rows (directory mode only).",
    )
    parser.add_argument(
        "--orientation_tag",
        default="Taman_edgeon",
        help="Orientation tag for cube discovery (directory mode only).",
    )
    add_redshift_arg(parser, default_z=1.0)
    parser.add_argument(
        "--colorbar-side",
        choices=["left", "right"],
        default="right",
        dest="colorbar_side",
        help="Side of the contour column to place the shared colorbar (default: right).",
    )
    parser.add_argument("--td_ms", type=float, default=30.0, help="Time delay in ms.")
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.5,
        dest="vmax_cap",
        help=(
            "Saturate the colorbar at this ε_RP value (default: 0.5). "
            "Pass a larger value or the actual data max to disable saturation."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PDF file path. If omitted, saves to "
            "figures/contour_omega_theta/combined_contour_waveform_I<...>_td<...>_z<...>_mcz<...>_<orientation_tag>.pdf"
        ),
    )
    args = parser.parse_args()

    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        cube_paths = _resolve_cube_paths(
            args.input[0], args.mcz_list, args.orientation_tag, args.z
        )
    else:
        cube_paths = args.input

    for p in cube_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Mismatch cube not found: {p}")

    contour_datasets = [load_contour_slice(p, args.td_ms) for p in cube_paths]

    if args.output is not None:
        output_path = args.output
    else:
        I_val = contour_datasets[0]["I"]
        I_token = _canonical_token(I_val)
        z_token = _canonical_z_token(args.z)
        mcz_token = "-".join(str(int(d["mcz_msun"])) for d in contour_datasets)
        output_path = os.path.join(
            "figures/contour_omega_theta",
            f"combined_contour_waveform_I{I_token}_td{int(args.td_ms)}_z{z_token}_mcz{mcz_token}_{args.orientation_tag}.pdf",
        )

    plot_combined(
        contour_datasets,
        output_path,
        colorbar_side=args.colorbar_side,
        vmax_cap=args.vmax_cap,
    )


if __name__ == "__main__":
    main()
