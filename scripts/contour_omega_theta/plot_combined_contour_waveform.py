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
from modules.plot_utils import (
    LBL_BRATIO_TS,
    LBL_F,
    LBL_OMEGA,
    LBL_PHASE_TS,
    LBL_THETA,
    apply_physics_paper_style,
    save_figure,
    add_colorbar_axes,
    format_colorbar_ticks,
)
from modules.waveform_plotting import (
    plot_best_match_overlay_from_contour,
    customize_2x1_axes_ratio,
)

SHARED_DATA_ROOT = "/work/10000/fairytien33/gw_shared_data"
DEFAULT_RUN_DIR = os.path.join(
    SHARED_DATA_ROOT,
    "mismatch_I0p5_z1_mcz5-45_td20-70_Taman_edgeon",
)
DEFAULT_MCZ_LIST = [5, 15, 25]

LINE_COLORS = ["black", "blue", "magenta"]
LINE_STYLES = ["-", "--", ":"]


# ============================================================================
# HDF5 Cube Readers
# ============================================================================


def _build_cube_path(run_dir: str, mcz_msun: float) -> str:
    mcz_token = str(int(mcz_msun)) if mcz_msun == int(mcz_msun) else str(mcz_msun)
    fname = (
        f"mismatch_cubes_z1_mcz{mcz_token}_I0p5"
        f"_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5"
    )
    return os.path.join(run_dir, "mismatch_cubes", fname)


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
        gamma_2d = gamma_best_grid[td_idx, :, :]

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
        extend="both",
    )

    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    min_omega = float(X[min_idx])
    min_theta = float(Y[min_idx])

    ax.plot(
        min_omega,
        min_theta,
        "x",
        color="white",
        markersize=10,
        markeredgewidth=2.5,
        label=(
            rf"$\varepsilon_{{\mathrm{{RP}}}}$ min"
            rf" $({min_omega:.2f},\,{min_theta:.2f})$"
        ),
    )
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.7,
        edgecolor="none",
        handletextpad=0.3,
    )

    if show_xlabel:
        ax.set_xlabel(LBL_OMEGA, fontsize=14)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(LBL_THETA, fontsize=14)

    mcz = contour_data["mcz_msun"]
    ax.text(
        0.03,
        0.96,
        rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {mcz:.3g}\,M_{{\odot}}$",
        transform=ax.transAxes,
        fontsize=12,
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
        rp_linestyle=LINE_STYLES[0],
        rp_label="best RP",
    )

    customize_2x1_axes_ratio([ax_amp, ax_phase])

    ax_phase.axhline(0.0, color=LINE_COLORS[0], linestyle=LINE_STYLES[0])

    for ax in (ax_amp, ax_phase):
        ax.set_xlim(f_min, float(summary["f_cut"]))

    ax_amp.set_xlabel("")
    ax_amp.tick_params(axis="x", labelbottom=False)

    if show_xlabel:
        ax_phase.set_xlabel(LBL_F, fontsize=14)
    else:
        ax_phase.tick_params(axis="x", labelbottom=False)

    ax_amp.set_ylabel(
        LBL_BRATIO_TS,
        fontsize=12,
        labelpad=4,
    )
    ax_phase.set_ylabel(
        LBL_PHASE_TS,
        fontsize=12,
        labelpad=4,
    )

    mcz = contour_data["mcz_msun"]
    omega = summary["omega_tilde"]
    theta = summary["theta_tilde"]
    gamma = summary["gamma_P"]
    epsilon = summary["epsilon"]

    box_text = (
        rf"$\tilde{{\Omega}}={omega:.2f}$, "
        rf"$\tilde{{\theta}}={theta:.2f}$, "
        rf"$\gamma_{{\mathrm{{P}}}}={gamma:.2f}$, "
        rf"$\varepsilon_{{\mathrm{{RP}}}}={epsilon:.2g}$"
    )
    ax_amp.text(
        0.5,
        1.08,
        box_text,
        transform=ax_amp.transAxes,
        fontsize=9,
        ha="center",
        va="bottom",
    )

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
) -> str:
    nrows = len(contour_datasets)

    apply_physics_paper_style(base_font=12, label_font=14, tick_font=11, legend_font=11)

    fig = plt.figure(figsize=(16, 5.5 * nrows))

    outer_gs = gridspec.GridSpec(
        nrows,
        2,
        figure=fig,
        width_ratios=[1, 1.3],
        wspace=0.30,
        hspace=0.25,
        left=0.07,
        right=0.88,
        bottom=0.06,
        top=0.94,
    )

    vmin = min(float(np.nanmin(d["epsilon_matrix"])) for d in contour_datasets)
    vmax = max(float(np.nanmax(d["epsilon_matrix"])) for d in contour_datasets)
    vmax = min(vmax, 0.5)

    contour_axes = []
    cf_last = None
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

        summary = plot_waveform_panel(
            ax_amp,
            ax_phase,
            data,
            f_min=f_min,
            npoints=npoints,
            show_xlabel=is_last,
        )
        summaries.append(summary)

    cax = add_colorbar_axes(fig, contour_axes, pad=0.015, width=0.015)
    cbar = fig.colorbar(cf_last, cax=cax)
    cbar.set_label(
        r"$\varepsilon_{\mathrm{RP}}$",
        fontsize=14,
    )
    format_colorbar_ticks(cbar, vmin, vmax, decimals=2)

    handles, labels = contour_axes[0].get_legend_handles_labels()

    wf_axes_first_row = fig.axes
    for ax in wf_axes_first_row:
        if ax not in contour_axes and ax is not cax:
            h, l = ax.get_legend_handles_labels()
            if h:
                fig.legend(
                    h,
                    l,
                    loc="upper center",
                    bbox_to_anchor=(0.72, 0.99),
                    ncol=3,
                    frameon=True,
                    fontsize=11,
                )
                break

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
        default=None,
        help="Mismatch cube HDF5 paths (one per chirp mass). "
        "If omitted, auto-resolved from --run_dir and --mcz_list.",
    )
    parser.add_argument(
        "--run_dir",
        default=DEFAULT_RUN_DIR,
        help="Pipeline run directory containing mismatch_cubes/.",
    )
    parser.add_argument(
        "--mcz_list",
        nargs="+",
        type=float,
        default=DEFAULT_MCZ_LIST,
        help="Chirp masses (Msun) to include as rows.",
    )
    parser.add_argument("--td_ms", type=float, default=30.0, help="Time delay in ms.")
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument("--npoints", type=int, default=10000)
    parser.add_argument(
        "--output_dir",
        default="figures/contour_omega_theta",
        help="Output directory.",
    )
    parser.add_argument(
        "--output_prefix",
        default="combined_contour_waveform",
        help="Output filename prefix.",
    )
    args = parser.parse_args()

    if args.input is not None:
        cube_paths = args.input
    else:
        cube_paths = [_build_cube_path(args.run_dir, mcz) for mcz in args.mcz_list]

    for p in cube_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Mismatch cube not found: {p}")

    contour_datasets = [load_contour_slice(p, args.td_ms) for p in cube_paths]

    os.makedirs(args.output_dir, exist_ok=True)
    mcz_token = "-".join(str(int(d["mcz_msun"])) for d in contour_datasets)
    output_path = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_mcz{mcz_token}_td{int(args.td_ms)}ms.pdf",
    )

    plot_combined(
        contour_datasets,
        output_path,
        f_min=args.f_min,
        npoints=args.npoints,
    )


if __name__ == "__main__":
    main()
