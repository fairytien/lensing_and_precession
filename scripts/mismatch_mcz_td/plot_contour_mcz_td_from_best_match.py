"""Create mismatch contour plots from one aggregated best-match HDF5 file.

This script requires an exact best-match file path produced by
python -m scripts.mismatch_mcz_td.aggregate_best_match.
All contour naming parameters are inferred from that file's metadata.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from modules.plot_utils import apply_physics_paper_style

apply_physics_paper_style()

from modules.filenames import (
    contour_mcz_td_filename,
    contour_run_dir,
)
from modules.bank_io import read_best_match_contour_data

# Import overlay functions from plot_cycles_and_extrema_mcz.py
from scripts.utils.plot_cycles_and_extrema_mcz import plot_cycle_lines, plot_mcz_extrema

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


VARIABLE_MAPPING = {
    "epsilon": {
        "dataset": "epsilon_min",
        "label": r"$\min_{\~\Omega, \~\theta, \gamma_P}$ $\epsilon(\tilde{h}_L, \tilde{h}_P)$",
        "suffix": "epsilon_min",
    },
    "omega": {
        "dataset": "omega_best",
        "label": r"$\tilde{\Omega}_{\rm best}$",
        "suffix": "omega_best",
    },
    "theta": {
        "dataset": "theta_best",
        "label": r"$\tilde{\theta}_{\rm best}$",
        "suffix": "theta_best",
    },
}


def main(
    input_path: str,
    output_dir: str,
    variable: str = "epsilon",
    overlay_cycles: bool = False,
    overlay_peaks: bool = False,
    overlay_troughs: bool = False,
    show_legend: bool = False,
    eta: float = 0.25,
    f_min: float = 20.0,
):
    """Plot mismatch contour from aggregated best-match file.

    Args:
        input_path: Exact path to an aggregated best-match HDF5 file.
        output_dir: Directory where the figure will be saved.
        variable: Variable to plot ("epsilon", "omega", or "theta").
        overlay_cycles: If True, overlay 1/2/3 lensing cycle lines.
        overlay_peaks: If True, overlay mcz peak points.
        overlay_troughs: If True, overlay mcz trough points.
        show_legend: If True, show legend for overlays.
        eta: Symmetric mass ratio (default 0.25).
        f_min: Minimum frequency in Hz (default 20.0).
    """
    if variable not in VARIABLE_MAPPING:
        raise ValueError(
            f"Invalid variable '{variable}'. Must be one of: {list(VARIABLE_MAPPING.keys())}"
        )

    var_info = VARIABLE_MAPPING[variable]
    best_match = read_best_match_contour_data(input_path, var_info["dataset"])

    if best_match["missing_mcz_count"] > 0:
        logging.warning(
            "Best-match file reports %d missing mcz rows; contour will include NaN gaps.",
            best_match["missing_mcz_count"],
        )

    logging.info(f"Using best-match file: {input_path}")

    output_dir = contour_run_dir(
        output_dir,
        I=best_match["I"],
        mcz_min=best_match["mcz_min"],
        mcz_max=best_match["mcz_max"],
        td_min_ms=best_match["td_min_ms"],
        td_max_ms=best_match["td_max_ms"],
        z=best_match["z"],
        orientation_tag=best_match["orientation_tag"],
    )
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Resolved figure output directory: {output_dir}")

    mcz_msun_arr = best_match["mcz"]
    td_arr = best_match["td"]
    Zmap = best_match["values"]

    # Convert td from seconds to ms for plotting
    td_arr_ms = td_arr * 1e3

    # Get actual data range for overlays
    mcz_msun_data_min, mcz_msun_data_max = mcz_msun_arr.min(), mcz_msun_arr.max()

    # Create contour plot
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_msun_arr)
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(TD, MCZ, Zmap, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    cbar.set_label(var_info["label"])
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # Overlay lensing cycle lines if requested
    if overlay_cycles:
        plot_cycle_lines(td_arr, td_arr_ms, eta=eta, f_min=f_min)

    # Overlay mcz extrema points if requested
    if overlay_troughs or overlay_peaks:
        plot_mcz_extrema(
            td_arr,
            mcz_msun_data_min,
            mcz_msun_data_max,
            eta=eta,
            plot_troughs=overlay_troughs,
            plot_peaks=overlay_peaks,
        )

    # Show legend if requested and there are labeled artists
    if show_legend:
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            plt.legend(loc="best")

    plt.tight_layout()

    # Generate filename with variable suffix and overlay suffixes
    base_path = contour_mcz_td_filename(
        output_dir,
        I=best_match["I"],
        mcz_min=best_match["mcz_min"],
        mcz_max=best_match["mcz_max"],
        mcz_pts=best_match["mcz_pts"],
        td_min_ms=best_match["td_min_ms"],
        td_max_ms=best_match["td_max_ms"],
        td_pts=best_match["td_pts"],
        orientation_tag=best_match["orientation_tag"],
        z=best_match["z"],
        ext="pdf",
    )

    # Build list of suffixes
    suffixes = []

    # Add variable suffix for non-epsilon variables
    if variable != "epsilon":
        suffixes.append(var_info["suffix"])

    # Add overlay suffix if any overlays are enabled
    if overlay_cycles or overlay_peaks or overlay_troughs:
        suffixes.append("overlayed")

    # Apply suffixes to filename
    path_without_ext, ext = os.path.splitext(base_path)
    if suffixes:
        fig_path = f"{path_without_ext}_{'_'.join(suffixes)}{ext}"
    else:
        fig_path = base_path

    plt.savefig(fig_path, dpi=200)
    logging.info(f"Figure saved as {fig_path}")
    plt.close()


if __name__ == "__main__":
    # Get project root directory (used for default paths)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    p = argparse.ArgumentParser(
        description=(
            "Plot mismatch contour from one aggregated best-match HDF5 file. "
            "Provide an exact --input_path from Stage 2 aggregation output."
        )
    )
    p.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Exact path to aggregated best-match HDF5 file under best_match/.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "figures", "mismatch"),
        help=(
            "Base directory where the figure will be saved. "
            "A canonical run-tagged subdirectory is inferred from file metadata."
        ),
    )
    p.add_argument(
        "--variable",
        type=str,
        default="epsilon",
        choices=["epsilon", "omega", "theta"],
        help="Variable to plot: 'epsilon' (mismatch), 'omega' (best-match Omega), or 'theta' (best-match theta).",
    )
    p.add_argument(
        "--overlay-cycles",
        action="store_true",
        help="Overlay 1/2/3 lensing cycle lines on the contour plot.",
    )
    p.add_argument(
        "--overlay-peaks",
        action="store_true",
        help="Overlay mcz peak points on the contour plot.",
    )
    p.add_argument(
        "--overlay-troughs",
        action="store_true",
        help="Overlay mcz trough points on the contour plot.",
    )
    p.add_argument(
        "--show-legend",
        action="store_true",
        help="Show legend for any plotted overlays (cycles, peaks, troughs).",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=0.25,
        help="Symmetric mass ratio for overlay calculations (default: 0.25).",
    )
    p.add_argument(
        "--f_min",
        type=float,
        default=20.0,
        help="Minimum frequency in Hz for cycle calculations (default: 20.0).",
    )

    args = p.parse_args()

    main(
        input_path=args.input_path,
        output_dir=args.output_dir,
        variable=args.variable,
        overlay_cycles=args.overlay_cycles,
        overlay_peaks=args.overlay_peaks,
        overlay_troughs=args.overlay_troughs,
        show_legend=args.show_legend,
        eta=args.eta,
        f_min=args.f_min,
    )
