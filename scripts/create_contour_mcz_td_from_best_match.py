"""Create mismatch contour plots from aggregated best-match HDF5 file.

This script reads the best-match file (produced by scripts/aggregate_best_match.py)
and generates a contour plot of the minimal mismatch across (td, mcz).

Pipeline:
1. scripts/compute_mismatch_cubes.py - compute per-mcz mismatch cubes
2. scripts/aggregate_best_match.py - consolidate cubes into best-match file
3. scripts/create_contour_mcz_td_from_best_match.py (this script) - plot contour
"""

import os, argparse, sys, glob

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.filenames import contour_mcz_td_filename, _format_min_precision

# Import overlay functions from plot_cycles_and_extrema_mcz.py
# Add scripts directory to path to allow importing from other scripts
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from plot_cycles_and_extrema_mcz import plot_cycle_lines, plot_mcz_extrema

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
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
        results_dir: Directory containing the best-match HDF5 file.
        td_min_ms: Minimum time delay in ms (for filename matching).
        td_max_ms: Maximum time delay in ms (for filename matching).
        mcz_min: Minimum chirp mass in Msun (for filename matching).
        mcz_max: Maximum chirp mass in Msun (for filename matching).
        orientation_tag: Orientation tag (for filename matching).
        output_dir: Directory where the figure will be saved.
        variable: Variable to plot ("epsilon", "omega", or "theta").
        overlay_cycles: If True, overlay 1/2/3 lensing cycle lines.
        overlay_peaks: If True, overlay mcz peak points.
        overlay_troughs: If True, overlay mcz trough points.
        show_legend: If True, show legend for overlays.
        eta: Symmetric mass ratio (default 0.25).
        f_min: Minimum frequency in Hz (default 20.0).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load best-match file (support optional resolution suffix in filename)
    # Pattern now includes I: best_match_I*_mcz*_td*_*_tag.h5
    pattern = os.path.join(
        results_dir,
        "best_match",
        f"best_match_I*_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms*_{orientation_tag}.h5",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            "No best-match file found. Expected pattern: " + pattern
        )
    if len(matches) > 1:
        # Choose the most recent if multiple files match
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    summary_path = matches[0]

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"Best-match file not found: {summary_path}\n"
            f"Please run scripts/aggregate_best_match.py first to create this file."
        )

    logging.info(f"Loading best-match data from: {summary_path}")

    # Define variable names and labels
    variable_mapping = {
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

    if variable not in variable_mapping:
        raise ValueError(
            f"Invalid variable '{variable}'. Must be one of: {list(variable_mapping.keys())}"
        )

    var_info = variable_mapping[variable]

    # Read all data and extract I from a single file open
    I_value = None
    with h5py.File(summary_path, "r") as h5:
        # Extract I from attributes
        if "I" in h5.attrs:
            I_value = float(h5.attrs["I"])

        # Validate that required datasets exist
        required_datasets = ["mcz", "td", var_info["dataset"]]
        missing = [ds for ds in required_datasets if ds not in h5]
        if missing:
            raise KeyError(
                f"Missing datasets in {summary_path}: {missing}. "
                f"Available datasets: {list(h5.keys())}"
            )

        mcz_arr = np.array(h5["mcz"])
        td_arr = np.array(h5["td"])
        Zmap = np.array(h5[var_info["dataset"]])

    if I_value is None:
        raise ValueError("Could not infer I value from best-match file")

    # Convert td from seconds to ms for plotting
    td_arr_ms = td_arr * 1e3

    # Get actual data range for overlays
    mcz_data_min, mcz_data_max = mcz_arr.min(), mcz_arr.max()

    # Create contour plot
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
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
            mcz_data_min,
            mcz_data_max,
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
        I=I_value,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        orientation_tag=orientation_tag,
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
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    p = argparse.ArgumentParser(
        description=(
            "Plot mismatch contour from aggregated best-match HDF5 file. "
            "Run scripts/aggregate_best_match.py first to create the best-match file."
        )
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(project_root, "data", "contours_td_mcz"),
        help="Directory containing the best-match HDF5 file.",
    )
    p.add_argument(
        "--td_min_ms",
        type=float,
        required=True,
        help="Minimum time delay in ms (for filename matching).",
    )
    p.add_argument(
        "--td_max_ms",
        type=float,
        required=True,
        help="Maximum time delay in ms (for filename matching).",
    )
    p.add_argument(
        "--mcz_min",
        type=float,
        required=True,
        help="Minimum chirp mass in Msun (for filename matching).",
    )
    p.add_argument(
        "--mcz_max",
        type=float,
        required=True,
        help="Maximum chirp mass in Msun (for filename matching).",
    )
    p.add_argument(
        "--orientation_tag",
        type=str,
        required=True,
        help="Orientation tag (for filename matching, e.g., 'Taman_edgeon').",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "figures"),
        help="Directory where the figure will be saved.",
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
        results_dir=args.results_dir,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        orientation_tag=args.orientation_tag,
        output_dir=args.output_dir,
        variable=args.variable,
        overlay_cycles=args.overlay_cycles,
        overlay_peaks=args.overlay_peaks,
        overlay_troughs=args.overlay_troughs,
        show_legend=args.show_legend,
        eta=args.eta,
        f_min=args.f_min,
    )
