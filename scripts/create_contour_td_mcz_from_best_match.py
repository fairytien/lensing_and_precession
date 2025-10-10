"""Create mismatch contour plots from aggregated best-match HDF5 file.

This script reads the best-match file (produced by scripts/aggregate_best_match.py)
and generates a contour plot of the minimal mismatch across (td, mcz).

Pipeline:
1. scripts/compute_mismatch_cubes.py - compute per-mcz mismatch cubes
2. scripts/aggregate_best_match.py - consolidate cubes into best-match file
3. scripts/create_contour_td_mcz_from_best_match.py (this script) - plot contour
"""

import os, argparse, sys

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.filenames import (
    best_match_filename,
    contour_td_mcz_filename,
)
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
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load best-match file
    summary_path = best_match_filename(
        results_dir, td_min_ms, td_max_ms, mcz_min, mcz_max, orientation_tag
    )

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"Best-match file not found: {summary_path}\n"
            f"Please run scripts/aggregate_best_match.py first to create this file."
        )

    logging.info(f"Loading best-match data from: {summary_path}")

    with h5py.File(summary_path, "r") as h5:
        mcz_arr = np.array(h5["mcz"])
        td_arr = np.array(h5["td"])
        Zmap = np.array(h5["epsilon_min"])

    # Convert td from seconds to ms for plotting
    td_arr_ms = td_arr * 1e3

    # Create contour plot
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(TD, MCZ, Zmap, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    cbar.set_label(
        r"$\min_{\~\Omega, \~\theta, \gamma_P}$ $\epsilon(\tilde{h}_L, \tilde{h}_P)$"
    )
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")
    plt.tight_layout()

    fig_path = contour_td_mcz_filename(
        output_dir,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        orientation_tag=orientation_tag,
        ext="pdf",
    )
    plt.savefig(fig_path, dpi=200)
    logging.info(f"Figure saved as {fig_path}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Plot mismatch contour from aggregated best-match HDF5 file. "
            "Run scripts/aggregate_best_match.py first to create the best-match file."
        )
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "contours",
        ),
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
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "figures",
        ),
        help="Directory where the figure will be saved.",
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
    )
