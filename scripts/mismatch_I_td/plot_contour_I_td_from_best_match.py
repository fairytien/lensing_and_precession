"""Create mismatch contour plots from one aggregated best-match HDF5 file.

This script requires an exact best-match file path produced by
python -m scripts.mismatch_I_td.aggregate_best_match.
All contour naming parameters are inferred from that file's metadata.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from modules.plot_utils import apply_physics_paper_style

apply_physics_paper_style()

from modules.filenames import contour_I_td_filename, contour_I_td_run_dir
from modules.bank_io import read_best_match_I_td_contour_data

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
):
    """Plot mismatch contour from aggregated best-match file.

    Args:
        input_path: Exact path to an aggregated best-match HDF5 file.
        output_dir: Directory where the figure will be saved.
        variable: Variable to plot ("epsilon", "omega", or "theta").
    """
    if variable not in VARIABLE_MAPPING:
        raise ValueError(
            f"Invalid variable '{variable}'. Must be one of: {list(VARIABLE_MAPPING.keys())}"
        )

    var_info = VARIABLE_MAPPING[variable]
    best_match = read_best_match_I_td_contour_data(input_path, var_info["dataset"])

    if best_match["missing_I_count"] > 0:
        logging.warning(
            "Best-match file reports %d missing I rows; contour will include NaN gaps.",
            best_match["missing_I_count"],
        )

    logging.info(f"Using best-match file: {input_path}")

    output_dir = contour_I_td_run_dir(
        output_dir,
        mcz=best_match["mcz"],
        I_min=best_match["I_min"],
        I_max=best_match["I_max"],
        td_min_ms=best_match["td_min_ms"],
        td_max_ms=best_match["td_max_ms"],
        z=best_match["z"],
        orientation_tag=best_match["orientation_tag"],
    )
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Resolved figure output directory: {output_dir}")

    I_arr = best_match["I"]
    td_arr = best_match["td"]
    Zmap = best_match["values"]

    # Convert td from seconds to ms for plotting
    td_arr_ms = td_arr * 1e3

    # Create contour plot: x = td (ms), y = I
    TD, I_GRID = np.meshgrid(td_arr_ms, I_arr)
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(TD, I_GRID, Zmap, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    cbar.set_label(var_info["label"])
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$I$")
    plt.title(
        rf"$\mathcal{{M}}_s = {best_match['mcz']:.1f}\ M_\odot$, $z = {best_match['z']:.2g}$"
    )

    plt.tight_layout()

    # Generate filename with variable suffix
    base_path = contour_I_td_filename(
        output_dir,
        mcz_msun=best_match["mcz"],
        I_min=best_match["I_min"],
        I_max=best_match["I_max"],
        I_pts=best_match["I_pts"],
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

    args = p.parse_args()

    main(
        input_path=args.input_path,
        output_dir=args.output_dir,
        variable=args.variable,
    )
