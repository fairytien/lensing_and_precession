"""Create mismatch contour plots from one aggregated best-match HDF5 file.

This script requires an exact best-match file path produced by
python -m scripts.mismatch_mcz_td.aggregate_best_match.
All contour naming parameters are inferred from that file's metadata.
"""

import os
import argparse

import matplotlib.pyplot as plt
from modules.plot_utils import apply_physics_paper_style, LBL_MCZ, LBL_TD

apply_physics_paper_style()

from modules.filenames import (
    contour_mcz_td_filename,
    contour_run_dir,
)
from modules.bank_io import read_best_match_mcz_td_data
from modules.cli_utils import add_cycle_extrema_overlay_args
from scripts.utils._best_match_plot import (
    VARIABLE_MAPPING,
    render_best_match_contour,
    build_figure_path,
)
from scripts.utils.plot_cycles_and_extrema import plot_cycle_lines, plot_mcz_extrema

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
    if variable not in VARIABLE_MAPPING:
        raise ValueError(
            f"Invalid variable '{variable}'. Must be one of: {list(VARIABLE_MAPPING.keys())}"
        )

    var_info = VARIABLE_MAPPING[variable]
    best_match = read_best_match_mcz_td_data(input_path, var_info["dataset"])

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
    td_arr_ms = td_arr * 1e3

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

    has_overlays = overlay_cycles or overlay_peaks or overlay_troughs
    fig_path = build_figure_path(base_path, variable, has_overlays=has_overlays)

    mcz_data_min, mcz_data_max = mcz_msun_arr.min(), mcz_msun_arr.max()

    def overlay_fn():
        if overlay_cycles:
            plot_cycle_lines(td_arr, td_arr_ms, eta=eta, f_min=f_min)
        if overlay_troughs or overlay_peaks:
            plot_mcz_extrema(
                td_arr,
                mcz_data_min,
                mcz_data_max,
                eta=eta,
                plot_troughs=overlay_troughs,
                plot_peaks=overlay_peaks,
            )
        if show_legend:
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                plt.legend(loc="best")

    render_best_match_contour(
        x_arr=td_arr_ms,
        y_arr=mcz_msun_arr,
        Zmap=best_match["values"],
        x_label=LBL_TD,
        y_label=LBL_MCZ,
        cbar_label=var_info["label"],
        title=None,
        output_path=fig_path,
        overlay_fn=overlay_fn if (has_overlays or show_legend) else None,
    )
    logging.info(f"Figure saved as {fig_path}")


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
    add_cycle_extrema_overlay_args(p)

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
