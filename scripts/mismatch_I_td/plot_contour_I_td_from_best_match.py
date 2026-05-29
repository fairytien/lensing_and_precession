"""Create mismatch contour plots from one aggregated best-match HDF5 file.

This script requires an exact best-match file path produced by
python -m scripts.mismatch_I_td.aggregate_best_match.
All contour naming parameters are inferred from that file's metadata.
"""

import os
import argparse

import matplotlib.pyplot as plt
from modules.plot_utils import apply_physics_paper_style

apply_physics_paper_style()

from modules.filenames import contour_I_td_filename, contour_I_td_run_dir
from modules.bank_io import read_best_match_I_td_data
from modules.cli_utils import add_cycle_extrema_overlay_args
from modules.cosmology import mcz_src_to_det
from scripts.utils._best_match_plot import (
    VARIABLE_MAPPING,
    render_best_match_contour,
    build_figure_path,
)
from scripts.utils.plot_cycles_and_extrema import (
    draw_fixed_mcz_overlays,
    make_fixed_mcz_overlay_legend_handles,
)

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
    best_match = read_best_match_I_td_data(input_path, var_info["dataset"])

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
    td_arr_ms = best_match["td"] * 1e3

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

    has_overlays = overlay_cycles or overlay_peaks or overlay_troughs
    fig_path = build_figure_path(base_path, variable, has_overlays=has_overlays)
    title = rf"$\mathcal{{M}}_{{\mathrm{{s}}}} = {best_match['mcz']:.1f}\,\mathrm{{M}}_\odot$, $z = {best_match['z']:.2g}$"

    z = best_match["z"]
    mcz_det = (
        float(mcz_src_to_det(float(best_match["mcz"]), float(z)))
        if z is not None
        else float(best_match["mcz"])
    )
    td_min_ms = float(td_arr_ms.min())
    td_max_ms = float(td_arr_ms.max())

    def overlay_fn():
        ax = plt.gca()
        positions = draw_fixed_mcz_overlays(
            ax,
            mcz_det,
            td_min_ms,
            td_max_ms,
            overlay_cycles=overlay_cycles,
            overlay_peaks=overlay_peaks,
            overlay_troughs=overlay_troughs,
            eta=eta,
            f_min=f_min,
        )
        if show_legend:
            handles = make_fixed_mcz_overlay_legend_handles(
                cycle_n_list=list(positions.keys()) if overlay_cycles else None,
                include_peaks=overlay_peaks,
                include_troughs=overlay_troughs,
            )
            if handles:
                ax.legend(handles=handles, loc="best")

    render_best_match_contour(
        x_arr=td_arr_ms,
        y_arr=I_arr,
        Zmap=best_match["values"],
        x_label=r"$\Delta t_{\mathrm{d}}\,[\mathrm{ms}]$",
        y_label=r"$I$",
        cbar_label=var_info["label"],
        title=title,
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
