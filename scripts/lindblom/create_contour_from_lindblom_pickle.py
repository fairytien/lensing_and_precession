"""Create contour plots from pickle file with Lindblom data."""

import os, sys, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.filenames import _format_min_precision
from modules.plot_utils import apply_physics_paper_style
from scripts.utils.plot_cycles_and_extrema import plot_cycle_lines, plot_mcz_extrema

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

apply_physics_paper_style()


def main(
    pickle_path: str,
    output_dir: str,
    variable: str = "lindblom",  # "lindblom" or "snr"
    overlay_cycles: bool = False,
    overlay_peaks: bool = False,
    overlay_troughs: bool = False,
    show_legend: bool = False,
    eta: float = 0.25,
    f_min: float = 20.0,
    contour_levels: int = 100,
    cmap: str = "jet",
    extend: str = "neither",
):
    """Plot contour from pickle file with Lindblom data.

    Args:
        pickle_path: Path to pickle file with Lindblom data.
        output_dir: Directory where the figure will be saved.
        variable: Which variable to plot ("lindblom" or "snr").
        overlay_cycles: If True, overlay 1/2/3 lensing cycle lines.
        overlay_peaks: If True, overlay mcz peak points.
        overlay_troughs: If True, overlay mcz trough points.
        show_legend: If True, show legend for overlays.
        eta: Symmetric mass ratio (default 0.25).
        f_min: Minimum frequency in Hz (default 20.0).
        contour_levels: Number of contour levels (default 100).
        cmap: Colormap name (default "jet").
        extend: Colorbar extension ("neither", "min", "max", "both").
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load pickle file
    logging.info(f"Loading data from: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    mcz_arr = np.array(data["mcz_arr"])
    td_arr = np.array(data["td_arr"])  # in seconds

    if variable == "lindblom":
        if "lindblom_matrix" not in data:
            raise KeyError(
                f"Missing 'lindblom_matrix' in pickle file. Available keys: {list(data.keys())}"
            )
        Zmap = np.array(data["lindblom_matrix"])
        label = r"$\epsilon(\tilde{h}_L, \tilde{h}_{\rm NP}) - \frac{1}{2\rho^2}$"
        title = "Lindblom Criterion: Lensed Sources vs Non-Precessing Templates"
    elif variable == "snr":
        if "snr_matrix" not in data:
            raise KeyError(
                f"Missing 'snr_matrix' in pickle file. Available keys: {list(data.keys())}"
            )
        Zmap = np.array(data["snr_matrix"])
        label = r"$\rho(\tilde{h}_L, \tilde{h}_{\rm NP})$"
        title = "SNR Between Lensed Sources and Non-Precessing Templates"
    else:
        raise ValueError(f"Unknown variable: {variable}. Use 'lindblom' or 'snr'")

    I_value = data.get("I", 0.5)

    # Convert td from seconds to ms for plotting
    td_arr_ms = td_arr * 1e3

    # Get actual data range for overlays
    mcz_data_min, mcz_data_max = mcz_arr.min(), mcz_arr.max()

    # Create contour plot
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
    plt.figure(figsize=(8, 6))

    # Create filled contours
    cf = plt.contourf(TD, MCZ, Zmap, levels=contour_levels, cmap=cmap, extend=extend)

    cbar = plt.colorbar(cf)
    cbar.set_label(label)
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # Add title
    plt.title(title)

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

    # Generate filename
    base_name = f"{variable}_contour_I{_format_min_precision(I_value)}_mcz{_format_min_precision(mcz_arr.min())}-{_format_min_precision(mcz_arr.max())}Msun_td{_format_min_precision(td_arr_ms.min())}-{_format_min_precision(td_arr_ms.max())}ms_NP"

    # Add overlay suffix if any overlays are enabled
    if overlay_cycles or overlay_peaks or overlay_troughs:
        base_name += "_overlayed"

    # Save as PDF
    fig_path_pdf = os.path.join(output_dir, f"{base_name}.pdf")
    plt.savefig(fig_path_pdf, dpi=200, bbox_inches="tight")
    logging.info(f"Figure saved as {fig_path_pdf}")

    # Save as PNG
    fig_path_png = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    logging.info(f"Figure saved as {fig_path_png}")

    plt.close()


if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    p = argparse.ArgumentParser(
        description="Create contour plots from pickle file with Lindblom data."
    )
    p.add_argument(
        "--pickle_path",
        type=str,
        required=True,
        help="Path to pickle file with Lindblom data.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "figures", "lindblom"),
        help="Directory where the figure will be saved.",
    )
    p.add_argument(
        "--variable",
        type=str,
        default="lindblom",
        choices=["lindblom", "snr"],
        help="Variable to plot: 'lindblom' or 'snr' (default: lindblom).",
    )
    p.add_argument(
        "--overlay-cycles",
        action="store_true",
        help="Overlay 1/2/3 lensing cycle lines.",
    )
    p.add_argument(
        "--overlay-peaks",
        action="store_true",
        help="Overlay mcz peak points.",
    )
    p.add_argument(
        "--overlay-troughs",
        action="store_true",
        help="Overlay mcz trough points.",
    )
    p.add_argument(
        "--show-legend",
        action="store_true",
        help="Show legend for overlays.",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=0.25,
        help="Symmetric mass ratio (default: 0.25).",
    )
    p.add_argument(
        "--f_min",
        type=float,
        default=20.0,
        help="Minimum frequency in Hz (default: 20.0).",
    )
    p.add_argument(
        "--contour-levels",
        type=int,
        default=100,
        help="Number of contour levels (default: 100).",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="Colormap name (default: jet).",
    )
    p.add_argument(
        "--extend",
        type=str,
        default="neither",
        choices=["neither", "min", "max", "both"],
        help="Colorbar extension (default: neither).",
    )

    args = p.parse_args()
    main(
        pickle_path=args.pickle_path,
        output_dir=args.output_dir,
        variable=args.variable,
        overlay_cycles=args.overlay_cycles,
        overlay_peaks=args.overlay_peaks,
        overlay_troughs=args.overlay_troughs,
        show_legend=args.show_legend,
        eta=args.eta,
        f_min=args.f_min,
        contour_levels=args.contour_levels,
        cmap=args.cmap,
        extend=args.extend,
    )
