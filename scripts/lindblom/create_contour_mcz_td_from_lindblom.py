"""Create Lindblom criterion contour plots from aggregated best-match HDF5 file.

This script reads the Lindblom best-match file (produced by scripts/aggregate_lindblom_best_match.py)
and generates a contour plot of the Lindblom criterion: (mismatch - 1/(2*SNR^2)) across (td, mcz).

The Lindblom criterion states that if mismatch >= 1/(2*SNR^2), the source waveform
is distinguishable from the template waveform. Regions where the Lindblom value
is positive indicate distinguishable waveforms.
"""

import os, argparse, sys, glob

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.filenames import _format_min_precision
from modules.plot_utils import apply_physics_paper_style
from modules.cli_utils import add_cycle_extrema_overlay_args

from scripts.utils.plot_cycles_and_extrema import plot_cycle_lines, plot_mcz_extrema

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

apply_physics_paper_style()


def main(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
    output_dir: str,
    overlay_cycles: bool = False,
    overlay_peaks: bool = False,
    overlay_troughs: bool = False,
    show_legend: bool = False,
    eta: float = 0.25,
    f_min: float = 20.0,
    contour_levels: int = 100,
    cmap: str = "RdBu_r",  # Red-Blue reversed: positive (red) = distinguishable, negative (blue) = not distinguishable
    zero_contour: bool = True,  # Draw a contour at zero to highlight the boundary
):
    """Plot Lindblom criterion contour from aggregated best-match file.

    Args:
        results_dir: Directory containing the Lindblom best-match HDF5 file.
        td_min_ms: Minimum time delay in ms (for filename matching).
        td_max_ms: Maximum time delay in ms (for filename matching).
        mcz_min: Minimum chirp mass in Msun (for filename matching).
        mcz_max: Maximum chirp mass in Msun (for filename matching).
        orientation_tag: Orientation tag (for filename matching).
        output_dir: Directory where the figure will be saved.
        overlay_cycles: If True, overlay 1/2/3 lensing cycle lines.
        overlay_peaks: If True, overlay mcz peak points.
        overlay_troughs: If True, overlay mcz trough points.
        show_legend: If True, show legend for overlays.
        eta: Symmetric mass ratio (default 0.25).
        f_min: Minimum frequency in Hz (default 20.0).
        contour_levels: Number of contour levels (default 100).
        cmap: Colormap name (default "RdBu_r").
        zero_contour: If True, draw a contour line at zero (default True).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load best-match file
    pattern = os.path.join(
        results_dir,
        "best_match",
        f"best_match_lindblom_I*_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms*_{orientation_tag}.h5",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            "No Lindblom best-match file found. Expected pattern: " + pattern
        )
    if len(matches) > 1:
        # Choose the most recent if multiple files match
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    summary_path = matches[0]

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"Lindblom best-match file not found: {summary_path}\n"
            f"Please run scripts/aggregate_lindblom_best_match.py first to create this file."
        )

    logging.info(f"Loading Lindblom best-match data from: {summary_path}")

    # Read data
    I_value = None
    with h5py.File(summary_path, "r") as h5:
        # Extract I from attributes
        if "I" in h5.attrs:
            I_value = float(h5.attrs["I"])

        # Validate that required datasets exist
        # Support both old name (lindblom_min) and new name (lindblom_at_best_match) for backward compatibility
        if "lindblom_at_best_match" in h5:
            lindblom_key = "lindblom_at_best_match"
        elif "lindblom_min" in h5:
            lindblom_key = "lindblom_min"  # Backward compatibility
        else:
            raise KeyError(
                f"Missing Lindblom dataset in {summary_path}. "
                f"Expected 'lindblom_at_best_match' or 'lindblom_min'. "
                f"Available datasets: {list(h5.keys())}"
            )

        required_datasets = ["mcz", "td"]
        missing = [ds for ds in required_datasets if ds not in h5]
        if missing:
            raise KeyError(
                f"Missing datasets in {summary_path}: {missing}. "
                f"Available datasets: {list(h5.keys())}"
            )

        mcz_arr = np.array(h5["mcz"])
        td_arr = np.array(h5["td"])
        Lmap = np.array(h5[lindblom_key])

    if I_value is None:
        raise ValueError("Could not infer I value from Lindblom best-match file")

    # Convert td from seconds to ms for plotting
    td_arr_ms = td_arr * 1e3

    # Get actual data range for overlays
    mcz_data_min, mcz_data_max = mcz_arr.min(), mcz_arr.max()

    # Create contour plot
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
    plt.figure(figsize=(8, 6))

    # Set vmin = -0.05 to saturate values below -0.05
    vmin = -0.05
    vmax = np.nanmax(Lmap)

    # Create contour levels from vmin to vmax
    if isinstance(contour_levels, int):
        levels = np.linspace(vmin, vmax, contour_levels)
    else:
        levels = contour_levels

    # Create filled contours with vmin saturation
    cf = plt.contourf(
        TD, MCZ, Lmap, levels=levels, cmap=cmap, extend="min", vmin=vmin, vmax=vmax
    )

    # Draw zero contour line if requested
    if zero_contour:
        cs = plt.contour(
            TD, MCZ, Lmap, levels=[0.0], colors="black", linewidths=2, linestyles="--"
        )
        plt.clabel(cs, inline=True, fontsize=10, fmt="0")

    cbar = plt.colorbar(cf)
    cbar.set_label(
        r"$\min_{\tilde{\Omega}, \tilde{\theta}, \gamma_P}$ $\epsilon(\tilde{h}_L, \tilde{h}_P) - \frac{1}{2\rho^2}$"
    )
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # Add title
    plt.title(
        "Lindblom Criterion: Distinguishability of Lensed Sources from Precessing Templates"
    )

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
    base_name = f"lindblom_contour_I{_format_min_precision(I_value)}_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms_{orientation_tag}"

    # Add overlay suffix if any overlays are enabled
    if overlay_cycles or overlay_peaks or overlay_troughs:
        base_name += "_overlayed"

    # Save as PDF
    fig_path_pdf = os.path.join(output_dir, f"{base_name}.pdf")
    plt.savefig(fig_path_pdf, dpi=200, bbox_inches="tight")
    logging.info(f"Figure saved as {fig_path_pdf}")

    # Save as PNG (higher DPI for better quality)
    fig_path_png = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    logging.info(f"Figure saved as {fig_path_png}")

    plt.close()


if __name__ == "__main__":
    # Get project root directory (used for default paths)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    p = argparse.ArgumentParser(
        description=(
            "Plot Lindblom criterion contour from aggregated best-match HDF5 file. "
            "Run scripts/aggregate_lindblom_best_match.py first to create the best-match file."
        )
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(project_root, "data", "contours_td_mcz"),
        help="Directory containing the Lindblom best-match HDF5 file.",
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
        help="Orientation tag (e.g., Taman_edgeon).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "figures", "lindblom"),
        help="Directory where the figure will be saved.",
    )
    add_cycle_extrema_overlay_args(p)
    p.add_argument(
        "--contour-levels",
        type=int,
        default=100,
        help="Number of contour levels (default: 100).",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="RdBu_r",
        help="Colormap name (default: RdBu_r).",
    )
    p.add_argument(
        "--no-zero-contour",
        action="store_true",
        help="Do not draw a contour line at zero.",
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
        overlay_cycles=args.overlay_cycles,
        overlay_peaks=args.overlay_peaks,
        overlay_troughs=args.overlay_troughs,
        show_legend=args.show_legend,
        eta=args.eta,
        f_min=args.f_min,
        contour_levels=args.contour_levels,
        cmap=args.cmap,
        zero_contour=not args.no_zero_contour,
    )
