"""Create contour plots of optimal SNR (SNR of source waveform) from HDF5 file."""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.filenames import _format_min_precision
from modules.plot_utils_v3 import apply_physics_paper_style
from scripts.utils.plot_cycles_and_extrema_mcz import plot_cycle_lines, plot_mcz_extrema

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

apply_physics_paper_style()


def main(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    I: float,
    orientation_tag: str,
    output_dir: str = "figures/lindblom",
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
    """Plot optimal SNR contour from HDF5 file.

    Args:
        results_dir: Directory containing the source SNR HDF5 file.
        td_min_ms: Minimum time delay in ms (for filename matching).
        td_max_ms: Maximum time delay in ms (for filename matching).
        mcz_min: Minimum chirp mass in Msun (for filename matching).
        mcz_max: Maximum chirp mass in Msun (for filename matching).
        I: Flux ratio (for filename matching).
        orientation_tag: Orientation tag (for filename matching).
        output_dir: Directory where the figure will be saved.
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

    # Find source SNR file
    from modules.filenames import best_match_mcz_td_filename

    summary_path = os.path.join(
        results_dir,
        "best_match",
        f"source_snr_I{_format_min_precision(I)}_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms_{orientation_tag}.h5",
    )

    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"Source SNR file not found: {summary_path}\n"
            f"Please run scripts/compute_source_snr_contour.py first."
        )

    logging.info(f"Loading source SNR data from: {summary_path}")

    # Read data
    I_value = None
    with h5py.File(summary_path, "r") as h5:
        # Extract I from attributes
        if "I" in h5.attrs:
            I_value = float(h5.attrs["I"])

        # Validate that required datasets exist
        required_datasets = ["mcz", "td", "source_snr"]
        missing = [ds for ds in required_datasets if ds not in h5]
        if missing:
            raise KeyError(
                f"Missing datasets in {summary_path}: {missing}. "
                f"Available datasets: {list(h5.keys())}"
            )

        mcz_arr = np.array(h5["mcz"])
        td_arr = np.array(h5["td"])
        Smap = np.array(h5["source_snr"])

    if I_value is None:
        I_value = I

    # Convert td from seconds to ms for plotting
    td_arr_ms = td_arr * 1e3

    # Get actual data range for overlays
    mcz_data_min, mcz_data_max = mcz_arr.min(), mcz_arr.max()

    # Create contour plot
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
    plt.figure(figsize=(8, 6))

    # Create filled contours
    cf = plt.contourf(TD, MCZ, Smap, levels=contour_levels, cmap=cmap, extend=extend)

    cbar = plt.colorbar(cf)
    cbar.set_label(r"$\rho(\tilde{h}_L)$")
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # Add title
    plt.title("SNR of Lensed Source Waveforms")

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
    base_name = f"source_snr_contour_I{_format_min_precision(I_value)}_mcz{_format_min_precision(mcz_min)}-{_format_min_precision(mcz_max)}Msun_td{_format_min_precision(td_min_ms)}-{_format_min_precision(td_max_ms)}ms_{orientation_tag}"

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
    # Get project root directory (used for default paths)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    p = argparse.ArgumentParser(
        description="Plot optimal SNR (SNR of source waveform) contour from HDF5 file."
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(project_root, "data", "contours_td_mcz"),
        help="Directory containing the source SNR HDF5 file.",
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
        "--I",
        type=float,
        default=0.5,
        help="Flux ratio (for filename matching, default: 0.5).",
    )
    p.add_argument(
        "--orientation_tag",
        type=str,
        required=True,
        help="Orientation tag (for filename matching, e.g., Taman_edgeon).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "figures", "lindblom"),
        help="Directory where the figure will be saved.",
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
        results_dir=args.results_dir,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        I=args.I,
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
        extend=args.extend,
    )
