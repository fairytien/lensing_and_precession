import os, argparse, pickle
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from modules.functions_v3 import mcz_for_n_lens_cycles
from modules.filenames import contour_td_mcz_filename


SOLMASS2SEC = 4.92624076e-6


def _mcz_extremum_for_n(td_s: float, n: float, eta: float = 0.25) -> float:
    """Calculate mcz extremum for given time delay and index n.

    For troughs: n = n_trough + 0.5
    For peaks: n = n_peak (integer >= 1)
    """
    return (eta ** (3 / 5) * td_s) / (6 ** (3 / 2) * np.pi * n) / SOLMASS2SEC


def _find_mcz_extrema(
    td_arr: np.ndarray,
    eta: float,
    mcz_min: float,
    mcz_max: float,
    n_start: float,
    n_increment: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generic function to find mcz extrema (troughs or peaks) within range.

    Parameters
    ----------
    td_arr : np.ndarray
        Array of time delays in seconds
    eta : float
        Symmetric mass ratio
    mcz_min, mcz_max : float
        Chirp mass range boundaries in solar masses
    n_start : float
        Starting value for n (0.5 for troughs, 1 for peaks)
    n_increment : float
        Increment for n (1.0 for both)

    Returns
    -------
    tuple
        (td_points, mcz_points) arrays
    """
    td_points = []
    mcz_points = []

    for td in td_arr:
        n = n_start
        while True:
            mcz = _mcz_extremum_for_n(td, n, eta)
            if mcz < mcz_min:
                break
            if mcz <= mcz_max:
                td_points.append(td)
                mcz_points.append(mcz)
            n += n_increment

    return np.array(td_points), np.array(mcz_points)


def find_mcz_troughs(
    td_arr: np.ndarray, eta: float = 0.25, mcz_min: float = 10.0, mcz_max: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Find mcz_trough points for each time delay within the mcz range."""
    return _find_mcz_extrema(
        td_arr, eta, mcz_min, mcz_max, n_start=0.5, n_increment=1.0
    )


def find_mcz_peaks(
    td_arr: np.ndarray, eta: float = 0.25, mcz_min: float = 10.0, mcz_max: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Find mcz_peak points for each time delay within the mcz range."""
    return _find_mcz_extrema(
        td_arr, eta, mcz_min, mcz_max, n_start=1.0, n_increment=1.0
    )


def _load_data(input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (mcz_arr [Msun], td_arr [s], epsilon_matrix) from .pkl or .h5 file.

    - Pickle must contain keys: 'mcz_arr' (Msun), 'td_arr' (seconds), 'epsilon_matrix'.
    - HDF5 (best_match) must contain datasets: 'mcz', 'td', 'epsilon_min'.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    if ext == ".pkl":
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        mcz_arr = np.asarray(data["mcz_arr"], dtype=float)
        td_arr = np.asarray(data["td_arr"], dtype=float)
        Z = np.asarray(data["epsilon_matrix"], dtype=float)
        return mcz_arr, td_arr, Z
    elif ext == ".h5":
        import h5py  # local import to avoid hard dep if unused

        with h5py.File(input_path, "r") as h5:
            if not all(k in h5 for k in ("mcz", "td", "epsilon_min")):
                raise ValueError(
                    "HDF5 must be a best_match file with datasets: 'mcz', 'td', 'epsilon_min'"
                )
            mcz_arr = np.asarray(h5["mcz"], dtype=float)
            td_arr = np.asarray(h5["td"], dtype=float)
            Z = np.asarray(h5["epsilon_min"], dtype=float)
            return mcz_arr, td_arr, Z
    else:
        raise ValueError(f"Unsupported input extension '{ext}'. Use .pkl or .h5")


def _infer_orientation_tag(input_path: str) -> str:
    """Infer orientation_tag from the input filename.

    Expects names like:
      - best_match_td20-70ms_mcz30-46Msun_Taman_edgeon.h5
      - mismatch_cubes_mcz30Msun_td20-70ms_Taman_edgeon.h5
      - contour_td20-70ms_mcz30-46Msun_Taman_edgeon.pkl
    Falls back to 'unknown' if no tag segment found.
    """
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    if len(parts) >= 2:
        # Orientation tag is usually the last token
        return parts[-1]
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Overlay mcz_1cyc, mcz_2cyc, mcz_3cyc, and mcz_extrema lines on a mismatch (L vs P) contour of td vs mcz."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input file (.pkl with mcz_arr, td_arr, epsilon_matrix) or best_match .h5",
    )
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--f_min", type=float, default=20.0)
    parser.add_argument(
        "--optimize_mcz",
        action="store_true",
        help="Use optimized mismatch over template chirp mass (affects filename)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix to append to figure filename",
    )
    parser.add_argument(
        "--no-troughs",
        dest="no_troughs",
        action="store_true",
        help="Do not plot mcz trough points",
    )
    parser.add_argument(
        "--no-peaks",
        dest="no_peaks",
        action="store_true",
        help="Do not plot mcz peak points",
    )
    parser.add_argument(
        "--no-cycles",
        dest="no_cycles",
        action="store_true",
        help="Do not plot 1/2/3 lensing cycle lines",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Show legend for any plotted overlays (cycles, peaks, troughs)",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    input_path = args.input_path

    mcz_arr, td_arr, Z = _load_data(input_path)

    # Validate data
    if mcz_arr.size == 0 or td_arr.size == 0:
        raise ValueError("Loaded arrays are empty")

    # Build grid for plotting
    td_arr_ms = td_arr * 1e3
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
    mcz_min, mcz_max = mcz_arr.min(), mcz_arr.max()

    # Compute 1/2/3-cycle lines (unless disabled)
    cycle_data = []
    if not args.no_cycles:
        for n_cyc, ls_style in [(1.0, "-"), (2.0, "--"), (3.0, ":")]:
            mcz_cyc = mcz_for_n_lens_cycles(
                n_cyc, td_arr, f_min=args.f_min, eta=args.eta
            )
            cycle_data.append((n_cyc, mcz_cyc, ls_style))

    # Find mcz_trough and mcz_peak points (unless disabled)
    td_trough_points, mcz_trough_points = (
        find_mcz_troughs(td_arr, args.eta, mcz_min, mcz_max)
        if not args.no_troughs
        else (np.array([]), np.array([]))
    )
    td_peak_points, mcz_peak_points = (
        find_mcz_peaks(td_arr, args.eta, mcz_min, mcz_max)
        if not args.no_peaks
        else (np.array([]), np.array([]))
    )

    # Plot
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(TD, MCZ, Z, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    if args.optimize_mcz:
        cbar.set_label(
            r"$\min_{\mathcal{M}_{\rm t}}$ $\epsilon(\tilde{h}_{\rm L}, \tilde{h}_{\rm P})$"
        )
    else:
        cbar.set_label(
            r"$\min_{\tilde{\Omega}, \, \tilde{\theta}, \, \gamma_P}\; \epsilon(\tilde{h}_L,\tilde{h}_P)$"
        )
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # Overlay cycle lines (unless disabled)
    for n_cyc, mcz_cyc, ls_style in cycle_data:
        label = f"{int(n_cyc)} cycle" if n_cyc == 1 else f"{int(n_cyc)} cycles"
        plt.plot(td_arr_ms, mcz_cyc, color="black", ls=ls_style, lw=2, label=label)

    # Overlay mcz extrema points
    extrema_config = [
        (td_trough_points, mcz_trough_points, "white", "mcz troughs"),
        (td_peak_points, mcz_peak_points, "red", "mcz peaks"),
    ]
    for td_pts, mcz_pts, color, label in extrema_config:
        if td_pts.size > 0:
            plt.scatter(
                td_pts * 1e3,  # Convert to ms
                mcz_pts,
                c=color,
                marker=".",
                s=5,
                alpha=0.8,
                label=label,
                zorder=5,
            )

    # Optionally show legend if there are labeled artists
    if args.show_legend:
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            plt.legend(loc="best")
    plt.tight_layout()

    # Generate output filename derived from source data + orientation tag, with 'overlayed' suffix
    td_min_ms = td_arr.min() * 1e3
    td_max_ms = td_arr.max() * 1e3
    orientation_tag = _infer_orientation_tag(input_path)

    # Build filename with overlayed suffix
    base_fig = contour_td_mcz_filename(
        fig_dir,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        orientation_tag=orientation_tag,
        ext="pdf",
    )
    base_name, base_ext = os.path.splitext(base_fig)

    # Add suffixes: _overlayed and optional user tag
    suffixes = ["overlayed"]
    if args.tag:
        suffixes.append(args.tag)
    out_path = f"{base_name}_{'_'.join(suffixes)}{base_ext}"
    plt.savefig(out_path, dpi=200)
    print("Figure saved as", out_path)


if __name__ == "__main__":
    main()
