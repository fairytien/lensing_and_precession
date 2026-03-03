import os, argparse, pickle
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from modules.functions_v3 import mcz_for_n_lens_cycles
from modules.filenames import contour_mcz_td_filename


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


def plot_mcz_extrema(
    td_arr: np.ndarray,
    mcz_min: float,
    mcz_max: float,
    eta: float = 0.25,
    plot_troughs: bool = True,
    plot_peaks: bool = True,
    ax=None,
) -> None:
    """Overlay mcz trough and/or peak points on specified matplotlib axes.

    Parameters
    ----------
    td_arr : np.ndarray
        Array of time delays in seconds
    mcz_min, mcz_max : float
        Chirp mass range boundaries in solar masses
    eta : float
        Symmetric mass ratio (default: 0.25)
    plot_troughs : bool
        If True, plot mcz trough points (default: True)
    plot_peaks : bool
        If True, plot mcz peak points (default: True)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    """
    if ax is None:
        ax = plt.gca()

    if plot_troughs:
        td_trough_pts, mcz_trough_pts = find_mcz_troughs(
            td_arr, eta=eta, mcz_min=mcz_min, mcz_max=mcz_max
        )
        if td_trough_pts.size > 0:
            ax.scatter(
                td_trough_pts * 1e3,  # Convert to ms
                mcz_trough_pts,
                c="white",
                marker=".",
                s=5,
                alpha=0.8,
                label="troughs",
                zorder=5,
            )

    if plot_peaks:
        td_peak_pts, mcz_peak_pts = find_mcz_peaks(
            td_arr, eta=eta, mcz_min=mcz_min, mcz_max=mcz_max
        )
        if td_peak_pts.size > 0:
            ax.scatter(
                td_peak_pts * 1e3,  # Convert to ms
                mcz_peak_pts,
                c="red",
                marker=".",
                s=5,
                alpha=0.8,
                label="peaks",
                zorder=5,
            )


def plot_cycle_lines(
    td_arr: np.ndarray,
    td_arr_ms: np.ndarray,
    eta: float = 0.25,
    f_min: float = 20.0,
    ax=None,
) -> None:
    """Overlay 1/2/3 lensing cycle lines on specified matplotlib axes.

    Parameters
    ----------
    td_arr : np.ndarray
        Array of time delays in seconds
    td_arr_ms : np.ndarray
        Array of time delays in milliseconds (for plotting)
    eta : float
        Symmetric mass ratio (default: 0.25)
    f_min : float
        Minimum frequency in Hz (default: 20.0)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    """
    if ax is None:
        ax = plt.gca()

    for n_cyc, ls_style in [(1.0, "-"), (2.0, "--"), (3.0, ":")]:
        mcz_cyc = mcz_for_n_lens_cycles(n_cyc, td_arr, f_min=f_min, eta=eta)
        label = f"{int(n_cyc)} cycle" if n_cyc == 1 else f"{int(n_cyc)} cycles"
        ax.plot(td_arr_ms, mcz_cyc, color="black", ls=ls_style, lw=2, label=label)


def _load_data(
    input_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """Load (mcz_arr [Msun], td_arr [s], epsilon_matrix, I) from .pkl or .h5 file.

    - Pickle must contain keys: 'mcz_arr' (Msun), 'td_arr' (seconds), 'epsilon_matrix'.
    - HDF5 (best_match) must contain datasets: 'mcz', 'td', 'epsilon_min', and 'I' attr.
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
        # Old pickle format may not have I; leave as None if missing
        I_value = float(data["I"]) if "I" in data else None
        return mcz_arr, td_arr, Z, I_value
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
            # Extract I from attributes (may be absent)
            I_value = float(h5.attrs["I"]) if "I" in h5.attrs else None
            return mcz_arr, td_arr, Z, I_value
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
        "--overlay-cycles",
        action="store_true",
        help="Overlay 1/2/3 lensing cycle lines on the contour plot",
    )
    parser.add_argument(
        "--overlay-peaks",
        action="store_true",
        help="Overlay mcz peak points on the contour plot",
    )
    parser.add_argument(
        "--overlay-troughs",
        action="store_true",
        help="Overlay mcz trough points on the contour plot",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Show legend for any plotted overlays (cycles, peaks, troughs)",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    fig_dir = os.path.join(base_dir, "figures", "utils")
    os.makedirs(fig_dir, exist_ok=True)

    input_path = args.input_path

    mcz_arr, td_arr, Z, I_value = _load_data(input_path)

    # Validate data
    if mcz_arr.size == 0 or td_arr.size == 0:
        raise ValueError("Loaded arrays are empty")

    # Build grid for plotting
    td_arr_ms = td_arr * 1e3
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
    mcz_min, mcz_max = mcz_arr.min(), mcz_arr.max()

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

    # Overlay cycle lines if requested
    if args.overlay_cycles:
        plot_cycle_lines(td_arr, td_arr_ms, eta=args.eta, f_min=args.f_min)

    # Overlay mcz extrema points if requested
    plot_mcz_extrema(
        td_arr,
        mcz_min,
        mcz_max,
        eta=args.eta,
        plot_troughs=args.overlay_troughs,
        plot_peaks=args.overlay_peaks,
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
    base_fig = contour_mcz_td_filename(
        fig_dir,
        I=I_value,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
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
