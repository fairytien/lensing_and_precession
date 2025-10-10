import os, argparse, pickle
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from modules.functions_v3 import mcz_for_n_lens_cycles
from modules.filenames import contour_td_mcz_filename


SOLMASS2SEC = 4.92624076e-6


def mcz_trough_for_n(td_s: float, n_trough: int, eta: float = 0.25) -> float:
    """Calculate mcz_trough for given time delay and trough number n_trough."""
    mcz_trough = (
        (eta ** (3 / 5) * td_s)
        / (6 ** (3 / 2) * np.pi * (n_trough + 1 / 2))
        / SOLMASS2SEC
    )
    return mcz_trough


def mcz_peak_for_n(td_s: float, n_peak: int, eta: float = 0.25) -> float:
    """Calculate mcz_peak for given time delay and peak number n_peak."""
    mcz_peak = (eta ** (3 / 5) * td_s) / (6 ** (3 / 2) * np.pi * n_peak) / SOLMASS2SEC
    return mcz_peak


def find_mcz_troughs(
    td_arr: np.ndarray, eta: float = 0.25, mcz_min: float = 10.0, mcz_max: float = 90.0
):
    """Find mcz_trough points for each time delay within the mcz range."""
    td_trough_points = []
    mcz_trough_points = []

    for td in td_arr:
        n_trough = 0
        while True:
            mcz_trough = mcz_trough_for_n(td, n_trough, eta)
            if mcz_trough < mcz_min:
                break
            if mcz_trough <= mcz_max:
                td_trough_points.append(td)
                mcz_trough_points.append(mcz_trough)
            n_trough += 1

    return np.array(td_trough_points), np.array(mcz_trough_points)


def find_mcz_peaks(
    td_arr: np.ndarray, eta: float = 0.25, mcz_min: float = 10.0, mcz_max: float = 90.0
):
    """Find mcz_peak points for each time delay within the mcz range."""
    td_peak_points = []
    mcz_peak_points = []

    for td in td_arr:
        n_peak = 1
        while True:
            mcz_peak = mcz_peak_for_n(td, n_peak, eta)
            if mcz_peak < mcz_min:
                break
            if mcz_peak <= mcz_max:
                td_peak_points.append(td)
                mcz_peak_points.append(mcz_peak)
            n_peak += 1

    return np.array(td_peak_points), np.array(mcz_peak_points)


def _load_data(input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (mcz_arr [Msun], td_arr [s], epsilon_matrix) from .pkl or .h5 file.

    - Pickle must contain keys: 'mcz_arr' (Msun), 'td_arr' (seconds), 'epsilon_matrix'.
    - HDF5 (best_match) must contain datasets: 'mcz', 'td', 'epsilon_min'.
    """
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
        description="Overlay mcz_1cyc, mcz_2cyc, and mcz_peaks lines on a mismatch contour (L vs P)."
    )
    # Back-compat: allow old --pkl_path or new --input_path (.pkl or .h5)
    parser.add_argument(
        "--pkl_path",
        type=str,
        default=None,
        help="[Deprecated] Path to pickle with mcz_arr, td_arr, epsilon_matrix",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
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
        help="Do not plot 1/2/3 lensing modulation lines",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    input_path = args.input_path or args.pkl_path
    if not input_path:
        raise SystemExit("Provide --input_path (.pkl or .h5) or legacy --pkl_path")

    mcz_arr, td_arr, Z = _load_data(input_path)

    # Build grid for plotting
    td_arr_ms = td_arr * 1e3
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)

    # Compute 1/2/3-cycle lines (unless disabled)
    if not args.no_cycles:
        mcz_1cyc = mcz_for_n_lens_cycles(1.0, td_arr, f_min=args.f_min, eta=args.eta)
        mcz_2cyc = mcz_for_n_lens_cycles(2.0, td_arr, f_min=args.f_min, eta=args.eta)
        mcz_3cyc = mcz_for_n_lens_cycles(3.0, td_arr, f_min=args.f_min, eta=args.eta)

    # Find mcz_trough and mcz_peak points (unless disabled)
    mcz_min, mcz_max = mcz_arr.min(), mcz_arr.max()
    if not args.no_troughs:
        td_trough_points, mcz_trough_points = find_mcz_troughs(
            td_arr, args.eta, mcz_min, mcz_max
        )
    else:
        td_trough_points, mcz_trough_points = np.array([]), np.array([])
    if not args.no_peaks:
        td_peak_points, mcz_peak_points = find_mcz_peaks(
            td_arr, args.eta, mcz_min, mcz_max
        )
    else:
        td_peak_points, mcz_peak_points = np.array([]), np.array([])

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
    if not args.no_cycles:
        plt.plot(
            td_arr_ms,
            mcz_1cyc,
            color="black",
            ls="-",
            lw=2,
            label="1 lensing modulation",
        )
        plt.plot(
            td_arr_ms,
            mcz_2cyc,
            color="black",
            ls="--",
            lw=2,
            label="2 lensing modulations",
        )
        plt.plot(
            td_arr_ms,
            mcz_3cyc,
            color="black",
            ls=":",
            lw=2,
            label="3 lensing modulations",
        )

    # Overlay mcz_trough points
    if len(td_trough_points) > 0:
        plt.scatter(
            td_trough_points * 1e3,  # Convert to ms
            mcz_trough_points,
            c="white",
            marker=".",
            s=5,
            alpha=0.8,
            label="mcz troughs",
            zorder=5,
        )

    # Overlay mcz_peak points
    if len(td_peak_points) > 0:
        plt.scatter(
            td_peak_points * 1e3,  # Convert to ms
            mcz_peak_points,
            c="red",
            marker=".",
            s=5,
            alpha=0.8,
            label="mcz peaks",
            zorder=5,
        )

    # plt.legend(loc="best")
    plt.tight_layout()

    # Generate output filename derived from source data + orientation tag, with 'overlayed' suffix
    td_min_ms = td_arr.min() * 1e3
    td_max_ms = td_arr.max() * 1e3
    mcz_min = mcz_arr.min()
    mcz_max = mcz_arr.max()
    orientation_tag = _infer_orientation_tag(input_path)

    # Start from the standard contour filename then add suffix 'overlayed'
    base_fig = contour_td_mcz_filename(
        fig_dir,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        orientation_tag=orientation_tag,
        ext="pdf",
    )
    # Append suffix
    base_name, base_ext = os.path.splitext(base_fig)
    out_path = f"{base_name}_overlayed{base_ext}"

    # Add tag if provided
    if args.tag:
        name, ext = os.path.splitext(out_path)
        out_path = f"{name}_{args.tag}{ext}"
    plt.savefig(out_path, dpi=200)
    print("Figure saved as", out_path)


if __name__ == "__main__":
    main()
