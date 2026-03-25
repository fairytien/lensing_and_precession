"""Plot mismatch contour over (omega_tilde, theta_tilde) for a given td and mcz.

This script loads a per-mcz mismatch cube HDF5 created by
python -m scripts.mismatch_mcz_td.compute_mismatch_cubes, extracts the
epsilon_min_grid slice at the td closest to the requested time delay, and
plots a contour over (omega_tilde, theta_tilde).

Notes:
- The aggregated best-match file best_match_*.h5 does NOT contain full (omega, theta)
  mismatch contours; it only stores the global minima and best-fit parameters per (mcz, td).
  Therefore, this script requires the matching per-mcz mismatch cube file
    discovered via modules.filenames.find_mismatch_cube_files.

Usage example:
    python -m scripts.mismatch_mcz_td.plot_omega_theta_from_cube \
    --mcz 50 --td_ms 35 \
        --results_root data/contours_td_mcz \
    --orientation_tag Taman_edgeon \
    --save_path figures/mismatch_cubes/mismatch_contour_mcz50_td35ms.png
"""

import os
import sys
import argparse
from typing import Optional, Tuple, List

import numpy as np
import h5py
import matplotlib.pyplot as plt

from modules.filenames import find_mismatch_cube_files


def _find_mismatch_cube(
    mcz: float,
    orientation_tag: str,
    results_roots: List[str],
) -> Optional[str]:
    """Search common results roots for the per-mcz mismatch cube file.

        Looks under:
            {root}/mismatch_cubes/*.h5
        and filters through canonical parsing rules.
    Returns the first matching path if found, else None.
    """
    for root in results_roots:
        matches = find_mismatch_cube_files(
            results_dir=root,
            td_min_ms=None,
            td_max_ms=None,
            orientation_tag=orientation_tag,
            mcz_msun=float(mcz),
        )
        if matches:
            return matches[0]
    return None


def _nearest_index(arr: np.ndarray, value: float) -> int:
    arr = np.asarray(arr, dtype=float)
    return int(np.argmin(np.abs(arr - value)))


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_contour_slice(
    cube_path: str,
    td_ms: float,
    out_path: Optional[str] = None,
    levels: int = 100,
    cmap: str = "jet",
    dpi: int = 200,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Load mismatch cube and render the contour at the nearest td to td_ms.

    Returns the Matplotlib (fig, ax). Optionally saves to out_path and/or shows.
    """
    with h5py.File(cube_path, "r") as h5:
        td = np.asarray(h5["td"], dtype=float)  # seconds
        theta = np.asarray(h5["theta"], dtype=float)
        omega = np.asarray(h5["omega"], dtype=float)
        Zcube = np.asarray(h5["epsilon_min_grid"], dtype=float)  # (td, theta, omega)

    # Convert td to ms for selection and labeling
    td_ms_arr = td * 1e3
    j = _nearest_index(td_ms_arr, td_ms)
    td_sel_ms = float(td_ms_arr[j])
    Z = Zcube[j, :, :]  # (theta, omega)

    # Build mesh for plotting (omega on x, theta on y)
    O, T = np.meshgrid(omega, theta)

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(O, T, Z, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax, label=r"$\epsilon(\tilde{h}_L, \tilde{h}_P)$")
    cs = ax.contour(O, T, Z, levels=levels, colors="k", alpha=0.3, linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlabel(r"$\tilde{\Omega}$")
    ax.set_ylabel(r"$\tilde{\theta}$ [deg]")
    ax.set_title(
        f"Mismatch contour at td={td_sel_ms:.2f} ms (nearest to {td_ms:.2f} ms)"
    )
    ax.grid(True, alpha=0.2)

    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def main():
    p = argparse.ArgumentParser(
        description=(
            "Plot mismatch contour over (omega_tilde, theta_tilde) for a given td and mcz\n"
            "Requires the per-mcz mismatch cube HDF5 produced by compute_mismatch_cubes.py."
        )
    )
    p.add_argument("--mcz", type=float, required=True, help="Chirp mass (Msun)")
    p.add_argument("--td_ms", type=float, required=True, help="Time delay (ms)")
    p.add_argument(
        "--orientation_tag",
        type=str,
        default="Taman_edgeon",
        help="Orientation tag used in filenames (default: Taman_edgeon)",
    )
    p.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=(
            "Root directory that contains mismatch_cubes/. If omitted, searches in "
            "['data/contours_td_mcz']"
        ),
    )
    p.add_argument(
        "--cube_path",
        type=str,
        default=None,
        help="Direct path to a mismatch cube HDF5; overrides results_root search",
    )
    p.add_argument(
        "--save_path", type=str, default=None, help="Path to save the figure"
    )
    p.add_argument(
        "--no_show", action="store_true", help="Do not display the plot window"
    )
    p.add_argument("--levels", type=int, default=100, help="Number of contour levels")
    p.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap")
    p.add_argument("--dpi", type=int, default=200, help="Saved figure DPI")
    args = p.parse_args()

    # Locate mismatch cube file
    cube_path = args.cube_path
    if cube_path is None:
        roots = (
            [args.results_root]
            if args.results_root
            else [
                os.path.join("data", "contours_td_mcz"),
            ]
        )
        cube_path = _find_mismatch_cube(args.mcz, args.orientation_tag, roots)

    if cube_path is None or not os.path.isfile(cube_path):
        print("ERROR: Could not find per-mcz mismatch cube for plotting.")
        print(
            "- The aggregated best_match_*.h5 does NOT contain full (omega, theta) grids."
        )
        print(
            "- Please run python -m scripts.mismatch_mcz_td.compute_mismatch_cubes for the requested mcz/td range, "
            "then retry."
        )
        print("Searched roots:")
        if args.results_root:
            print(f"  - {args.results_root}")
        else:
            print("  - data/contours_td_mcz")
        sys.exit(2)

    print(f"Loading mismatch cube: {cube_path}")
    plot_contour_slice(
        cube_path=cube_path,
        td_ms=args.td_ms,
        out_path=args.save_path,
        levels=args.levels,
        cmap=args.cmap,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
