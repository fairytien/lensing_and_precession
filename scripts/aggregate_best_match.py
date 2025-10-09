"""Aggregate per-mcz mismatch cubes into one best-match HDF5 and plot.

Scans results_dir/mismatch_cubes for per-mcz cubes, reduces each across
(theta, omega), stacks over mcz, writes a combined best_match_*.h5, and
optionally generates the final contour figure.
"""

import os, sys, argparse, glob
import h5py
import numpy as np

# Ensure project root is on path for local invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.filenames import best_match_filename, contour_td_mcz_filename
from modules.functions_v3 import timer_decorator


@timer_decorator
def main(
    results_dir: str,
    td_min_ms: float,
    td_max_ms: float,
    mcz_min: float,
    mcz_max: float,
    orientation_tag: str,
    no_plot: bool,
):
    cube_paths_all = sorted(
        glob.glob(
            os.path.join(
                results_dir,
                "mismatch_cubes",
                f"mismatch_cubes_mcz*Msun_td{td_min_ms:.0f}-{td_max_ms:.0f}ms_{orientation_tag}.h5",
            )
        )
    )
    # Filter by requested mcz range by parsing the filename token 'mcz<val>Msun'
    cube_paths = []
    for p in cube_paths_all:
        base = os.path.basename(p)
        try:
            # Example: mismatch_cubes_mcz47Msun_td20-70ms_TAG.h5
            token = base.split("_mcz", 1)[1]
            mcz_str = token.split("Msun", 1)[0]
            mcz_val = float(mcz_str)
        except Exception:
            continue
        if mcz_min <= mcz_val <= mcz_max:
            cube_paths.append(p)
    if not cube_paths:
        raise FileNotFoundError("No mismatch cube files found")

    mcz_vals = []
    Z_rows = []
    O_rows = []
    T_rows = []
    G_rows = []
    td_arr = None

    for p in cube_paths:
        # Skip unreadable/corrupted files gracefully
        try:
            h5 = h5py.File(p, "r")
        except Exception as e:
            print(f"Warning: Skipping unreadable file: {p} ({e})")
            continue
        with h5:
            mcz = float(np.array(h5["mcz"]).item())
            if td_arr is None:
                td_arr = np.array(h5["td"])  # (td,)
            ep_min_grid = np.array(h5["epsilon_min_grid"])  # (td, theta, omega)
            g_best_grid = np.array(h5["gamma_best_grid"])  # (td, theta, omega)
            theta_arr = np.array(h5["theta"])  # (theta,)
            omega_arr = np.array(h5["omega"])  # (omega,)

            # For each td, find global min over (theta, omega)
            Z = np.zeros(td_arr.shape[0], dtype=np.float32)
            O = np.zeros_like(Z)
            T = np.zeros_like(Z)
            G = np.zeros_like(Z)
            for j in range(td_arr.shape[0]):
                Zgrid = ep_min_grid[j]
                idx = np.unravel_index(int(np.nanargmin(Zgrid)), Zgrid.shape)
                Z[j] = float(Zgrid[idx])
                T[j] = float(theta_arr[idx[0]])
                O[j] = float(omega_arr[idx[1]])
                G[j] = float(g_best_grid[j, idx[0], idx[1]])

            mcz_vals.append(mcz)
            Z_rows.append(Z)
            O_rows.append(O)
            T_rows.append(T)
            G_rows.append(G)

    # Build full mcz grid [mcz_min, ..., mcz_max] with blanks (NaNs) for missing entries
    desired_mcz = np.arange(float(mcz_min), float(mcz_max) + 1.0, 1.0, dtype=np.float64)
    td_len = td_arr.shape[0]
    Zmap = np.full((desired_mcz.shape[0], td_len), np.nan, dtype=np.float32)
    Omap = np.full_like(Zmap, np.nan)
    Tmap = np.full_like(Zmap, np.nan)
    Gmap = np.full_like(Zmap, np.nan)

    # Place available rows at the correct indices
    present_mcz = np.array(mcz_vals, dtype=np.float64)
    order = np.argsort(present_mcz)
    present_mcz_sorted = present_mcz[order]
    Z_rows_sorted = [Z_rows[i] for i in order]
    O_rows_sorted = [O_rows[i] for i in order]
    T_rows_sorted = [T_rows[i] for i in order]
    G_rows_sorted = [G_rows[i] for i in order]
    index_map = {val: idx for idx, val in enumerate(desired_mcz)}
    for val, Zr, Or, Tr, Gr in zip(
        present_mcz_sorted, Z_rows_sorted, O_rows_sorted, T_rows_sorted, G_rows_sorted
    ):
        if val in index_map:
            j = index_map[val]
            Zmap[j, :] = Zr
            Omap[j, :] = Or
            Tmap[j, :] = Tr
            Gmap[j, :] = Gr

    # Save combined best-match file
    summary_path = best_match_filename(
        results_dir, td_min_ms, td_max_ms, mcz_min, mcz_max, orientation_tag
    )
    with h5py.File(summary_path, "w") as h5:
        h5.create_dataset("mcz", data=desired_mcz)
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("epsilon_min", data=Zmap.astype(np.float32))
        h5.create_dataset("omega_best", data=Omap.astype(np.float32))
        h5.create_dataset("theta_best", data=Tmap.astype(np.float32))
        h5.create_dataset("gamma_best", data=Gmap.astype(np.float32))
    print(f"Saved aggregated best-match results: {summary_path}")

    if not no_plot:
        import matplotlib.pyplot as plt

        TD, MCZ = np.meshgrid(td_arr, desired_mcz)
        plt.figure(figsize=(8, 6))
        # Mask NaNs so missing mcz rows appear as blank space
        import numpy.ma as ma

        Zmasked = ma.masked_invalid(Zmap)
        cf = plt.contourf(TD * 1e3, MCZ, Zmasked, levels=100, cmap="jet")
        cbar = plt.colorbar(cf)
        cbar.set_label(
            r"$\min_{\~\Omega, \~\theta, \gamma_P}$ $\epsilon(\tilde{h}_L, \tilde{h}_P)$"
        )
        plt.xlabel(r"$\Delta t_d$ [ms]")
        plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")
        plt.tight_layout()
        # Save figures into the project-level figures directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fig_dir = os.path.join(project_root, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = contour_td_mcz_filename(
            fig_dir, td_min_ms, td_max_ms, mcz_min, mcz_max, orientation_tag, ext="pdf"
        )
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved as {fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Aggregate per-mcz mismatch cubes into a combined best-match file and optional plot."
    )
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--td_min_ms", type=float, required=True)
    p.add_argument("--td_max_ms", type=float, required=True)
    p.add_argument("--mcz_min", type=float, required=True)
    p.add_argument("--mcz_max", type=float, required=True)
    p.add_argument("--orientation_tag", type=str, required=True)
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    main(
        results_dir=args.results_dir,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        orientation_tag=args.orientation_tag,
        no_plot=args.no_plot,
    )
