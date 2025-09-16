import os
import sys
import argparse
import pickle
import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from modules.functions_v3 import mcz_for_n_lens_cycles


SOLMASS2SEC = 4.92624076e-6


def f_cut_from_mcz(mcz_msun: np.ndarray, eta: float = 0.25) -> np.ndarray:
    return (eta ** (3.0 / 5.0)) / ((6.0**1.5) * math.pi * mcz_msun * SOLMASS2SEC)


def mcz_trough_for_n(td_s: float, n_trough: int, eta: float = 0.25) -> float:
    """Calculate mcz_trough for given time delay and trough number n_trough."""
    solar_mass = SOLMASS2SEC
    mcz_trough = (
        (eta ** (3 / 5) * td_s)
        / (6 ** (3 / 2) * np.pi * (n_trough + 1 / 2))
        / solar_mass
    )
    return mcz_trough


def mcz_peak_for_n(td_s: float, n_peak: int, eta: float = 0.25) -> float:
    """Calculate mcz_peak for given time delay and peak number n_peak."""
    solar_mass = SOLMASS2SEC
    mcz_peak = (eta ** (3 / 5) * td_s) / (6 ** (3 / 2) * np.pi * n_peak) / solar_mass
    return mcz_peak


def find_column_minima(
    mcz_arr: np.ndarray, z_col: np.ndarray, max_peaks: int = 3
) -> List[float]:
    # Deprecated: Not used anymore (kept for reference)
    return []


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


def main():
    parser = argparse.ArgumentParser(
        description="Overlay mcz_1cyc, mcz_2cyc, and mcz_peaks lines on a mismatch contour (L vs NP)."
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        required=True,
        help="Path to mismatch pickle (contains mcz_arr, td_arr, epsilon_matrix)",
    )
    parser.add_argument("--max_peaks", type=int, default=3)
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
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    mcz_arr = np.asarray(data["mcz_arr"])  # Msun
    td_arr = np.asarray(data["td_arr"])  # seconds
    Z = np.asarray(data["epsilon_matrix"])  # shape (len(mcz), len(td))

    # Build grid for plotting
    td_arr_ms = td_arr * 1e3
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)

    # Compute 1-cycle and 2-cycle lines
    mcz_1cyc = mcz_for_n_lens_cycles(1.0, td_arr, f_min=args.f_min, eta=args.eta)
    mcz_2cyc = mcz_for_n_lens_cycles(2.0, td_arr, f_min=args.f_min, eta=args.eta)

    # Deprecated: Minima peak lines are no longer computed or plotted

    # Find mcz_trough and mcz_peak points
    mcz_min, mcz_max = mcz_arr.min(), mcz_arr.max()
    td_trough_points, mcz_trough_points = find_mcz_troughs(
        td_arr, args.eta, mcz_min, mcz_max
    )
    td_peak_points, mcz_peak_points = find_mcz_peaks(td_arr, args.eta, mcz_min, mcz_max)

    # Plot
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(TD, MCZ, Z, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    if args.optimize_mcz:
        cbar.set_label(
            r"$\min_{\mathcal{M}_{\rm t}}$ $\epsilon(\tilde{h}_{\rm L}, \tilde{h}_{\rm NP})$"
        )
    else:
        cbar.set_label(r"$\epsilon(\tilde{h}_\mathrm{L}, \tilde{h}_\mathrm{NP})$")
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")

    # Overlay cycle lines
    plt.plot(td_arr_ms, mcz_1cyc, color="black", ls="-", label="1 cycle")
    plt.plot(td_arr_ms, mcz_2cyc, color="black", ls="--", label="2 cycles")

    # No minima peak lines

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

    # Generate output filename based on optimize_mcz option
    if args.optimize_mcz:
        filename_suffix = "overlay_cycles_peaks_opt_mcz"
    else:
        filename_suffix = "overlay_cycles_peaks"

    fig_name = f"mismatch_contour_L_NP_mcz_td_{filename_suffix}"
    if args.tag:
        fig_name = f"{fig_name}_{args.tag}"
    out_path = os.path.join(fig_dir, f"{fig_name}.pdf")
    plt.savefig(out_path, dpi=200)
    print("Figure saved as", out_path)


if __name__ == "__main__":
    main()
