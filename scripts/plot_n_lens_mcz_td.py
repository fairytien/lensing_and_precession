import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


SOLMASS2SEC = 4.92624076e-6  # solar mass in seconds


def f_cut_from_mcz(mcz_msun: float, eta: float = 0.25) -> float:
    # f_cut [Hz] with mcz in solar masses
    return (eta ** (3.0 / 5.0)) / ((6.0**1.5) * math.pi * mcz_msun * SOLMASS2SEC)


def compute_lens_cycles(
    mcz_msun: np.ndarray, td_ms: np.ndarray, f_min: float = 20.0, eta: float = 0.25
) -> np.ndarray:
    # Broadcast over grids: cycles = (f_cut(mcz) - f_min) * td
    f_cut = f_cut_from_mcz(mcz_msun, eta)
    td_s = td_ms / 1e3
    return (f_cut - f_min) * td_s


def main():
    parser = argparse.ArgumentParser(
        description="Plot number of lens cycles vs mcz and td"
    )
    parser.add_argument("--mcz_min", type=float, default=10.0)
    parser.add_argument("--mcz_max", type=float, default=90.0)
    parser.add_argument("--mcz_points", type=int, default=81)
    parser.add_argument("--td_min_ms", type=float, default=20.0)
    parser.add_argument("--td_max_ms", type=float, default=70.0)
    parser.add_argument("--td_points", type=int, default=51)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--f_min", type=float, default=20.0)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    mcz_arr = np.linspace(args.mcz_min, args.mcz_max, args.mcz_points)
    td_arr_ms = np.linspace(args.td_min_ms, args.td_max_ms, args.td_points)
    TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)

    Z = compute_lens_cycles(MCZ, TD, f_min=args.f_min, eta=args.eta)

    plt.figure(figsize=(8, 6))
    cf = plt.contourf(TD, MCZ, Z, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    cbar.set_label(r"$N_\mathrm{lens}$")
    plt.xlabel(r"$\Delta t_d$ [ms]")
    plt.ylabel(r"$\mathcal{M}_c\ [M_\odot]$")
    plt.tight_layout()

    out_path = os.path.join(fig_dir, "n_lens_mcz_td.pdf")
    plt.savefig(out_path, dpi=200)
    print("Figure saved as", out_path)


if __name__ == "__main__":
    main()
