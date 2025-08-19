import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


SOLMASS2SEC = 4.92624076e-6  # solar mass in seconds


def f_cut_from_mcz(mcz_msun: float, eta: float = 0.25) -> float:
    # f_cut [Hz] with mcz in solar masses
    return (eta ** (3.0 / 5.0)) / ((6.0**1.5) * math.pi * mcz_msun * SOLMASS2SEC)


def total_mass_seconds(mcz_msun: np.ndarray, eta: float = 0.25) -> np.ndarray:
    # Total mass from chirp mass [seconds]
    mcz_sec = mcz_msun * SOLMASS2SEC
    return mcz_sec / (eta ** (3.0 / 5.0))


def phi_LJ_amplitude(mcz_msun: np.ndarray, eta: float = 0.25) -> np.ndarray:
    # Amplitude factor A such that phi_LJ(f) = A * (1/FMIN - 1/f) + gamma_P
    M_sec = total_mass_seconds(mcz_msun, eta)
    mcz_sec = mcz_msun * SOLMASS2SEC
    f_cut = f_cut_from_mcz(mcz_msun, eta)
    denom = (
        (M_sec / SOLMASS2SEC)
        * (math.pi ** (8.0 / 3.0))
        * (mcz_sec ** (5.0 / 3.0))
        * (f_cut ** (5.0 / 3.0))
    )
    # A ∝ omega_tilde, we return the coefficient for omega_tilde=1; caller multiplies by omega_tilde
    return (5000.0 / 96.0) / denom


def compute_prec_cycles(
    mcz_msun: np.ndarray,
    omega_tilde: np.ndarray,
    f_min: float = 20.0,
    eta: float = 0.25,
) -> np.ndarray:
    # n_prec = [phi_LJ(f_cut) - phi_LJ(f_min)] / (2*pi)
    A0 = phi_LJ_amplitude(mcz_msun, eta)  # coefficient for omega_tilde=1
    f_cut = f_cut_from_mcz(mcz_msun, eta)
    delta = (1.0 / f_min) - (1.0 / f_cut)
    # Broadcast over grid: A0(mcz) * omega(mcz,omega) * delta(mcz)
    return (A0[:, None] * omega_tilde[None, :] * delta[:, None]) / (2.0 * math.pi)


def main():
    parser = argparse.ArgumentParser(
        description="Plot number of precession cycles vs mcz and omega_tilde"
    )
    parser.add_argument("--mcz_min", type=float, default=10.0)
    parser.add_argument("--mcz_max", type=float, default=90.0)
    parser.add_argument("--mcz_points", type=int, default=81)
    parser.add_argument("--omega_min", type=float, default=0.0)
    parser.add_argument("--omega_max", type=float, default=5.0)
    parser.add_argument("--omega_points", type=int, default=51)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--f_min", type=float, default=20.0)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    mcz_arr = np.linspace(args.mcz_min, args.mcz_max, args.mcz_points)
    omega_arr = np.linspace(args.omega_min, args.omega_max, args.omega_points)

    Z = compute_prec_cycles(mcz_arr, omega_arr, f_min=args.f_min, eta=args.eta)

    OMEGA, MCZ = np.meshgrid(omega_arr, mcz_arr)
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(OMEGA, MCZ, Z, levels=100, cmap="jet")
    cbar = plt.colorbar(cf)
    cbar.set_label(r"$N_\mathrm{prec}$")
    plt.xlabel(r"$\tilde{\Omega}$")
    plt.ylabel(r"$\mathcal{M}_c\ [M_\odot]$")
    plt.tight_layout()

    out_path = os.path.join(fig_dir, "n_prec_mcz_omega.pdf")
    plt.savefig(out_path, dpi=200)
    print("Figure saved as", out_path)


if __name__ == "__main__":
    main()
