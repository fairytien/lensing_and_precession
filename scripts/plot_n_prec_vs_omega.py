import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Constants reused from scripts/plot_n_prec_mcz_omega.py
SOLMASS2SEC = 4.92624076e-6  # solar mass in seconds


def f_cut_from_mcz(mcz_msun: float, eta: float = 0.25) -> float:
    """
    Compute the cutoff frequency f_cut [Hz] given chirp mass in solar masses and eta.
    """
    return (eta ** (3.0 / 5.0)) / ((6.0**1.5) * math.pi * mcz_msun * SOLMASS2SEC)


def total_mass_seconds(mcz_msun: np.ndarray, eta: float = 0.25) -> np.ndarray:
    """
    Total mass in seconds from chirp mass in solar masses.
    """
    mcz_sec = mcz_msun * SOLMASS2SEC
    return mcz_sec / (eta ** (3.0 / 5.0))


def phi_LJ_amplitude(mcz_msun: np.ndarray, eta: float = 0.25) -> np.ndarray:
    """
    Amplitude factor A such that phi_LJ(f) = (A * omega_tilde) * (1/FMIN - 1/f) + gamma_P.
    Returns the coefficient A for omega_tilde = 1.
    """
    M_sec = total_mass_seconds(mcz_msun, eta)
    mcz_sec = mcz_msun * SOLMASS2SEC
    f_cut = f_cut_from_mcz(mcz_msun, eta)
    denom = (
        (M_sec / SOLMASS2SEC)
        * (math.pi ** (8.0 / 3.0))
        * (mcz_sec ** (5.0 / 3.0))
        * (f_cut ** (5.0 / 3.0))
    )
    return (5000.0 / 96.0) / denom


def compute_prec_cycles(
    mcz_msun: np.ndarray,
    omega_tilde: np.ndarray,
    f_min: float = 20.0,
    eta: float = 0.25,
) -> np.ndarray:
    """
    Number of precession cycles: N_prec = [phi_LJ(f_cut) - phi_LJ(f_min)] / (2*pi)
    with phi_LJ ∝ omega_tilde.
    """
    A0 = phi_LJ_amplitude(mcz_msun, eta)  # coefficient for omega_tilde=1
    f_cut = f_cut_from_mcz(mcz_msun, eta)
    delta = (1.0 / f_min) - (1.0 / f_cut)
    return (A0[:, None] * omega_tilde[None, :] * delta[:, None]) / (2.0 * math.pi)


def main():
    parser = argparse.ArgumentParser(
        description="Plot number of precession cycles vs omega_tilde at fixed chirp mass"
    )
    parser.add_argument("--mcz", type=float, default=20.0, help="Chirp mass [M_sun]")
    parser.add_argument("--omega_min", type=float, default=3.5)
    parser.add_argument("--omega_max", type=float, default=4.0)
    parser.add_argument("--omega_points", type=int, default=201)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--f_min", type=float, default=20.0)
    # Angles included for completeness in title (do not affect N_prec in this model)
    parser.add_argument("--theta_S", type=float, default=math.pi / 3)
    parser.add_argument("--phi_S", type=float, default=math.pi / 4)
    parser.add_argument("--theta_J", type=float, default=math.pi / 6)
    parser.add_argument("--phi_J", type=float, default=math.pi / 3)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    omega_arr = np.linspace(args.omega_min, args.omega_max, args.omega_points)

    # Compute N_prec for the specified mcz over omega_tilde
    nprec_arr = compute_prec_cycles(
        np.array([args.mcz]), omega_arr, f_min=args.f_min, eta=args.eta
    )[0]

    plt.figure(figsize=(7.5, 5.5))
    plt.plot(omega_arr, nprec_arr, lw=2.0)
    plt.xlabel(r"$\tilde{\Omega}$")
    plt.ylabel(r"$N_\mathrm{prec}$")
    plt.title(
        rf"$\mathcal{{M}}_c={args.mcz:.0f}\,M_\odot$, $\theta_S=\pi/3$, $\phi_S=\pi/4$, $\theta_J=\pi/6$, $\phi_J=\pi/3$"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(fig_dir, f"n_prec_vs_omega_mcz{int(round(args.mcz))}.pdf")
    plt.savefig(out_path, dpi=200)
    print("Figure saved as", out_path)


if __name__ == "__main__":
    main()
