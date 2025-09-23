import os
import argparse
from typing import Optional

import numpy as np

# Ensure project root on path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.template_bank import build_and_save_bank, orientation_tag
from modules.default_params_v3 import RP_params_1
from modules.functions_v3 import set_orientation
from modules.default_params_v3 import orient_params


def main(
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
    mcz_min: float,
    mcz_max: float,
    mcz_pts: int,
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
    theta_pts: int,
    gamma_pts: int,
    f_min: float,
    delta_f: float,
    bank_dir: str,
    bank_prefix: str,
    n_workers: Optional[int],
):
    # Base RP params with orientation
    base_params = set_orientation(orient_params["Taman"]["edgeon"], RP_params_1)[0]
    if theta_J is not None:
        base_params["theta_J"] = theta_J
    if phi_J is not None:
        base_params["phi_J"] = phi_J
    if theta_S is not None:
        base_params["theta_S"] = theta_S
    if phi_S is not None:
        base_params["phi_S"] = phi_S

    tag = orientation_tag(theta_J, phi_J, theta_S, phi_S)

    mcz_arr = np.linspace(mcz_min, mcz_max, mcz_pts)
    for i, mcz in enumerate(mcz_arr):
        print(
            f"[{i+1}/{len(mcz_arr)}] Building bank for mcz={mcz:.1f} Msun with omega {omega_min:.0f}-{omega_max:.0f} x{omega_pts}, theta {theta_min:.0f}-{theta_max:.0f} x{theta_pts}, gamma x{gamma_pts}"
        )
        out_path = build_and_save_bank(
            base_params,
            mcz,
            omega_min,
            omega_max,
            omega_pts,
            theta_min,
            theta_max,
            theta_pts,
            gamma_pts,
            f_min,
            delta_f,
            bank_dir,
            tag,
            bank_prefix,
            n_workers,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build RP template banks across mcz range and save as HDF5."
    )
    p.add_argument("--theta_J", type=float, default=None)
    p.add_argument("--phi_J", type=float, default=None)
    p.add_argument("--theta_S", type=float, default=None)
    p.add_argument("--phi_S", type=float, default=None)
    p.add_argument("--mcz_min", type=float, default=10.0)
    p.add_argument("--mcz_max", type=float, default=80.0)
    p.add_argument("--mcz_pts", type=int, default=71)
    p.add_argument("--omega_min", type=float, default=0.0)
    p.add_argument("--omega_max", type=float, default=6.0)
    p.add_argument("--omega_pts", type=int, default=61)
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=15.0)
    p.add_argument("--theta_pts", type=int, default=151)
    p.add_argument("--gamma_pts", type=int, default=101)
    p.add_argument("--f_min", type=float, default=20.0)
    p.add_argument("--delta_f", type=float, default=0.25)
    p.add_argument(
        "--bank_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "template_banks",
        ),
    )
    p.add_argument("--bank_prefix", type=str, default="rp_bank")
    p.add_argument("--n_workers", type=int, default=None)
    args = p.parse_args()

    main(
        theta_J=args.theta_J,
        phi_J=args.phi_J,
        theta_S=args.theta_S,
        phi_S=args.phi_S,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        mcz_pts=args.mcz_pts,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        omega_pts=args.omega_pts,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_pts=args.theta_pts,
        gamma_pts=args.gamma_pts,
        f_min=args.f_min,
        delta_f=args.delta_f,
        bank_dir=args.bank_dir,
        bank_prefix=args.bank_prefix,
        n_workers=args.n_workers,
    )
