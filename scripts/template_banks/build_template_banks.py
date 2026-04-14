"""Build RP template banks across an mcz grid with streaming HDF5 output.

Streams templates directly to HDF5 to avoid high RAM usage and supports
SLURM-style chunking via CLI or environment variables.
"""

import os, argparse
from typing import Optional

import numpy as np

from modules.template_bank import build_and_save_bank
from modules.filenames import template_bank_run_dir
from modules.orientation import resolve_orientation, allowed_orient_presets
from modules.default_params import RP_params_1
from modules.functions import timer_decorator

# set_orientation is used internally by resolve_orientation; no direct import needed here
from modules.default_params import orient_params
import logging
from modules.cluster_utils import get_env_int, chunk_bounds
from modules.cli_utils import (
    add_orientation_args,
    add_mcz_grid_args,
    add_template_grid_args,
    add_frequency_args,
    add_redshift_arg,
    add_chunking_args,
    resolve_grid_array,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@timer_decorator
def main(
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
    orient_preset: Optional[str],
    mcz_min: float,
    mcz_max: float,
    mcz_pts: Optional[int],
    mcz_step: Optional[float],
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
    theta_pts: int,
    gamma_pts: int,
    f_min: float,
    delta_f: float,
    z: Optional[float],
    bank_dir: str,
    bank_prefix: str,
    n_workers: Optional[int],
    dtype: str,
    mcz_chunk_index: Optional[int],
    mcz_chunk_count: Optional[int],
):
    # Base RP params and orientation/tag handling via shared helper
    base_params, tag = resolve_orientation(
        orient_preset,
        theta_J,
        phi_J,
        theta_S,
        phi_S,
        RP_params_1,
        orient_params,
        default_author="Taman",
        default_orientation="edgeon",
    )

    bank_dir = template_bank_run_dir(bank_dir, z, orientation_tag=tag)
    os.makedirs(bank_dir, exist_ok=True)
    logging.info(f"Resolved bank output directory: {bank_dir}")

    mcz_msun_arr = resolve_grid_array(
        mcz_min, mcz_max, pts=mcz_pts, step=mcz_step, label="mcz"
    )
    mcz_pts_eff = len(mcz_msun_arr)

    # Resolve chunking from CLI or SLURM env vars
    env_idx = get_env_int("SLURM_ARRAY_TASK_ID")
    env_cnt = get_env_int("SLURM_ARRAY_TASK_COUNT")
    if mcz_chunk_index is None:
        mcz_chunk_index = env_idx
    if mcz_chunk_count is None:
        mcz_chunk_count = env_cnt

    if (
        mcz_chunk_index is not None
        and mcz_chunk_count is not None
        and mcz_chunk_count > 1
    ):
        start, end = chunk_bounds(mcz_pts_eff, mcz_chunk_count, mcz_chunk_index)
        sel = range(start, end)
        logging.info(
            f"Chunking mcz across {mcz_chunk_count} chunks: running indices [{start}:{end})"
        )
    else:
        sel = range(mcz_pts_eff)

    for i in sel:
        mcz_msun = float(mcz_msun_arr[i])
        logging.info(
            f"[{i+1}/{len(mcz_msun_arr)}] Building bank for mcz={mcz_msun} Msun with omega {omega_min}-{omega_max} o{omega_pts}, theta {theta_min}-{theta_max} t{theta_pts}, gamma g{gamma_pts}"
        )
        out_path = build_and_save_bank(
            base_params,
            mcz_msun,
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
            z,
            bank_prefix,
            n_workers,
            dtype,
        )
        logging.info(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build RP template banks across mcz range and save as HDF5."
    )
    # Build dynamic choices list from orient_params to avoid drift
    dynamic_choices = allowed_orient_presets(orient_params)
    add_orientation_args(p, orient_choices=dynamic_choices)
    add_mcz_grid_args(p, default_min=10.0, default_max=90.0, default_pts=81)
    add_template_grid_args(
        p,
        omega_min=0.0,
        omega_max=6.0,
        omega_pts=61,
        theta_min=0.0,
        theta_max=15.0,
        theta_pts=151,
        gamma_pts=101,
    )
    add_frequency_args(p, f_min=20.0, delta_f=0.25)
    add_redshift_arg(p, default_z=None)
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
    p.add_argument(
        "--dtype",
        type=str,
        choices=["complex64", "complex128"],
        default="complex128",
        help="Data type for stored complex strain arrays.",
    )
    add_chunking_args(p)

    args = p.parse_args()

    main(
        theta_J=args.theta_J,
        phi_J=args.phi_J,
        theta_S=args.theta_S,
        phi_S=args.phi_S,
        orient_preset=args.orient_preset,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        mcz_pts=getattr(args, "mcz_pts", None),
        mcz_step=args.mcz_step,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        omega_pts=args.omega_pts,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_pts=args.theta_pts,
        gamma_pts=args.gamma_pts,
        f_min=args.f_min,
        delta_f=args.delta_f,
        z=args.z,
        bank_dir=args.bank_dir,
        bank_prefix=args.bank_prefix,
        n_workers=args.n_workers,
        dtype=args.dtype,
        mcz_chunk_index=args.mcz_chunk_index,
        mcz_chunk_count=args.mcz_chunk_count,
    )
