"""Compute per-mcz mismatch cubes from prebuilt RP template banks.

This script streams templates directly from HDF5, computes per-(theta, omega)
minima across gamma in parallel, and writes per-mcz mismatch cubes incrementally.
Designed for array-job chunking and low-memory operation.

Outputs per-mcz HDF5 files to results_dir/mismatch_cubes/ containing:
  - epsilon_min_grid (td, theta, omega)
  - gamma_best_grid (td, theta, omega)
  - optional mismatch (td, theta, omega, gamma) if --save_full_mismatch

Use scripts/aggregate_best_match.py to consolidate cubes into a single best-match file.
Use scripts/create_contour_mcz_td_from_best_match.py to plot the contour from the best-match file.
"""

import os, argparse, sys
from typing import Optional

import numpy as np
from multiprocessing import Pool, cpu_count

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.functions_v3 import (
    get_gw,
    get_y_from_I,
    get_MLz_from_td,
    Sn,
    timer_decorator,
    get_fcut_from_mcz,
)
from modules.default_params_v3 import SOLMASS2SEC, lens_params_1, orient_params
from modules.orientation import resolve_orientation, allowed_orient_presets
from modules.filenames import bank_filename, mismatch_cube_filename
from modules.match_utils import (
    build_source_strain_for_td,
    init_mismatch_worker,
    mismatch_gamma_job,
)
from modules.bank_io import open_bank_readonly, create_mismatch_cube
import logging
from modules.cluster_utils import get_env_int, chunk_bounds
from modules.chunking import choose_gamma_chunk

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@timer_decorator
def main(
    I: float,
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
    orient_preset: Optional[str],
    mcz_min: float,
    mcz_max: float,
    mcz_pts: int,
    td_min_ms: float,
    td_max_ms: float,
    td_pts: int,
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
    compare_both: bool,
    use_opt_match: bool,
    save_full_mismatch: bool,
    results_dir: str,
    mcz_chunk_index: Optional[int] = None,
    mcz_chunk_count: Optional[int] = None,
):
    os.makedirs(results_dir, exist_ok=True)

    # Axes arrays
    mcz_arr = np.linspace(mcz_min, mcz_max, mcz_pts)
    td_arr_ms = np.linspace(td_min_ms, td_max_ms, td_pts)
    td_arr = td_arr_ms / 1e3

    # Orientation/tag used to find matching banks and to set source orientation
    lens_base, tag = resolve_orientation(
        orient_preset,
        theta_J,
        phi_J,
        theta_S,
        phi_S,
        lens_params_1,
        orient_params,
        default_author="Taman",
        default_orientation="edgeon",
    )

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
        start, end = chunk_bounds(mcz_pts, mcz_chunk_count, mcz_chunk_index)
        sel = range(start, end)
        logging.info(
            f"Chunking mcz across {mcz_chunk_count} chunks: running indices [{start}:{end})"
        )
    else:
        sel = range(mcz_pts)

    # Loop over mcz values
    for i in sel:
        mcz = float(mcz_arr[i])
        logging.info(
            f"[{i+1}/{len(mcz_arr)}] Processing mcz={mcz} Msun with td {td_min_ms}-{td_max_ms}ms td{td_pts}, omega {omega_min}-{omega_max} o{omega_pts}, theta {theta_min}-{theta_max} t{theta_pts}, gamma g{gamma_pts}"
        )

        # Bank path (must have been created already)
        bank_path = bank_filename(
            bank_dir,
            mcz,
            omega_min,
            omega_max,
            omega_pts,
            theta_min,
            theta_max,
            theta_pts,
            gamma_pts,
            tag,
            prefix=bank_prefix,
        )
        if not os.path.isfile(bank_path):
            raise FileNotFoundError(f"Template bank not found: {bank_path}")

        # Open bank for slicing without loading to memory
        h5, omega_arr, theta_arr, gamma_arr, bank, _ = open_bank_readonly(bank_path)
        with h5:
            n_theta, n_omega, n_gamma, n_freq = bank.shape

            assert (
                n_theta == theta_pts and n_omega == omega_pts and n_gamma == gamma_pts
            )

            # Set source mcz
            y = get_y_from_I(I)
            lens_params = dict(lens_base)
            lens_params["mcz"] = float(mcz) * SOLMASS2SEC
            lens_params["y"] = float(y)

            # Precompute PSD once for this mcz (independent of td)
            # Define f-array using mcz -> f_cut
            f_cut = float(get_fcut_from_mcz(mcz, eta=lens_params["eta"]))
            s_f = np.arange(f_min, f_cut, delta_f)
            psd = Sn(s_f, f_min=f_min, delta_f=delta_f)

            # Prepare HDF5 output for mismatch cubes (per-mcz)
            mm_out_path = mismatch_cube_filename(
                results_dir,
                mcz_msun=mcz,
                I=I,
                td_min_ms=td_min_ms,
                td_max_ms=td_max_ms,
                td_pts=td_pts,
                omega_pts=omega_pts,
                theta_pts=theta_pts,
                gamma_pts=gamma_pts,
                orientation_tag=tag,
            )
            mmh5, dsets = create_mismatch_cube(
                filepath=mm_out_path,
                td_pts=td_pts,
                theta_arr=theta_arr,
                omega_arr=omega_arr,
                gamma_arr=gamma_arr,
                mcz=mcz,
                td_arr=td_arr,
                save_full_mismatch=save_full_mismatch,
            )
            with mmh5:
                # Store source parameters as HDF5 attributes for later aggregation
                mmh5.attrs["I"] = float(I)
                mmh5.attrs["theta_J"] = np.nan if theta_J is None else float(theta_J)
                mmh5.attrs["phi_J"] = np.nan if phi_J is None else float(phi_J)
                mmh5.attrs["theta_S"] = np.nan if theta_S is None else float(theta_S)
                mmh5.attrs["phi_S"] = np.nan if phi_S is None else float(phi_S)

                mm_dset = dsets.get("mismatch")
                ep_min_grid_dset = dsets["epsilon_min_grid"]
                g_best_grid_dset = dsets["gamma_best_grid"]

                # Iterate over td values
                for j, td in enumerate(td_arr):
                    lens_params_j = dict(lens_params)
                    lens_params_j["MLz"] = float(get_MLz_from_td(td, y) * SOLMASS2SEC)
                    s_strain = build_source_strain_for_td(
                        get_gw, lens_params_j, f_min=f_min, delta_f=delta_f
                    )

                    # Prepare jobs across (theta, omega) using indices only
                    total_jobs = int(n_theta) * int(n_omega)
                    if n_workers is None:
                        n_workers_eff = min(cpu_count(), total_jobs)
                    else:
                        n_workers_eff = int(n_workers)

                    Zgrid = np.zeros((n_theta, n_omega), dtype=np.float32)
                    Ggrid = np.zeros_like(Zgrid)

                    # Stream results to reduce memory; open bank inside workers
                    gamma_chunk = choose_gamma_chunk(int(n_gamma))
                    with Pool(
                        n_workers_eff,
                        initializer=init_mismatch_worker,
                        initargs=(
                            s_strain,
                            psd,
                            delta_f,
                            compare_both,
                            use_opt_match,
                            bank_path,
                            gamma_arr,
                            gamma_chunk,
                        ),
                        maxtasksperchild=256,
                    ) as pool:
                        job_iter = (
                            (r, c) for r in range(n_theta) for c in range(n_omega)
                        )
                        for r, c, ep_vec, ep_min, g_best in pool.imap_unordered(
                            mismatch_gamma_job, job_iter, chunksize=1
                        ):
                            if save_full_mismatch and mm_dset is not None:
                                mm_dset[j, r, c, :] = ep_vec
                            Zgrid[r, c] = ep_min
                            Ggrid[r, c] = g_best

                    # Save per-td min grids
                    ep_min_grid_dset[j, :, :] = Zgrid
                    g_best_grid_dset[j, :, :] = Ggrid

        logging.info(f"Saved mismatch cube: {mm_out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Compute per-mcz mismatch cubes between lensed sources and RP templates using precomputed banks."
        )
    )
    p.add_argument(
        "--I", type=float, default=0.5, help="Flux ratio I (0<I<1). Default 0.5"
    )
    p.add_argument("--theta_J", type=float, default=None)
    p.add_argument("--phi_J", type=float, default=None)
    p.add_argument("--theta_S", type=float, default=None)
    p.add_argument("--phi_S", type=float, default=None)
    p.add_argument(
        "--orient_preset",
        type=str,
        default=None,
        help=(
            "Optional orientation preset to use for both params and tag."
            "If not provided, angles (theta_J, phi_J, theta_S, phi_S) form the tag."
        ),
    )
    p.add_argument("--mcz_min", type=float, default=10.0)
    p.add_argument("--mcz_max", type=float, default=80.0)
    p.add_argument("--mcz_pts", type=int, default=71)
    p.add_argument("--td_min_ms", type=float, default=20.0)
    p.add_argument("--td_max_ms", type=float, default=70.0)
    p.add_argument("--td_pts", type=int, default=51)
    p.add_argument("--omega_min", type=float, default=0.0)
    p.add_argument("--omega_max", type=float, default=6.0)
    p.add_argument("--omega_pts", type=int, default=61)
    p.add_argument("--theta_min", type=float, default=0.0)
    p.add_argument("--theta_max", type=float, default=15.0)
    p.add_argument("--theta_pts", type=int, default=151)
    p.add_argument("--gamma_pts", type=int, default=51)
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
    p.add_argument("--compare_both", action="store_true")
    p.add_argument("--use_opt_match", action="store_true")
    p.add_argument("--save_full_mismatch", action="store_true")
    p.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "contours_td_mcz",
        ),
    )
    p.add_argument(
        "--mcz_chunk_index",
        type=int,
        default=None,
        help="Chunk index for mcz splitting (0-based). Defaults to SLURM_ARRAY_TASK_ID if set.",
    )
    p.add_argument(
        "--mcz_chunk_count",
        type=int,
        default=None,
        help="Total chunks for mcz splitting. Defaults to SLURM_ARRAY_TASK_COUNT if set.",
    )
    # Build dynamic choices list from orient_params to avoid drift
    dynamic_choices = allowed_orient_presets(orient_params)
    # Repoint choices on orient_preset action
    for action in p._actions:
        if getattr(action, "dest", None) == "orient_preset":
            action.choices = dynamic_choices
            break

    args = p.parse_args()

    main(
        I=args.I,
        theta_J=args.theta_J,
        phi_J=args.phi_J,
        theta_S=args.theta_S,
        phi_S=args.phi_S,
        orient_preset=args.orient_preset,
        mcz_min=args.mcz_min,
        mcz_max=args.mcz_max,
        mcz_pts=args.mcz_pts,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        td_pts=args.td_pts,
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
        compare_both=args.compare_both,
        use_opt_match=args.use_opt_match,
        save_full_mismatch=args.save_full_mismatch,
        results_dir=args.results_dir,
        mcz_chunk_index=args.mcz_chunk_index,
        mcz_chunk_count=args.mcz_chunk_count,
    )
