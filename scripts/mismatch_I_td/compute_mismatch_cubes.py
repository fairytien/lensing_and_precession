"""Compute per-I mismatch cubes from prebuilt RP template banks.

This script streams templates directly from HDF5, computes per-(theta, omega)
minima across gamma in parallel, and writes per-I mismatch cubes incrementally.
Designed for array-job chunking and low-memory operation.

Unlike the mcz_td pipeline (which varies mcz and needs a different bank per mcz),
this I-td pipeline varies I (flux ratio) at a fixed mcz. Since template banks
depend only on mcz (not I), all I values share the SAME template bank.

Outputs per-I HDF5 files to run_dir/mismatch_cubes/ containing:
  - epsilon_min_grid (td, theta, omega)
  - gamma_best_grid (td, theta, omega)
  - optional mismatch (td, theta, omega, gamma) if --save_full_mismatch

Use python -m scripts.mismatch_I_td.aggregate_best_match to consolidate cubes into a single best-match file.
Use python -m scripts.mismatch_I_td.plot_contour_I_td_from_best_match to plot the contour from the best-match file.
"""

import os
import argparse
from typing import Optional

import numpy as np
from multiprocessing import Pool, cpu_count

from modules.functions import (
    get_gw,
    get_y_from_I,
    get_MLz_from_td,
    Sn,
    timer_decorator,
    get_fcut_from_mcz,
)
from modules.cosmology import apply_z
from modules.default_params import SOLMASS2SEC, lens_params_1, orient_params
from modules.orientation import resolve_orientation, allowed_orient_presets
from modules.filenames import (
    bank_filename,
    mismatch_I_cube_filename,
    template_bank_run_dir,
    contour_I_td_run_dir,
)
from modules.match_utils import (
    build_source_strain_for_td,
    init_mismatch_worker,
    mismatch_gamma_job,
)
from modules.bank_io import (
    safe_open_bank_readonly,
    create_I_mismatch_cube,
    write_source_attrs,
    write_I_td_grid_attrs,
    write_orientation_attr,
    write_scalar_attr_with_unit,
    write_parameter_attrs,
    write_dataset_units,
    extract_prefixed_params,
)
import logging
from modules.cluster_utils import get_env_int, chunk_bounds
from modules.chunking import choose_gamma_chunk
from modules.cli_utils import (
    add_orientation_args,
    add_I_grid_args,
    add_td_grid_args,
    add_template_grid_args,
    add_frequency_args,
    add_redshift_arg,
    add_I_chunking_args,
    resolve_grid_array,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@timer_decorator
def main(
    mcz: float,
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
    orient_preset: Optional[str],
    I_min: float,
    I_max: float,
    I_pts: Optional[int],
    I_step: Optional[float],
    td_min_ms: float,
    td_max_ms: float,
    td_pts: Optional[int],
    td_step_ms: Optional[float],
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
    compare_both: bool,
    use_opt_match: bool,
    save_full_mismatch: bool,
    run_dir: str,
    I_chunk_index: Optional[int] = None,
    I_chunk_count: Optional[int] = None,
):
    z_val = None if z is None else float(z)
    mcz_src_msun = float(mcz)

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

    bank_dir = template_bank_run_dir(bank_dir, z_val, orientation_tag=tag)
    run_dir = contour_I_td_run_dir(
        run_dir,
        mcz=mcz_src_msun,
        I_min=I_min,
        I_max=I_max,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        z=z_val,
        orientation_tag=tag,
    )
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Resolved bank input directory: {bank_dir}")
    logging.info(f"Resolved mismatch output directory: {run_dir}")

    # Axes arrays
    I_arr = resolve_grid_array(I_min, I_max, pts=I_pts, step=I_step, label="I")
    td_arr_ms = resolve_grid_array(
        td_min_ms, td_max_ms, pts=td_pts, step=td_step_ms, label="td_ms"
    )
    td_arr = td_arr_ms / 1e3
    I_pts_eff = len(I_arr)
    td_pts_eff = len(td_arr_ms)

    # Resolve chunking from CLI or SLURM env vars
    env_idx = get_env_int("SLURM_ARRAY_TASK_ID")
    env_cnt = get_env_int("SLURM_ARRAY_TASK_COUNT")
    if I_chunk_index is None:
        I_chunk_index = env_idx
    if I_chunk_count is None:
        I_chunk_count = env_cnt
    if I_chunk_index is not None and I_chunk_count is not None and I_chunk_count > 1:
        start, end = chunk_bounds(I_pts_eff, I_chunk_count, I_chunk_index)
        sel = range(start, end)
        logging.info(
            f"Chunking I across {I_chunk_count} chunks: running indices [{start}:{end})"
        )
    else:
        sel = range(I_pts_eff)

    z_str = "None" if z_val is None else f"{z_val:g}"
    processed_count = 0
    skipped_count = 0

    def skip_I(I_val: float, message: str, *fmt_args):
        nonlocal skipped_count
        logging.warning(
            "Skipping I=%s: " + message,
            I_val,
            *fmt_args,
        )
        skipped_count += 1

    # All I values share the SAME template bank (banks depend only on mcz, not I)
    # Open the bank once outside the I loop
    lens_params = dict(lens_base)
    lens_params["mcz"] = mcz_src_msun * SOLMASS2SEC
    if z_val is not None:
        lens_params = apply_z(lens_params, z_val, mcz_is_source=True)
    mcz_det_msun = float(lens_params["mcz"] / SOLMASS2SEC)
    lens_params["z"] = z_val
    lens_params["mcz_source_msun"] = mcz_src_msun
    lens_params["mcz_detector_msun"] = mcz_det_msun

    # Bank path (must have been created already)
    bank_path = bank_filename(
        bank_dir,
        mcz_src_msun,
        omega_min,
        omega_max,
        omega_pts,
        theta_min,
        theta_max,
        theta_pts,
        gamma_pts,
        tag,
        z=z_val,
        prefix=bank_prefix,
    )
    logging.info(f"Using template bank for mcz_src={mcz_src_msun} Msun: {bank_path}")

    # Open bank for slicing without loading to memory
    bank_payload, bank_open_error = safe_open_bank_readonly(bank_path)
    if bank_payload is None:
        logging.error(
            "Template bank unavailable: %s (%s)",
            bank_path,
            bank_open_error or "unknown error",
        )
        return

    h5_bank, omega_arr, theta_arr, gamma_arr, bank, _ = bank_payload

    with h5_bank:
        n_theta, n_omega, n_gamma, _ = bank.shape

        if not (n_theta == theta_pts and n_omega == omega_pts and n_gamma == gamma_pts):
            logging.error(
                "Bank grid mismatch in %s (expected theta=%d omega=%d gamma=%d, got theta=%d omega=%d gamma=%d)",
                bank_path,
                theta_pts,
                omega_pts,
                gamma_pts,
                n_theta,
                n_omega,
                n_gamma,
            )
            return

        # Precompute PSD once for this mcz (independent of td and I)
        f_cut = float(get_fcut_from_mcz(mcz_det_msun, eta=lens_params["eta"]))
        s_f = np.arange(f_min, f_cut, delta_f)
        psd = Sn(s_f, f_min=f_min, delta_f=delta_f)

        # Loop over I values
        for i in sel:
            I_val = float(I_arr[i])
            y = float(get_y_from_I(I_val))

            # Update lens_params with this I's y value
            lens_params_i = dict(lens_params)
            lens_params_i["y"] = y
            lens_params_i["I"] = I_val

            logging.info(
                f"[{i+1}/{len(I_arr)}] Processing I={I_val}, mcz_src={mcz_src_msun} Msun, "
                f"z={z_str}, mcz_det={mcz_det_msun} Msun with td {td_min_ms}-{td_max_ms}ms td{td_pts_eff}, "
                f"omega {omega_min}-{omega_max} o{omega_pts}, theta {theta_min}-{theta_max} t{theta_pts}, gamma g{gamma_pts}"
            )

            # Compute MLz array for all td values at this I's y
            mlz_arr = np.array(
                [float(get_MLz_from_td(td, y) * SOLMASS2SEC) for td in td_arr],
                dtype=np.float64,
            )

            # Prepare HDF5 output for mismatch cubes (per-I)
            mm_out_path = mismatch_I_cube_filename(
                run_dir,
                I=I_val,
                mcz_msun=mcz_src_msun,
                td_min_ms=td_min_ms,
                td_max_ms=td_max_ms,
                td_pts=td_pts_eff,
                omega_min=omega_min,
                omega_max=omega_max,
                omega_pts=omega_pts,
                theta_min=theta_min,
                theta_max=theta_max,
                theta_pts=theta_pts,
                gamma_pts=gamma_pts,
                orientation_tag=tag,
                z=z_val,
            )
            try:
                mmh5, dsets = create_I_mismatch_cube(
                    filepath=mm_out_path,
                    td_pts=td_pts_eff,
                    theta_arr=theta_arr,
                    omega_arr=omega_arr,
                    gamma_arr=gamma_arr,
                    I_val=I_val,
                    mcz=mcz_src_msun,
                    td_arr=td_arr,
                    save_full_mismatch=save_full_mismatch,
                )
            except Exception as exc:
                skip_I(
                    I_val,
                    "failed to create mismatch cube file %s (%s)",
                    mm_out_path,
                    exc,
                )
                continue

            I_failed_exc = None
            with mmh5:
                # Store source parameters as HDF5 attributes for later aggregation
                write_source_attrs(
                    mmh5,
                    I_val,
                    lens_params_i.get("theta_J"),
                    lens_params_i.get("phi_J"),
                    lens_params_i.get("theta_S"),
                    lens_params_i.get("phi_S"),
                )
                write_orientation_attr(mmh5, tag)
                write_parameter_attrs(
                    mmh5,
                    {**lens_params_i, "I": float(I_val)},
                    prefix="source_param_",
                    include_units=True,
                )
                # Carry template-generation metadata from the source bank file.
                write_parameter_attrs(
                    mmh5,
                    extract_prefixed_params(h5_bank.attrs, "template_param_"),
                    prefix="template_param_",
                    include_units=True,
                )
                # Store the intended I grid so aggregation can detect missing rows
                write_I_td_grid_attrs(mmh5, I_min, I_max, I_pts_eff)
                write_scalar_attr_with_unit(mmh5, "z", z_val, none_as_nan=True)
                write_scalar_attr_with_unit(mmh5, "mcz_source_msun", mcz_src_msun)
                mmh5.create_dataset("MLz", data=mlz_arr)
                write_dataset_units(mmh5, {"MLz": "s"})

                mm_dset = dsets.get("mismatch")
                ep_min_grid_dset = dsets["epsilon_min_grid"]
                g_best_grid_dset = dsets["gamma_best_grid"]

                # Iterate over td values
                for j, td in enumerate(td_arr):
                    lens_params_j = dict(lens_params_i)
                    lens_params_j["MLz"] = float(mlz_arr[j])
                    s_strain = build_source_strain_for_td(
                        get_gw, lens_params_j, f_min=f_min, delta_f=delta_f
                    )

                    # Prepare jobs across (theta, omega) using indices only
                    total_jobs = int(n_theta) * int(n_omega)
                    n_workers_eff = (
                        int(n_workers)
                        if n_workers is not None
                        else min(cpu_count(), total_jobs)
                    )

                    Zgrid = np.zeros((n_theta, n_omega), dtype=np.float32)
                    Ggrid = np.zeros_like(Zgrid)

                    # Stream results to reduce memory; open bank inside workers
                    gamma_chunk = choose_gamma_chunk(int(n_gamma))
                    try:
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
                    except Exception as exc:
                        I_failed_exc = exc
                        break

                    # Save per-td min grids
                    ep_min_grid_dset[j, :, :] = Zgrid
                    g_best_grid_dset[j, :, :] = Ggrid

            if I_failed_exc is not None:
                if mm_out_path is not None and os.path.isfile(mm_out_path):
                    try:
                        os.remove(mm_out_path)
                        logging.warning(
                            "Removed partial mismatch cube after failure: %s",
                            mm_out_path,
                        )
                    except OSError as exc:
                        logging.warning(
                            "Could not remove partial mismatch cube %s (%s)",
                            mm_out_path,
                            exc,
                        )
                skip_I(
                    I_val,
                    "mismatch evaluation failed for bank %s (%s)",
                    bank_path,
                    I_failed_exc,
                )
                continue

            logging.info(f"Saved mismatch cube: {mm_out_path}")
            processed_count += 1

    logging.info(
        "Mismatch cube computation completed: processed=%d skipped=%d",
        processed_count,
        skipped_count,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Compute per-I mismatch cubes between lensed sources and RP templates using precomputed banks."
        )
    )
    p.add_argument(
        "--mcz",
        type=float,
        default=20.0,
        help="Source-frame chirp mass in Msun (fixed for all I values). Default 20.0",
    )
    # Build dynamic choices list from orient_params to avoid drift
    dynamic_choices = allowed_orient_presets(orient_params)
    add_orientation_args(p, orient_choices=dynamic_choices)
    add_I_grid_args(p, default_min=0.1, default_max=0.9, default_pts=41)
    add_td_grid_args(p, default_min_ms=20.0, default_max_ms=70.0, default_pts=51)
    add_template_grid_args(
        p,
        omega_min=0.0,
        omega_max=6.0,
        omega_pts=61,
        theta_min=0.0,
        theta_max=15.0,
        theta_pts=151,
        gamma_pts=51,
    )
    add_frequency_args(p, f_min=20.0, delta_f=0.25)
    add_redshift_arg(p, default_z=1.0)
    p.add_argument(
        "--bank_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
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
        "--run_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data",
            "mismatch",
        ),
        help=(
            "Base contour run directory used to write mismatch_cubes/. "
            "Final tagged run directory is auto-derived if needed."
        ),
    )
    add_I_chunking_args(p)

    args = p.parse_args()

    main(
        mcz=args.mcz,
        theta_J=args.theta_J,
        phi_J=args.phi_J,
        theta_S=args.theta_S,
        phi_S=args.phi_S,
        orient_preset=args.orient_preset,
        I_min=args.I_min,
        I_max=args.I_max,
        I_pts=getattr(args, "I_pts", None),
        I_step=args.I_step,
        td_min_ms=args.td_min_ms,
        td_max_ms=args.td_max_ms,
        td_pts=getattr(args, "td_pts", None),
        td_step_ms=args.td_step_ms,
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
        compare_both=args.compare_both,
        use_opt_match=args.use_opt_match,
        save_full_mismatch=args.save_full_mismatch,
        run_dir=args.run_dir,
        I_chunk_index=args.I_chunk_index,
        I_chunk_count=args.I_chunk_count,
    )
