import os
import argparse
from typing import Optional, Tuple

import numpy as np
import h5py
from multiprocessing import Pool, cpu_count

# Ensure project root is on path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.functions_v3 import (
    get_gw,
    get_y_from_I,
    get_MLz_from_td,
    mismatch_from_strains,
    Sn,
    timer_decorator,
    get_fcut_from_mcz,
)
from modules.default_params_v3 import SOLMASS2SEC, lens_params_1, orient_params
from modules.orientation import resolve_orientation, allowed_orient_presets
from modules.filenames import (
    bank_filename,
    mismatch_cubes_filename,
    best_match_filename,
    contour_td_mcz_filename,
)
from modules.match_utils import cast_to_match_precision, ensure_same_length
from modules.bank_io import open_bank_readonly
import logging


# Globals for worker processes
_S_STRAIN = None
_PSD = None
_DELTA_F = None
_COMPARE_BOTH = False
_USE_OPT_MATCH = True
_BANK_H5 = None
_BANK_DSET = None
_GAMMA_ARR = None
_GAMMA_CHUNK = None


def _init_worker(
    s_strain,
    psd,
    delta_f,
    compare_both,
    use_opt_match,
    bank_path,
    gamma_arr,
    gamma_chunk,
):
    import atexit

    global _S_STRAIN, _PSD, _DELTA_F, _COMPARE_BOTH, _USE_OPT_MATCH, _BANK_H5, _BANK_DSET, _GAMMA_ARR, _GAMMA_CHUNK
    _S_STRAIN = s_strain
    _PSD = psd
    _DELTA_F = delta_f
    _COMPARE_BOTH = bool(compare_both)
    _USE_OPT_MATCH = bool(use_opt_match)
    _BANK_H5 = h5py.File(bank_path, "r")
    _BANK_DSET = _BANK_H5["bank"]
    _GAMMA_ARR = gamma_arr
    _GAMMA_CHUNK = int(gamma_chunk) if gamma_chunk is not None else None
    # Harmonize dtype for PyCBC: use complex128 for strains to match PSD double precision
    try:
        _S_STRAIN = cast_to_match_precision(_S_STRAIN)
    except Exception:
        pass
    atexit.register(lambda: _BANK_H5.close())


def _mismatch_gamma_job(args: tuple) -> tuple:
    r, c = args
    n_gamma = _BANK_DSET.shape[2]
    ep_vec = np.empty(n_gamma, dtype=np.float32)
    best_ep = np.inf
    best_gamma = 0.0
    # Iterate over gamma in chunks to keep memory low
    chunk = _GAMMA_CHUNK or max(1, min(32, n_gamma))
    for k0 in range(0, n_gamma, chunk):
        k1 = min(n_gamma, k0 + chunk)
        gamma_block = _BANK_DSET[int(r), int(c), k0:k1, :]  # shape (g, n_freq)
        # Ensure numpy array and cast to complex128 to match source/PSD precision
        gamma_block = cast_to_match_precision(gamma_block)
        for local_idx in range(gamma_block.shape[0]):
            k = k0 + local_idx
            t_arr = gamma_block[local_idx]
            # Length guard: ensure template matches source length
            t_arr, _ = ensure_same_length(t_arr, _S_STRAIN)
            res = mismatch_from_strains(
                t_arr,
                _S_STRAIN,
                f_min=20.0,  # unused since psd is provided
                delta_f=_DELTA_F,
                psd=_PSD,
                use_opt_match=_USE_OPT_MATCH,
                compare_both=_COMPARE_BOTH,
            )
            ep = float(res["mismatch"])
            ep_vec[k] = ep
            if ep < best_ep:
                best_ep = ep
                best_gamma = float(_GAMMA_ARR[k])
    return int(r), int(c), ep_vec, float(best_ep), float(best_gamma)


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
    no_plot: bool,
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Axes arrays
    mcz_arr = np.linspace(mcz_min, mcz_max, mcz_pts)
    td_arr_ms = np.linspace(td_min_ms, td_max_ms, td_pts)
    td_arr = td_arr_ms / 1e3

    # Output maps
    Zmap = np.zeros((mcz_pts, td_pts), dtype=float)
    Omap = np.zeros_like(Zmap)
    Tmap = np.zeros_like(Zmap)
    Gmap = np.zeros_like(Zmap)

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

    # Loop over mcz values
    for i, mcz in enumerate(mcz_arr):
        logging.info(
            f"[{i+1}/{len(mcz_arr)}] Processing mcz={mcz:.1f} Msun (omega {omega_min:.0f}-{omega_max:.0f}, theta {theta_min:.0f}-{theta_max:.0f})"
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
            mm_out_path = mismatch_cubes_filename(
                results_dir,
                mcz_msun=mcz,
                td_min_ms=td_min_ms,
                td_max_ms=td_max_ms,
                orientation_tag=tag,
            )
            with h5py.File(mm_out_path, "w") as mmh5:
                mmh5.create_dataset("mcz", data=np.array([mcz], dtype=np.float64))
                mmh5.create_dataset("td", data=td_arr.astype(np.float64))
                mmh5.create_dataset("omega", data=omega_arr.astype(np.float64))
                mmh5.create_dataset("theta", data=theta_arr.astype(np.float64))
                mmh5.create_dataset("gamma", data=gamma_arr.astype(np.float64))

                if save_full_mismatch:
                    mm_dset = mmh5.create_dataset(
                        "mismatch",  # shape (td, theta, omega, gamma)
                        shape=(td_pts, n_theta, n_omega, n_gamma),
                        dtype=np.float32,
                        chunks=(1, min(16, n_theta), min(16, n_omega), n_gamma),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                        fletcher32=True,
                    )
                else:
                    mm_dset = None

                # Per-td min over gamma grids (stored for convenience)
                ep_min_grid_dset = mmh5.create_dataset(
                    "epsilon_min_grid",  # (td, theta, omega)
                    shape=(td_pts, n_theta, n_omega),
                    dtype=np.float32,
                    chunks=(1, min(16, n_theta), min(16, n_omega)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    fletcher32=True,
                )
                g_best_grid_dset = mmh5.create_dataset(
                    "gamma_best_grid",  # (td, theta, omega)
                    shape=(td_pts, n_theta, n_omega),
                    dtype=np.float32,
                    chunks=(1, min(16, n_theta), min(16, n_omega)),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    fletcher32=True,
                )

                # Iterate over td values
                for j, td in enumerate(td_arr):
                    lens_params_j = dict(lens_params)
                    lens_params_j["MLz"] = float(get_MLz_from_td(td, y) * SOLMASS2SEC)
                    s = get_gw(lens_params_j, f_min=f_min, delta_f=delta_f)
                    s_strain = s["strain"]

                    # Prepare jobs across (theta, omega) using indices only
                    total_jobs = int(n_theta) * int(n_omega)
                    if n_workers is None:
                        n_workers_eff = min(cpu_count(), total_jobs)
                    else:
                        n_workers_eff = int(n_workers)

                    Zgrid = np.zeros((n_theta, n_omega), dtype=np.float32)
                    Ggrid = np.zeros_like(Zgrid)

                    # Stream results to reduce memory; open bank inside workers
                    gamma_chunk = max(1, min(32, int(n_gamma)))
                    with Pool(
                        n_workers_eff,
                        initializer=_init_worker,
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
                            _mismatch_gamma_job, job_iter, chunksize=1
                        ):
                            if save_full_mismatch and mm_dset is not None:
                                mm_dset[j, r, c, :] = ep_vec
                            Zgrid[r, c] = ep_min
                            Ggrid[r, c] = g_best

                    # Save per-td min grids
                    ep_min_grid_dset[j, :, :] = Zgrid
                    g_best_grid_dset[j, :, :] = Ggrid

                    # Extract overall minima across (theta, omega)
                    idx = np.unravel_index(int(np.nanargmin(Zgrid)), Zgrid.shape)
                    Zmap[i, j] = float(Zgrid[idx])
                    Omap[i, j] = float(omega_arr[idx[1]])
                    Tmap[i, j] = float(theta_arr[idx[0]])
                    Gmap[i, j] = float(Ggrid[idx])

        logging.info(f"Saved mismatch data: {mm_out_path}")

    # Save best-match results across all mcz
    summary_path = best_match_filename(
        results_dir,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        mcz_min=mcz_min,
        mcz_max=mcz_max,
        orientation_tag=tag,
    )
    with h5py.File(summary_path, "w") as h5:
        h5.create_dataset("mcz", data=mcz_arr.astype(np.float64))
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("epsilon_min", data=Zmap.astype(np.float32))
        h5.create_dataset("omega_best", data=Omap.astype(np.float32))
        h5.create_dataset("theta_best", data=Tmap.astype(np.float32))
        h5.create_dataset("gamma_best", data=Gmap.astype(np.float32))
        h5.attrs["I"] = float(I)
        h5.attrs["theta_J"] = np.nan if theta_J is None else float(theta_J)
        h5.attrs["phi_J"] = np.nan if phi_J is None else float(phi_J)
        h5.attrs["theta_S"] = np.nan if theta_S is None else float(theta_S)
        h5.attrs["phi_S"] = np.nan if phi_S is None else float(phi_S)
    logging.info(f"Saved best-match results: {summary_path}")

    # Plot contour of minimal mismatch
    if not no_plot:
        import matplotlib.pyplot as plt

        TD, MCZ = np.meshgrid(td_arr_ms, mcz_arr)
        plt.figure(figsize=(8, 6))
        cf = plt.contourf(TD, MCZ, Zmap, levels=100, cmap="jet")
        cbar = plt.colorbar(cf)
        cbar.set_label(
            r"$\min_{\~\Omega, \~\theta, \gamma_P}$ $\epsilon(\tilde{h}_L, \tilde{h}_P)$"
        )
        plt.xlabel(r"$\Delta t_d$ [ms]")
        plt.ylabel(r"$\mathcal{M}_s\ [M_\odot]$")
        plt.tight_layout()
        fig_path = contour_td_mcz_filename(
            fig_dir,
            td_min_ms=td_min_ms,
            td_max_ms=td_max_ms,
            mcz_min=mcz_min,
            mcz_max=mcz_max,
            orientation_tag=tag,
            ext="pdf",
        )
        plt.savefig(fig_path, dpi=200)
        logging.info(f"Figure saved as {fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Compute mismatch between lensed sources and best-matching RP templates across td (x) and mcz (y) using precomputed banks."
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
            "contours",
        ),
    )
    p.add_argument("--no_plot", action="store_true")
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
        no_plot=args.no_plot,
    )
