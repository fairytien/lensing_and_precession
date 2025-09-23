import os
import copy
from typing import Tuple, Optional, Dict

import numpy as np
import h5py
from multiprocessing import Pool, cpu_count

# Make project modules importable when used as a script
try:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except Exception:
    pass

from modules.functions_v3 import get_gw
from modules.default_params_v3 import SOLMASS2SEC
from modules.Classes_v2 import Precessing as P2
from modules.filenames import bank_filename
from modules.chunking import choose_bank_chunks
from modules.bank_io import create_bank_writer


def _grid_arrays(
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
    theta_pts: int,
    gamma_pts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega_arr = np.linspace(float(omega_min), float(omega_max), int(omega_pts))
    theta_arr = np.linspace(float(theta_min), float(theta_max), int(theta_pts))
    gamma_arr = np.linspace(0.0, 2.0 * np.pi, int(gamma_pts), endpoint=False)
    return omega_arr, theta_arr, gamma_arr


def _template_job(args: tuple) -> tuple:
    (r, c, k, omega_val, theta_val, gamma_val, base_params, f_min, delta_f) = args
    t_params = copy.deepcopy(base_params)
    t_params["omega_tilde"] = float(omega_val)
    t_params["theta_tilde"] = float(theta_val)
    t_params["gamma_P"] = float(gamma_val)
    gw = get_gw(
        t_params, f_min=f_min, delta_f=delta_f, prec_Class=P2, frequencySeries=False
    )
    return (r, c, k), np.asarray(gw["strain"], dtype=np.complex128)


def build_bank_for_mcz(
    base_rp_params: Dict,
    omega_min: float,
    omega_max: float,
    omega_pts: int,
    theta_min: float,
    theta_max: float,
    theta_pts: int,
    gamma_pts: int,
    f_min: float,
    delta_f: float,
    n_workers: Optional[int] = None,
    dtype: str = "complex128",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build a 4D RP template bank for a fixed mcz and orientation.

    Returns (omega_arr, theta_arr, gamma_arr, bank, delta_f_actual)
    where bank has shape (theta_pts, omega_pts, gamma_pts, n_freq).
    """
    omega_arr, theta_arr, gamma_arr = _grid_arrays(
        omega_min,
        omega_max,
        omega_pts,
        theta_min,
        theta_max,
        theta_pts,
        gamma_pts,
    )

    # Determine frequency axis from one probe template
    probe_params = copy.deepcopy(base_rp_params)
    probe_params["omega_tilde"] = float(omega_arr[0])
    probe_params["theta_tilde"] = float(theta_arr[0])
    probe_params["gamma_P"] = float(gamma_arr[0])
    probe = get_gw(
        probe_params, f_min=f_min, delta_f=delta_f, prec_Class=P2, frequencySeries=False
    )
    target_dtype = np.complex64 if str(dtype) == "complex64" else np.complex128
    probe_strain = np.asarray(probe["strain"], dtype=target_dtype)
    n_freq = int(probe_strain.shape[0])
    delta_f_actual = float(delta_f)

    bank = np.empty((theta_pts, omega_pts, gamma_pts, n_freq), dtype=target_dtype)

    jobs = []
    for r in range(theta_pts):
        for c in range(omega_pts):
            for k in range(gamma_pts):
                jobs.append(
                    (
                        r,
                        c,
                        k,
                        omega_arr[c],
                        theta_arr[r],
                        gamma_arr[k],
                        base_rp_params,
                        f_min,
                        delta_f,
                    )
                )

    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))

    with Pool(n_workers) as pool:
        for (r, c, k), strain in pool.map(_template_job, jobs):
            # Resize to match probe length if needed (safety)
            arr = np.asarray(strain, dtype=target_dtype)
            if arr.shape[0] != n_freq:
                # pad or truncate to n_freq
                if arr.shape[0] < n_freq:
                    pad = np.zeros((n_freq - arr.shape[0],), dtype=target_dtype)
                    arr = np.concatenate([arr, pad], axis=0)
                else:
                    arr = arr[:n_freq]
            bank[r, c, k, :] = arr

    return omega_arr, theta_arr, gamma_arr, bank, delta_f_actual


def save_bank_hdf5(
    filepath: str,
    omega_arr: np.ndarray,
    theta_arr: np.ndarray,
    gamma_arr: np.ndarray,
    freq_meta: Dict[str, float],
    bank: np.ndarray,
) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as h5:
        h5.create_dataset("omega", data=np.asarray(omega_arr, dtype=np.float64))
        h5.create_dataset("theta", data=np.asarray(theta_arr, dtype=np.float64))
        h5.create_dataset("gamma", data=np.asarray(gamma_arr, dtype=np.float64))
        dset = h5.create_dataset(
            "bank",
            data=bank,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )
        for k, v in freq_meta.items():
            dset.attrs[k] = v
        # Also store grid metadata on file attrs for clarity
        h5.attrs["omega_pts"] = int(omega_arr.shape[0])
        h5.attrs["theta_pts"] = int(theta_arr.shape[0])
        h5.attrs["gamma_pts"] = int(gamma_arr.shape[0])
        h5.attrs["omega_min"] = float(omega_arr.min()) if omega_arr.size else np.nan
        h5.attrs["omega_max"] = float(omega_arr.max()) if omega_arr.size else np.nan
        h5.attrs["theta_min"] = float(theta_arr.min()) if theta_arr.size else np.nan
        h5.attrs["theta_max"] = float(theta_arr.max()) if theta_arr.size else np.nan
    return filepath


def load_bank_hdf5(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    with h5py.File(filepath, "r") as h5:
        omega = np.array(h5["omega"])  # (omega_pts,)
        theta = np.array(h5["theta"])  # (theta_pts,)
        gamma = np.array(h5["gamma"])  # (gamma_pts,)
        bank = np.array(h5["bank"])  # (theta, omega, gamma, n_freq)
        attrs = dict(h5["bank"].attrs.items())
    return omega, theta, gamma, bank, attrs


# bank_filename now imported from modules.filenames


def build_and_save_bank(
    base_rp_params: Dict,
    mcz_msun: float,
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
    orientation_tag: str,
    bank_prefix: str = "rp_bank",
    n_workers: Optional[int] = None,
    dtype: str = "complex128",
) -> str:
    # Prepare parameters
    params = copy.deepcopy(base_rp_params)
    params["mcz"] = float(mcz_msun) * SOLMASS2SEC

    # Build grid arrays
    omega_arr, theta_arr, gamma_arr = _grid_arrays(
        omega_min,
        omega_max,
        omega_pts,
        theta_min,
        theta_max,
        theta_pts,
        gamma_pts,
    )

    # Probe one template to determine n_freq and dtype
    probe_params = copy.deepcopy(params)
    probe_params["omega_tilde"] = float(omega_arr[0])
    probe_params["theta_tilde"] = float(theta_arr[0])
    probe_params["gamma_P"] = float(gamma_arr[0])
    probe = get_gw(
        probe_params, f_min=f_min, delta_f=delta_f, prec_Class=P2, frequencySeries=False
    )
    target_dtype = np.complex64 if str(dtype) == "complex64" else np.complex128
    n_freq = int(np.asarray(probe["strain"], dtype=target_dtype).shape[0])
    df_actual = float(delta_f)

    # Output path
    path = bank_filename(
        bank_dir,
        mcz_msun,
        omega_min,
        omega_max,
        omega_pts,
        theta_min,
        theta_max,
        theta_pts,
        gamma_pts,
        orientation_tag,
        prefix=bank_prefix,
    )

    # Choose chunks and create bank writer
    t_chunk, o_chunk, chunk_gamma, chunk_freq = choose_bank_chunks(
        theta_pts, omega_pts, gamma_pts, n_freq
    )
    dset_attrs = {
        "f_min": float(f_min),
        "delta_f": float(df_actual),
        "mcz_msun": float(mcz_msun),
    }
    with create_bank_writer(
        path,
        shape=(theta_pts, omega_pts, gamma_pts, n_freq),
        dtype=target_dtype,
        chunking=(t_chunk, o_chunk, chunk_gamma, chunk_freq),
        dset_attrs=dset_attrs,
    ) as (h5, dset):
        # Write axis datasets and file attrs
        h5.create_dataset("omega", data=np.asarray(omega_arr, dtype=np.float64))
        h5.create_dataset("theta", data=np.asarray(theta_arr, dtype=np.float64))
        h5.create_dataset("gamma", data=np.asarray(gamma_arr, dtype=np.float64))
        h5.attrs["omega_pts"] = int(omega_arr.shape[0])
        h5.attrs["theta_pts"] = int(theta_arr.shape[0])
        h5.attrs["gamma_pts"] = int(gamma_arr.shape[0])
        h5.attrs["omega_min"] = float(omega_arr.min()) if omega_arr.size else np.nan
        h5.attrs["omega_max"] = float(omega_arr.max()) if omega_arr.size else np.nan
        h5.attrs["theta_min"] = float(theta_arr.min()) if theta_arr.size else np.nan
        h5.attrs["theta_max"] = float(theta_arr.max()) if theta_arr.size else np.nan

        # Lazy job iterator to avoid materializing a huge job list
        def _job_iter():
            for r in range(theta_pts):
                for c in range(omega_pts):
                    for k in range(gamma_pts):
                        yield (
                            r,
                            c,
                            k,
                            omega_arr[c],
                            theta_arr[r],
                            gamma_arr[k],
                            params,
                            f_min,
                            delta_f,
                        )

        total_jobs = int(theta_pts) * int(omega_pts) * int(gamma_pts)
        workers = n_workers if n_workers is not None else min(cpu_count(), total_jobs)

        # Stream results as they complete; write each directly into HDF5
        with Pool(workers, maxtasksperchild=256) as pool:
            for (r, c, k), strain in pool.imap_unordered(
                _template_job, _job_iter(), chunksize=1
            ):
                arr = np.asarray(strain, dtype=target_dtype)
                if arr.shape[0] != n_freq:
                    # pad or truncate to n_freq
                    if arr.shape[0] < n_freq:
                        pad = np.zeros((n_freq - arr.shape[0],), dtype=target_dtype)
                        arr = np.concatenate([arr, pad], axis=0)
                    else:
                        arr = arr[:n_freq]
                dset[int(r), int(c), int(k), :] = arr

    return path


# orientation helpers moved to modules.orientation
