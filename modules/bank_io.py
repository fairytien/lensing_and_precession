"""HDF5 I/O helpers for template banks and mismatch cubes.

Provides:
- open_bank_readonly(path) -> (h5, omega, theta, gamma, bank, attrs)
- create_bank_writer(path, shape, dtype, chunking, dset_attrs)
- create_mismatch_cube(path, td_pts, theta_arr, omega_arr, gamma_arr, mcz, td_arr, save_full_mismatch)

Design goals: streaming-friendly writes, gzip compression, shuffle filter, and
fletcher32 checksums for robustness.
"""

import os
from contextlib import contextmanager
from typing import Tuple, Dict, Any

import numpy as np
import h5py


def open_bank_readonly(
    filepath: str,
) -> Tuple[h5py.File, np.ndarray, np.ndarray, np.ndarray, h5py.Dataset, Dict[str, Any]]:
    """
    Open a bank HDF5 file for read-only access and return handles/arrays.

    Returns (h5_file, omega, theta, gamma, bank_dataset, bank_attrs)
    Caller is responsible for closing the returned h5_file.
    """
    h5 = h5py.File(filepath, "r")
    omega = np.array(h5["omega"]).astype(float)
    theta = np.array(h5["theta"]).astype(float)
    gamma = np.array(h5["gamma"]).astype(float)
    bank = h5["bank"]
    attrs = dict(bank.attrs.items())
    return h5, omega, theta, gamma, bank, attrs


@contextmanager
def create_bank_writer(
    filepath: str,
    shape: Tuple[int, int, int, int],
    dtype: np.dtype,
    chunking: Tuple[int, int, int, int],
    dset_attrs: Dict[str, Any],
):
    """
    Context-managed creator for the 4D bank dataset designed for streaming writes.

    Yields (h5_file, bank_dataset). Caller may write axis datasets (omega/theta/gamma)
    and assign file attrs as needed within the context.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    h5 = h5py.File(filepath, "w")
    try:
        bank = h5.create_dataset(
            "bank",
            shape=shape,
            dtype=dtype,
            chunks=chunking,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )
        for k, v in dset_attrs.items():
            bank.attrs[k] = v
        yield h5, bank
    finally:
        h5.close()


def create_mismatch_cube(
    filepath: str,
    td_pts: int,
    theta_arr: np.ndarray,
    omega_arr: np.ndarray,
    gamma_arr: np.ndarray,
    mcz: float,
    td_arr: np.ndarray,
    save_full_mismatch: bool = False,
):
    """
    Create an HDF5 file with per-mcz mismatch cube datasets.

    Returns a tuple (h5, datasets) where datasets is a dict containing:
      - 'mismatch' (optional): (td, theta, omega, gamma)
      - 'epsilon_min_grid': (td, theta, omega)
      - 'gamma_best_grid': (td, theta, omega)
    Caller is responsible for closing the returned h5 file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    h5 = h5py.File(filepath, "w")
    h5.create_dataset("mcz", data=np.array([mcz], dtype=np.float64))
    h5.create_dataset("td", data=td_arr.astype(np.float64))
    h5.create_dataset("omega", data=omega_arr.astype(np.float64))
    h5.create_dataset("theta", data=theta_arr.astype(np.float64))
    h5.create_dataset("gamma", data=gamma_arr.astype(np.float64))

    n_theta = int(theta_arr.shape[0])
    n_omega = int(omega_arr.shape[0])
    n_gamma = int(gamma_arr.shape[0])

    datasets: Dict[str, Any] = {}
    if save_full_mismatch:
        datasets["mismatch"] = h5.create_dataset(
            "mismatch",
            shape=(int(td_pts), n_theta, n_omega, n_gamma),
            dtype=np.float32,
            chunks=(1, min(16, n_theta), min(16, n_omega), n_gamma),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )

    datasets["epsilon_min_grid"] = h5.create_dataset(
        "epsilon_min_grid",
        shape=(int(td_pts), n_theta, n_omega),
        dtype=np.float32,
        chunks=(1, min(16, n_theta), min(16, n_omega)),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
    )

    datasets["gamma_best_grid"] = h5.create_dataset(
        "gamma_best_grid",
        shape=(int(td_pts), n_theta, n_omega),
        dtype=np.float32,
        chunks=(1, min(16, n_theta), min(16, n_omega)),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
    )

    return h5, datasets
