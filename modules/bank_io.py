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


def _float_or_nan(value):
    return np.nan if value is None else float(value)


def write_source_attrs(
    h5: h5py.File,
    I: float,
    theta_J,
    phi_J,
    theta_S,
    phi_S,
) -> None:
    """Write source metadata attributes to an open HDF5 file."""
    h5.attrs["I"] = float(I)
    h5.attrs["theta_J"] = _float_or_nan(theta_J)
    h5.attrs["phi_J"] = _float_or_nan(phi_J)
    h5.attrs["theta_S"] = _float_or_nan(theta_S)
    h5.attrs["phi_S"] = _float_or_nan(phi_S)


def read_source_attrs(h5: h5py.File) -> Dict[str, Any]:
    """Read source metadata attributes from an open HDF5 file if present."""
    attrs: Dict[str, Any] = {}
    for key in ("I", "theta_J", "phi_J", "theta_S", "phi_S"):
        if key in h5.attrs:
            attrs[key] = h5.attrs[key]
    return attrs


def write_mcz_grid_attrs(
    h5: h5py.File,
    mcz_min: float,
    mcz_max: float,
    mcz_pts: int,
) -> None:
    """Write intended Stage 1 mcz grid metadata to an open HDF5 file."""
    h5.attrs["mcz_min"] = float(mcz_min)
    h5.attrs["mcz_max"] = float(mcz_max)
    h5.attrs["mcz_pts"] = int(mcz_pts)


def read_mcz_grid_attrs(h5: h5py.File) -> Dict[str, Any]:
    """Read mcz grid metadata from an open HDF5 file if present."""
    attrs: Dict[str, Any] = {}
    for key in ("mcz_min", "mcz_max", "mcz_pts"):
        if key in h5.attrs:
            attrs[key] = h5.attrs[key]
    return attrs


def read_mismatch_cube_shape(h5: h5py.File) -> Tuple[int, int, int, int]:
    """Return axis sizes for a mismatch cube as (td, theta, omega, gamma)."""
    return (
        int(h5["td"].shape[0]),
        int(h5["theta"].shape[0]),
        int(h5["omega"].shape[0]),
        int(h5["gamma"].shape[0]),
    )


def mcz_grid_meta_consistent(
    reference_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    tol: float = 1e-6,
) -> bool:
    """Return True when two mcz grid metadata dicts are numerically consistent."""
    if not reference_meta or not candidate_meta:
        return True
    return (
        abs(
            float(candidate_meta.get("mcz_min", np.nan))
            - float(reference_meta.get("mcz_min", np.nan))
        )
        <= tol
        and abs(
            float(candidate_meta.get("mcz_max", np.nan))
            - float(reference_meta.get("mcz_max", np.nan))
        )
        <= tol
        and int(candidate_meta.get("mcz_pts", -1))
        == int(reference_meta.get("mcz_pts", -1))
    )


def write_missing_mcz_metadata(
    h5: h5py.File,
    expected_mcz: np.ndarray,
    missing_mcz: np.ndarray,
) -> None:
    """Write aggregation completeness metadata to an open HDF5 file."""
    missing = np.asarray(missing_mcz, dtype=np.float64)
    expected = np.asarray(expected_mcz, dtype=np.float64)
    h5.attrs["missing_mcz_count"] = int(missing.shape[0])
    if missing.shape[0] > 0:
        h5.create_dataset("missing_mcz", data=missing)
    h5.create_dataset("expected_mcz", data=expected)


def read_missing_mcz_metadata(h5: h5py.File) -> Dict[str, Any]:
    """Read aggregation completeness metadata from an open HDF5 file."""
    missing = (
        np.array(h5["missing_mcz"], dtype=np.float64)
        if "missing_mcz" in h5
        else np.array([], dtype=np.float64)
    )
    count = int(h5.attrs.get("missing_mcz_count", missing.shape[0]))
    expected = (
        np.array(h5["expected_mcz"], dtype=np.float64) if "expected_mcz" in h5 else None
    )
    return {
        "missing_mcz_count": count,
        "missing_mcz": missing,
        "expected_mcz": expected,
    }


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
