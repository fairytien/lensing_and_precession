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
