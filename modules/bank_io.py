"""HDF5 I/O helpers for template banks and mismatch cubes.

Provides:
- open_bank_readonly(path) -> (h5, omega, theta, gamma, bank, attrs)
- safe_open_bank_readonly(path) -> (payload_or_none, error_message_or_none)
- create_bank_writer(path, shape, dtype, chunking, dset_attrs)
- create_mismatch_mcz_cube(path, td_pts, theta_arr, omega_arr, gamma_arr, mcz, td_arr, save_full_mismatch)

Design goals: streaming-friendly writes, gzip compression, shuffle filter, and
fletcher32 checksums for robustness.
"""

import os
from contextlib import contextmanager
from typing import Tuple, Dict, Any, Optional

import numpy as np
import h5py


PARAM_UNITS: Dict[str, str] = {
    "I": "dimensionless",
    "theta_S": "rad",
    "phi_S": "rad",
    "theta_J": "rad",
    "phi_J": "rad",
    "mcz": "s",
    "dist": "s",
    "eta": "dimensionless",
    "t_c": "s",
    "phi_c": "rad",
    "y": "dimensionless",
    "MLz": "s",
    "theta_tilde": "dimensionless",
    "omega_tilde": "dimensionless",
    "gamma_P": "rad",
    "f_min": "Hz",
    "delta_f": "Hz",
    "mcz_msun": "Msun",
    "mcz_det_msun": "Msun",
    "mcz_source_msun": "Msun",
    "mcz_detector_msun": "Msun",
    "z": "dimensionless",
}


BankReadonlyPayload = Tuple[
    h5py.File,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    h5py.Dataset,
    Dict[str, Any],
]


def _is_scalar_metadata_value(value: Any) -> bool:
    return np.isscalar(value) or isinstance(value, (str, bytes))


def _normalize_attr_value(value: Any, *, none_as_nan: bool = False) -> Any:
    if value is None and none_as_nan:
        return np.nan
    if isinstance(value, (str, bytes, bytearray)):
        return bytes(value) if isinstance(value, bytearray) else value
    if np.isscalar(value):
        return value
    raise TypeError(f"Unsupported metadata value type: {type(value)}")


def _write_attrs(
    h5: h5py.File, attrs: Dict[str, Any], *, none_as_nan: bool = False
) -> None:
    for key, value in attrs.items():
        h5.attrs[str(key)] = _normalize_attr_value(value, none_as_nan=none_as_nan)


def write_orientation_attr(h5: h5py.File, orientation_tag: str) -> None:
    """Write orientation tag as file metadata."""
    _write_attrs(h5, {"orientation_tag": str(orientation_tag)})


def write_scalar_attr_with_unit(
    h5: h5py.File,
    key: str,
    value: Any,
    *,
    unit: str = None,
    none_as_nan: bool = False,
) -> None:
    """Write one scalar file attr and optional `unit_<key>` companion attr."""
    _write_attrs(h5, {key: value}, none_as_nan=none_as_nan)
    unit_val = PARAM_UNITS.get(key) if unit is None else unit
    if unit_val is not None:
        _write_attrs(h5, {f"unit_{key}": unit_val})


def write_parameter_attrs(
    h5: h5py.File,
    params: Dict[str, Any],
    *,
    prefix: str,
    include_units: bool = True,
) -> None:
    """Write scalar parameter snapshot into file attrs with optional unit attrs."""
    for key, value in params.items():
        if not _is_scalar_metadata_value(value):
            continue
        attr_key = f"{prefix}{key}"
        _write_attrs(h5, {attr_key: value}, none_as_nan=True)
        if include_units and key in PARAM_UNITS:
            _write_attrs(h5, {f"{prefix}unit_{key}": PARAM_UNITS[key]})


def write_dataset_units(h5: h5py.File, dataset_units: Dict[str, str]) -> None:
    """Attach units attrs to datasets when they exist in the file."""
    for name, unit in dataset_units.items():
        if name in h5:
            h5[name].attrs["units"] = str(unit)


def extract_prefixed_params(attrs: Any, prefix: str) -> Dict[str, Any]:
    """Extract prefixed scalar attrs as a plain param dict, skipping unit attrs."""
    out: Dict[str, Any] = {}
    unit_prefix = f"{prefix}unit_"
    for key in attrs.keys():
        key_str = str(key)
        if not key_str.startswith(prefix) or key_str.startswith(unit_prefix):
            continue
        out[key_str.replace(prefix, "", 1)] = attrs[key]
    return out


def write_source_attrs(
    h5: h5py.File,
    I: float,
    theta_J,
    phi_J,
    theta_S,
    phi_S,
) -> None:
    """Write source metadata attributes to an open HDF5 file."""
    _write_attrs(
        h5,
        {
            "I": float(I),
            "theta_J": theta_J,
            "phi_J": phi_J,
            "theta_S": theta_S,
            "phi_S": phi_S,
        },
        none_as_nan=True,
    )


def read_source_attrs(h5: h5py.File) -> Dict[str, Any]:
    """Read source metadata attributes from an open HDF5 file if present."""
    attrs: Dict[str, Any] = {}
    for key in ("I", "theta_J", "phi_J", "theta_S", "phi_S"):
        if key in h5.attrs:
            attrs[key] = h5.attrs[key]
    # Propagate orientation and any source/template parameter snapshots.
    if "orientation_tag" in h5.attrs:
        attrs["orientation_tag"] = h5.attrs["orientation_tag"]
    for key in h5.attrs.keys():
        if str(key).startswith("source_param_") or str(key).startswith(
            "template_param_"
        ):
            attrs[key] = h5.attrs[key]
    return attrs


def write_mcz_grid_attrs(
    h5: h5py.File,
    mcz_min: float,
    mcz_max: float,
    mcz_pts: int,
) -> None:
    """Write intended Stage 1 mcz grid metadata to an open HDF5 file."""
    _write_attrs(
        h5,
        {
            "mcz_min": float(mcz_min),
            "mcz_max": float(mcz_max),
            "mcz_pts": int(mcz_pts),
        },
    )


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


# ==============================================================================
# I-td Pipeline Helpers (flux ratio grid metadata)
# ==============================================================================


def write_I_grid_attrs(
    h5: h5py.File,
    I_min: float,
    I_max: float,
    I_pts: int,
) -> None:
    """Write intended Stage 1 I grid metadata to an open HDF5 file."""
    _write_attrs(
        h5,
        {
            "I_min": float(I_min),
            "I_max": float(I_max),
            "I_pts": int(I_pts),
        },
    )


def read_I_grid_attrs(h5: h5py.File) -> Dict[str, Any]:
    """Read I grid metadata from an open HDF5 file if present."""
    attrs: Dict[str, Any] = {}
    for key in ("I_min", "I_max", "I_pts"):
        if key in h5.attrs:
            attrs[key] = h5.attrs[key]
    return attrs


def I_grid_meta_consistent(
    reference_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    tol: float = 1e-6,
) -> bool:
    """Return True when two I grid metadata dicts are numerically consistent."""
    if not reference_meta or not candidate_meta:
        return True
    return (
        abs(
            float(candidate_meta.get("I_min", np.nan))
            - float(reference_meta.get("I_min", np.nan))
        )
        <= tol
        and abs(
            float(candidate_meta.get("I_max", np.nan))
            - float(reference_meta.get("I_max", np.nan))
        )
        <= tol
        and int(candidate_meta.get("I_pts", -1)) == int(reference_meta.get("I_pts", -1))
    )


def write_missing_I_metadata(
    h5: h5py.File,
    expected_I: np.ndarray,
    missing_I: np.ndarray,
) -> None:
    """Write aggregation completeness metadata for I-td pipeline to an open HDF5 file."""
    missing = np.asarray(missing_I, dtype=np.float64)
    expected = np.asarray(expected_I, dtype=np.float64)
    _write_attrs(h5, {"missing_I_count": int(missing.shape[0])})
    if missing.shape[0] > 0:
        h5.create_dataset("missing_I", data=missing)
    h5.create_dataset("expected_I", data=expected)


def read_missing_I_metadata(h5: h5py.File) -> Dict[str, Any]:
    """Read aggregation completeness metadata for I-td pipeline from an open HDF5 file."""
    missing = (
        np.array(h5["missing_I"], dtype=np.float64)
        if "missing_I" in h5
        else np.array([], dtype=np.float64)
    )
    expected = (
        np.array(h5["expected_I"], dtype=np.float64)
        if "expected_I" in h5
        else np.array([], dtype=np.float64)
    )
    return {
        "missing_I_count": int(h5.attrs.get("missing_I_count", 0)),
        "missing_I": missing,
        "expected_I": expected,
    }


def write_missing_mcz_metadata(
    h5: h5py.File,
    expected_mcz: np.ndarray,
    missing_mcz: np.ndarray,
) -> None:
    """Write aggregation completeness metadata to an open HDF5 file."""
    missing = np.asarray(missing_mcz, dtype=np.float64)
    expected = np.asarray(expected_mcz, dtype=np.float64)
    _write_attrs(h5, {"missing_mcz_count": int(missing.shape[0])})
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


def _decode_string_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_optional_float_attr(attrs: Any, key: str) -> Optional[float]:
    if key not in attrs:
        return None
    value = float(attrs[key])
    if np.isnan(value):
        return None
    return value


def read_best_match_mcz_td_contour_data(
    input_path: str, value_dataset: str
) -> Dict[str, Any]:
    """Load one best-match contour dataset and infer plotting metadata.

    Returns a dict with arrays and metadata needed by contour plotting scripts.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Best-match file not found: {input_path}")

    with h5py.File(input_path, "r") as h5:
        required = ("mcz", "td", value_dataset)
        missing = [name for name in required if name not in h5]
        if missing:
            raise KeyError(
                f"Missing datasets in {input_path}: {missing}. "
                f"Available datasets: {list(h5.keys())}"
            )

        mcz_arr = np.array(h5["mcz"], dtype=np.float64)
        td_arr = np.array(h5["td"], dtype=np.float64)
        values = np.array(h5[value_dataset], dtype=np.float64)

        if mcz_arr.ndim != 1 or td_arr.ndim != 1:
            raise ValueError(
                f"Expected 1D axes in {input_path}, got mcz ndim={mcz_arr.ndim}, td ndim={td_arr.ndim}."
            )
        if mcz_arr.size == 0 or td_arr.size == 0:
            raise ValueError(f"Empty axis dataset found in {input_path}.")

        expected_shape = (int(mcz_arr.shape[0]), int(td_arr.shape[0]))
        if values.shape != expected_shape:
            raise ValueError(
                f"Dataset '{value_dataset}' in {input_path} has shape {values.shape}, "
                f"expected {expected_shape}."
            )

        if "I" not in h5.attrs:
            raise ValueError(f"Missing required attribute 'I' in {input_path}.")
        if "orientation_tag" not in h5.attrs:
            raise ValueError(
                f"Missing required attribute 'orientation_tag' in {input_path}."
            )

        i_raw = np.asarray(h5.attrs["I"])
        if i_raw.size != 1:
            raise ValueError(f"Attribute 'I' must be scalar in {input_path}.")
        I_value = float(i_raw.reshape(-1)[0])
        orientation_tag = _decode_string_attr(h5.attrs["orientation_tag"]).strip()
        if not orientation_tag:
            raise ValueError(f"Attribute 'orientation_tag' is empty in {input_path}.")

        z_value = _read_optional_float_attr(h5.attrs, "z")
        missing_meta = read_missing_mcz_metadata(h5)

    return {
        "mcz": mcz_arr,
        "td": td_arr,
        "values": values,
        "I": I_value,
        "orientation_tag": orientation_tag,
        "z": z_value,
        "mcz_min": float(np.round(np.nanmin(mcz_arr), decimals=10)),
        "mcz_max": float(np.round(np.nanmax(mcz_arr), decimals=10)),
        "mcz_pts": int(mcz_arr.shape[0]),
        "td_min_ms": float(np.round(np.nanmin(td_arr) * 1e3, decimals=10)),
        "td_max_ms": float(np.round(np.nanmax(td_arr) * 1e3, decimals=10)),
        "td_pts": int(td_arr.shape[0]),
        "missing_mcz_count": int(missing_meta["missing_mcz_count"]),
    }


def read_best_match_I_td_contour_data(
    input_path: str, value_dataset: str
) -> Dict[str, Any]:
    """Load one best-match I-td contour dataset and infer plotting metadata.

    Returns a dict with arrays and metadata needed by I-td contour plotting scripts.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Best-match file not found: {input_path}")

    with h5py.File(input_path, "r") as h5:
        required = ("I", "td", value_dataset)
        missing = [name for name in required if name not in h5]
        if missing:
            raise KeyError(
                f"Missing datasets in {input_path}: {missing}. "
                f"Available datasets: {list(h5.keys())}"
            )

        I_arr = np.array(h5["I"], dtype=np.float64)
        td_arr = np.array(h5["td"], dtype=np.float64)
        values = np.array(h5[value_dataset], dtype=np.float64)

        if I_arr.ndim != 1 or td_arr.ndim != 1:
            raise ValueError(
                f"Expected 1D axes in {input_path}, got I ndim={I_arr.ndim}, td ndim={td_arr.ndim}."
            )
        if I_arr.size == 0 or td_arr.size == 0:
            raise ValueError(f"Empty axis dataset found in {input_path}.")

        expected_shape = (int(I_arr.shape[0]), int(td_arr.shape[0]))
        if values.shape != expected_shape:
            raise ValueError(
                f"Dataset '{value_dataset}' in {input_path} has shape {values.shape}, "
                f"expected {expected_shape}."
            )

        # Read mcz (scalar or array with one value)
        if "mcz" not in h5:
            raise ValueError(f"Missing required dataset 'mcz' in {input_path}.")
        mcz_raw = np.asarray(h5["mcz"])
        if mcz_raw.size != 1:
            raise ValueError(f"Dataset 'mcz' must be scalar in {input_path}.")
        mcz_value = float(mcz_raw.reshape(-1)[0])

        if "orientation_tag" not in h5.attrs:
            raise ValueError(
                f"Missing required attribute 'orientation_tag' in {input_path}."
            )

        orientation_tag = _decode_string_attr(h5.attrs["orientation_tag"]).strip()
        if not orientation_tag:
            raise ValueError(f"Attribute 'orientation_tag' is empty in {input_path}.")

        z_value = _read_optional_float_attr(h5.attrs, "z")
        missing_meta = read_missing_I_metadata(h5)

    return {
        "I": I_arr,
        "td": td_arr,
        "values": values,
        "mcz": mcz_value,
        "orientation_tag": orientation_tag,
        "z": z_value,
        "I_min": float(np.round(np.nanmin(I_arr), decimals=10)),
        "I_max": float(np.round(np.nanmax(I_arr), decimals=10)),
        "I_pts": int(I_arr.shape[0]),
        "td_min_ms": float(np.round(np.nanmin(td_arr) * 1e3, decimals=10)),
        "td_max_ms": float(np.round(np.nanmax(td_arr) * 1e3, decimals=10)),
        "td_pts": int(td_arr.shape[0]),
        "missing_I_count": int(missing_meta["missing_I_count"]),
    }


def open_bank_readonly(
    filepath: str,
) -> BankReadonlyPayload:
    """
    Open a bank HDF5 file for read-only access and return handles/arrays.

    Returns (h5_file, omega, theta, gamma, bank_dataset, bank_attrs)
    Caller is responsible for closing the returned h5_file.
    """
    h5 = h5py.File(filepath, "r")
    try:
        omega = np.array(h5["omega"]).astype(float)
        theta = np.array(h5["theta"]).astype(float)
        gamma = np.array(h5["gamma"]).astype(float)
        bank = h5["bank"]
        attrs = dict(bank.attrs.items())
        return h5, omega, theta, gamma, bank, attrs
    except Exception:
        h5.close()
        raise


def safe_open_bank_readonly(
    filepath: str,
) -> Tuple[Optional[BankReadonlyPayload], Optional[str]]:
    """Best-effort wrapper for opening a bank file read-only.

    Returns:
      (payload, None) on success
      (None, error_message) on failure
    """
    try:
        return open_bank_readonly(filepath), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


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


def create_mismatch_mcz_cube(
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
    write_dataset_units(
        h5,
        {
            "mcz": "Msun",
            "td": "s",
            "omega": "dimensionless",
            "theta": "dimensionless",
            "gamma": "rad",
        },
    )

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

    datasets["epsilon_min_grid"].attrs["axis_order"] = "td,theta,omega"
    datasets["gamma_best_grid"].attrs["axis_order"] = "td,theta,omega"
    if "mismatch" in datasets:
        datasets["mismatch"].attrs["axis_order"] = "td,theta,omega,gamma"

    return h5, datasets


def create_mismatch_I_cube(
    filepath: str,
    td_pts: int,
    theta_arr: np.ndarray,
    omega_arr: np.ndarray,
    gamma_arr: np.ndarray,
    I_val: float,
    mcz: float,
    td_arr: np.ndarray,
    save_full_mismatch: bool = False,
):
    """
    Create an HDF5 file with per-I mismatch cube datasets (I-td pipeline).

    Returns a tuple (h5, datasets) where datasets is a dict containing:
      - 'mismatch' (optional): (td, theta, omega, gamma)
      - 'epsilon_min_grid': (td, theta, omega)
      - 'gamma_best_grid': (td, theta, omega)
    Caller is responsible for closing the returned h5 file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    h5 = h5py.File(filepath, "w")
    h5.create_dataset("I", data=np.array([I_val], dtype=np.float64))
    h5.create_dataset("mcz", data=np.array([mcz], dtype=np.float64))
    h5.create_dataset("td", data=td_arr.astype(np.float64))
    h5.create_dataset("omega", data=omega_arr.astype(np.float64))
    h5.create_dataset("theta", data=theta_arr.astype(np.float64))
    h5.create_dataset("gamma", data=gamma_arr.astype(np.float64))
    write_dataset_units(
        h5,
        {
            "I": "dimensionless",
            "mcz": "Msun",
            "td": "s",
            "omega": "dimensionless",
            "theta": "dimensionless",
            "gamma": "rad",
        },
    )

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

    datasets["epsilon_min_grid"].attrs["axis_order"] = "td,theta,omega"
    datasets["gamma_best_grid"].attrs["axis_order"] = "td,theta,omega"
    if "mismatch" in datasets:
        datasets["mismatch"].attrs["axis_order"] = "td,theta,omega,gamma"

    return h5, datasets
