"""HDF5 I/O helpers for template banks and pipeline mismatch cubes.

Sections:
- Shared metadata and dataset helpers
- Template-bank readers/writers
- `mcz_td` pipeline helpers
- `I_td` pipeline helpers

Provides:
- open_bank_readonly(path) -> (h5, omega, theta, gamma, bank, attrs)
- safe_open_bank_readonly(path) -> (payload_or_none, error_message_or_none)
- create_bank_writer(path, shape, dtype, chunking, dset_attrs)
- create_mcz_mismatch_cube(...)
- create_I_mismatch_cube(...)

Design goals: streaming-friendly writes, gzip compression, shuffle filter, and
fletcher32 checksums for robustness.
"""

import os
from contextlib import contextmanager
from typing import Tuple, Dict, Any, Optional, cast

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


_COMPRESSED_DATASET_KWARGS = {
    "compression": "gzip",
    "compression_opts": 4,
    "shuffle": True,
    "fletcher32": True,
}
_MISMATCH_CHUNK_LIMIT = 16


# ==============================================================================
# Shared Metadata and Dataset Helpers
# ==============================================================================


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


def _read_named_attrs(attrs: Any, keys: Tuple[str, ...]) -> Dict[str, Any]:
    return {key: attrs[key] for key in keys if key in attrs}


def _get_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    obj = h5[name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected '{name}' to be an HDF5 dataset.")
    return cast(h5py.Dataset, obj)


def _write_grid_attrs(
    h5: h5py.File,
    axis_name: str,
    axis_min: float,
    axis_max: float,
    axis_pts: int,
) -> None:
    _write_attrs(
        h5,
        {
            f"{axis_name}_min": float(axis_min),
            f"{axis_name}_max": float(axis_max),
            f"{axis_name}_pts": int(axis_pts),
        },
    )


def _read_grid_attrs(h5: h5py.File, axis_name: str) -> Dict[str, Any]:
    return _read_named_attrs(
        h5.attrs,
        (f"{axis_name}_min", f"{axis_name}_max", f"{axis_name}_pts"),
    )


def _grid_meta_consistent(
    reference_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    axis_name: str,
    tol: float = 1e-6,
) -> bool:
    if not reference_meta or not candidate_meta:
        return True
    return (
        abs(
            float(candidate_meta.get(f"{axis_name}_min", np.nan))
            - float(reference_meta.get(f"{axis_name}_min", np.nan))
        )
        <= tol
        and abs(
            float(candidate_meta.get(f"{axis_name}_max", np.nan))
            - float(reference_meta.get(f"{axis_name}_max", np.nan))
        )
        <= tol
        and int(candidate_meta.get(f"{axis_name}_pts", -1))
        == int(reference_meta.get(f"{axis_name}_pts", -1))
    )


def _write_missing_axis_metadata(
    h5: h5py.File,
    axis_name: str,
    expected_values: np.ndarray,
    missing_values: np.ndarray,
) -> None:
    missing = np.asarray(missing_values, dtype=np.float64)
    expected = np.asarray(expected_values, dtype=np.float64)
    _write_attrs(h5, {f"missing_{axis_name}_count": int(missing.shape[0])})
    if missing.shape[0] > 0:
        h5.create_dataset(f"missing_{axis_name}", data=missing)
    h5.create_dataset(f"expected_{axis_name}", data=expected)


def _read_missing_axis_metadata(
    h5: h5py.File,
    axis_name: str,
    *,
    expected_default: Any,
) -> Dict[str, Any]:
    missing_name = f"missing_{axis_name}"
    expected_name = f"expected_{axis_name}"
    missing = (
        np.array(h5[missing_name], dtype=np.float64)
        if missing_name in h5
        else np.array([], dtype=np.float64)
    )
    if expected_name in h5:
        expected = np.array(h5[expected_name], dtype=np.float64)
    elif expected_default is None:
        expected = None
    else:
        expected = np.array(expected_default, dtype=np.float64)
    return {
        f"missing_{axis_name}_count": int(
            h5.attrs.get(f"missing_{axis_name}_count", missing.shape[0])
        ),
        missing_name: missing,
        expected_name: expected,
    }


def write_orientation_attr(h5: h5py.File, orientation_tag: str) -> None:
    """Write orientation tag as file metadata."""
    _write_attrs(h5, {"orientation_tag": str(orientation_tag)})


def write_scalar_attr_with_unit(
    h5: h5py.File,
    key: str,
    value: Any,
    *,
    unit: Optional[str] = None,
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
    attrs = _read_named_attrs(h5.attrs, ("I", "theta_J", "phi_J", "theta_S", "phi_S"))
    # Propagate orientation and any source/template parameter snapshots.
    if "orientation_tag" in h5.attrs:
        attrs["orientation_tag"] = h5.attrs["orientation_tag"]
    for key in h5.attrs.keys():
        if str(key).startswith("source_param_") or str(key).startswith(
            "template_param_"
        ):
            attrs[key] = h5.attrs[key]
    return attrs


def read_mismatch_cube_shape(h5: h5py.File) -> Tuple[int, int, int, int]:
    """Return axis sizes for a mismatch cube as (td, theta, omega, gamma)."""
    return (
        int(_get_dataset(h5, "td").shape[0]),
        int(_get_dataset(h5, "theta").shape[0]),
        int(_get_dataset(h5, "omega").shape[0]),
        int(_get_dataset(h5, "gamma").shape[0]),
    )


# ==============================================================================
# `mcz_td` Pipeline Helpers
# ==============================================================================


def write_mcz_td_grid_attrs(
    h5: h5py.File,
    mcz_min: float,
    mcz_max: float,
    mcz_pts: int,
) -> None:
    """Write intended Stage 1 mcz grid metadata for the `mcz_td` pipeline."""
    _write_grid_attrs(h5, "mcz", mcz_min, mcz_max, mcz_pts)


def read_mcz_td_grid_attrs(h5: h5py.File) -> Dict[str, Any]:
    """Read `mcz_td` mcz grid metadata from an open HDF5 file if present."""
    return _read_grid_attrs(h5, "mcz")


def mcz_td_grid_meta_consistent(
    reference_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    tol: float = 1e-6,
) -> bool:
    """Return True when two `mcz_td` mcz grid metadata dicts match."""
    return _grid_meta_consistent(reference_meta, candidate_meta, "mcz", tol=tol)


def write_missing_mcz_td_metadata(
    h5: h5py.File,
    expected_mcz: np.ndarray,
    missing_mcz: np.ndarray,
) -> None:
    """Write aggregation completeness metadata for the `mcz_td` pipeline."""
    _write_missing_axis_metadata(h5, "mcz", expected_mcz, missing_mcz)


def read_missing_mcz_td_metadata(h5: h5py.File) -> Dict[str, Any]:
    """Read aggregation completeness metadata for the `mcz_td` pipeline."""
    return _read_missing_axis_metadata(h5, "mcz", expected_default=None)


# ==============================================================================
# `I_td` Pipeline Helpers
# ==============================================================================


def write_I_td_grid_attrs(
    h5: h5py.File,
    I_min: float,
    I_max: float,
    I_pts: int,
) -> None:
    """Write intended Stage 1 I grid metadata for the `I_td` pipeline."""
    _write_grid_attrs(h5, "I", I_min, I_max, I_pts)


def read_I_td_grid_attrs(h5: h5py.File) -> Dict[str, Any]:
    """Read `I_td` I grid metadata from an open HDF5 file if present."""
    return _read_grid_attrs(h5, "I")


def I_td_grid_meta_consistent(
    reference_meta: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    tol: float = 1e-6,
) -> bool:
    """Return True when two `I_td` I grid metadata dicts match."""
    return _grid_meta_consistent(reference_meta, candidate_meta, "I", tol=tol)


def write_missing_I_td_metadata(
    h5: h5py.File,
    expected_I: np.ndarray,
    missing_I: np.ndarray,
) -> None:
    """Write aggregation completeness metadata for the `I_td` pipeline."""
    _write_missing_axis_metadata(h5, "I", expected_I, missing_I)


def read_missing_I_td_metadata(h5: h5py.File) -> Dict[str, Any]:
    """Read aggregation completeness metadata for the `I_td` pipeline."""
    return _read_missing_axis_metadata(
        h5,
        "I",
        expected_default=np.array([], dtype=np.float64),
    )


# ==============================================================================
# Best-Match Contour Readers
# ==============================================================================


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


def _require_datasets(h5: h5py.File, input_path: str, names: Tuple[str, ...]) -> None:
    missing = [name for name in names if name not in h5]
    if missing:
        raise KeyError(
            f"Missing datasets in {input_path}: {missing}. "
            f"Available datasets: {list(h5.keys())}"
        )


def _read_best_match_axes_and_values(
    h5: h5py.File,
    input_path: str,
    axis_name: str,
    value_dataset: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_datasets(h5, input_path, (axis_name, "td", value_dataset))

    axis_arr = np.array(h5[axis_name], dtype=np.float64)
    td_arr = np.array(h5["td"], dtype=np.float64)
    values = np.array(h5[value_dataset], dtype=np.float64)

    if axis_arr.ndim != 1 or td_arr.ndim != 1:
        raise ValueError(
            f"Expected 1D axes in {input_path}, got {axis_name} ndim={axis_arr.ndim}, td ndim={td_arr.ndim}."
        )
    if axis_arr.size == 0 or td_arr.size == 0:
        raise ValueError(f"Empty axis dataset found in {input_path}.")

    expected_shape = (int(axis_arr.shape[0]), int(td_arr.shape[0]))
    if values.shape != expected_shape:
        raise ValueError(
            f"Dataset '{value_dataset}' in {input_path} has shape {values.shape}, "
            f"expected {expected_shape}."
        )
    return axis_arr, td_arr, values


def _read_required_scalar_attr(h5: h5py.File, input_path: str, key: str) -> float:
    if key not in h5.attrs:
        raise ValueError(f"Missing required attribute '{key}' in {input_path}.")
    raw = np.asarray(h5.attrs[key])
    if raw.size != 1:
        raise ValueError(f"Attribute '{key}' must be scalar in {input_path}.")
    return float(raw.reshape(-1)[0])


def _read_required_scalar_dataset(h5: h5py.File, input_path: str, key: str) -> float:
    if key not in h5:
        raise ValueError(f"Missing required dataset '{key}' in {input_path}.")
    raw = np.asarray(h5[key])
    if raw.size != 1:
        raise ValueError(f"Dataset '{key}' must be scalar in {input_path}.")
    return float(raw.reshape(-1)[0])


def _read_required_orientation_tag(h5: h5py.File, input_path: str) -> str:
    if "orientation_tag" not in h5.attrs:
        raise ValueError(
            f"Missing required attribute 'orientation_tag' in {input_path}."
        )
    orientation_tag = _decode_string_attr(h5.attrs["orientation_tag"]).strip()
    if not orientation_tag:
        raise ValueError(f"Attribute 'orientation_tag' is empty in {input_path}.")
    return orientation_tag


def read_best_match_mcz_td_contour_data(
    input_path: str, value_dataset: str
) -> Dict[str, Any]:
    """Load one best-match contour dataset and infer plotting metadata.

    Returns a dict with arrays and metadata needed by contour plotting scripts.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Best-match file not found: {input_path}")

    with h5py.File(input_path, "r") as h5:
        mcz_arr, td_arr, values = _read_best_match_axes_and_values(
            h5,
            input_path,
            "mcz",
            value_dataset,
        )
        I_value = _read_required_scalar_attr(h5, input_path, "I")
        orientation_tag = _read_required_orientation_tag(h5, input_path)
        z_value = _read_optional_float_attr(h5.attrs, "z")
        missing_meta = read_missing_mcz_td_metadata(h5)

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
        I_arr, td_arr, values = _read_best_match_axes_and_values(
            h5,
            input_path,
            "I",
            value_dataset,
        )
        mcz_value = _read_required_scalar_dataset(h5, input_path, "mcz")
        orientation_tag = _read_required_orientation_tag(h5, input_path)
        z_value = _read_optional_float_attr(h5.attrs, "z")
        missing_meta = read_missing_I_td_metadata(h5)

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
        omega = np.asarray(_get_dataset(h5, "omega"), dtype=float)
        theta = np.asarray(_get_dataset(h5, "theta"), dtype=float)
        gamma = np.asarray(_get_dataset(h5, "gamma"), dtype=float)
        bank = _get_dataset(h5, "bank")
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


# ==============================================================================
# Template-Bank Readers/Writers
# ==============================================================================


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
            **_COMPRESSED_DATASET_KWARGS,
        )
        for k, v in dset_attrs.items():
            bank.attrs[k] = v
        yield h5, bank
    finally:
        h5.close()


# ==============================================================================
# Shared Mismatch-Cube Builders
# ==============================================================================


def _create_compressed_dataset(
    h5: h5py.File,
    name: str,
    shape: Tuple[int, ...],
    chunks: Tuple[int, ...],
) -> h5py.Dataset:
    return h5.create_dataset(
        name,
        shape=shape,
        dtype=np.float32,
        chunks=chunks,
        **_COMPRESSED_DATASET_KWARGS,
    )


def _create_mismatch_cube_datasets(
    h5: h5py.File,
    td_pts: int,
    n_theta: int,
    n_omega: int,
    n_gamma: int,
    save_full_mismatch: bool,
) -> Dict[str, h5py.Dataset]:
    theta_chunk = min(_MISMATCH_CHUNK_LIMIT, n_theta)
    omega_chunk = min(_MISMATCH_CHUNK_LIMIT, n_omega)
    grid_shape = (int(td_pts), n_theta, n_omega)
    grid_chunks = (1, theta_chunk, omega_chunk)

    datasets: Dict[str, h5py.Dataset] = {}
    if save_full_mismatch:
        datasets["mismatch"] = _create_compressed_dataset(
            h5,
            "mismatch",
            shape=(int(td_pts), n_theta, n_omega, n_gamma),
            chunks=(1, theta_chunk, omega_chunk, n_gamma),
        )

    for name in ("epsilon_min_grid", "gamma_best_grid"):
        datasets[name] = _create_compressed_dataset(
            h5,
            name,
            shape=grid_shape,
            chunks=grid_chunks,
        )

    datasets["epsilon_min_grid"].attrs["axis_order"] = "td,theta,omega"
    datasets["gamma_best_grid"].attrs["axis_order"] = "td,theta,omega"
    if "mismatch" in datasets:
        datasets["mismatch"].attrs["axis_order"] = "td,theta,omega,gamma"
    return datasets


def _create_mismatch_cube_file(
    filepath: str,
    td_pts: int,
    theta_arr: np.ndarray,
    omega_arr: np.ndarray,
    gamma_arr: np.ndarray,
    td_arr: np.ndarray,
    scalar_datasets: Dict[str, float],
    dataset_units: Dict[str, str],
    save_full_mismatch: bool,
) -> Tuple[h5py.File, Dict[str, h5py.Dataset]]:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    h5 = h5py.File(filepath, "w")
    for name, value in scalar_datasets.items():
        h5.create_dataset(name, data=np.array([value], dtype=np.float64))
    h5.create_dataset("td", data=np.asarray(td_arr, dtype=np.float64))
    h5.create_dataset("omega", data=np.asarray(omega_arr, dtype=np.float64))
    h5.create_dataset("theta", data=np.asarray(theta_arr, dtype=np.float64))
    h5.create_dataset("gamma", data=np.asarray(gamma_arr, dtype=np.float64))
    write_dataset_units(h5, dataset_units)
    datasets = _create_mismatch_cube_datasets(
        h5,
        td_pts=int(td_pts),
        n_theta=int(theta_arr.shape[0]),
        n_omega=int(omega_arr.shape[0]),
        n_gamma=int(gamma_arr.shape[0]),
        save_full_mismatch=save_full_mismatch,
    )
    return h5, datasets


# ==============================================================================
# Artifact Mismatch-Cube Writers
# ==============================================================================


def create_mcz_mismatch_cube(
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
    return _create_mismatch_cube_file(
        filepath,
        td_pts,
        theta_arr,
        omega_arr,
        gamma_arr,
        td_arr,
        scalar_datasets={"mcz": mcz},
        dataset_units={
            "mcz": "Msun",
            "td": "s",
            "omega": "dimensionless",
            "theta": "dimensionless",
            "gamma": "rad",
        },
        save_full_mismatch=save_full_mismatch,
    )


def create_I_mismatch_cube(
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
    Create an HDF5 file with per-I mismatch cube datasets.

    Returns a tuple (h5, datasets) where datasets is a dict containing:
      - 'mismatch' (optional): (td, theta, omega, gamma)
      - 'epsilon_min_grid': (td, theta, omega)
      - 'gamma_best_grid': (td, theta, omega)
    Caller is responsible for closing the returned h5 file.
    """
    return _create_mismatch_cube_file(
        filepath,
        td_pts,
        theta_arr,
        omega_arr,
        gamma_arr,
        td_arr,
        scalar_datasets={"I": I_val, "mcz": mcz},
        dataset_units={
            "I": "dimensionless",
            "mcz": "Msun",
            "td": "s",
            "omega": "dimensionless",
            "theta": "dimensionless",
            "gamma": "rad",
        },
        save_full_mismatch=save_full_mismatch,
    )


# ==============================================================================
# Backward-Compatible Aliases
# ==============================================================================


write_mcz_grid_attrs = write_mcz_td_grid_attrs
read_mcz_grid_attrs = read_mcz_td_grid_attrs
mcz_grid_meta_consistent = mcz_td_grid_meta_consistent
write_missing_mcz_metadata = write_missing_mcz_td_metadata
read_missing_mcz_metadata = read_missing_mcz_td_metadata
create_mcz_td_mismatch_cube = create_mcz_mismatch_cube
create_mismatch_mcz_cube = create_mcz_mismatch_cube

write_I_grid_attrs = write_I_td_grid_attrs
read_I_grid_attrs = read_I_td_grid_attrs
I_grid_meta_consistent = I_td_grid_meta_consistent
write_missing_I_metadata = write_missing_I_td_metadata
read_missing_I_metadata = read_missing_I_td_metadata
create_I_td_mismatch_cube = create_I_mismatch_cube
create_mismatch_I_cube = create_I_mismatch_cube
