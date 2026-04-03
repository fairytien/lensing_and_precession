"""Convert an aggregated best-match HDF5 file between redshifts.

This remaps source-mass axes from z_from to z_to while preserving the
detector-frame mismatch surface values.
"""

import argparse
import os
import re
from typing import Optional

import h5py
import numpy as np

from modules.cosmology import source_mass_redshift_scale, z_to_DL
from modules.default_params_v3 import GIGAPC2SEC
from modules.filenames import best_match_mcz_td_filename


def _token_to_float(token: str) -> float:
    return float(str(token).replace("p", "."))


def _to_filename_token(value: float) -> str:
    s = str(float(value))
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _clean_endpoint(value: float, tol: float = 1e-6) -> float:
    nearest_int = round(float(value))
    if abs(float(value) - float(nearest_int)) <= tol:
        return float(nearest_int)
    return float(np.round(value, 8))


def _parse_template_grid_tokens(path: str):
    base = os.path.basename(path)
    match = re.search(
        r"_omega([^_]+)-([^x_]+)x(\d+)_theta([^_]+)-([^x_]+)x(\d+)_gamma0-2pix(\d+)_",
        base,
    )
    if not match:
        return None
    return {
        "omega_min": _token_to_float(match.group(1)),
        "omega_max": _token_to_float(match.group(2)),
        "omega_pts": int(match.group(3)),
        "theta_min": _token_to_float(match.group(4)),
        "theta_max": _token_to_float(match.group(5)),
        "theta_pts": int(match.group(6)),
        "gamma_pts": int(match.group(7)),
    }


def _default_output_path(
    input_path: str,
    attrs,
    mcz_new: np.ndarray,
    td: np.ndarray,
    z_to: float,
) -> str:
    # results_dir is parent of "best_match"
    results_dir = os.path.dirname(os.path.dirname(input_path))
    orientation_tag = str(attrs.get("orientation_tag", "orientation"))
    i_value = float(attrs["I"])
    grid_tokens = _parse_template_grid_tokens(input_path) or {}

    return best_match_mcz_td_filename(
        results_dir=results_dir,
        I=i_value,
        mcz_min=_clean_endpoint(float(np.nanmin(mcz_new))),
        mcz_max=_clean_endpoint(float(np.nanmax(mcz_new))),
        mcz_pts=int(mcz_new.shape[0]),
        td_min_ms=float(np.nanmin(td) * 1e3),
        td_max_ms=float(np.nanmax(td) * 1e3),
        td_pts=int(td.shape[0]),
        omega_min=grid_tokens.get("omega_min"),
        omega_max=grid_tokens.get("omega_max"),
        omega_pts=grid_tokens.get("omega_pts"),
        theta_min=grid_tokens.get("theta_min"),
        theta_max=grid_tokens.get("theta_max"),
        theta_pts=grid_tokens.get("theta_pts"),
        gamma_pts=grid_tokens.get("gamma_pts"),
        orientation_tag=orientation_tag,
        z=z_to,
    )


def convert_best_match(
    input_path: str,
    output_path: Optional[str],
    z_from: Optional[float],
    z_to: float,
    overwrite: bool,
) -> str:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    z_to = float(z_to)

    with h5py.File(input_path, "r") as src:
        if "mcz" not in src or "td" not in src:
            raise KeyError("Input file must contain datasets 'mcz' and 'td'")

        attrs = src.attrs
        z_from_val = float(attrs["z"]) if z_from is None else float(z_from)

        mcz_scale = source_mass_redshift_scale(z_from_val, z_to)
        mcz_new = np.asarray(src["mcz"][:], dtype=np.float64) * mcz_scale
        td = np.asarray(src["td"][:], dtype=np.float64)

        if output_path is None:
            output_path = _default_output_path(input_path, attrs, mcz_new, td, z_to)

        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. Use --overwrite to replace it."
            )

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with h5py.File(output_path, "w") as dst:
            for key, value in attrs.items():
                dst.attrs[key] = value

            for name in src.keys():
                src.copy(name, dst, name=name)

            for dset_name in ("mcz", "expected_mcz", "missing_mcz"):
                if dset_name in dst:
                    dst[dset_name][...] = (
                        np.asarray(dst[dset_name][:], dtype=np.float64) * mcz_scale
                    )

            dst.attrs["z"] = z_to
            if "source_param_z" in dst.attrs:
                dst.attrs["source_param_z"] = z_to

            dist_to = float(z_to_DL(z_to) * GIGAPC2SEC)
            for key in ("source_param_dist", "template_param_dist"):
                if key in dst.attrs:
                    dst.attrs[key] = dist_to

            if "source_param_mcz_source_msun" in dst.attrs:
                dst.attrs["source_param_mcz_source_msun"] = (
                    float(dst.attrs["source_param_mcz_source_msun"]) * mcz_scale
                )

            dst.attrs["redshift_conversion_from_z"] = z_from_val
            dst.attrs["redshift_conversion_to_z"] = z_to
            dst.attrs["redshift_conversion_mcz_source_scale"] = mcz_scale

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an aggregated mcz_td best-match HDF5 file from one redshift to another. "
            "Source-mass axes are rescaled by (1+z_from)/(1+z_to)."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input best_match HDF5")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to output HDF5. If omitted, a canonical best_match filename is built "
            "in the same run directory."
        ),
    )
    parser.add_argument(
        "--z_from",
        type=float,
        default=None,
        help="Source file redshift. If omitted, read from input file attribute 'z'.",
    )
    parser.add_argument("--z_to", type=float, default=1.0, help="Target redshift")
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing existing output file"
    )
    args = parser.parse_args()

    out_path = convert_best_match(
        input_path=args.input,
        output_path=args.output,
        z_from=args.z_from,
        z_to=args.z_to,
        overwrite=args.overwrite,
    )

    z_from_text = "attr:z" if args.z_from is None else _to_filename_token(args.z_from)
    print(f"Saved converted file: {out_path}")
    print(f"z_from={z_from_text}, z_to={_to_filename_token(args.z_to)}")


if __name__ == "__main__":
    main()
