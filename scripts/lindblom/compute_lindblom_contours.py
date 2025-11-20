"""Compute Lindblom criterion contours from mismatch cubes.

This script computes the Lindblom criterion: (mismatch - 1/(2*SNR^2))
where SNR^2 = ⟨h_s|h_t⟩ is the inner product between source and template waveforms.

The Lindblom criterion states that if mismatch >= 1/(2*SNR^2), the source waveform
is distinguishable from the template waveform.

For each point in the mismatch cube, we:
1. Reconstruct the source (lensed) waveform
2. Reconstruct the template (precessing) waveform using best gamma
3. Compute inner product ⟨h_s|h_t⟩ = 4 Re ∫ [h_s(f)h_t*(f) / S_n(f)] df
4. Compute SNR^2 = ⟨h_s|h_t⟩
5. Compute (mismatch - 1/(2*SNR^2))
6. Create contours of this quantity
"""

import os
import sys
import argparse
import h5py
import numpy as np
from scipy.integrate import simpson
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.Classes_v3 import Precessing
from modules.functions_v3 import (
    get_gw,
    get_y_from_I,
    get_MLz_from_td,
    Sn,
    get_fcut_from_mcz,
    timer_decorator,
)
from modules.default_params_v3 import SOLMASS2SEC, lens_params_1, orient_params
from modules.orientation import resolve_orientation, allowed_orient_presets
from modules.filenames import mismatch_cube_filename, bank_filename
from modules.bank_io import open_bank_readonly
from modules.match_utils import ensure_same_length, cast_to_match_precision


def inner_product(
    h_s: np.ndarray,
    h_t: np.ndarray,
    f_array: np.ndarray,
    psd: np.ndarray,
) -> float:
    """
    Compute the inner product between source and template waveforms.

    According to the paper: ⟨h_s|h_t⟩ = 4 Re ∫(from f_min to f_cut) [h_s(f)h_t*(f) / S_n(f)] df

    Parameters
    ----------
    h_s : np.ndarray
        Source waveform (complex frequency domain)
    h_t : np.ndarray
        Template waveform (complex frequency domain)
    f_array : np.ndarray
        Frequency array
    psd : np.ndarray
        Power spectral density array

    Returns
    -------
    float
        Inner product ⟨h_s|h_t⟩
    """
    # Ensure waveforms have same length
    h_t_aligned, h_s_aligned = ensure_same_length(h_t, h_s)

    # Compute h_s * h_t* (complex conjugate of template)
    h_t_conj = np.conj(h_t_aligned)
    product = h_s_aligned * h_t_conj

    # Divide by PSD
    integrand = product / psd

    # Integrate: 4 Re ∫ [h_s(f)h_t*(f) / S_n(f)] df
    integrated = simpson(integrand, x=f_array)
    inner_prod = 4 * np.real(integrated)

    return float(inner_prod)


def compute_lindblom_for_point(
    args: Tuple,
) -> Tuple[int, int, int, float]:
    """
    Compute Lindblom criterion for a single (td_idx, theta_idx, omega_idx) point.

    Parameters
    ----------
    args : Tuple
        (td_idx, theta_idx, omega_idx, td, theta_val, omega_val, gamma_val,
         lens_params, template_base_params, f_min, delta_f, psd, mismatch_val)

    Returns
    -------
    Tuple[int, int, int, float]
        (td_idx, theta_idx, omega_idx, lindblom_value)
    """
    (
        td_idx,
        theta_idx,
        omega_idx,
        td,
        theta_val,
        omega_val,
        gamma_val,
        lens_params,
        template_base_params,
        f_min,
        delta_f,
        psd,
        mismatch_val,
    ) = args

    try:
        # Reconstruct source waveform (lensed)
        lens_params_td = dict(lens_params)
        lens_params_td["MLz"] = float(
            get_MLz_from_td(td, lens_params["y"]) * SOLMASS2SEC
        )
        source_gw = get_gw(lens_params_td, f_min=f_min, delta_f=delta_f)
        h_s = cast_to_match_precision(source_gw["strain"])
        f_array_s = source_gw["f_array"]

        # Reconstruct template waveform (precessing)
        template_params = dict(template_base_params)
        template_params["omega_tilde"] = float(omega_val)
        template_params["theta_tilde"] = float(theta_val)
        template_params["gamma_P"] = float(gamma_val)
        template_gw = get_gw(
            template_params, f_min=f_min, delta_f=delta_f, prec_Class=Precessing
        )
        h_t = cast_to_match_precision(template_gw["strain"])
        f_array_t = template_gw["f_array"]

        # Use the minimum f_cut to ensure both waveforms are on the same grid
        f_cut = min(f_array_s[-1] + delta_f, f_array_t[-1] + delta_f)
        f_array = np.arange(f_min, f_cut, delta_f)

        # Truncate waveforms to common frequency range
        n_freq = len(f_array)
        h_s_trunc = (
            h_s[:n_freq]
            if len(h_s) >= n_freq
            else np.pad(h_s, (0, n_freq - len(h_s)), mode="constant")
        )
        h_t_trunc = (
            h_t[:n_freq]
            if len(h_t) >= n_freq
            else np.pad(h_t, (0, n_freq - len(h_t)), mode="constant")
        )

        # Recompute PSD for the actual frequency array
        psd = Sn(f_array, f_min=f_min, delta_f=delta_f)

        # Compute inner product
        snr_squared = inner_product(h_s_trunc, h_t_trunc, f_array, psd)

        # Compute SNR = sqrt(SNR^2)
        if snr_squared > 0:
            snr_value = np.sqrt(snr_squared)
            # Compute Lindblom criterion: (mismatch - 1/(2*SNR^2))
            lindblom_value = mismatch_val - 1.0 / (2.0 * snr_squared)
        else:
            snr_value = np.nan
            lindblom_value = np.nan

    except Exception as e:
        print(
            f"Error computing Lindblom for (td={td_idx}, theta={theta_idx}, omega={omega_idx}): {e}"
        )
        lindblom_value = np.nan
        snr_value = np.nan

    return td_idx, theta_idx, omega_idx, lindblom_value, snr_value


@timer_decorator
def compute_lindblom_cube(
    cube_path: str,
    bank_dir: str,
    bank_prefix: str,
    f_min: float = 20.0,
    delta_f: float = 0.25,
    n_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Lindblom criterion cube from a mismatch cube.

    Parameters
    ----------
    cube_path : str
        Path to mismatch cube HDF5 file
    bank_dir : str
        Directory containing template banks
    bank_prefix : str
        Prefix for template bank filenames
    f_min : float
        Minimum frequency in Hz
    delta_f : float
        Frequency spacing in Hz
    n_workers : int | None
        Number of parallel workers. If None, uses cpu_count()

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (td_arr, theta_arr, omega_arr, lindblom_cube, snr_cube)
        where lindblom_cube and snr_cube have shape (n_td, n_theta, n_omega)
    """
    # Load mismatch cube
    with h5py.File(cube_path, "r") as h5:
        mcz = float(np.array(h5["mcz"]).item())
        td_arr = np.array(h5["td"], dtype=float)  # seconds
        theta_arr = np.array(h5["theta"], dtype=float)
        omega_arr = np.array(h5["omega"], dtype=float)
        gamma_arr = np.array(h5["gamma"], dtype=float)
        epsilon_min_grid = np.array(
            h5["epsilon_min_grid"], dtype=float
        )  # (td, theta, omega)
        gamma_best_grid = np.array(
            h5["gamma_best_grid"], dtype=float
        )  # (td, theta, omega)

        # Extract source parameters from attributes
        I = float(h5.attrs.get("I", 0.5))
        theta_J = h5.attrs.get("theta_J", np.nan)
        phi_J = h5.attrs.get("phi_J", np.nan)
        theta_S = h5.attrs.get("theta_S", np.nan)
        phi_S = h5.attrs.get("phi_S", np.nan)

        if np.isnan(theta_J):
            # Try to resolve from orientation preset
            orient_preset = h5.attrs.get("orient_preset", None)
            if orient_preset:
                orient_dict = orient_params.get(orient_preset, {}).get("edgeon", {})
                theta_J = orient_dict.get("theta_J", np.pi / 2)
                phi_J = orient_dict.get("phi_J", np.pi / 2)
                theta_S = orient_dict.get("theta_S", np.pi / 4)
                phi_S = orient_dict.get("phi_S", 0.0)

    # Set up source (lensed) parameters
    y = get_y_from_I(I)

    # Resolve orientation parameters
    lens_base, _ = resolve_orientation(
        orient_preset=None,
        theta_J=theta_J if not np.isnan(theta_J) else None,
        phi_J=phi_J if not np.isnan(phi_J) else None,
        theta_S=theta_S if not np.isnan(theta_S) else None,
        phi_S=phi_S if not np.isnan(phi_S) else None,
        base_params=lens_params_1,
        orient_params=orient_params,
    )

    # Set mcz and y for lensed source
    lens_base["mcz"] = float(mcz) * SOLMASS2SEC
    lens_base["y"] = float(y)

    # Set up template (precessing) base parameters
    template_base = dict(lens_base)
    template_base["mcz"] = float(mcz) * SOLMASS2SEC
    template_base["dist"] = lens_base["dist"]
    template_base["eta"] = lens_base["eta"]
    template_base["t_c"] = 0.0
    template_base["phi_c"] = 0.0

    # Find template bank
    # Extract bank parameters from cube filename or use defaults
    # For now, we'll try to infer from the cube structure
    omega_min = float(omega_arr.min())
    omega_max = float(omega_arr.max())
    omega_pts = len(omega_arr)
    theta_min = float(theta_arr.min())
    theta_max = float(theta_arr.max())
    theta_pts = len(theta_arr)
    gamma_pts = len(gamma_arr)

    # Try to get orientation tag from cube path or use default
    orientation_tag = "Taman_edgeon"  # Default, should be extracted from filename

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
        orientation_tag,
        prefix=bank_prefix,
    )

    if not os.path.isfile(bank_path):
        # Try alternative bank filename format (with integer mcz, no Msun)
        bank_path_int = os.path.join(
            bank_dir,
            f"{bank_prefix}_mcz{int(mcz)}_omega{omega_min:.0f}-{omega_max:.0f}_theta{theta_min:.0f}-{theta_max:.0f}_o{omega_pts}-t{theta_pts}-g{gamma_pts}_{orientation_tag}.h5",
        )
        if os.path.isfile(bank_path_int):
            bank_path = bank_path_int
        else:
            raise FileNotFoundError(
                f"Template bank not found. Tried:\n  {bank_path}\n  {bank_path_int}\n"
                f"Please ensure the bank file exists with matching resolution: omega={omega_pts}, theta={theta_pts}, gamma={gamma_pts}"
            )

    # Precompute PSD once for this mcz
    f_cut = get_fcut_from_mcz(mcz, eta=lens_base["eta"])
    f_array_base = np.arange(f_min, f_cut, delta_f)
    psd_base = Sn(f_array_base, f_min=f_min, delta_f=delta_f)

    # Initialize output cubes
    n_td = len(td_arr)
    n_theta = len(theta_arr)
    n_omega = len(omega_arr)
    lindblom_cube = np.full((n_td, n_theta, n_omega), np.nan, dtype=np.float32)
    snr_cube = np.full((n_td, n_theta, n_omega), np.nan, dtype=np.float32)

    # Prepare jobs
    jobs = []
    for td_idx, td in enumerate(td_arr):
        for theta_idx, theta_val in enumerate(theta_arr):
            for omega_idx, omega_val in enumerate(omega_arr):
                gamma_val = gamma_best_grid[td_idx, theta_idx, omega_idx]
                mismatch_val = epsilon_min_grid[td_idx, theta_idx, omega_idx]

                jobs.append(
                    (
                        td_idx,
                        theta_idx,
                        omega_idx,
                        td,
                        theta_val,
                        omega_val,
                        gamma_val,
                        lens_base,
                        template_base,
                        f_min,
                        delta_f,
                        psd_base,
                        mismatch_val,
                    )
                )

    # Compute in parallel
    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))

    print(
        f"Computing Lindblom criterion for {len(jobs)} points using {n_workers} workers..."
    )

    with Pool(n_workers) as pool:
        results = pool.map(compute_lindblom_for_point, jobs)

    # Fill output cubes
    for result in results:
        td_idx, theta_idx, omega_idx, lindblom_val, snr_val = result
        lindblom_cube[td_idx, theta_idx, omega_idx] = lindblom_val
        snr_cube[td_idx, theta_idx, omega_idx] = snr_val

    return td_arr, theta_arr, omega_arr, lindblom_cube, snr_cube


def save_lindblom_cube(
    output_path: str,
    td_arr: np.ndarray,
    theta_arr: np.ndarray,
    omega_arr: np.ndarray,
    lindblom_cube: np.ndarray,
    snr_cube: np.ndarray,
    source_attrs: dict,
):
    """Save Lindblom criterion and SNR cubes to HDF5 file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("td", data=td_arr.astype(np.float64))
        h5.create_dataset("theta", data=theta_arr.astype(np.float64))
        h5.create_dataset("omega", data=omega_arr.astype(np.float64))
        h5.create_dataset("lindblom_cube", data=lindblom_cube.astype(np.float32))
        h5.create_dataset("snr_cube", data=snr_cube.astype(np.float32))

        # Copy source attributes
        for key, value in source_attrs.items():
            h5.attrs[key] = value


def main():
    parser = argparse.ArgumentParser(
        description="Compute Lindblom criterion contours from mismatch cubes"
    )
    parser.add_argument(
        "--cube_path",
        type=str,
        required=True,
        help="Path to mismatch cube HDF5 file",
    )
    parser.add_argument(
        "--bank_dir",
        type=str,
        required=True,
        help="Directory containing template banks",
    )
    parser.add_argument(
        "--bank_prefix",
        type=str,
        default="bank",
        help="Prefix for template bank filenames (default: bank)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for Lindblom cube. If None, auto-generated.",
    )
    parser.add_argument(
        "--f_min",
        type=float,
        default=20.0,
        help="Minimum frequency in Hz (default: 20.0)",
    )
    parser.add_argument(
        "--delta_f",
        type=float,
        default=0.25,
        help="Frequency spacing in Hz (default: 0.25)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count())",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.cube_path):
        raise FileNotFoundError(f"Mismatch cube not found: {args.cube_path}")

    # Compute Lindblom cube and SNR cube
    td_arr, theta_arr, omega_arr, lindblom_cube, snr_cube = compute_lindblom_cube(
        cube_path=args.cube_path,
        bank_dir=args.bank_dir,
        bank_prefix=args.bank_prefix,
        f_min=args.f_min,
        delta_f=args.delta_f,
        n_workers=args.n_workers,
    )

    # Determine output path
    if args.output_path is None:
        base_dir = os.path.dirname(args.cube_path)
        base_name = os.path.splitext(os.path.basename(args.cube_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_lindblom.h5")
    else:
        output_path = args.output_path

    # Load source attributes from original cube
    with h5py.File(args.cube_path, "r") as h5:
        source_attrs = dict(h5.attrs.items())

    # Save Lindblom cube and SNR cube
    save_lindblom_cube(
        output_path=output_path,
        td_arr=td_arr,
        theta_arr=theta_arr,
        omega_arr=omega_arr,
        lindblom_cube=lindblom_cube,
        snr_cube=snr_cube,
        source_attrs=source_attrs,
    )

    print(f"Lindblom and SNR cubes saved to: {output_path}")
    print(f"Cube shape: {lindblom_cube.shape}")
    print(
        f"Lindblom value range: {np.nanmin(lindblom_cube):.6f} to {np.nanmax(lindblom_cube):.6f}"
    )
    print(f"SNR value range: {np.nanmin(snr_cube):.6f} to {np.nanmax(snr_cube):.6f}")


if __name__ == "__main__":
    main()
