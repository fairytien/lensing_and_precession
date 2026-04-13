"""Compute Lindblom criterion from pickle file with mismatch data.

This script loads a pickle file containing mismatch data between lensed sources
and non-precessing templates, computes SNR for each point, and calculates the
Lindblom criterion: mismatch - 1/(2*SNR^2).
"""

import os, sys, argparse, pickle
import numpy as np
from scipy.integrate import simpson

# Ensure project root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.functions import (
    get_gw,
    get_y_from_I,
    get_MLz_from_td,
    Sn,
    get_fcut_from_mcz,
    timer_decorator,
)
from modules.default_params import SOLMASS2SEC, lens_params_1, orient_params
from modules.orientation import resolve_orientation
from pycbc.types import FrequencySeries

import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
        Source strain (complex frequency series)
    h_t : np.ndarray
        Template strain (complex frequency series)
    f_array : np.ndarray
        Frequency array
    psd : np.ndarray
        Power spectral density array

    Returns
    -------
    float
        The inner product value.
    """
    # Ensure waveforms have same length
    min_len = min(len(h_s), len(h_t), len(psd))
    h_s = h_s[:min_len]
    h_t = h_t[:min_len]
    psd = psd[:min_len]
    f_array = f_array[:min_len]

    # Compute h_s * h_t* (complex conjugate of template)
    h_t_conj = np.conj(h_t)
    product = h_s * h_t_conj

    # Divide by PSD
    integrand = product / psd

    # Integrate: 4 Re ∫ [h_s(f)h_t*(f) / S_n(f)] df
    integrated = simpson(integrand, x=f_array)
    inner_prod = 4 * np.real(integrated)

    return float(inner_prod)


def compute_lindblom_for_point(args):
    """Compute Lindblom value and SNR for a single (mcz, td) point."""
    (
        mcz_idx,
        td_idx,
        mcz,
        td,
        I,
        mismatch_val,
        f_min,
        delta_f,
        lens_base,
        template_base,
    ) = args

    try:
        # Reconstruct source waveform (lensed)
        lens_params_td = dict(lens_base)
        lens_params_td["MLz"] = float(get_MLz_from_td(td, lens_base["y"]) * SOLMASS2SEC)
        source_gw = get_gw(lens_params_td, f_min=f_min, delta_f=delta_f)
        h_s = np.asarray(source_gw["strain"])
        f_array_s = source_gw["f_array"]

        # Reconstruct template waveform (non-precessing)
        template_gw = get_gw(template_base, f_min=f_min, delta_f=delta_f)
        h_t = np.asarray(template_gw["strain"])
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

        return mcz_idx, td_idx, lindblom_value, snr_value

    except Exception as e:
        logging.warning(
            f"Error computing Lindblom for mcz={mcz:.1f}, td={td*1e3:.1f}ms: {e}"
        )
        return mcz_idx, td_idx, np.nan, np.nan


@timer_decorator
def main(
    pickle_path: str,
    output_dir: str,
    f_min: float = 20.0,
    delta_f: float = 0.25,
    n_workers: int = None,
):
    """Compute Lindblom criterion from pickle file.

    Parameters
    ----------
    pickle_path : str
        Path to pickle file with mismatch data
    output_dir : str
        Directory to save output files
    f_min : float
        Minimum frequency in Hz
    delta_f : float
        Frequency spacing in Hz
    n_workers : int | None
        Number of parallel workers. If None, uses cpu_count()
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load pickle file
    logging.info(f"Loading mismatch data from: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    mcz_arr = np.array(data["mcz_arr"])
    td_arr = np.array(data["td_arr"])  # in seconds
    epsilon_matrix = np.array(data["epsilon_matrix"])  # (mcz, td)
    I = float(data["I"])
    location = data.get("location", "Taman_edgeon")

    logging.info(f"Data shape: mcz={len(mcz_arr)}, td={len(td_arr)}")
    logging.info(f"I={I}, location={location}")

    # Parse location to get orientation.
    # Canonical format is "Taman_edgeon"; legacy dot separators are normalized.
    location_norm = str(location).replace(".", "_")
    if "_" in location_norm:
        location_parts = location_norm.split("_", 1)
        location_name = location_parts[0]
        orientation_name = location_parts[1] if len(location_parts) > 1 else "edgeon"
    else:
        location_name = "Taman"
        orientation_name = "edgeon"

    # Get orientation parameters
    if (
        location_name in orient_params
        and orientation_name in orient_params[location_name]
    ):
        orient_dict = orient_params[location_name][orientation_name]
    else:
        logging.warning(f"Unknown location/orientation: {location}, using Taman_edgeon")
        orient_dict = orient_params["Taman"]["edgeon"]

    # Set up source (lensed) parameters
    y = get_y_from_I(I)
    from modules.functions import set_orientation
    from modules.default_params import NP_params_1

    lens_base = set_orientation(orient_dict, lens_params_1)[0]
    lens_base["y"] = float(y)

    # Set up template (non-precessing) base parameters
    # Use NP_params_1 and set orientation to match lens_base
    template_base = set_orientation(orient_dict, NP_params_1)[0]
    # Ensure precession parameters are zero for non-precessing
    template_base["theta_tilde"] = 0.0
    template_base["omega_tilde"] = 0.0
    template_base["gamma_P"] = 0.0
    template_base["t_c"] = 0.0
    template_base["phi_c"] = 0.0
    # Remove any lensing parameters that might have been copied
    template_base.pop("MLz", None)
    template_base.pop("y", None)

    # Initialize output arrays
    n_mcz = len(mcz_arr)
    n_td = len(td_arr)
    lindblom_matrix = np.full((n_mcz, n_td), np.nan, dtype=np.float32)
    snr_matrix = np.full((n_mcz, n_td), np.nan, dtype=np.float32)

    # Prepare jobs
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = min(cpu_count(), n_mcz * n_td)

    jobs = []
    for mcz_idx, mcz in enumerate(mcz_arr):
        # Set mcz for both source and template
        lens_params_mcz = dict(lens_base)
        lens_params_mcz["mcz"] = float(mcz) * SOLMASS2SEC
        template_params_mcz = dict(template_base)
        template_params_mcz["mcz"] = float(mcz) * SOLMASS2SEC

        for td_idx, td in enumerate(td_arr):
            mismatch_val = epsilon_matrix[mcz_idx, td_idx]
            jobs.append(
                (
                    mcz_idx,
                    td_idx,
                    mcz,
                    td,
                    I,
                    mismatch_val,
                    f_min,
                    delta_f,
                    lens_params_mcz,
                    template_params_mcz,
                )
            )

    # Compute in parallel
    logging.info(f"Computing Lindblom values using {n_workers} workers...")
    with Pool(n_workers) as pool:
        results = pool.map(compute_lindblom_for_point, jobs)

    # Fill output arrays
    for mcz_idx, td_idx, lindblom_val, snr_val in results:
        lindblom_matrix[mcz_idx, td_idx] = lindblom_val
        snr_matrix[mcz_idx, td_idx] = snr_val

    # Save results
    output_data = {
        "mcz_arr": mcz_arr,
        "td_arr": td_arr,
        "lindblom_matrix": lindblom_matrix,
        "snr_matrix": snr_matrix,
        "epsilon_matrix": epsilon_matrix,
        "I": I,
        "location": location,
        "template": "NP",
    }

    output_path = os.path.join(output_dir, "lindblom.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)
    logging.info(f"Results saved to: {output_path}")

    # Print statistics
    valid_lindblom = lindblom_matrix[~np.isnan(lindblom_matrix)]
    valid_snr = snr_matrix[~np.isnan(snr_matrix)]
    if len(valid_lindblom) > 0:
        logging.info(
            f"Lindblom value range: {valid_lindblom.min():.6f} to {valid_lindblom.max():.6f}"
        )
    if len(valid_snr) > 0:
        logging.info(f"SNR value range: {valid_snr.min():.6f} to {valid_snr.max():.6f}")

    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute Lindblom criterion from pickle file with mismatch data."
    )
    p.add_argument(
        "--pickle_path",
        type=str,
        required=True,
        help="Path to pickle file with mismatch data",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/lindblom",
        help="Directory to save output files",
    )
    p.add_argument(
        "--f_min",
        type=float,
        default=20.0,
        help="Minimum frequency in Hz (default: 20.0)",
    )
    p.add_argument(
        "--delta_f",
        type=float,
        default=0.25,
        help="Frequency spacing in Hz (default: 0.25)",
    )
    p.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count())",
    )

    args = p.parse_args()
    main(
        pickle_path=args.pickle_path,
        output_dir=args.output_dir,
        f_min=args.f_min,
        delta_f=args.delta_f,
        n_workers=args.n_workers,
    )
