# NOTE: This module is built to experiment with vectorization of precessing waveform calculations.

#############################
# Section 1: Import Modules #
#############################


import numpy as np

error_handler = np.seterr(invalid="raise")
from scipy.integrate import odeint
from pycbc.types import FrequencySeries
from legacy.modules.default_params_v2 import (
    NP_params_0,
    NP_params_1,
    RP_params_0,
    RP_params_1,
    error_handler,
    giga_parsec,
    lens_params_0,
    lens_params_1,
    loc_params,
    np,
    omega_theta_tilde_pairs,
    sky_locs_J_E,
    sky_locs_J_S,
    sky_locs_S_E,
    sky_locs_S_S,
    solar_mass,
    year,
)
from scipy.integrate import cumulative_trapezoid

NEAR_ZERO_THRESHOLD = 1e-8

SOLMASS2SEC = 4.92624076 * 1e-6  # solar mass -> seconds
GIGAPC2SEC = 1.02927125 * 1e17  # gigaparsec -> seconds
FMIN = 20  # lower frequency of the detector sensitivity band [Hz]


###################################
# Section 2: Precessing Functions #
###################################


def P_total_mass(mcz=20 * solar_mass, eta=0.25, **kwargs):
    """Total mass from chirp mass [seconds]"""
    return mcz / (eta ** (3 / 5))


def P_f_cut(**kwargs):
    """f_cut"""
    return 1 / (6 ** (3 / 2) * np.pi * P_total_mass(**kwargs))


def P_theta_LJ(f, theta_tilde=4.0, **kwargs):
    """theta_LJ_new"""
    theta_tilde_arr = theta_tilde[:, np.newaxis]  # Make it 2D for broadcasting
    f = np.tile(f, theta_tilde_arr.shape)
    return 0.1 * theta_tilde_arr * (f / P_f_cut(**kwargs)) ** (1 / 3)


def P_phi_LJ(f, mcz=20 * solar_mass, omega_tilde=2.0, gamma_P=0.0, **kwargs):
    """phi_LJ"""
    omega_tilde_arr = omega_tilde[:, np.newaxis]  # Make it 2D for broadcasting
    f = np.tile(f, omega_tilde_arr.shape)
    num = (5000 / 96) * omega_tilde_arr
    deno = (
        (P_total_mass(**kwargs) / SOLMASS2SEC)
        * (np.pi ** (8 / 3))
        * (mcz ** (5 / 3))
        * (P_f_cut(**kwargs) ** (5 / 3))
    )
    phi_LJ_amp = num / deno
    return phi_LJ_amp * (1 / FMIN - 1 / f) + gamma_P


def P_amp_prefactor(mcz=20 * solar_mass, dist=1.5 * giga_parsec, **kwargs):
    """amplitude prefactor calculated using chirp mass and distance"""
    amp_prefactor = np.sqrt(5 / 96) * (np.pi ** (-2 / 3)) * (mcz ** (5 / 6)) / dist
    return amp_prefactor


def P_precession_angles(
    theta_S=np.pi / 4, phi_S=0, theta_J=8 * np.pi / 9, phi_J=np.pi / 4, **kwargs
):
    """some angles"""

    if phi_J == phi_S:
        if theta_J == theta_S:
            cos_i_JN = 1
        else:
            cos_i_JN = np.cos(theta_J - theta_S)

    else:
        cos_i_JN = np.sin(theta_J) * np.sin(theta_S) * np.cos(phi_J - phi_S) + np.cos(
            theta_J
        ) * np.cos(theta_S)

    sin_i_JN = np.sqrt(1 - cos_i_JN**2.0)

    if np.abs(sin_i_JN) < NEAR_ZERO_THRESHOLD:
        cos_o_XH = 1
        sin_o_XH = 0
    else:
        cos_o_XH = (
            np.cos(theta_S) * np.sin(theta_J) * np.cos(phi_J - phi_S)
            - np.sin(theta_S) * np.cos(theta_J)
        ) / (
            sin_i_JN
        )  # seems to be cos Omega_{XH}
        sin_o_XH = (np.sin(theta_J) * np.sin(phi_J - phi_S)) / (sin_i_JN)
    return cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH


def P_LdotN(f, **kwargs):
    cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = P_precession_angles(**kwargs)
    LdotN = (
        np.sin(P_theta_LJ(f, **kwargs)) * sin_i_JN * np.sin(P_phi_LJ(f, **kwargs))
        + np.cos(P_theta_LJ(f, **kwargs)) * cos_i_JN
    )
    return LdotN


def P_polarization_amplitude_and_phase(
    f, theta_S=np.pi / 4, phi_S=0, theta_J=8 * np.pi / 9, phi_J=np.pi / 4, **kwargs
):
    cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = P_precession_angles(**kwargs)
    # for C
    C_amp = np.sqrt(
        0.25 * (1 + (np.cos(theta_S)) ** 2) ** 2 * ((np.cos(2 * phi_S)) ** 2)
        + ((np.cos(theta_S)) ** 2 * (np.sin(2 * phi_S)) ** 2)
    )

    # define alpha
    sin_alpha = np.cos(theta_S) * np.sin(2 * phi_S) / C_amp
    cos_alpha = (1 + np.cos(theta_S) ** 2) * np.cos(2 * phi_S) / (2 * C_amp)

    # define tan_psi
    num_psi = (
        np.sin(P_theta_LJ(f, **kwargs))
        * (
            np.cos(P_phi_LJ(f, **kwargs)) * sin_o_XH
            + np.sin(P_phi_LJ(f, **kwargs)) * cos_i_JN * cos_o_XH
        )
        - np.cos(P_theta_LJ(f, **kwargs)) * sin_i_JN * cos_o_XH
    )
    den_psi = (
        np.sin(P_theta_LJ(f, **kwargs))
        * (
            np.cos(P_phi_LJ(f, **kwargs)) * cos_o_XH
            - np.sin(P_phi_LJ(f, **kwargs)) * cos_i_JN * sin_o_XH
        )
        + np.cos(P_theta_LJ(f, **kwargs)) * sin_i_JN * sin_o_XH
    )
    if phi_S == phi_J:
        if theta_S == theta_J:
            tan_psi = np.tan(P_phi_LJ(f, **kwargs))
        else:
            tan_psi = num_psi / den_psi

    else:
        tan_psi = num_psi / den_psi

    # define  2 * Psi + alpha
    sin_2pa = (2 * cos_alpha * tan_psi + sin_alpha * (1 - (tan_psi) ** 2)) / (
        1 + (tan_psi) ** 2
    )
    cos_2pa = (cos_alpha * (1 - (tan_psi) ** 2) - 2 * sin_alpha * tan_psi) / (
        1 + (tan_psi) ** 2
    )

    return C_amp, sin_2pa, cos_2pa


### get the amplitude
def P_amplitude(f, **kwargs) -> np.ndarray:
    """NP/Unlensed amplitude"""
    LdotN = P_LdotN(f, **kwargs)
    C_amp, sin_2pa, cos_2pa = P_polarization_amplitude_and_phase(f, **kwargs)

    amp = (
        P_amp_prefactor(**kwargs)
        * C_amp
        * f ** (-7 / 6)
        * np.sqrt(4 * LdotN**2 * sin_2pa**2 + cos_2pa**2 * (1 + LdotN**2) ** 2)
    )
    return amp


### get the phase phi_P
def P_phase_phi_P(f, **kwargs):
    """phi_p"""
    LdotN = P_LdotN(f, **kwargs)
    C_amp, sin_2pa, cos_2pa = P_polarization_amplitude_and_phase(f, **kwargs)

    phi_p_temp = np.arctan2(2 * LdotN * sin_2pa, (1 + LdotN**2) * cos_2pa)
    phi_p = np.unwrap(phi_p_temp, discont=np.pi)
    return phi_p


def P_f_dot(f, mcz=20 * solar_mass, **kwargs):
    """df/dt from Cutler Flanagan 1994"""
    prefactor = (96 / 5) * np.pi ** (8 / 3) * mcz ** (5 / 3) * f ** (11 / 3)
    return prefactor


### get the delta phi_P
def P_integrand_delta_phi(x, f, omega_tilde=2.0, **kwargs):
    """integrand for delta phi p (equations in Apostolatos 1994, and appendix of Evangelos in prep)"""
    LdotN = P_LdotN(f, **kwargs)
    cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = P_precession_angles(**kwargs)
    f_dot = P_f_dot(f, **kwargs)

    Omega_LJ = (
        1000
        * omega_tilde
        * (f / P_f_cut(**kwargs)) ** (5 / 3)
        / (P_total_mass(**kwargs) / SOLMASS2SEC)
    )

    if (
        np.abs(1 - cos_i_JN) < NEAR_ZERO_THRESHOLD
    ):  # face-on (precessing & non-precessing)
        integrand_delta_phi = -Omega_LJ * np.cos(P_theta_LJ(f, **kwargs)) / f_dot

    else:
        integrand_delta_phi = (
            (LdotN / (1 - LdotN**2))
            * Omega_LJ
            * np.sin(P_theta_LJ(f, **kwargs))
            * (
                np.cos(P_theta_LJ(f, **kwargs))
                * sin_i_JN
                * np.sin(P_phi_LJ(f, **kwargs))
                - np.sin(P_theta_LJ(f, **kwargs)) * cos_i_JN
            )
            / f_dot
        )

    return integrand_delta_phi


def P_phase_delta_phi(f, **kwargs):
    """integrate the delta_phi integrand"""
    integral = odeint(P_integrand_delta_phi, 0, f, **kwargs)
    return np.squeeze(integral)


def P_Psi(f, mcz=20 * solar_mass, eta=0.25, t_c=0.0, phi_c=0.0, **kwargs):
    """GW phase"""
    x = (np.pi * P_total_mass(**kwargs) * f) ** (2 / 3)
    Psi = (
        (2 * np.pi * f * t_c)
        - phi_c
        - np.pi / 4
        + ((3 / 4) * (8 * np.pi * mcz * f) ** (-5 / 3))
        * (1 + (20 / 9) * (743 / 336 + (11 / 4) * eta) * x - 16 * np.pi * x ** (3 / 2))
    )
    return Psi


def P_strain(f, delta_f=0.25, frequencySeries=True, **kwargs):
    """precessing GW"""
    strain = P_amplitude(f, **kwargs) * np.exp(
        1j
        * (
            P_Psi(f, **kwargs)
            - P_phase_phi_P(f, **kwargs)
            - 2 * P_phase_delta_phi(f, **kwargs)
        )
    )
    if frequencySeries:
        return FrequencySeries(strain, delta_f)
    return strain


###############################
# Section 3: Vectorized Bank  #
###############################


def precessing_strain_bank(
    base_params: dict,
    omega_tilde_vals: np.ndarray,
    theta_tilde_vals: np.ndarray,
    gamma_P_vals: np.ndarray,
    f: np.ndarray,
    delta_f: float = 0.25,
    return_frequency_series: bool = False,
    chunk_limit: int | None = None,
):
    """Generate a bank of precessing strains over a 3D grid of precession parameters.

    This function vectorizes over (omega_tilde, theta_tilde, gamma_P) while
    broadcasting over frequency f, producing an array of shape
    (n_omega, n_theta, n_gamma, n_f).

    Parameters
    ----------
    base_params : dict
        Dictionary of source (non-precession) parameters required by Precessing
        excluding 'omega_tilde', 'theta_tilde', 'gamma_P'.
    omega_tilde_vals : array-like
        1D array of omega_tilde values.
    theta_tilde_vals : array-like
        1D array of theta_tilde values.
    gamma_P_vals : array-like
        1D array of gamma_P values.
    f : array-like
        1D frequency array (assumed monotonically increasing for phase unwrap).
    delta_f : float, optional
        Frequency spacing, used only if return_frequency_series=True.
    return_frequency_series : bool, optional
        If True, returns a nested dict mapping parameter tuples to FrequencySeries
        instead of a 4D ndarray (slower, more memory per object; useful for PyCBC
        interoperability).
    chunk_limit : int | None, optional
        If not None, yields a generator over chunks of the leading flattened
        parameter combinations with at most this many waveforms per chunk. This
        avoids allocating the full 4D array in memory. In generator mode the
        function yields tuples (slice_indices, strain_chunk) where slice_indices
        is a list of (i_omega, i_theta, i_gamma) index triples.

    Returns
    -------
    strains : np.ndarray or dict or generator
        4D complex ndarray (n_omega, n_theta, n_gamma, n_f) if
        chunk_limit is None and return_frequency_series is False.
        Dict[(i_omega, i_theta, i_gamma)] -> FrequencySeries if
        return_frequency_series is True (and chunk_limit is None).
        Generator of chunks otherwise.

    Notes
    -----
    * Full vectorization can be memory intensive. Approximate memory (bytes)
      = n_omega * n_theta * n_gamma * n_f * 16 (complex128). Use chunk_limit
      to iterate without holding everything at once.
    * This reproduces the formulas inside Precessing without per-parameter
      Python object creation overhead.
    """

    # Convert grids to 1D arrays
    omega_tilde_vals = np.asarray(omega_tilde_vals)
    theta_tilde_vals = np.asarray(theta_tilde_vals)
    gamma_P_vals = np.asarray(gamma_P_vals)
    f = np.asarray(f)

    n_om = omega_tilde_vals.size
    n_th = theta_tilde_vals.size
    n_gp = gamma_P_vals.size
    n_f = f.size

    # Extract scalar base params
    required = [
        "theta_S",
        "phi_S",
        "theta_J",
        "phi_J",
        "mcz",
        "dist",
        "eta",
        "t_c",
        "phi_c",
    ]
    missing = [k for k in required if k not in base_params]
    if missing:
        raise ValueError(f"Missing base parameters: {missing}")

    theta_S = base_params["theta_S"]
    phi_S = base_params["phi_S"]
    theta_J = base_params["theta_J"]
    phi_J = base_params["phi_J"]
    mcz = base_params["mcz"]
    dist = base_params["dist"]
    eta = base_params["eta"]
    t_c = base_params["t_c"]
    phi_c = base_params["phi_c"]

    # Helpers (scalars)
    total_mass = mcz / (eta ** (3 / 5))
    f_cut = 1 / (6 ** (3 / 2) * np.pi * total_mass)
    amp_prefactor = np.sqrt(5 / 96) * (np.pi ** (-2 / 3)) * (mcz ** (5 / 6)) / dist

    # Geometry angles (scalar) - reuse logic from Precessing.precession_angles
    if phi_J == phi_S:
        if theta_J == theta_S:
            cos_i_JN = 1.0
        else:
            cos_i_JN = np.cos(theta_J - theta_S)
    else:
        cos_i_JN = np.sin(theta_J) * np.sin(theta_S) * np.cos(phi_J - phi_S) + np.cos(
            theta_J
        ) * np.cos(theta_S)
    sin_i_JN = np.sqrt(1 - cos_i_JN**2)
    if np.abs(sin_i_JN) < NEAR_ZERO_THRESHOLD:
        cos_o_XH = 1.0
        sin_o_XH = 0.0
    else:
        cos_o_XH = (
            np.cos(theta_S) * np.sin(theta_J) * np.cos(phi_J - phi_S)
            - np.sin(theta_S) * np.cos(theta_J)
        ) / sin_i_JN
        sin_o_XH = (np.sin(theta_J) * np.sin(phi_J - phi_S)) / sin_i_JN

    # Pre-compute constant C_amp, sin_alpha, cos_alpha (scalar)
    C_amp = np.sqrt(
        0.25 * (1 + (np.cos(theta_S)) ** 2) ** 2 * (np.cos(2 * phi_S) ** 2)
        + (np.cos(theta_S) ** 2 * (np.sin(2 * phi_S)) ** 2)
    )
    sin_alpha = np.cos(theta_S) * np.sin(2 * phi_S) / C_amp
    cos_alpha = (1 + np.cos(theta_S) ** 2) * np.cos(2 * phi_S) / (2 * C_amp)

    # Shape param grid with singleton frequency axis for broadcasting
    # param_grid_shape = (n_om, n_th, n_gp, 1)
    omega_grid = omega_tilde_vals[:, None, None, None]
    theta_grid = theta_tilde_vals[None, :, None, None]
    gamma_grid = gamma_P_vals[None, None, :, None]
    f_grid = f[None, None, None, :]

    # Core angle evolutions
    theta_LJ = 0.1 * theta_grid * (f_grid / f_cut) ** (1 / 3)
    # phi_LJ amplitude factor
    num = (5000 / 96) * omega_grid
    deno = (
        (total_mass / SOLMASS2SEC)
        * (np.pi ** (8 / 3))
        * (mcz ** (5 / 3))
        * (f_cut ** (5 / 3))
    )
    phi_LJ_amp = num / deno
    phi_LJ = phi_LJ_amp * (1 / FMIN - 1 / f_grid) + gamma_grid

    sin_theta_LJ = np.sin(theta_LJ)
    cos_theta_LJ = np.cos(theta_LJ)
    sin_phi_LJ = np.sin(phi_LJ)
    cos_phi_LJ = np.cos(phi_LJ)

    # LdotN
    LdotN = sin_theta_LJ * sin_i_JN * sin_phi_LJ + cos_theta_LJ * cos_i_JN

    # tan_psi components
    num_psi = (
        sin_theta_LJ * (cos_phi_LJ * sin_o_XH + sin_phi_LJ * cos_i_JN * cos_o_XH)
        - cos_theta_LJ * sin_i_JN * cos_o_XH
    )
    den_psi = (
        sin_theta_LJ * (cos_phi_LJ * cos_o_XH - sin_phi_LJ * cos_i_JN * sin_o_XH)
        + cos_theta_LJ * sin_i_JN * sin_o_XH
    )
    # Handle special aligned case (phi_S == phi_J)
    if phi_S == phi_J:
        if theta_S == theta_J:
            tan_psi = np.tan(phi_LJ)
        else:
            tan_psi = num_psi / den_psi
    else:
        tan_psi = num_psi / den_psi

    tan_psi_sq = tan_psi**2
    # 2*Psi + alpha trig components
    sin_2pa = (2 * cos_alpha * tan_psi + sin_alpha * (1 - tan_psi_sq)) / (
        1 + tan_psi_sq
    )
    cos_2pa = (cos_alpha * (1 - tan_psi_sq) - 2 * sin_alpha * tan_psi) / (
        1 + tan_psi_sq
    )

    # Amplitude factor (per waveform & frequency)
    amp = (
        amp_prefactor
        * C_amp
        * f_grid ** (-7 / 6)
        * np.sqrt(4 * LdotN**2 * sin_2pa**2 + cos_2pa**2 * (1 + LdotN**2) ** 2)
    )

    # phi_p and unwrap along frequency axis (last axis)
    phi_p = np.arctan2(2 * LdotN * sin_2pa, (1 + LdotN**2) * cos_2pa)
    phi_p = np.unwrap(phi_p, axis=-1, discont=np.pi)

    # f_dot and Omega_LJ for delta_phi integrand
    f_dot = (96 / 5) * np.pi ** (8 / 3) * mcz ** (5 / 3) * f_grid ** (11 / 3)
    Omega_LJ = (
        1000.0
        * omega_grid
        * (f_grid / f_cut) ** (5.0 / 3.0)
        / (total_mass / SOLMASS2SEC)
    )

    face_on = np.abs(1.0 - np.abs(cos_i_JN)) < NEAR_ZERO_THRESHOLD
    if face_on:
        integrand = -Omega_LJ * cos_theta_LJ / f_dot
    else:
        integrand = (
            (LdotN / (1.0 - LdotN**2))
            * Omega_LJ
            * sin_theta_LJ
            * (cos_theta_LJ * sin_i_JN * sin_phi_LJ - sin_theta_LJ * cos_i_JN)
            / f_dot
        )
    # Integrate delta_phi along frequency axis
    delta_phi = cumulative_trapezoid(integrand, f, axis=-1, initial=0.0)

    # Carrier phase Psi(f)
    x = (np.pi * total_mass * f_grid) ** (2 / 3)
    Psi = (
        (2 * np.pi * f_grid * t_c)
        - phi_c
        - np.pi / 4
        + ((3 / 4) * (8 * np.pi * mcz * f_grid) ** (-5 / 3))
        * (1 + (20 / 9) * (743 / 336 + (11 / 4) * eta) * x - 16 * np.pi * x ** (3 / 2))
    )

    strain = amp * np.exp(1j * (Psi - phi_p - 2 * delta_phi))

    if chunk_limit is not None:
        # Generator mode
        flat_indices = [
            (i, j, k) for i in range(n_om) for j in range(n_th) for k in range(n_gp)
        ]
        total = len(flat_indices)

        def chunk_generator():
            for start in range(0, total, chunk_limit):
                idx_chunk = flat_indices[start : start + chunk_limit]
                # Gather slice
                arr_chunk = np.stack(
                    [strain[i, j, k, :] for (i, j, k) in idx_chunk], axis=0
                )  # shape (chunk, n_f)
                yield idx_chunk, arr_chunk

        return chunk_generator()

    if return_frequency_series:
        out = {}
        for i in range(n_om):
            for j in range(n_th):
                for k in range(n_gp):
                    out[(i, j, k)] = FrequencySeries(strain[i, j, k, :], delta_f)
        return out

    return strain  # shape (n_omega, n_theta, n_gamma, n_f)


__all__ = [
    "Lensing",
    "LensingGeo",
    "Precessing",
    "precessing_strain_bank",
]
