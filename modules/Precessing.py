import numpy as np

error_handler = np.seterr(invalid="raise")
from scipy.integrate import odeint
from pycbc.types import FrequencySeries
from modules.default_params_ver2 import *

NEAR_ZERO_THRESHOLD = 1e-8

SOLMASS2SEC = 4.92624076 * 1e-6  # solar mass -> seconds
GIGAPC2SEC = 1.02927125 * 1e17  # gigaparsec -> seconds
FMIN = 20  # lower frequency of the detector sensitivity band [Hz]


def P_total_mass(mcz=20 * solar_mass, eta=0.25, **kwargs):
    """Total mass from chirp mass [seconds]"""
    return mcz / (eta ** (3 / 5))


def P_f_cut(**kwargs):
    """f_cut"""
    return 1 / (6 ** (3 / 2) * np.pi * P_total_mass(**kwargs))


def P_theta_LJ(f, theta_tilde=4.0, **kwargs):
    """theta_LJ_new"""
    return 0.1 * theta_tilde * (f / P_f_cut(**kwargs)) ** (1 / 3)


def P_phi_LJ(f, mcz=20 * solar_mass, omega_tilde=2.0, gamma_P=0.0, **kwargs):
    """phi_LJ"""
    num = (5000 / 96) * omega_tilde
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
