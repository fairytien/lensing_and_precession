#############################
# Section 1: Import Modules #
#############################

# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

import numpy as np
from numpy.lib import NumpyVersion
from typing import Optional

try:
    import numba  # type: ignore
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# Compatibility shim for NumPy 1.24+ where several aliases were removed
if not hasattr(np, "alen"):
    try:
        setattr(np, "alen", lambda a: len(a))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - environment-specific safeguard
        # fallback: provide a local alias if numpy cannot be modified
        def _local_alen(a):
            return len(a)

        alen = _local_alen
if NumpyVersion(np.__version__) < NumpyVersion("1.24.0"):
    for _name, _alias in (
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("complex", complex),
        ("object", object),
    ):
        if not hasattr(np, _name):
            setattr(np, _name, _alias)

# Be robust to environments where np.seterr may be unavailable or altered
try:
    error_handler = np.seterr(invalid="raise")
except Exception:  # pragma: no cover - environment-specific safeguard
    error_handler = None

from scipy.integrate import odeint
import scipy.special as sc
import mpmath as mp
from pycbc.types import FrequencySeries

# Define converters/constants
NEAR_ZERO_THRESHOLD = 1e-10
SOLMASS2SEC = 4.92624076 * 1e-6  # solar mass -> seconds
GIGAPC2SEC = 1.02927125 * 1e17  # gigaparsec -> seconds
FMIN = 20  # lower frequency of the detector sensitivity band [Hz]


# Compiled integrand for precession phase correction (Step 3: numba JIT)
if HAS_NUMBA:
    @numba.njit
    def _integrand_delta_phi_compiled(
        f,
        theta_tilde,
        omega_tilde,
        eta,
        mcz,
        theta_lj_coeff,
        phi_lj_amp,
        f_dot_prefactor,
        omega_lj_prefactor,
        cos_i_JN,
        sin_i_JN,
        cos_o_XH,
        sin_o_XH,
        face_on,
        gamma_P,
        f_min,
    ):
        """Compiled integrand for precession phase correction.
        
        Step 2 & 3: Cache per-call powers (f^1/3, f^5/3, f^11/3) and use numba JIT
        to eliminate numpy dispatch overhead for scalar odeint callbacks.
        """
        if theta_tilde == 0:
            return 0.0

        # Step 2: Compute powers once per call and reuse
        f_pow_13 = f ** (1.0 / 3.0)
        f_pow_53 = f_pow_13 * f_pow_13 * f_pow_13 * f_pow_13 * f_pow_13  # f^(5/3)
        f_pow_113 = f_pow_53 * f_pow_53 * f_pow_13 * f_pow_13  # f^(11/3)

        # Compute theta_LJ and phi_LJ using cached coefficients
        theta_lj = theta_lj_coeff * f_pow_13
        phi_lj = phi_lj_amp * (1.0 / f_min - 1.0 / f) + gamma_P

        # Compute trig values
        sin_theta = np.sin(theta_lj)
        cos_theta = np.cos(theta_lj)
        sin_phi = np.sin(phi_lj)
        cos_phi = np.cos(phi_lj)

        # Compute derived quantities
        ldotn = sin_theta * sin_i_JN * sin_phi + cos_theta * cos_i_JN
        f_dot = f_dot_prefactor * f_pow_113
        omega_lj = omega_lj_prefactor * f_pow_53

        # Face-on case
        if face_on:
            return -omega_lj * cos_theta / f_dot

        # Generic (non face-on) case
        denom = 1.0 - ldotn * ldotn
        if abs(denom) < 1e-14:
            return 0.0

        base = (
            (ldotn / denom)
            * omega_lj
            * sin_theta
            * (cos_theta * sin_i_JN * sin_phi - sin_theta * cos_i_JN)
            / f_dot
        )

        corr = (ldotn / denom) * (-(theta_lj / (3.0 * f)) * cos_phi * sin_i_JN)

        return base + corr
else:
    def _integrand_delta_phi_compiled(*args, **kwargs):
        """Fallback if numba is not available (should not reach production)."""
        raise NotImplementedError(
            "numba not available; integrand_delta_phi requires numba for optimization"
        )

############################
# Section 2: Lensing Class #
############################


class Lensing:
    """
    Point-mass lensing model for gravitational wave signals.

    This class implements the wave optics treatment of gravitational lensing
    by a point mass, computing lensed gravitational waveforms.

    Notes
    -----
    Based on Takahashi & Nakamura (2003) and related works on gravitational
    wave lensing in the wave optics regime.
    """

    def __init__(self, params):
        """
        Initialize gravitational lensing model.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters for unlensed signal and lens:
            - theta_S : Sky inclination (detector frame)
            - phi_S : Sky azimuthal angle (detector frame)
            - theta_J : Binary plane inclination (J == L, no precession)
            - phi_J : Binary plane azimuthal angle
            - mcz : Chirp mass [s]
            - dist : Luminosity distance
            - eta : Symmetric mass ratio
            - t_c : Coalescence time
            - phi_c : Coalescence phase
            - MLz : Lens mass (redshifted) [s]
            - y : Dimensionless source position
        """
        self.params = params

        assert type(self.params == dict), "Parameters should be a dictionary"

        # unlensed parameters
        self.theta_S = params["theta_S"]
        self.phi_S = params["phi_S"]
        self.theta_J = params["theta_J"]  # J == L (no precession)
        self.phi_J = params["phi_J"]  # J == L (no precession)
        self.mcz = params["mcz"]
        self.dist = params["dist"]
        self.eta = params["eta"]
        self.t_c = params["t_c"]
        self.phi_c = params["phi_c"]

        # lensed parameters
        self.M_Lz = params["MLz"]
        self.y = params["y"]

    def total_mass(self):
        """
        Calculate total mass from chirp mass.

        Returns
        -------
        float
            Total mass [s].
        """
        return self.mcz / (self.eta ** (3 / 5))

    def f_cut(self):
        """
        Calculate ISCO cut-off frequency.

        Returns
        -------
        float
            Cut-off frequency [Hz].
        """
        return 1 / (6 ** (3 / 2) * np.pi * self.total_mass())

    def LdotN(self):
        """
        Calculate dot product between orbital angular momentum and line of sight.

        Returns
        -------
        float
            Dot product L · N (for non-precessing case, L == J).
        """
        cos_term = np.cos(self.theta_S) * np.cos(self.theta_J)
        sin_term = (
            np.sin(self.theta_S)
            * np.sin(self.theta_J)
            * np.cos(self.phi_S - self.phi_J)
        )
        inner_prod = cos_term + sin_term
        return inner_prod

    def amp(self):
        """
        Calculate amplitude prefactor for unlensed waveform following equation 3.13 in Cutler-Flanaghan 1994.

        Returns
        -------
        float
            Amplitude prefactor.
        """
        amplitude = (
            np.sqrt(5 / 96) * np.pi ** (-2 / 3) * self.mcz ** (5 / 6) / (self.dist)
        )
        return amplitude

    def Psi(self, f):
        """
        Calculate GW phase to 2 PN order.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].

        Returns
        -------
        float or ndarray
            GW phase.

        Notes
        -----
        Implements Equation 3.13 from Cutler & Flanagan (1994).
        """
        x = (np.pi * self.total_mass() * f) ** (2 / 3)
        term1 = 2 * np.pi * f * self.t_c - self.phi_c - np.pi / 4
        prefactor = (3 / 4) * (8 * np.pi * self.mcz * f) ** (-5 / 3)
        term2 = (
            1
            + (20 / 9) * (743 / 336 + (11 / 4) * self.eta) * x
            - 16 * np.pi * x ** (3 / 2)
        )
        Psi = term1 + prefactor * term2
        return Psi

    def psi_s(self):
        """psi_s that goes into F_plus and F_cross"""

        numerator = np.cos(self.theta_J) - np.cos(self.theta_S) * (self.LdotN())
        denominator = (
            np.sin(self.theta_S)
            * np.sin(self.theta_J)
            * np.sin(self.phi_J - self.phi_S)
        )

        psi_s_val = np.arctan2(numerator, denominator)
        return psi_s_val

    def fIp(self):
        """F_plus"""

        term_1 = (
            1
            / 2
            * (1 + np.power(np.cos(self.theta_S), 2))
            * np.cos(2 * self.phi_S)
            * np.cos(2 * self.psi_s())
        )
        term_2 = (
            np.cos(self.theta_S) * np.sin(2 * self.phi_S) * np.sin(2 * self.psi_s())
        )

        fIp_val = term_1 - term_2
        return fIp_val

    def fIc(self):
        """F_cross"""

        term_1 = (
            1
            / 2
            * (1 + np.power(np.cos(self.theta_S), 2))
            * np.cos(2 * self.phi_S)
            * np.sin(2 * self.psi_s())
        )
        term_2 = (
            np.cos(self.theta_S) * np.sin(2 * self.phi_S) * np.cos(2 * self.psi_s())
        )

        fIc_val = term_1 + term_2
        return fIc_val

    def lambdaI(self):
        """|F_plus (1+L.N**2) - i (2*F_cross*L.N)|"""

        term_1 = np.power(2 * self.LdotN() * self.fIc(), 2)
        term_2 = np.power((1 + np.power(self.LdotN(), 2)) * self.fIp(), 2)
        lambdaI_val = np.sqrt(term_1 + term_2)
        return lambdaI_val

    def phi_pI(self):
        """tan-1((2*F_cross*L.N)/F_plus (1+L.N**2))"""

        numerator = 2 * self.LdotN() * self.fIc()
        denominator = (1 + np.power(self.LdotN(), 2)) * self.fIp()

        phi_pI_val = np.arctan2(numerator, denominator)
        return phi_pI_val

    def hI(self, f):
        """
        Calculate the unlensed gravitational waveform.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].

        Returns
        -------
        complex or ndarray
            Unlensed strain h_I(f).
        """
        term_1 = self.lambdaI()
        term_2 = np.exp(-1j * self.phi_pI())
        term_3 = self.amp() * f ** (-7 / 6)
        term_4 = np.exp(1j * self.Psi(f))

        signal_I = term_1 * term_2 * term_3 * term_4

        return signal_I

    def F(self, f):
        """
        Calculate the point-mass amplification factor in wave optics regime.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].

        Returns
        -------
        complex or ndarray
            Amplification factor F(f).

        Notes
        -----
        Implements Equation 17 from Takahashi & Nakamura (2003) using the
        confluent hypergeometric function.
        """
        self.w = 8 * np.pi * self.M_Lz * f
        x_m = 0.5 * (self.y + np.sqrt(self.y**2 + 4))
        phi_m = np.power((x_m - self.y), 2) / 2 - np.log(x_m)

        term1 = np.exp(
            np.pi * self.w / 4 + 1j * (self.w / 2) * (np.log(self.w / 2) - 2 * phi_m)
        )
        term2 = sc.gamma(1 - 1j * (self.w / 2))

        # broadcasting mp hyp1f1 function to NumPy ufunc
        hyp1f1_np = np.frompyfunc(mp.hyp1f1, 3, 1)

        term3 = hyp1f1_np(1j * self.w / 2, 1, 1j * (self.w / 2) * (self.y**2))

        F_val = np.complex128(term1 * term2 * term3)

        return F_val

    def strain(self, f, delta_f=0.25, frequencySeries=True):
        """
        Calculate the lensed gravitational wave strain.

        Parameters
        ----------
        f : array_like
            Frequencies at which to evaluate [Hz].
        delta_f : float, optional
            Frequency spacing for FrequencySeries output (default: 0.25 Hz).
        frequencySeries : bool, optional
            If True, return pycbc.FrequencySeries; if False, return ndarray
            (default: True).

        Returns
        -------
        FrequencySeries or ndarray
            Lensed strain h_L(f) = h_I(f) * F(f).
        """
        hL = self.hI(f) * self.F(f)

        if frequencySeries:
            return FrequencySeries(hL, delta_f)

        return hL


class LensingGeo(Lensing):
    """
    Geometric optics approximation for gravitational wave lensing.

    This class implements the geometric optics limit of
    point-mass gravitational lensing, which is computationally faster than
    the full wave optics treatment.

    Notes
    -----
    Valid when f >> f_0 = c^3/(4*pi*G*M_L) ~ 0.4 Hz / (M_L/M_sun).
    Based on Takahashi & Nakamura (2003) and Saif et al. (2023).
    """

    def __init__(self, params):
        """
        Initialize geometric optics lensing model.

        Parameters
        ----------
        params : dict
            Same parameters as Lensing class.
        """
        super().__init__(params)

    def mu_plus(self):
        """
        Calculate the magnification of the plus (primary) image.

        Returns
        -------
        complex
            Magnification mu_+.

        Notes
        -----
        Implements Equation 18 from Takahashi & Nakamura (2003) and
        Equation 16a from Saif et al. (2023).
        """
        mu_plus_val = (
            1 / 2 + (self.y**2 + 2) / (2 * self.y * np.sqrt(self.y**2 + 4)) + 0j
        )
        return mu_plus_val

    def mu_minus(self):
        """
        Calculate the magnification of the minus (secondary) image.

        Returns
        -------
        complex
            Magnification mu_-.

        Notes
        -----
        Implements Equation 18 from Takahashi & Nakamura (2003) and
        Equation 16a from Saif et al. (2023).
        """
        mu_minus_val = (
            1 / 2 - (self.y**2 + 2) / (2 * self.y * np.sqrt(self.y**2 + 4)) + 0j
        )
        return mu_minus_val

    def I(self):
        """
        Calculate the flux ratio between images.

        Returns
        -------
        float
            Flux ratio |mu_-| / |mu_+|.

        Notes
        -----
        Implements Equation 17a from Saif et al. (2023).
        """
        I_val = np.abs(self.mu_minus()) / np.abs(self.mu_plus())
        return I_val

    def td(self):
        """
        Calculate the time delay between the two images.

        Returns
        -------
        float
            Time delay [s].

        Notes
        -----
        Implements Equation 16b from Saif et al. (2023).
        """
        td_val = (
            2
            * self.M_Lz
            * (
                self.y * np.sqrt(self.y**2 + 4)
                + 2
                * np.log(
                    (np.sqrt(self.y**2 + 4) + self.y)
                    / (np.sqrt(self.y**2 + 4) - self.y)
                )
            )
        )
        return td_val

    def F(self, f):
        """
        Calculate amplification factor in geometric optics limit.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].

        Returns
        -------
        complex or ndarray
            Amplification factor F(f) in geometric optics approximation.

        Notes
        -----
        Implements Equation 18 from Takahashi & Nakamura (2003).
        Superposition of two images with magnifications and time delay.
        """
        F_val = np.sqrt(np.abs(self.mu_plus())) - 1j * np.sqrt(
            np.abs(self.mu_minus())
        ) * np.exp(2j * np.pi * f * self.td())
        return F_val


###############################
# Section 3: Precessing Class #
###############################


class Precessing:
    """
    Regular Precession: When L moves in a cone around J with an opening angle theta_tilde
    that changes on a radiation reaction timescale, with frequency omega_tilde (also changing
    on the same timescale) and a phase gamma_P.

    Model presented in following paper: arXiv:2509.10628 [gr-qc]
    """

    def __init__(self, params):
        """
        Initialize Regular Precession model.

        Parameters
        ----------
        params : dict
            Dictionary containing physical parameters:
            - theta_S : Sky inclination (polar angle for line of sight in detector frame)
            - phi_S : Sky azimuthal angle (azimuthal angle for line of sight in detector frame)
            - theta_J : Source binary plane inclination (polar angle for J in detector frame)
            - phi_J : Source binary plane azimuthal angle (azimuthal angle for J in detector frame)
            - mcz : Chirp mass [s] - M_c = (m1*m2)**(3/5) / (m1 + m2)**(1/5)
            - dist : Distance to the source
            - eta : Symmetric mass ratio - eta = m1*m2 / (m1 + m2)**2
            - t_c : Coalescence time
            - phi_c : Coalescence phase
            - theta_tilde : Dimensionless precession amplitude (10x opening angle at r = 6M)
            - omega_tilde : Dimensionless precession frequency (1000x frequency at r = 6M for solar mass binary)
            - gamma_P : Phase of the precession when binary enters detector band
        """
        self.params = params

        assert type(self.params == dict), "Parameters should be a dictionary"

        # non-precession/unlensed parameters
        self.theta_S = params["theta_S"]
        self.phi_S = params["phi_S"]
        self.theta_J = params["theta_J"]
        self.phi_J = params["phi_J"]
        self.mcz = params["mcz"]
        self.dist = params["dist"]
        self.eta = params["eta"]
        self.t_c = params["t_c"]
        self.phi_c = params["phi_c"]

        # regular precession parameters
        self.theta_tilde = params["theta_tilde"]
        self.omega_tilde = params["omega_tilde"]
        self.gamma_P = params["gamma_P"]

        # Cache constants used repeatedly in hot paths (waveform + ODE RHS).
        self._total_mass = self.mcz / (self.eta ** (3 / 5))
        self._f_cut = 1 / (6 ** (3 / 2) * np.pi * self._total_mass)
        self._inv_f_cut = 1.0 / self._f_cut
        self._theta_lj_coeff = (
            (0.1 / (4 * self.eta)) * self.theta_tilde * (self._inv_f_cut ** (1 / 3))
        )
        self._phi_lj_amp = ((5000 / 96) * self.omega_tilde) / (
            (self._total_mass / SOLMASS2SEC)
            * (np.pi ** (8 / 3))
            * (self.mcz ** (5 / 3))
            * (self._f_cut ** (5 / 3))
        )
        self._amp_prefactor = (
            np.sqrt(5 / 96) * (np.pi ** (-2 / 3)) * (self.mcz ** (5 / 6)) / self.dist
        )
        self._f_dot_prefactor = (96 / 5) * np.pi ** (8 / 3) * self.mcz ** (5 / 3)
        self._omega_lj_prefactor = (
            (1000 * self.omega_tilde)
            * (self._inv_f_cut ** (5 / 3))
            / (self._total_mass / SOLMASS2SEC)
        )

        # Cache geometry terms that depend only on source orientation.
        self._precession_angles = self._compute_precession_angles()
        self._face_on = (
            np.abs(1 - np.abs(self._precession_angles[0])) < NEAR_ZERO_THRESHOLD
        )

    def _compute_precession_angles(self):
        """Compute (cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH) once per instance."""
        if self.phi_J == self.phi_S:
            if self.theta_J == self.theta_S:
                cos_i_JN = 1
            else:
                cos_i_JN = np.cos(self.theta_J - self.theta_S)
        else:
            cos_i_JN = np.sin(self.theta_J) * np.sin(self.theta_S) * np.cos(
                self.phi_J - self.phi_S
            ) + np.cos(self.theta_J) * np.cos(self.theta_S)

        sin_i_JN = np.sqrt(1 - cos_i_JN**2.0)
        if np.abs(sin_i_JN) < NEAR_ZERO_THRESHOLD:
            cos_o_XH = 1
            sin_o_XH = 0
        else:
            cos_o_XH = (
                np.cos(self.theta_S)
                * np.sin(self.theta_J)
                * np.cos(self.phi_J - self.phi_S)
                - np.sin(self.theta_S) * np.cos(self.theta_J)
            ) / sin_i_JN
            sin_o_XH = (np.sin(self.theta_J) * np.sin(self.phi_J - self.phi_S)) / sin_i_JN
        return cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH

    def _theta_phi_terms(self, f):
        """Return theta/phi and their trig values for a given frequency.
        
        Uses scalar power ** instead of np.power for scalar f (Step 2 optimization).
        """
        # Step 2: Use Python ** for scalar f to avoid numpy dispatch overhead
        if np.isscalar(f):
            f_pow_13 = f ** (1.0 / 3.0)
            theta_lj = self._theta_lj_coeff * f_pow_13
        else:
            theta_lj = self._theta_lj_coeff * np.power(f, 1 / 3)
        
        phi_lj = self._phi_lj_amp * (1 / FMIN - 1 / f) + self.gamma_P
        sin_theta = np.sin(theta_lj)
        cos_theta = np.cos(theta_lj)
        sin_phi = np.sin(phi_lj)
        cos_phi = np.cos(phi_lj)
        return theta_lj, phi_lj, sin_theta, cos_theta, sin_phi, cos_phi

    def total_mass(self):
        """
        Calculate the total mass [seconds] of the binary system from the chirp mass and symmetric mass ratio.

        Returns
        -------
        total_mass : float
            Total mass of the binary system [s]: M = M_c / eta**(3/5)
        """
        return self._total_mass

    def f_cut(self):
        """
        Compute the cut-off frequency where the binary coalesces.

        Returns
        -------
        f_cut : float
            Cut-off frequency [Hz]: f_cut = 1/(r_{ISCO}**(3/2) * pi * M)
        """
        return self._f_cut

    def theta_LJ(self, f):
        """
        Compute the opening angle between L and J at a given frequency.

        Parameters
        ----------
        f : float or array_like
            Frequency at which to calculate the angle [Hz].

        Returns
        -------
        float or ndarray
            Opening angle between L and J [rad].

        Notes
        -----
        Implements Equation 18a from Taman's paper: theta_LJ = 0.1/(4*eta) * theta_tilde * (f/f_cut)**(1/3)
        """
        return self._theta_lj_coeff * np.power(f, 1 / 3)

    def phi_LJ(self, f):
        """
        Compute the azimuthal precession angle at a given frequency.

        Parameters
        ----------
        f : float or array_like
            Frequency at which to calculate the angle [Hz].

        Returns
        -------
        float or ndarray
            Azimuthal angle of L in the source frame [rad].

        Notes
        -----
        Implements Equations 18b and 19:
        phi_LJ = phi_LJ_amp * (1/f_min - 1/f) + gamma_P
        where phi_LJ_amp depends on omega_tilde and system parameters.
        """
        return self._phi_lj_amp * (1 / FMIN - 1 / f) + self.gamma_P

    def amp_prefactor(self):
        """
        Calculate the gravitational wave amplitude prefactor.

        Returns
        -------
        float
            Amplitude prefactor.

        Notes
        -----
        Implements Equation 6: A = sqrt(5/96) * pi^(-2/3) * M_c^(5/6) / D_L
        """
        return self._amp_prefactor

    def precession_angles(self):
        """
        Compute coordinate transformation angles for the precession calculation.

        Returns
        -------
        cos_i_JN : float
            Cosine of angle between J and line of sight N.
        sin_i_JN : float
            Sine of angle between J and line of sight N.
        cos_o_XH : float
            Cosine of angle Omega_XH in detector frame.
        sin_o_XH : float
            Sine of angle Omega_XH in detector frame.

        Notes
        -----
        Implements Equations A4, A6a, and A6b for coordinate transformation
        between detector frame and source frame.
        """

        return self._precession_angles

    def LdotN(self, f):
        """
        Compute the dot product between orbital angular momentum and line of sight.

        Parameters
        ----------
        f : float or array_like
            Frequency at which to evaluate [Hz].

        Returns
        -------
        float or ndarray
            Dot product L · N (cosine of angle between L and N).
        """
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        theta_lj, phi_lj, sin_theta, cos_theta, sin_phi, cos_phi = self._theta_phi_terms(f)
        LdotN = sin_theta * sin_i_JN * sin_phi + cos_theta * cos_i_JN
        return LdotN

    def polarization_amplitude_and_phase(self, f):
        """
        Calculate beam pattern amplitude and polarization angle components.

        Parameters
        ----------
        f : float or array_like
            Frequency at which to evaluate [Hz].

        Returns
        -------
        C_amp : float or ndarray
            Amplitude of beam pattern function C.
        sin_2pa : float or ndarray
            sin(2*psi + alpha), related to cross polarization.
        cos_2pa : float or ndarray
            cos(2*psi + alpha), related to plus polarization.

        Notes
        -----
        Implements Equations 3, 4a, and 4b in Taman et al. 2025. Combines the polarization angle psi
        with the detector orientation angle alpha.
        """
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        theta_lj, phi_lj, sin_theta, cos_theta, sin_phi, cos_phi = self._theta_phi_terms(f)

        # For C
        C_amp = np.sqrt(
            0.25
            * (1 + (np.cos(self.theta_S)) ** 2) ** 2
            * ((np.cos(2 * self.phi_S)) ** 2)
            + ((np.cos(self.theta_S)) ** 2 * (np.sin(2 * self.phi_S)) ** 2)
        )

        # Define alpha based on equation 4b
        sin_alpha = np.cos(self.theta_S) * np.sin(2 * self.phi_S) / C_amp
        cos_alpha = (
            (1 + np.cos(self.theta_S) ** 2) * np.cos(2 * self.phi_S) / (2 * C_amp)
        )

        # Define tan_psi from equation 3
        num_psi = (
            sin_theta
            * (
                cos_phi * sin_o_XH
                + sin_phi * cos_i_JN * cos_o_XH
            )
            - cos_theta * sin_i_JN * cos_o_XH
        )
        den_psi = (
            sin_theta
            * (
                cos_phi * cos_o_XH
                - sin_phi * cos_i_JN * sin_o_XH
            )
            + cos_theta * sin_i_JN * sin_o_XH
        )
        # Hybrid handling: tolerant face-on special case uses Taman's closed-form tan(psi),
        # otherwise use the generic quotient (algebraic and stable).
        face_on = self._face_on
        if face_on:
            o_XH = np.arctan2(sin_o_XH, cos_o_XH)
            tan_psi = np.tan(o_XH + np.sign(cos_i_JN) * phi_lj)
        else:
            tan_psi = num_psi / den_psi

        # if den_psi.all() == 0:  # True for face-on and theta_tilde = 0
        #     if self.theta_tilde == 0:  # WRONG!!! Refer to Eq A14 in Taman's paper!
        #         return C_amp, 0, -1

        # Define  2 * Psi + alpha
        # Algebraic forms (naturally stable when tan_psi is finite)
        sin_2pa_alg = (2 * cos_alpha * tan_psi + sin_alpha * (1 - (tan_psi) ** 2)) / (
            1 + (tan_psi) ** 2
        )
        cos_2pa_alg = (cos_alpha * (1 - (tan_psi) ** 2) - 2 * sin_alpha * tan_psi) / (
            1 + (tan_psi) ** 2
        )

        # Asymptotic guard: when tan_psi blows up or den_psi is tiny, use limits
        # sin(2ψ+α) -> -sin α, cos(2ψ+α) -> -cos α as tan_psi -> ±∞
        T_bad = (
            ~np.isfinite(tan_psi) | (np.abs(tan_psi) > 1e12) | (np.abs(den_psi) < 1e-12)
        )
        sin_2pa = np.where(T_bad, -sin_alpha, sin_2pa_alg)
        cos_2pa = np.where(T_bad, -cos_alpha, cos_2pa_alg)

        return C_amp, sin_2pa, cos_2pa

    def amplitude(self, f):
        """
        Calculate the non-precessing/unlensed gravitational wave amplitude.

        Parameters
        ----------
        f : float or array_like
            Frequency at which to evaluate [Hz].

        Returns
        -------
        ndarray
            GW amplitude.

        Notes
        -----
        Implements Equation 10 from Apostolatos et al. (1994).
        """
        LdotN = self.LdotN(f)
        C_amp, sin_2pa, cos_2pa = self.polarization_amplitude_and_phase(f)

        amp = (
            self.amp_prefactor()
            * C_amp
            * f ** (-7 / 6)
            * np.sqrt(4 * LdotN**2 * sin_2pa**2 + cos_2pa**2 * (1 + LdotN**2) ** 2)
        )
        return amp

    def phase_phi_P(self, f):
        """
        Calculate the polarization phase.

        Parameters
        ----------
        f : float or array_like
            Frequency at which to evaluate [Hz].

        Returns
        -------
        ndarray
            Polarization phase phi_P (unwrapped).

        Notes
        -----
        Implements Equation 11 from Apostolatos et al. (1994).
        """
        LdotN = self.LdotN(f)
        C_amp, sin_2pa, cos_2pa = self.polarization_amplitude_and_phase(f)

        phi_p_temp = np.arctan2(2 * LdotN * sin_2pa, (1 + LdotN**2) * cos_2pa)
        phi_p = np.unwrap(phi_p_temp, discont=np.pi)
        return phi_p

    def f_dot(self, f):
        """
        Calculate the rate of change of frequency with time.

        Parameters
        ----------
        f : float or array_like
            Frequency at which to evaluate [Hz].

        Returns
        -------
        float or ndarray
            Time derivative df/dt [Hz/s].

        Notes
        -----
        Implements leading order term from Cutler & Flanagan (1994).
        Higher order PN corrections are commented out.
        """
        return self._f_dot_prefactor * np.power(f, 11 / 3)  # * (1 - (743/336 + (11/4) * self.eta) * (np.pi * self.total_mass() * f)**(2/3) + 4 * np.pi * (np.pi * self.total_mass() * f))

    def integrand_delta_phi(self, y, f):
        """
        Compute the integrand for the precession phase correction delta_phi_P.

        This function is designed to be used as the right-hand side for ODE integrators such as `scipy.integrate.odeint`, which require the signature `func(y, f)`.
        The integrand is based on Equation A18 from Apostolatos 1994 or Equation A19 from Taman et al. 2025.
        Handles special cases for non-precessing, face-on, and generic precessing binaries.

        Parameters
        ----------
        y : float
            Dummy variable for ODE integrator compatibility (unused).
        f : float or array_like
            Frequency at which to evaluate [Hz].

        Returns
        -------
        float or ndarray
            Integrand value d(delta_phi_P)/df at frequency f.

        Notes
        -----
        Added correction term following Equation A19 from Taman et al. 2025.
        Uses numba-compiled core for scalar f (Step 3) and caches powers (Step 2).
        """
        # Step 3: Use compiled integrand for scalar f to avoid Python dispatch overhead
        if HAS_NUMBA and np.isscalar(f):
            cos_i_JN, sin_i_JN, *_ = self.precession_angles()
            return _integrand_delta_phi_compiled(
                float(f),
                float(self.theta_tilde),
                float(self.omega_tilde),
                float(self.eta),
                float(self.mcz),
                float(self._theta_lj_coeff),
                float(self._phi_lj_amp),
                float(self._f_dot_prefactor),
                float(self._omega_lj_prefactor),
                float(cos_i_JN),
                float(sin_i_JN),
                0.0,  # cos_o_XH (not used in core integrand)
                0.0,  # sin_o_XH (not used in core integrand)
                bool(self._face_on),
                float(self.gamma_P),
                float(FMIN),
            )
        
        # Fallback for array f or when numba is unavailable (legacy path)
        cos_i_JN, sin_i_JN, *_ = self.precession_angles()
        theta_LJ, phi_LJ, sin_theta, cos_theta, sin_phi, cos_phi = self._theta_phi_terms(f)
        LdotN = sin_theta * sin_i_JN * sin_phi + cos_theta * cos_i_JN
        # Step 2: Use ** for scalar f instead of np.power when possible
        if np.isscalar(f):
            f_pow_113 = f ** (11.0 / 3.0)
            f_pow_53 = f ** (5.0 / 3.0)
        else:
            f_pow_113 = np.power(f, 11 / 3)
            f_pow_53 = np.power(f, 5 / 3)
        f_dot = self._f_dot_prefactor * f_pow_113
        Omega_LJ = self._omega_lj_prefactor * f_pow_53

        if self.theta_tilde == 0:  # non-precessing
            return 0

        # Face-on case (precessing & non-precessing)
        face_on = self._face_on
        if face_on:
            return -Omega_LJ * cos_theta / f_dot

        # Generic (non face-on) expression (matches original formula, just factored)
        denom = 1 - LdotN**2
        if np.abs(denom) < NEAR_ZERO_THRESHOLD:
            return 0.0
        
        base = (
            (LdotN / denom)
            * Omega_LJ
            * np.sin(theta_LJ)
            * (
                np.cos(theta_LJ) * sin_i_JN * np.sin(phi_LJ)
                - sin_theta * cos_i_JN
            )
            / f_dot
        )

        # Added correction term with theta_LJ/3f cos phi_LJ sin i_JN term (from Taman/regular_precession.py)
        corr = (LdotN / denom) * (
            -(theta_LJ / (3.0 * f)) * np.cos(phi_LJ) * sin_i_JN
        )

        return base + corr

    def phase_delta_phi(self, f, rtol=1.49012e-8, atol=1.49012e-8):
        """
        Integrate the delta_phi integrand over the given frequency array.

        Parameters
        ----------
        f : array_like
            Frequencies at which to compute the phase correction [Hz].
        rtol : float, optional
            Relative tolerance for odeint (default: 1.49012e-8).
        atol : float, optional
            Absolute tolerance for odeint (default: 1.49012e-8).

        Returns
        -------
        ndarray
            Precession phase correction delta_phi at each frequency.

        Notes
        -----
        Uses scipy.integrate.odeint to numerically integrate the integrand from f_min to each frequency value.
        """
        integral = odeint(self.integrand_delta_phi, 0, f, rtol=rtol, atol=atol)
        return np.squeeze(integral)

    def Psi(self, f):
        """
        Calculate GW phase to 2 PN order.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].

        Returns
        -------
        float or ndarray
            GW phase.

        Notes
        -----
        Implements Equation 3.13 from Cutler & Flanagan (1994).
        """
        x = (np.pi * self.total_mass() * f) ** (2 / 3)
        Psi = (
            (2 * np.pi * f * self.t_c)
            - self.phi_c
            - np.pi / 4
            + ((3 / 4) * (8 * np.pi * self.mcz * f) ** (-5 / 3))
            * (
                1
                + (20 / 9) * (743 / 336 + (11 / 4) * self.eta) * x
                - 16 * np.pi * x ** (3 / 2)
            )
        )
        return Psi

    def cos_theta_L(self, f):
        """
        Evolution of the orbital angular momentum vector in the detector frame (cosine of polar angle).

        Parameters
        ----------
        f : float
            Frequency at which the angle is to be calculated [Hz].

        Returns
        -------
        L_z : float
            Cosine of the polar angle for the orbital angular momentum vector.

        Notes
        -----
        Implements Equation A8 from Taman et al. 2025.
        """
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        # L_H = np.sin(self.theta_LJ(f)) * (np.cos(self.phi_LJ(f)) * cos_o_XH - np.sin(self.phi_LJ(f)) * cos_i_JN * sin_o_XH) + sin_i_JN * sin_o_XH * np.cos(self.theta_LJ(f))
        # L_V = np.sin(self.theta_LJ(f)) * (np.cos(self.phi_LJ(f)) * sin_o_XH + np.sin(self.phi_LJ(f)) * cos_i_JN * cos_o_XH) - sin_i_JN * cos_o_XH * np.cos(self.theta_LJ(f))
        # L_N = np.sin(self.theta_LJ(f)) * np.sin(self.phi_LJ(f)) * sin_i_JN + np.cos(self.theta_LJ(f)) * cos_i_JN

        L_z = (
            np.sin(self.theta_LJ(f))
            * (
                np.cos(self.phi_LJ(f)) * sin_o_XH
                + np.sin(self.phi_LJ(f)) * cos_i_JN * cos_o_XH
            )
            - sin_i_JN * cos_o_XH * np.cos(self.theta_LJ(f))
        ) * np.sin(self.theta_S) + (
            np.sin(self.theta_LJ(f)) * np.sin(self.phi_LJ(f)) * sin_i_JN
            + np.cos(self.theta_LJ(f)) * cos_i_JN
        ) * np.cos(
            self.theta_S
        )
        return L_z

    def phi_L(self, f):
        """
        Evolution of the orbital angular momentum vector in the detector frame (azimuthal angle).

        Parameters
        ----------
        f : float
            Frequency at which the angle is to be calculated [Hz].

        Returns
        -------
        Phi_L : float
            Phase of the orbital angular momentum vector.

        Notes
        -----
        Implements Equation A8 from Taman et al. 2025.
        """
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        L_H = np.sin(self.theta_LJ(f)) * (
            np.cos(self.phi_LJ(f)) * cos_o_XH
            - np.sin(self.phi_LJ(f)) * cos_i_JN * sin_o_XH
        ) + sin_i_JN * sin_o_XH * np.cos(self.theta_LJ(f))
        L_V = np.sin(self.theta_LJ(f)) * (
            np.cos(self.phi_LJ(f)) * sin_o_XH
            + np.sin(self.phi_LJ(f)) * cos_i_JN * cos_o_XH
        ) - sin_i_JN * cos_o_XH * np.cos(self.theta_LJ(f))
        L_N = (
            np.sin(self.theta_LJ(f)) * np.sin(self.phi_LJ(f)) * sin_i_JN
            + np.cos(self.theta_LJ(f)) * cos_i_JN
        )

        L_x = (
            -np.sin(self.phi_S) * L_H
            - np.cos(self.theta_S) * np.cos(self.phi_S) * L_V
            + np.sin(self.theta_S) * np.cos(self.phi_S) * L_N
        )
        L_y = (
            np.cos(self.phi_S) * L_H
            - np.cos(self.theta_S) * np.sin(self.phi_S) * L_V
            + np.sin(self.theta_S) * np.sin(self.phi_S) * L_N
        )
        Phi_L = np.arctan2(L_y, L_x)
        # Phi_L_ur = np.unwrap(Phi_L, discont = np.pi)
        return Phi_L  # _ur

    def strain(self, f, delta_f=0.25, frequencySeries=True):
        """
        Calculate the complete gravitational wave strain with regular precession.

        Parameters
        ----------
        f : array_like
            Frequencies at which to evaluate the strain [Hz].
        delta_f : float, optional
            Frequency spacing for FrequencySeries output (default: 0.25 Hz).
        frequencySeries : bool, optional
            If True, return pycbc.FrequencySeries; if False, return ndarray (default: True).

        Returns
        -------
        FrequencySeries or ndarray
            Complex gravitational wave strain h(f) = A(f) * exp(i*Phi(f)).

        Notes
        -----
        The total phase is Phi = Psi - phi_P - 2*delta_phi_P,
        combining the orbital phase, polarization phase, and precession correction.
        """
        strain = self.amplitude(f) * np.exp(
            1j * (self.Psi(f) - self.phase_phi_P(f) - 2 * self.phase_delta_phi(f))
        )
        if frequencySeries:
            return FrequencySeries(strain, delta_f)
        return strain
