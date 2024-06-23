###########
# UPDATES #
###########
# 1. 2024-02-07: changed C_amp in polarization_amplitude_and_phase in Precessing class to the correct equation (4a in Evangelos's)

#############################
# Section 1: Import Modules #
#############################

# if running on Google Colab, uncomment the following lines
# import sys
# !{sys.executable} -m pip install pycbc ligo-common --no-cache-dir

import numpy as np

error_handler = np.seterr(invalid="raise")
from scipy.integrate import odeint
import scipy.special as sc
import mpmath as mp
from pycbc.types import FrequencySeries

############################
# Section 2: Lensing Class #
############################


class Lensing:
    def __init__(self, params):
        self.params = params

        assert type(self.params == dict), "Parameters should be a dictionary"

        # unlensed parameters
        self.theta_s = params["theta_S"]
        self.phi_s = params["phi_S"]
        self.theta_l = params["theta_L"]
        self.phi_l = params["phi_L"]
        self.mcz = params["mcz"]
        self.dist = params["dist"]
        self.eta = params["eta"]
        self.tc = params["tc"]
        self.phi_c = params["phi_c"]

        # lensed parameters
        self.M_Lz = params["MLz"]
        self.y = params["y"]

    def total_mass(self):
        """Total mass from chirp mass [seconds]"""
        return self.mcz / (self.eta ** (3 / 5))

    def f_cut(self):
        """f_cut"""
        return 1 / (6 ** (3 / 2) * np.pi * self.total_mass())

    def l_dot_n(self):
        """(cosine angle between l and n)"""
        cos_term = np.cos(self.theta_s) * np.cos(self.theta_l)
        sin_term = (
            np.sin(self.theta_s)
            * np.sin(self.theta_l)
            * np.cos(self.phi_s - self.phi_l)
        )
        inner_prod = cos_term + sin_term
        return inner_prod

    def amp(self):
        """A for h(f)"""
        amplitude = (
            np.sqrt(5 / 96) * np.pi ** (-2 / 3) * self.mcz ** (5 / 6) / (self.dist)
        )
        return amplitude

    def psi(self, f):
        """eqn 3.13 in Cutler-Flanaghan 1994"""
        x = (np.pi * self.total_mass() * f) ** (2 / 3)
        term1 = 2 * np.pi * f * self.tc - self.phi_c - np.pi / 4
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

        numerator = np.cos(self.theta_l) - np.cos(self.theta_s) * (self.l_dot_n())
        denominator = (
            np.sin(self.theta_s)
            * np.sin(self.theta_l)
            * np.sin(self.phi_l - self.phi_s)
        )

        psi_s_val = np.arctan2(numerator, denominator)
        return psi_s_val

    def fIp(self):
        """F_plus"""

        term_1 = (
            1
            / 2
            * (1 + np.power(np.cos(self.theta_s), 2))
            * np.cos(2 * self.phi_s)
            * np.cos(2 * self.psi_s())
        )
        term_2 = (
            np.cos(self.theta_s) * np.sin(2 * self.phi_s) * np.sin(2 * self.psi_s())
        )

        fIp_val = term_1 - term_2
        return fIp_val

    def fIc(self):
        """F_cross"""

        term_1 = (
            1
            / 2
            * (1 + np.power(np.cos(self.theta_s), 2))
            * np.cos(2 * self.phi_s)
            * np.sin(2 * self.psi_s())
        )
        term_2 = (
            np.cos(self.theta_s) * np.sin(2 * self.phi_s) * np.cos(2 * self.psi_s())
        )

        fIc_val = term_1 + term_2
        return fIc_val

    def lambdaI(self):
        """|F_plus (1+L.N**2) - i (2*F_cross*L.N)|"""

        term_1 = np.power(2 * self.l_dot_n() * self.fIc(), 2)
        term_2 = np.power((1 + np.power(self.l_dot_n(), 2)) * self.fIp(), 2)
        lambdaI_val = np.sqrt(term_1 + term_2)
        return lambdaI_val

    def phi_pI(self):
        """tan-1((2*F_cross*L.N)/F_plus (1+L.N**2))"""

        numerator = 2 * self.l_dot_n() * self.fIc()
        denominator = (1 + np.power(self.l_dot_n(), 2)) * self.fIp()

        phi_pI_val = np.arctan2(numerator, denominator)
        return phi_pI_val

    def hI(self, f):
        """Unlensed Waveform"""

        term_1 = self.lambdaI()
        term_2 = np.exp(-1j * self.phi_pI())
        term_3 = self.amp() * f ** (-7 / 6)
        term_4 = np.exp(1j * self.psi(f))

        signal_I = term_1 * term_2 * term_3 * term_4

        return signal_I

    def F(self, f):
        """PM amplification factor in exact form, equation 17 in Takahashi & Nakamura 2003"""
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
        """lensed strain = unlensed strain * amplification factor
        Args:
            f (numpy array): frequency range
            delta_f (float): interval length of frequency. Default at 0.25 Hz.
            frequencySeries (bool): True for FrequencySeries. False otherwise.

        Returns:
            hL (numpy array): lensed strain
        """
        hL = self.hI(f) * self.F(f)

        if frequencySeries:
            return FrequencySeries(hL, delta_f)

        return hL


class LensingGeo(Lensing):
    def __init__(self, params):
        super().__init__(params)

    def mu_plus(self):
        """plus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023"""
        mu_plus_val = (
            1 / 2 + (self.y**2 + 2) / (2 * self.y * np.sqrt(self.y**2 + 4)) + 0j
        )
        return mu_plus_val

    def mu_minus(self):
        """minus magnification, equation 18 in Takahashi & Nakamura 2003, also 16a in Saif et al. 2023"""
        mu_minus_val = (
            1 / 2 - (self.y**2 + 2) / (2 * self.y * np.sqrt(self.y**2 + 4)) + 0j
        )
        return mu_minus_val

    def I(self):
        """flux ratio, equation 17a in Saif et al. 2023"""
        I_val = np.abs(self.mu_minus()) / np.abs(self.mu_plus())
        return I_val

    def Delta_td(self):
        """time delay, equation 16b in Saif et al. 2023"""
        Delta_td_val = (
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
        return Delta_td_val

    def F(self, f):
        """PM amplification factor in geometric optics limit, equation 18 in Takahashi & Nakamura 2003"""
        F_val = np.sqrt(np.abs(self.mu_plus())) - 1j * np.sqrt(
            np.abs(self.mu_minus())
        ) * np.exp(2j * np.pi * f * self.Delta_td())
        return F_val


###############################
# Section 3: Precessing Class #
###############################


class Precessing:
    def __init__(self, params):
        self.params = params

        # non-precession/unlensed parameters
        self.theta_S = params["theta_S"]
        self.phi_S = params["phi_S"]
        self.theta_J = params["theta_J"]
        self.phi_J = params["phi_J"]
        self.mcz = params["mcz"]
        self.dist = params["dist"]
        self.eta = params["eta"]
        self.tc = params["tc"]
        self.phi_c = params["phi_c"]

        # regular precession parameters
        self.theta_tilde = params["theta_tilde"]
        self.omega_tilde = params["omega_tilde"]
        self.gamma_P = params["gamma_P"]

        # some converters/constants

        self.SOLMASS2SEC = 4.92624076 * 1e-6  # solar mass -> seconds
        self.GIGAPC2SEC = 1.02927125 * 1e17  # gigaparsec -> seconds
        self.FMIN = 20  # lower frequency of the detector sensitivity band [Hz]

    def total_mass(self):
        """Total mass from chirp mass [seconds]"""
        return self.mcz / (self.eta ** (3 / 5))

    def f_cut(self):
        """f_cut"""
        return 1 / (6 ** (3 / 2) * np.pi * self.total_mass())

    def theta_LJ(self, f):
        """theta_LJ_new"""
        return 0.1 * self.theta_tilde * (f / self.f_cut()) ** (1 / 3)

    def phi_LJ(self, f):
        """phi_LJ"""
        num = (5000 / 96) * self.omega_tilde
        deno = (
            (self.total_mass() / self.SOLMASS2SEC)
            * (np.pi ** (8 / 3))
            * (self.mcz ** (5 / 3))
            * (self.f_cut() ** (5 / 3))
        )
        phi_LJ_amp = num / deno
        return phi_LJ_amp * (1 / self.FMIN - 1 / f) + self.gamma_P

    def amp_prefactor(self) -> float:
        """amplitude prefactor calculated using chirp mass and distance"""
        amp_prefactor = (
            np.sqrt(5 / 96) * (np.pi ** (-2 / 3)) * (self.mcz ** (5 / 6)) / self.dist
        )
        return amp_prefactor

    def precession_angles(self):
        """some angles"""

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

        if sin_i_JN == 0:
            cos_o_XH = 1
            sin_o_XH = 0
        else:
            cos_o_XH = (
                np.cos(self.theta_S)
                * np.sin(self.theta_J)
                * np.cos(self.phi_J - self.phi_S)
                - np.sin(self.theta_S) * np.cos(self.theta_J)
            ) / (
                sin_i_JN
            )  # seems to be cos Omega_{XH}
            sin_o_XH = (np.sin(self.theta_J) * np.sin(self.phi_J - self.phi_S)) / (
                sin_i_JN
            )
        return cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH

    def LdotN(self, f):
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        LdotN = (
            np.sin(self.theta_LJ(f)) * sin_i_JN * np.sin(self.phi_LJ(f))
            + np.cos(self.theta_LJ(f)) * cos_i_JN
        )
        return LdotN

    def polarization_amplitude_and_phase(self, f):
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        # for C
        C_amp = np.sqrt(
            0.25
            * (1 + (np.cos(self.theta_S)) ** 2) ** 2
            * ((np.cos(2 * self.phi_S)) ** 2)
            + ((np.cos(self.theta_S)) ** 2 * (np.sin(2 * self.phi_S)) ** 2)
        )

        # define alpha
        sin_alpha = np.cos(self.theta_S) * np.sin(2 * self.phi_S) / C_amp
        cos_alpha = (
            (1 + np.cos(self.theta_S) ** 2) * np.cos(2 * self.phi_S) / (2 * C_amp)
        )

        # define tan_psi
        num_psi = (
            np.sin(self.theta_LJ(f))
            * (
                np.cos(self.phi_LJ(f)) * sin_o_XH
                + np.sin(self.phi_LJ(f)) * cos_i_JN * cos_o_XH
            )
            - np.cos(self.theta_LJ(f)) * sin_i_JN * cos_o_XH
        )
        den_psi = (
            np.sin(self.theta_LJ(f))
            * (
                np.cos(self.phi_LJ(f)) * cos_o_XH
                - np.sin(self.phi_LJ(f)) * cos_i_JN * sin_o_XH
            )
            + np.cos(self.theta_LJ(f)) * sin_i_JN * sin_o_XH
        )
        if self.phi_S == self.phi_J:
            if self.theta_S == self.theta_J:
                tan_psi = np.tan(self.phi_LJ(f))
            else:
                tan_psi = num_psi / den_psi

        else:
            tan_psi = num_psi / den_psi

        if den_psi.all() == 0:
            if self.theta_tilde == 0:
                return C_amp, 0, -1

        # define  2 * psi + alpha
        sin_2pa = (2 * cos_alpha * tan_psi + sin_alpha * (1 - (tan_psi) ** 2)) / (
            1 + (tan_psi) ** 2
        )
        cos_2pa = (cos_alpha * (1 - (tan_psi) ** 2) - 2 * sin_alpha * tan_psi) / (
            1 + (tan_psi) ** 2
        )

        return C_amp, sin_2pa, cos_2pa

    ### get the amplitude
    def amplitude(self, f) -> np.ndarray:
        """NP/Unlensed amplitude"""
        LdotN = self.LdotN(f)
        C_amp, sin_2pa, cos_2pa = self.polarization_amplitude_and_phase(f)

        amp = (
            self.amp_prefactor()
            * C_amp
            * f ** (-7 / 6)
            * np.sqrt(4 * LdotN**2 * sin_2pa**2 + cos_2pa**2 * (1 + LdotN**2) ** 2)
        )
        return amp

    ### get the phase phi_P
    def phase_phi_P(self, f):
        """phi_p"""
        LdotN = self.LdotN(f)
        C_amp, sin_2pa, cos_2pa = self.polarization_amplitude_and_phase(f)

        phi_p_temp = np.arctan2(2 * LdotN * sin_2pa, (1 + LdotN**2) * cos_2pa)
        phi_p = np.unwrap(phi_p_temp, discont=np.pi)
        return phi_p

    def f_dot(self, f):
        """df/dt from Cutler Flanagan 1994"""
        prefactor = (96 / 5) * np.pi ** (8 / 3) * self.mcz ** (5 / 3) * f ** (11 / 3)
        return prefactor  # * (1 - (743/336 + (11/4) * self.eta) * (np.pi * self.total_mass() * f)**(2/3) + 4 * np.pi * (np.pi * self.total_mass() * f))

    ### get the delta phi_P
    def integrand_delta_phi(self, y, f):
        """integrand for delta phi p (equations in Apostolatos 1994, and appendix of Evangelos in prep)"""
        LdotN = self.LdotN(f)
        cos_i_JN, sin_i_JN, cos_o_XH, sin_o_XH = self.precession_angles()
        f_dot = self.f_dot(f)

        Omega_LJ = (
            1000
            * self.omega_tilde
            * (f / self.f_cut()) ** (5 / 3)
            / (self.total_mass() / self.SOLMASS2SEC)
        )

        # if self.theta_tilde == 0:  # non-precessing
        #     integrand_delta_phi = 0
        #     # not necessary to include this case, but just in case, check equations 17, 18a, A18 in Evangelos

        if cos_i_JN == 1:  # face-on (precessing & non-precessing)
            if self.theta_tilde == 0:
                integrand_delta_phi = 0
            else:
                integrand_delta_phi = -Omega_LJ * np.cos(self.theta_LJ(f)) / f_dot

        # elif LdotN == 1: # TODO: check this case
        #     # NOT face-on & STILL precessing, when L and N are aligned at some point in the precession cycle
        #     # very rare, L aligns with N only ONCE as it spirals out --> blows up???
        #     integrand_delta_phi = 0

        else:
            integrand_delta_phi = (
                (LdotN / (1 - LdotN**2))
                * Omega_LJ
                * np.sin(self.theta_LJ(f))
                * (
                    np.cos(self.theta_LJ(f)) * sin_i_JN * np.sin(self.phi_LJ(f))
                    - np.sin(self.theta_LJ(f)) * cos_i_JN
                )
                / f_dot
            )

        return integrand_delta_phi

    def phase_delta_phi(self, f):
        """integrate the delta_phi integrand"""
        integral = odeint(self.integrand_delta_phi, 0, f)
        return np.squeeze(integral)

    def Psi(self, f):
        """GW phase"""
        x = (np.pi * self.total_mass() * f) ** (2 / 3)
        Psi = (
            (2 * np.pi * f * self.tc)
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
        """for figure 2 in Evangelos"""
        # from equation A8
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
        """for figure 2 in Evangelos"""
        # from equation A8
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
        """precessing GW"""
        strain = self.amplitude(f) * np.exp(
            1j * (self.Psi(f) - self.phase_phi_P(f) - 2 * self.phase_delta_phi(f))
        )
        if frequencySeries:
            return FrequencySeries(strain, delta_f)
        return strain
