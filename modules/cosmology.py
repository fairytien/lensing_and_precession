"""
Cosmology utilities for redshift ↔ luminosity distance and mass frame conversions.

Uses Planck 2018 FlatΛCDM cosmology (H0=67.4, Ωm=0.315).

Primary implementation uses astropy; falls back to pure scipy if astropy is
unavailable or incompatible with the installed NumPy version.
"""

from typing import Union
import numpy as np

from .default_params_v3 import SOLMASS2SEC, GIGAPC2SEC

ZMIN = 1e-8
ZMAX = 20.0


#############################
# Section 1: Cosmology Setup #
#############################

# Planck 2018 cosmological parameters
H0 = 67.4  # km/s/Mpc
OM0 = 0.315  # matter density
OL0 = 1.0 - OM0  # dark energy density (flat universe)

# NumPy compatibility shim for older astropy versions (< 5.0) that expect np.asscalar
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item()  # type: ignore[attr-defined]

# Try to use astropy; fall back to scipy-based implementation if unavailable
_USE_ASTROPY = False
try:
    from astropy.cosmology import FlatLambdaCDM, z_at_value
    from astropy import units as u

    COSMO = FlatLambdaCDM(H0=H0, Om0=OM0)
    _USE_ASTROPY = True
except Exception:
    # astropy unavailable or incompatible — will use fallback functions
    pass


###########################################
# Section 2: Fallback (scipy-based) Impl. #
###########################################

if not _USE_ASTROPY:
    from scipy.optimize import brentq
    from scipy.integrate import quad

    C_KM_S = 299792.458  # speed of light [km/s]
    DH = C_KM_S / H0  # Hubble distance [Mpc]

    def _Ez(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        return np.sqrt(OM0 * (1 + z) ** 3 + OL0)

    def _comoving_distance_mpc(z: float) -> float:
        """Comoving distance in Mpc via numerical integration."""
        if z <= 0:
            return 0.0
        integral, _ = quad(lambda zp: 1.0 / _Ez(zp), 0.0, z)
        return DH * integral


##################################
# Section 3: z ↔ D_L Conversions #
##################################


def z_to_DL(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert redshift to luminosity distance.

    Args:
        z: Redshift (scalar or array).

    Returns:
        Luminosity distance in Gpc.
    """
    if _USE_ASTROPY:
        return COSMO.luminosity_distance(z).to(u.Gpc).value

    # Fallback: scipy-based
    z = np.asarray(z)
    scalar_input = z.ndim == 0
    z = np.atleast_1d(z)

    DL_mpc = np.array([(1 + zi) * _comoving_distance_mpc(zi) for zi in z])
    DL_gpc = DL_mpc / 1000.0  # Mpc -> Gpc

    if scalar_input:
        return float(DL_gpc[0])
    return DL_gpc


def DL_to_z(DL_gpc: float, zmin: float = ZMIN, zmax: float = ZMAX) -> float:
    """
    Convert luminosity distance to redshift by numerical inversion.

    Args:
        DL_gpc: Luminosity distance in Gpc.
        zmin: Lower bound for root finding. Default ZMIN.
        zmax: Upper bound for root finding. Default ZMAX.

    Returns:
        Redshift corresponding to the given luminosity distance.

    Raises:
        ValueError: If DL_gpc <= 0.
    """
    if DL_gpc <= 0:
        raise ValueError("Luminosity distance must be positive.")

    if _USE_ASTROPY:
        z = z_at_value(COSMO.luminosity_distance, DL_gpc * u.Gpc, zmin=zmin, zmax=zmax)
        return float(z)

    # Fallback: scipy brentq
    from scipy.optimize import brentq

    def residual(z):
        return z_to_DL(z) - DL_gpc

    z = brentq(residual, zmin, zmax)
    return float(z)


#####################################
# Section 3: Mass Frame Conversions #
#####################################


def mcz_src_to_det(
    mcz_src: Union[float, np.ndarray], z: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert source-frame chirp mass to detector-frame chirp mass.

    M_det = M_src * (1 + z)

    Args:
        mcz_src: Source-frame chirp mass (any units).
        z: Redshift.

    Returns:
        Detector-frame chirp mass (same units as input).
    """
    return mcz_src * (1 + z)


def mcz_det_to_src(
    mcz_det: Union[float, np.ndarray], z: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert detector-frame chirp mass to source-frame chirp mass.

    M_src = M_det / (1 + z)

    Args:
        mcz_det: Detector-frame chirp mass (any units).
        z: Redshift.

    Returns:
        Source-frame chirp mass (same units as input).
    """
    return mcz_det / (1 + z)


def source_mass_redshift_scale(z_from: float, z_to: float) -> float:
    """Scale source-frame masses when remapping between redshifts.

    For a fixed detector-frame mass,
    m_src(z_to) = m_src(z_from) * ((1 + z_from) / (1 + z_to)).

    Args:
        z_from: Original source redshift.
        z_to: Target source redshift.

    Returns:
        Multiplicative source-mass scale factor.

    Raises:
        ValueError: If either redshift is non-finite or <= 0.
    """
    z_from = float(z_from)
    z_to = float(z_to)
    if not np.isfinite(z_from) or not np.isfinite(z_to):
        raise ValueError("z_from and z_to must be finite")
    if z_from <= 0 or z_to <= 0:
        raise ValueError("z_from and z_to must be > 0")
    return (1.0 + z_from) / (1.0 + z_to)


#######################################
# Section 4: Parameter Dict Utilities #
#######################################


def apply_z(params: dict, z: float, mcz_is_source: bool = True) -> dict:
    """
    Apply redshift to a parameter dict, updating `dist` and optionally `mcz`.

    Args:
        params: Parameter dict with keys `mcz` (in seconds) and `dist` (in seconds).
        z: Redshift to apply.
        mcz_is_source: If True (default), treats `params["mcz"]` as source-frame
            and converts to detector-frame. If False, leaves `mcz` unchanged.

    Returns:
        A shallow-copied dict with `dist` set from z and `mcz` redshifted if
        requested. The input `params` dict is not mutated.

    Raises:
        ValueError: If z is not finite or z <= 0.
    """
    z = float(z)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(
            "apply_z requires a finite redshift z > 0 because it maps z to luminosity distance. "
            "Use a small positive value such as z=1e-8 for near-zero cosmology, or skip apply_z to keep dist unchanged."
        )

    out = params.copy()
    out["dist"] = z_to_DL(z) * GIGAPC2SEC

    if mcz_is_source:
        out["mcz"] = mcz_src_to_det(out["mcz"], z)

    return out
