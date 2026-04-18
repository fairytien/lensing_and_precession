"""Orientation utilities for parameter resolution and tagging.

Provides helpers to:
- Build a stable orientation tag from explicit angles
- Resolve an orientation preset (author_orientation) into parameters and tag
"""

from typing import Optional, Dict, Tuple
import logging

from modules.waveform import set_orientation


def orientation_tag(
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
) -> str:
    """Return a filename-safe tag from explicit angles or default preset tag.

    If any angle is provided, produce a deterministic string including their
    values (with 'nan' for missing angles). Otherwise return a default
    'Author_orientation' tag.
    """
    vals = [theta_J, phi_J, theta_S, phi_S]
    if any(v is not None for v in vals):

        def fmt(x):
            return "nan" if x is None else f"{float(x):.3f}"

        return (
            f"custom_thetaJ{fmt(theta_J)}_phiJ{fmt(phi_J)}_"
            f"thetaS{fmt(theta_S)}_phiS{fmt(phi_S)}"
        )
    return "Taman_edgeon"


def resolve_orientation(
    orient_preset: Optional[str],
    theta_J: Optional[float],
    phi_J: Optional[float],
    theta_S: Optional[float],
    phi_S: Optional[float],
    base_params: Dict,
    orient_params: Dict,
    default_author: str = "Taman",
    default_orientation: str = "edgeon",
) -> Tuple[Dict, str]:
    """Resolve orientation parameters and filename tag.

    If orient_preset is provided (e.g., 'Taman_edgeon'), use it to set params
    and tag. If not, start from defaults and apply overrides from explicit
    angles, and build a tag from those angles.
    Returns (params, tag).
    """
    if orient_preset:
        try:
            author, orientation = orient_preset.split("_", 1)
            preset = orient_params[author][orientation]
        except Exception as exc:
            raise ValueError(
                f"Invalid orient_preset '{orient_preset}'. Use one of the allowed choices."
            ) from exc
        params = set_orientation(preset, base_params)[0]
        if any(v is not None for v in (theta_J, phi_J, theta_S, phi_S)):
            logging.info(
                "orient_preset provided; ignoring explicit angle overrides for tag and params."
            )
        return params, orient_preset

    # No preset: start from default, then apply any explicit angles and tag by angles
    params = set_orientation(
        orient_params[default_author][default_orientation], base_params
    )[0]
    if theta_J is not None:
        params["theta_J"] = theta_J
    if phi_J is not None:
        params["phi_J"] = phi_J
    if theta_S is not None:
        params["theta_S"] = theta_S
    if phi_S is not None:
        params["phi_S"] = phi_S

    tag = orientation_tag(theta_J, phi_J, theta_S, phi_S)
    return params, tag


def allowed_orient_presets(_orient_params: Dict) -> list:
    """Return a sorted list of preset names as 'Author_orientation'."""
    return sorted(
        [f"{author}_{name}" for author, sub in _orient_params.items() for name in sub]
    )
