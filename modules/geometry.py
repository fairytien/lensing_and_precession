"""Geometry helpers for source-orientation calculations.

This module is a source-of-truth implementation for orientation utilities
formerly kept in `functions_v3.py`.
"""

from typing import Tuple, Union

import numpy as np


def calculate_cosJN_params(params: dict) -> float:
    """Return cos(J,N) from parameter dictionary entries."""
    return np.sin(params["theta_J"]) * np.sin(params["theta_S"]) * np.cos(
        params["phi_J"] - params["phi_S"]
    ) + np.cos(params["theta_J"]) * np.cos(params["theta_S"])


def calculate_cosJN(
    phi_S: Union[float, np.ndarray],
    theta_S: Union[float, np.ndarray],
    phi_J: Union[float, np.ndarray],
    theta_J: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Return cos(J,N) from explicit angle arguments."""
    print("order of arguments: phi_S, theta_S, phi_J, theta_J")
    return np.sin(theta_J) * np.sin(theta_S) * np.cos(phi_J - phi_S) + np.cos(
        theta_J
    ) * np.cos(theta_S)


def find_FaceOn_coords(
    fix: str, fixed_phi: float, fixed_theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Find angle-grid coordinates that are face-on (|cos JN| ~ 1)."""
    n_pts = 151
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = calculate_cosJN(fixed_phi, fixed_theta, X, Y)
    else:
        Z = calculate_cosJN(X, Y, fixed_phi, fixed_theta)

    cond = np.isclose(np.abs(Z), 1, rtol=0, atol=1e-3)
    return X[cond], Y[cond]


def find_EdgeOn_coords(
    fix: str, fixed_phi: float, fixed_theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Find angle-grid coordinates that are edge-on (|cos JN| ~ 0)."""
    n_pts = 151
    phi_arr = np.linspace(0, 2 * np.pi, n_pts)
    theta_arr = np.linspace(0, np.pi, n_pts)
    X, Y = np.meshgrid(phi_arr, theta_arr)

    if fix == "S":
        Z = calculate_cosJN(fixed_phi, fixed_theta, X, Y)
    else:
        Z = calculate_cosJN(X, Y, fixed_phi, fixed_theta)

    cond = np.isclose(np.abs(Z), 0, rtol=0, atol=1e-2)
    return X[cond], Y[cond]
