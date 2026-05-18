from typing import Sequence, Tuple

import numpy as np

from modules.default_params import SOLMASS2SEC
from modules.waveform import get_fcut_from_mcz


def _mcz_extremum_for_n(td_s: float, n: float, eta: float = 0.25) -> float:
    """Calculate mcz extremum for given time delay and index n.

    For troughs: n = n_trough + 0.5
    For peaks: n = n_peak (integer >= 1)
    """
    return (eta ** (3 / 5) * td_s) / (6 ** (3 / 2) * np.pi * n) / SOLMASS2SEC


def _find_mcz_extrema(
    td_arr: np.ndarray,
    eta: float,
    mcz_min: float,
    mcz_max: float,
    n_start: float,
    n_increment: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generic function to find mcz extrema (troughs or peaks) within range.

    Parameters
    ----------
    td_arr : np.ndarray
        Array of time delays in seconds
    eta : float
        Symmetric mass ratio
    mcz_min, mcz_max : float
        Chirp mass range boundaries in solar masses
    n_start : float
        Starting value for n (0.5 for troughs, 1 for peaks)
    n_increment : float
        Increment for n (1.0 for both)

    Returns
    -------
    tuple
        (td_points, mcz_points) arrays
    """
    td_points = []
    mcz_points = []

    for td in td_arr:
        n = n_start
        while True:
            mcz = _mcz_extremum_for_n(td, n, eta)
            if mcz < mcz_min:
                break
            if mcz <= mcz_max:
                td_points.append(td)
                mcz_points.append(mcz)
            n += n_increment

    return np.array(td_points), np.array(mcz_points)


def find_mcz_troughs(
    td_arr: np.ndarray, eta: float = 0.25, mcz_min: float = 10.0, mcz_max: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Find mcz_trough points for each time delay within the mcz range."""
    return _find_mcz_extrema(
        td_arr, eta, mcz_min, mcz_max, n_start=0.5, n_increment=1.0
    )


def find_mcz_peaks(
    td_arr: np.ndarray, eta: float = 0.25, mcz_min: float = 10.0, mcz_max: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Find mcz_peak points for each time delay within the mcz range."""
    return _find_mcz_extrema(
        td_arr, eta, mcz_min, mcz_max, n_start=1.0, n_increment=1.0
    )


def fixed_mcz_cycle_positions_ms(
    mcz_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    *,
    eta: float = 0.25,
    f_min: float = 20.0,
    cycle_counts: Sequence[int] = (1, 2, 3),
) -> dict[int, float]:
    """Return visible fixed-mass lensing-cycle positions on a td axis."""
    f_cut = float(get_fcut_from_mcz(float(mcz_msun), eta=eta))
    delta_f = f_cut - float(f_min)
    if delta_f <= 0:
        return {}
    positions: dict[int, float] = {}
    for cycle_count in cycle_counts:
        td_ms = 1e3 * float(cycle_count) / delta_f
        if td_min_ms <= td_ms <= td_max_ms:
            positions[int(cycle_count)] = td_ms
    return positions


def _fixed_mcz_extrema_positions_ms(
    mcz_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    *,
    eta: float,
    n_start: float,
) -> np.ndarray:
    f_cut = float(get_fcut_from_mcz(float(mcz_msun), eta=eta))
    if f_cut <= 0:
        return np.array([], dtype=float)

    positions_ms = []
    n_value = float(n_start)
    while True:
        td_ms = 1e3 * n_value / f_cut
        if td_ms > td_max_ms:
            break
        if td_ms >= td_min_ms:
            positions_ms.append(td_ms)
        n_value += 1.0
    return np.asarray(positions_ms, dtype=float)


def fixed_mcz_peak_positions_ms(
    mcz_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    *,
    eta: float = 0.25,
) -> np.ndarray:
    """Return visible fixed-mass peak positions on a td axis."""
    return _fixed_mcz_extrema_positions_ms(
        mcz_msun,
        td_min_ms,
        td_max_ms,
        eta=eta,
        n_start=1.0,
    )


def fixed_mcz_trough_positions_ms(
    mcz_msun: float,
    td_min_ms: float,
    td_max_ms: float,
    *,
    eta: float = 0.25,
) -> np.ndarray:
    """Return visible fixed-mass trough positions on a td axis."""
    return _fixed_mcz_extrema_positions_ms(
        mcz_msun,
        td_min_ms,
        td_max_ms,
        eta=eta,
        n_start=0.5,
    )
