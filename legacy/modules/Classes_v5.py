#############################
# Section: Precessing v5     #
#############################

import numpy as np
from typing import Optional
from scipy.integrate import quad

from .Classes_v3 import Precessing as _PrecessingV3


class Precessing(_PrecessingV3):
    """Precessing class v5 (quad-based integrator).

    Computes `phase_delta_phi` using SciPy's adaptive quadrature `scipy.integrate.quad`.
    Integration is performed piecewise between successive points of the provided
    frequency array and accumulated to build the cumulative phase.

    Notes
    -----
    - The integrand is taken from `self.integrand_delta_phi(y, f)` with the first
      argument ignored (set to 0.0) since the integrand does not depend on the
      integrated value in this model.
    - `quad` tolerances map to `epsrel` (relative) and `epsabs` (absolute).
      If these are left as "default", SciPy's defaults are used.
    """

    def phase_delta_phi(
        self,
        f,
        epsrel: Optional[float] = "default",
        epsabs: Optional[float] = "default",
        limit: Optional[int] = "default",
    ):
        """Compute delta phi_P using adaptive quadrature (quad).

        Parameters
        ----------
        f : array-like
            Strictly increasing frequency array where the cumulative integral is evaluated.
        epsrel : float, optional
            Relative error tolerance for `quad`. If "default", uses SciPy's default.
        epsabs : float, optional
            Absolute error tolerance for `quad`. If "default", uses SciPy's default.
        limit : int, optional
            An upper bound on the number of subintervals used by `quad`. If "default", SciPy's default is used.

        Returns
        -------
        np.ndarray
            Cumulative integral values with the same shape as `f`.
        """
        f = np.asarray(f)
        if f.ndim != 1:
            f = f.ravel()
        if f.size == 0:
            return f.astype(float)
        if not np.all(np.diff(f) > 0):
            raise ValueError("Frequency array f must be strictly increasing.")

        def _integrand(freq: float) -> float:
            # Use y=0.0 since the integrand is independent of y for this problem
            return float(self.integrand_delta_phi(0.0, float(freq)))

        quad_kwargs = {}
        if epsrel != "default":
            quad_kwargs["epsrel"] = epsrel
        if epsabs != "default":
            quad_kwargs["epsabs"] = epsabs
        if limit != "default":
            quad_kwargs["limit"] = int(limit)

        # Vectorized approach: use numpy.vectorize for quad calls
        def _quad_integrate(f_start, f_end):
            val, _err = quad(_integrand, f_start, f_end, **quad_kwargs)
            return val

        # Vectorize the quad integration over frequency intervals
        quad_vec = np.vectorize(_quad_integrate)

        # Create intervals: [f[0], f[1]], [f[1], f[2]], ..., [f[n-2], f[n-1]]
        f_starts = f[:-1]
        f_ends = f[1:]

        # Compute integrals for all intervals at once
        interval_integrals = quad_vec(f_starts, f_ends)

        # Cumulative sum to get the final result
        cumulative = np.concatenate([[0.0], np.cumsum(interval_integrals)])

        return cumulative
