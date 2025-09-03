#############################
# Section: Precessing v4     #
#############################

import numpy as np
from scipy.integrate import solve_ivp
from typing import Union, Sequence, Dict

from .Classes_v3 import Precessing as _PrecessingV3
from pycbc.types import FrequencySeries


class Precessing(_PrecessingV3):
    """Precessing class v4 (solve_ivp-only variant).

    Computes phase_delta_phi exclusively with SciPy's solve_ivp.
    Accepts a single method name or a list of method names and returns either
    a single array or a dict of arrays keyed by method.
    """

    def phase_delta_phi(
        self,
        f,
        ivp_method: Union[str, Sequence[str]] = "LSODA",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: float = np.inf,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Compute delta phi_P using solve_ivp.

        Args:
            f (array-like): Strictly increasing frequency array.
            ivp_method (str | Sequence[str]): One or more solve_ivp methods
                (e.g., "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA").
            rtol (float): Relative tolerance for solve_ivp.
            atol (float): Absolute tolerance for solve_ivp.
            max_step (float): Maximum allowed step size.

        Returns:
            If ivp_method is a string, returns np.ndarray (len(f),).
            If ivp_method is a sequence, returns dict method -> np.ndarray.
        """
        f = np.asarray(f)
        if f.ndim != 1:
            f = f.ravel()
        if not np.all(np.diff(f) > 0):
            raise ValueError("Frequency array f must be strictly increasing.")

        def _solve_with_method(method_name: str) -> np.ndarray:
            # Use solve_ivp
            def rhs(freq, y):
                # dy/df = integrand_delta_phi(y, f)
                # Support both y.shape == (1,) and vectorized calls with y.shape == (1, m)
                try:
                    # Compute scalar integrand value (independent of y in our model)
                    val = float(self.integrand_delta_phi(0.0, freq))
                except Exception:
                    val = float(self.integrand_delta_phi(y, freq))

                y_arr = np.asarray(y)
                if y_arr.ndim == 0:
                    return np.asarray([val], dtype=float)
                # Broadcast to match shape of y
                return np.full_like(y_arr, fill_value=val, dtype=float)

            sol = solve_ivp(
                rhs,
                (float(f[0]), float(f[-1])),
                y0=[0.0],
                t_eval=f,
                method=method_name,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            if not sol.success:
                raise RuntimeError(
                    f"solve_ivp failed with method '{method_name}': {sol.message}"
                )
            return sol.y[0]

        if isinstance(ivp_method, str):
            return _solve_with_method(ivp_method)

        results: Dict[str, np.ndarray] = {}
        failures = []
        for method_name in ivp_method:
            try:
                results[str(method_name)] = _solve_with_method(str(method_name))
            except Exception as exc:
                failures.append((str(method_name), str(exc)))

        if results:
            return results
        raise RuntimeError(f"All solve_ivp methods failed: {failures}")

    def strain(
        self,
        f,
        delta_f=0.25,
        frequencySeries=True,
        ivp_method: str = "RK45",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_step: float = np.inf,
    ):
        """precessing GW with selectable ODE solver for phase_delta_phi

        Args:
            f (array-like): Frequency array.
            delta_f (float): Frequency spacing for FrequencySeries.
            frequencySeries (bool): Return FrequencySeries if True.
            ivp_method (str): ODE solver method for phase_delta_phi.
            rtol (float): Relative tolerance for ODE solver.
            atol (float): Absolute tolerance for ODE solver.
            max_step (float): Maximum step size for ODE solver.

        Returns:
            FrequencySeries or np.ndarray: GW strain.
        """
        strain = self.amplitude(f) * np.exp(
            1j
            * (
                self.Psi(f)
                - self.phase_phi_P(f)
                - 2 * self.phase_delta_phi(f, ivp_method, rtol, atol, max_step)
            )
        )
        if frequencySeries:
            return FrequencySeries(strain, delta_f)
        return strain
