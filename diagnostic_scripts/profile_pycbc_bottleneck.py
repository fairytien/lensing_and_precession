#!/usr/bin/env python3
"""
Profile PyCBC matched filtering to identify where 500ms per mismatch is spent.

Breaks down time cost of each sub-operation in both:
  - match_utils.optimized_match_bounded  (our wrapper)
  - pycbc.filter.match                   (coarse integer-sample alignment)
  - pycbc.filter.optimized_match         (brent sub-sample refinement)

Run with:
    python diagnostic_scripts/profile_pycbc_bottleneck.py
"""

import sys
import time
import cProfile
import pstats
from io import StringIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SOLMASS2SEC = 4.92624076e-6
GIGAPC2SEC  = 1.02927125e17

def _make_params(theta_tilde=0.5, omega_tilde=0.5):
    mcz = 25 * SOLMASS2SEC
    eta = 0.22   # ~(30,20) Msun pair
    return {
        "theta_S": np.pi / 4, "phi_S": 0.0,
        "theta_J": np.pi / 2, "phi_J": np.pi / 2,
        "mcz": mcz, "dist": 1.5 * GIGAPC2SEC,
        "eta": eta, "t_c": 0.0, "phi_c": 0.0,
        "y": 0.25, "MLz": 1e3 * SOLMASS2SEC,
        "theta_tilde": theta_tilde, "omega_tilde": omega_tilde,
        "gamma_P": 0.0,
    }


def _timer(label, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), print elapsed, return result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"  {label:<45s}: {dt*1000:8.2f} ms")
    return result, dt


# ──────────────────────────────────────────────────────────────────────────────
# Build two waveform strains
# ──────────────────────────────────────────────────────────────────────────────

print("Loading modules …")
t_load = time.perf_counter()

from modules.Classes import Precessing
from modules.functions import Sn
from pycbc.filter import match as pycbc_match, optimized_match as pycbc_opt_match
from pycbc.filter.matchedfilter import (
    make_frequency_series, matched_filter_core, get_cutoff_indices, sigmasq,
)
from modules.match_utils import optimized_match_bounded
from scipy.optimize import minimize_scalar

print(f"  modules loaded in {(time.perf_counter()-t_load)*1000:.0f} ms\n")

f_min, f_max, delta_f = 20.0, 256.0, 0.25
frequencies = np.arange(f_min, f_max, delta_f)

params_t = _make_params(theta_tilde=0.50)
params_s = _make_params(theta_tilde=0.55)

print("Building waveform strains …")
prec_t = Precessing(params_t)
prec_s = Precessing(params_s)

_, t_strain = _timer("template strain()", prec_t.strain, frequencies, delta_f=delta_f)
_, t_strain = _timer("signal  strain()", prec_s.strain, frequencies, delta_f=delta_f)

htilde = prec_t.strain(frequencies, delta_f=delta_f)
stilde = prec_s.strain(frequencies, delta_f=delta_f)
htilde.resize(len(stilde))

# Build PSD on the same frequency grid
psd = Sn(frequencies, f_min=f_min, delta_f=delta_f)

print()

# ──────────────────────────────────────────────────────────────────────────────
# WARMUP — trigger FFTW plan compilation before any timed section
# ──────────────────────────────────────────────────────────────────────────────

print("Warming up FFTW (full optimized_match_bounded call) …")
_t_warm = time.perf_counter()
optimized_match_bounded(htilde, stilde, psd=psd,
                        low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
print(f"  warmup done in {(time.perf_counter()-_t_warm)*1000:.0f} ms\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 – top-level timing of every distinct layer
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("SECTION 1 — TOP-LEVEL OPERATION TIMING")
print("=" * 65)

# 1a. make_frequency_series  (called twice inside every match)
_, t_mfs_h = _timer("make_frequency_series(htilde)", make_frequency_series, htilde)
_, t_mfs_s = _timer("make_frequency_series(stilde)", make_frequency_series, stilde)

# 1b. sigmasq (called for norm_1 and norm_2)
_, t_sq1 = _timer("sigmasq(htilde)", sigmasq, htilde, psd, f_min, f_max)
_, t_sq2 = _timer("sigmasq(stilde)", sigmasq, stilde, psd, f_min, f_max)

# 1c. matched_filter_core  (the FFT correlation at the heart of match())
_, t_mfc = _timer("matched_filter_core()", matched_filter_core,
                  htilde, stilde, psd,
                  low_frequency_cutoff=f_min,
                  high_frequency_cutoff=f_max)

# 1d. pycbc match()  – integer-sample alignment
_, t_match = _timer("pycbc match()", pycbc_match,
                    htilde, stilde, psd=psd,
                    low_frequency_cutoff=f_min,
                    high_frequency_cutoff=f_max,
                    return_phase=True)

# 1e. cyclic_time_shift  (called once per match to align waveforms)
_, max_id, _ = pycbc_match(htilde, stilde, psd=psd,
                           low_frequency_cutoff=f_min,
                           high_frequency_cutoff=f_max,
                           return_phase=True)
delta_t = stilde.delta_t
_, t_shift = _timer("cyclic_time_shift()", stilde.cyclic_time_shift, -max_id * delta_t)
stilde_shifted = stilde.cyclic_time_shift(-max_id * delta_t)

# 1f. numpy setup for the sub-sample optimization loop
N = (len(stilde_shifted) - 1) * 2
kmin, kmax = get_cutoff_indices(f_min, f_max, delta_f, N)
mask = slice(kmin, kmax)

w1 = htilde.numpy()[mask]
w2 = stilde_shifted.numpy()[mask]
freqs = stilde_shifted.sample_frequencies.numpy()[mask]
psd_arr = psd.numpy()[mask]

t0 = time.perf_counter()
_ = htilde.numpy()
t_np1 = time.perf_counter() - t0
print(f"  {'htilde.numpy()  (array extraction)':<45s}: {t_np1*1000:8.2f} ms")

# 1g. Number of frequencies in the inner-product window
n_freqs = kmax - kmin
print(f"\n  Freq. window: {n_freqs} bins  "
      f"({f_min:.0f}–{f_max:.0f} Hz, delta_f={delta_f} Hz, "
      f"N_fft={N})")

# 1h. Single inner-product evaluation (what minimize_scalar calls each time)
def inner_product(dt):
    offset = np.exp(2j * np.pi * freqs * dt)
    integral = np.sum(np.conj(w1) * (w2 * offset) / psd_arr) * delta_f
    return 4 * abs(integral)

N_CALLS = 50
t0 = time.perf_counter()
for _ in range(N_CALLS):
    inner_product(0.0)
t_inner_single = (time.perf_counter() - t0) / N_CALLS
print(f"  {'single inner-product eval (avg of 50)':<45s}: {t_inner_single*1e6:8.2f} µs")

# 1i. minimize_scalar (brent) – the sub-sample refinement
def to_minimize(dt):
    return -inner_product(dt)

_, t_opt = _timer("minimize_scalar (brent)", minimize_scalar,
                  to_minimize, method="brent", bracket=(-delta_t, delta_t))

# 1j. minimize_scalar (bounded) – our variant
_, t_opt_b = _timer("minimize_scalar (bounded)", minimize_scalar,
                    to_minimize, method="bounded", bounds=(-delta_t, delta_t))

# 1k. pycbc optimized_match (uses brent internally)
_, t_opt_match = _timer("pycbc optimized_match()", pycbc_opt_match,
                        htilde, stilde, psd=psd,
                        low_frequency_cutoff=f_min,
                        high_frequency_cutoff=f_max)

# 1l. our optimized_match_bounded wrapper
_, t_omb = _timer("optimized_match_bounded() [ours]", optimized_match_bounded,
                  htilde, stilde, psd=psd,
                  low_frequency_cutoff=f_min,
                  high_frequency_cutoff=f_max)

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 – inner-product scaling with frequency resolution
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("SECTION 2 — INNER-PRODUCT SCALING WITH delta_f")
print("=" * 65)
print(f"  (single inner-product eval at different frequency resolutions)")

for df_test in [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]:
    freqs_t = np.arange(f_min, f_max, df_test)
    n = len(freqs_t)
    w1_t = np.ones(n, dtype=np.complex128)
    w2_t = np.ones(n, dtype=np.complex128)
    psd_t = np.ones(n, dtype=np.float64)

    N_CALLS = 20
    t0 = time.perf_counter()
    for _ in range(N_CALLS):
        offset = np.exp(2j * np.pi * freqs_t * 0.0)
        integral = np.sum(np.conj(w1_t) * (w2_t * offset) / psd_t) * df_test
        _ = 4 * abs(integral)
    t_avg = (time.perf_counter() - t0) / N_CALLS
    print(f"  delta_f={df_test:6.4f} Hz  →  {n:5d} bins  →  {t_avg*1e6:7.2f} µs per eval")

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – how many times does minimize_scalar call the objective?
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("SECTION 3 — minimize_scalar CALL COUNT")
print("=" * 65)

call_counts = {"brent": 0, "bounded": 0}

def to_min_counted_brent(dt):
    call_counts["brent"] += 1
    return -inner_product(dt)

def to_min_counted_bounded(dt):
    call_counts["bounded"] += 1
    return -inner_product(dt)

minimize_scalar(to_min_counted_brent,   method="brent",   bracket=(-delta_t, delta_t))
minimize_scalar(to_min_counted_bounded, method="bounded", bounds=(-delta_t, delta_t))

print(f"  brent   objective calls: {call_counts['brent']}")
print(f"  bounded objective calls: {call_counts['bounded']}")
total_brent_ms   = call_counts["brent"]   * t_inner_single * 1000
total_bounded_ms = call_counts["bounded"] * t_inner_single * 1000
print(f"  estimated brent   inner-product cost: {total_brent_ms:.2f} ms")
print(f"  estimated bounded inner-product cost: {total_bounded_ms:.2f} ms")

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – cProfile on the full optimized_match_bounded
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("SECTION 4 — cPROFILE (top 20 cumulative, optimized_match_bounded)")
print("=" * 65)

prof = cProfile.Profile()
prof.enable()
for _ in range(3):
    optimized_match_bounded(htilde, stilde, psd=psd,
                            low_frequency_cutoff=f_min,
                            high_frequency_cutoff=f_max)
prof.disable()

stream = StringIO()
ps = pstats.Stats(prof, stream=stream).sort_stats("cumulative")
ps.print_stats(20)
print(stream.getvalue())

# ──────────────────────────────────────────────────────────────────────────────
# Section 5 – summary table
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("SECTION 5 — SUMMARY")
print("=" * 65)
print(f"  {'operation':<45s}  {'ms':>8}")
print(f"  {'-'*45}  {'--------':>8}")
print(f"  {'matched_filter_core (FFT correlation)':<45s}  {t_mfc*1000:>8.2f}")
print(f"  {'sigmasq x2 (norm_1 + norm_2)':<45s}  {(t_sq1+t_sq2)*1000:>8.2f}")
print(f"  {'pycbc match() (integer-sample align)':<45s}  {t_match*1000:>8.2f}")
print(f"  {'cyclic_time_shift':<45s}  {t_shift*1000:>8.2f}")
print(f"  {'minimize_scalar (brent)':<45s}  {t_opt*1000:>8.2f}")
print(f"  {'minimize_scalar (bounded)':<45s}  {t_opt_b*1000:>8.2f}")
print(f"  {'optimized_match_bounded (full ours)':<45s}  {t_omb*1000:>8.2f}")
print()
print("  NOTE: 'pycbc match' already calls matched_filter_core internally,")
print("  so total cost ≈ match() + cyclic_time_shift + minimize_scalar overhead.")
