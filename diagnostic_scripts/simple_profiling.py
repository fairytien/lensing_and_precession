#!/usr/bin/env python3
"""
Simplified performance analysis focusing on key bottlenecks.
"""

import sys
import time
import numpy as np
from pathlib import Path
import gc

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Loading modules...")

# Quick test of import times
import_times = {}

start = time.perf_counter()
import numpy as np
import_times['numpy'] = time.perf_counter() - start

start = time.perf_counter()
from scipy.integrate import odeint
import_times['scipy.integrate'] = time.perf_counter() - start

start = time.perf_counter()
from modules.waveform import get_fcut_from_mcz
import_times['waveform'] = time.perf_counter() - start

start = time.perf_counter()
from modules.Classes import LensingGeo, Precessing
import_times['Classes'] = time.perf_counter() - start

start = time.perf_counter()
from modules.match_utils import mismatch_from_params
import_times['match_utils'] = time.perf_counter() - start

print("\nImport times:")
for name, t in import_times.items():
    print(f"  {name:30s}: {t*1000:8.2f} ms")

print("\n" + "="*70)
print("PERFORMANCE ANALYSIS - BOTTLENECK IDENTIFICATION")
print("="*70)

# Create simple test case
print("\nCreating test waveform objects...")
m1, m2, s1z, s2z = 30, 20, 0.5, -0.3
theta_tilde, omega_tilde = 0.7, 0.5
f_min = 20

# Create params dict
SOLMASS2SEC = 4.92624076 * 1e-6
GIGAPC2SEC = 1.02927125 * 1e17
mcz = 25 * SOLMASS2SEC  # ~25 solar masses
eta = m1 * m2 / (m1 + m2) ** 2
dist = 1.5 * GIGAPC2SEC

params_lensing = {
    "theta_S": np.pi / 4,
    "phi_S": 0.0,
    "theta_J": np.pi / 2,
    "phi_J": np.pi / 2,
    "mcz": mcz,
    "dist": dist,
    "eta": eta,
    "t_c": 0.0,
    "phi_c": 0.0,
    "y": 0.25,
    "MLz": 1e3 * SOLMASS2SEC,
}

params_precessing = params_lensing.copy()
params_precessing.update({
    "theta_tilde": theta_tilde,
    "omega_tilde": omega_tilde,
    "s1z": s1z,
    "s2z": s2z,
    "gamma_P": 0.0,
})

# Time object initialization
start = time.perf_counter()
lensing = LensingGeo(params_lensing)
t_lensing_init = time.perf_counter() - start

start = time.perf_counter()
precessing = Precessing(params_precessing)
t_precessing_init = time.perf_counter() - start

print(f"LensingGeo init:  {t_lensing_init*1000:8.2f} ms")
print(f"Precessing init:  {t_precessing_init*1000:8.2f} ms")

# Test strain generation
print("\n" + "-"*70)
print("STRAIN GENERATION PROFILING")
print("-"*70)

f_min, f_max, delta_f = 20, 256, 0.25
frequencies = np.arange(f_min, f_max, delta_f)
print(f"Frequency grid: {len(frequencies)} points ({f_min}-{f_max} Hz, delta_f={delta_f})")

# Warm up JIT compiler
_ = precessing.strain(frequencies, delta_f=delta_f)

# Time strain generation
num_trials = 3
times_strain = []
for _ in range(num_trials):
    gc.collect()
    start = time.perf_counter()
    strain = precessing.strain(frequencies, delta_f=delta_f)
    t = time.perf_counter() - start
    times_strain.append(t)

avg_strain = np.mean(times_strain)
print(f"Strain generation: {avg_strain*1000:8.2f} ms (avg of {num_trials} trials)")
print(f"  Per frequency: {avg_strain/len(frequencies)*1e6:8.2f} µs")

# Test phase correction (if precessing)
print("\n" + "-"*70)
print("PHASE CORRECTION PROFILING (integrand + ODE solver)")
print("-"*70)

f_test = np.logspace(np.log10(f_min), np.log10(f_max), 100)
print(f"Test frequencies: {len(f_test)} points (log-spaced from {f_min} to {f_max} Hz)")

# Warm up
_ = precessing.phase_delta_phi(f_test)

# Time phase correction
times_phase = []
for _ in range(num_trials):
    gc.collect()
    start = time.perf_counter()
    phase = precessing.phase_delta_phi(f_test)
    t = time.perf_counter() - start
    times_phase.append(t)

avg_phase = np.mean(times_phase)
print(f"Phase correction: {avg_phase*1000:8.2f} ms (avg of {num_trials} trials)")
print(f"  Per frequency: {avg_phase/len(f_test)*1e6:8.2f} µs")

# Estimate ODE calls
print(f"\nEstimated ODE integration calls: ~{int(avg_phase/1e-5)} (if ~10µs per call)")

# Test mismatch computation
print("\n" + "-"*70)
print("MISMATCH COMPUTATION PROFILING")
print("-"*70)

print("Computing mismatch timings (cold + warm)...")

# Create two different parameter sets
params_template = params_precessing.copy()
params_signal = params_precessing.copy()
params_signal['theta_tilde'] = 0.75  # Slightly different precession

# First call includes one-time setup costs (numba/FFTW planning)
start = time.perf_counter()
result_cold = mismatch_from_params(
    params_template,
    params_signal,
    f_min=20,
    delta_f=0.25
)
t_mismatch_cold = time.perf_counter() - start
print(f"Cold mismatch (first call): {t_mismatch_cold*1000:8.2f} ms")

# Warm/steady-state timings (what matters for large sweeps)
warm_trials = 5
warm_times = []
for _ in range(warm_trials):
    start = time.perf_counter()
    result_warm = mismatch_from_params(
        params_template,
        params_signal,
        f_min=20,
        delta_f=0.25
    )
    warm_times.append(time.perf_counter() - start)
t_mismatch = float(np.mean(warm_times))
print(f"Warm mismatch (avg of {warm_trials}): {t_mismatch*1000:8.2f} ms")

result = result_warm
if isinstance(result, dict):
    print(f"  Mismatch value: {result.get('mismatch', result.get('match', 'N/A'))}")
else:
    print(f"  Mismatch value: {result:.6f}")

# Project scaling
print("\n" + "-"*70)
print("SCALING ANALYSIS")
print("-"*70)

print(f"\nAssuming steady-state performance:")
print(f"  Per-mismatch time (warm): {t_mismatch*1000:.2f} ms")
print(f"  Mismatches per second: {1.0/t_mismatch:.1f}")
print(f"\nProjected times for contour generation:")
for n_points in [100, 500, 1000, 5000]:
    total_time = n_points * t_mismatch
    print(f"  {n_points:5d} points: {total_time/3600:8.2f} hours")

# Breakdown analysis
print("\n" + "-"*70)
print("TIME BREAKDOWN ANALYSIS")
print("-"*70)

total_time = t_mismatch
print(f"\nTotal mismatch time: {total_time*1000:.2f} ms = 100%")

# Estimate components (rough)
strain_fraction = (2 * avg_strain) / total_time  # Two strains (template and signal)
phase_fraction = (avg_phase) / total_time  # Phase correction for precessing
other_fraction = 1.0 - strain_fraction - phase_fraction

print(f"  Estimated strain generation:  {strain_fraction*100:6.1f}%")
print(f"  Estimated phase correction:   {phase_fraction*100:6.1f}%")
print(f"  Estimated other (matching):   {other_fraction*100:6.1f}%")

print("\n" + "="*70)
