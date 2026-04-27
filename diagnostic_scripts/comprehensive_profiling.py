#!/usr/bin/env python3
"""
Comprehensive performance profiling of the mismatch computation pipeline.

Measures:
- Waveform generation time (strain computation)
- Phase correction time (ODE integration)
- Matched filtering time
- Mismatch computation time
- Memory usage and call frequencies

Run with:
    python diagnostic_scripts/comprehensive_profiling.py [--sample-size N] [--profile-level basic|detailed|full]
"""

import sys
import time
import numpy as np
from pathlib import Path
import cProfile
import pstats
from io import StringIO
import gc
import psutil
import os

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.Classes import LensingGeo, Precessing
from modules.match_utils import mismatch_from_params
from modules.waveform import MCz_to_mtotal_eta
import argparse


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def profile_waveform_generation(waveform_obj, f_min, f_max, delta_f, num_trials=5):
    """Profile waveform generation for a single object."""
    frequencies = np.arange(f_min, f_max, delta_f)
    
    # Warmup
    _ = waveform_obj.strain(frequencies, f_min, f_max, delta_f)
    
    times = []
    for _ in range(num_trials):
        gc.collect()
        start = time.perf_counter()
        strain = waveform_obj.strain(frequencies, f_min, f_max, delta_f)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'num_frequencies': len(frequencies),
    }


def profile_phase_correction(waveform_obj, f_min, f_max, num_trials=3):
    """Profile phase correction computation."""
    frequencies = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    
    # Warmup
    _ = waveform_obj.phase_delta_phi(frequencies)
    
    times = []
    for _ in range(num_trials):
        gc.collect()
        start = time.perf_counter()
        phase = waveform_obj.phase_delta_phi(frequencies)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'num_frequencies': len(frequencies),
    }


def profile_mismatch_computation(params_list, num_trials=3):
    """Profile full mismatch computation for a set of parameter points."""
    times = []
    mismatches = []
    
    for _ in range(num_trials):
        gc.collect()
        start = time.perf_counter()
        
        results = []
        for params in params_list:
            result = mismatch_from_params(
                *params,
                f_min=20,
                f_max=256,
                delta_f=0.25
            )
            results.append(result)
        
        end = time.perf_counter()
        times.append(end - start)
        mismatches = results
    
    total_time = np.sum(times)
    avg_per_point = total_time / len(params_list) / num_trials
    
    return {
        'total_time': total_time,
        'mean_per_trial': np.mean(times),
        'std_per_trial': np.std(times),
        'avg_per_point': avg_per_point,
        'num_points': len(params_list),
        'num_trials': num_trials,
    }


def run_detailed_profiling(sample_size=5):
    """Run detailed profiling with cProfile."""
    print("\n" + "="*70)
    print("DETAILED CPROFILE ANALYSIS")
    print("="*70)
    
    # Create test parameters
    params_list = []
    np.random.seed(42)
    for i in range(sample_size):
        m1 = np.random.uniform(10, 50)
        m2 = np.random.uniform(5, m1)
        s1z = np.random.uniform(-1, 1)
        s2z = np.random.uniform(-1, 1)
        theta_tilde = np.random.uniform(0, 1)
        omega_tilde = np.random.uniform(0, 1)
        params_list.append((m1, m2, s1z, s2z, theta_tilde, omega_tilde))
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run mismatch computation
    for params in params_list:
        try:
            mismatch_from_params(
                *params,
                f_min=20,
                f_max=256,
                delta_f=0.25
            )
        except Exception as e:
            print(f"Error computing mismatch: {e}")
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(s.getvalue())
    
    return profiler


def main():
    parser = argparse.ArgumentParser(description='Comprehensive performance profiling')
    parser.add_argument('--sample-size', type=int, default=5, help='Number of parameter points')
    parser.add_argument('--profile-level', choices=['basic', 'detailed', 'full'], 
                        default='basic', help='Profiling detail level')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MISMATCH COMPUTATION PROFILING")
    print("="*70)
    
    # Setup
    mem_start = get_memory_usage()
    print(f"\nInitial memory: {mem_start:.1f} MB")
    
    f_min, f_max, delta_f = 20, 256, 0.25
    
    # Create test parameters
    np.random.seed(42)
    params_list = []
    for i in range(args.sample_size):
        m1 = np.random.uniform(10, 50)
        m2 = np.random.uniform(5, m1)
        s1z = np.random.uniform(-1, 1)
        s2z = np.random.uniform(-1, 1)
        theta_tilde = np.random.uniform(0, 1)
        omega_tilde = np.random.uniform(0, 1)
        params_list.append((m1, m2, s1z, s2z, theta_tilde, omega_tilde))
    
    print(f"\nTest parameters: {args.sample_size} points")
    print(f"Frequency range: {f_min}-{f_max} Hz, delta_f={delta_f} Hz")
    
    # Test individual waveform generation
    print("\n" + "-"*70)
    print("WAVEFORM GENERATION PROFILING")
    print("-"*70)
    
    try:
        m1, m2, s1z, s2z, theta_tilde, omega_tilde = params_list[0]
        
        # LensingGeo waveform
        print("\nLensingGeo strain generation:")
        lensing_geo = LensingGeo(m1, m2, s1z, s2z, f_min=f_min)
        result = profile_waveform_generation(lensing_geo, f_min, f_max, delta_f, num_trials=3)
        print(f"  Mean time: {result['mean']*1000:.2f} ms")
        print(f"  Std:       {result['std']*1000:.2f} ms")
        print(f"  Frequencies: {result['num_frequencies']}")
        
        # Precessing waveform
        print("\nPrecessing strain generation:")
        precessing = Precessing(m1, m2, s1z, s2z, theta_tilde, omega_tilde, f_min=f_min)
        result = profile_waveform_generation(precessing, f_min, f_max, delta_f, num_trials=3)
        print(f"  Mean time: {result['mean']*1000:.2f} ms")
        print(f"  Std:       {result['std']*1000:.2f} ms")
        print(f"  Frequencies: {result['num_frequencies']}")
        
    except Exception as e:
        print(f"Error in waveform profiling: {e}")
    
    # Test phase correction
    print("\n" + "-"*70)
    print("PHASE CORRECTION PROFILING (precession phase_delta_phi)")
    print("-"*70)
    
    try:
        for i, (m1, m2, s1z, s2z, theta_tilde, omega_tilde) in enumerate(params_list[:3]):
            if theta_tilde < 0.01:  # Skip non-precessing
                continue
            print(f"\nParameter set {i}: theta_tilde={theta_tilde:.3f}, omega_tilde={omega_tilde:.3f}")
            precessing = Precessing(m1, m2, s1z, s2z, theta_tilde, omega_tilde, f_min=f_min)
            result = profile_phase_correction(precessing, f_min, f_max, num_trials=3)
            print(f"  Mean time: {result['mean']*1000:.2f} ms")
            print(f"  Std:       {result['std']*1000:.2f} ms")
            print(f"  Frequencies: {result['num_frequencies']}")
            print(f"  Time per frequency: {result['mean']/result['num_frequencies']*1e6:.2f} µs")
    except Exception as e:
        print(f"Error in phase profiling: {e}")
    
    # Test full mismatch computation
    print("\n" + "-"*70)
    print("MISMATCH COMPUTATION PROFILING")
    print("-"*70)
    
    try:
        result = profile_mismatch_computation(params_list, num_trials=2)
        print(f"\nFull mismatch computation ({result['num_points']} points, {result['num_trials']} trials):")
        print(f"  Total time:        {result['total_time']:.3f} s")
        print(f"  Mean per trial:    {result['mean_per_trial']:.3f} s")
        print(f"  Avg per point:     {result['avg_per_point']*1000:.2f} ms")
        print(f"  Points per second: {1.0/result['avg_per_point']:.1f}")
    except Exception as e:
        print(f"Error in mismatch profiling: {e}")
    
    # Detailed profiling if requested
    if args.profile_level in ['detailed', 'full']:
        run_detailed_profiling(min(3, args.sample_size))
    
    # Memory summary
    mem_end = get_memory_usage()
    print("\n" + "-"*70)
    print("MEMORY SUMMARY")
    print("-"*70)
    print(f"Start:  {mem_start:.1f} MB")
    print(f"End:    {mem_end:.1f} MB")
    print(f"Delta:  {mem_end - mem_start:+.1f} MB")
    
    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
