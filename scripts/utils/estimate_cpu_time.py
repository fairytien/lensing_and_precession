"""Estimate CPU time for template bank builds and mismatch calculations based on runlog_mcz_td.csv."""

import csv
import sys
from datetime import timedelta
import numpy as np


def parse_time(time_str):
    """Parse time string like '0:4:5.09' (h:m:s) to seconds"""
    if not time_str or time_str.strip() == "":
        return None
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    return None


def estimate_template_bank_time():
    """Estimate time for building template banks"""
    print("=" * 80)
    print("TEMPLATE BANK BUILD ESTIMATE")
    print("=" * 80)

    # Target specifications
    n_mcz = 161
    omega_pts = 61
    theta_pts = 151
    gamma_pts = 51

    # Current typical specs from CSV (same as target - no scaling)
    omega_ref = 61
    theta_ref = 151
    gamma_ref = 51

    # Resolution scaling factor (1.0x - no scaling needed)
    resolution_factor = 1.0
    print(
        f"\nTarget resolution: omega={omega_pts}, theta={theta_pts}, gamma={gamma_pts}"
    )
    print(
        f"Reference resolution: omega={omega_ref}, theta={theta_ref}, gamma={gamma_ref}"
    )
    print(
        f"Resolution scaling factor: {resolution_factor:.2f}x (no scaling - matches reference)"
    )

    # Extract build times from CSV
    build_times = []
    with open("data/run_logs/runlog_mcz_td.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("stage") == "build" and row["total_time"]:
                time_sec = parse_time(row["total_time"])
                if time_sec:
                    omega = int(row.get("omega_pts", 0) or 61)
                    theta = int(row.get("theta_pts", 0) or 151)
                    gamma = int(row.get("gamma_pts", 0) or 51)
                    # Only use data matching reference resolution
                    if omega == omega_ref and theta == theta_ref and gamma == gamma_ref:
                        build_times.append(time_sec)

    if build_times:
        avg_time = np.mean(build_times)
        median_time = np.median(build_times)
        min_time = np.min(build_times)
        max_time = np.max(build_times)

        print(f"\nBuild time statistics (reference resolution):")
        print(f"  Mean: {avg_time:.1f}s ({timedelta(seconds=int(avg_time))})")
        print(f"  Median: {median_time:.1f}s ({timedelta(seconds=int(median_time))})")
        print(f"  Min: {min_time:.1f}s ({timedelta(seconds=int(min_time))})")
        print(f"  Max: {max_time:.1f}s ({timedelta(seconds=int(max_time))})")
        print(f"  Sample size: {len(build_times)}")

        # Scale to target resolution
        scaled_time_per_bank = avg_time * resolution_factor
        total_time = scaled_time_per_bank * n_mcz

        print(
            f"\nEstimated time per bank (target resolution): {scaled_time_per_bank:.1f}s ({timedelta(seconds=int(scaled_time_per_bank))})"
        )
        print(
            f"Total time for {n_mcz} banks: {total_time:.1f}s ({timedelta(seconds=int(total_time))})"
        )
        print(f"  = {total_time/3600:.2f} CPU hours")
        print(f"  = {total_time/86400:.2f} CPU days")

        # Estimate for parallel execution (assuming 161 jobs)
        print(
            f"\nIf parallelized across {n_mcz} jobs: {scaled_time_per_bank:.1f}s wall time ({timedelta(seconds=int(scaled_time_per_bank))})"
        )
    else:
        print("No suitable build time data found in CSV")


def estimate_mismatch_time():
    """Estimate time for mismatch calculations"""
    print("\n" + "=" * 80)
    print("MISMATCH CALCULATION ESTIMATE")
    print("=" * 80)

    # Target specifications
    n_mcz = 161
    n_td = 101
    n_I = 81
    omega_pts = 61  # Template bank resolution
    theta_pts = 151
    gamma_pts = 51

    # Reference specs from CSV (contour jobs)
    omega_ref = 61
    theta_ref = 151
    gamma_ref = 51
    td_ref = 51  # Typical from CSV

    print(f"\nTarget parameters:")
    print(f"  mcz: {n_mcz} values (10-90)")
    print(f"  td: {n_td} values (20-70 ms)")
    print(f"  I (flux ratio): {n_I} values (0.1-0.9)")
    print(f"  omega: {omega_pts} points")
    print(f"  theta: {theta_pts} points")
    print(f"  gamma: {gamma_pts} points")

    # Extract mismatch computation times from CSV
    mismatch_times = []
    with open("data/run_logs/runlog_mcz_td.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("stage") == "mismatch" and row["total_time"]:
                time_sec = parse_time(row["total_time"])
                if time_sec:
                    # Extract job parameters
                    mcz_min = float(row.get("mcz_min", 0) or 0)
                    mcz_max = float(row.get("mcz_max", 0) or 0)
                    n_mcz_job = (
                        int((mcz_max - mcz_min) / 1.0) + 1 if mcz_max > mcz_min else 1
                    )
                    td_min = float(row.get("td_min_ms", 0) or 20)
                    td_max = float(row.get("td_max_ms", 0) or 70)
                    n_td_job = int(td_max - td_min) + 1  # Approximate

                    if n_mcz_job > 0 and n_td_job > 0:
                        # Time per (mcz, td) pair
                        time_per_pair = time_sec / (n_mcz_job * n_td_job)
                        mismatch_times.append(time_per_pair)

    if mismatch_times:
        avg_time_per_pair = np.mean(mismatch_times)
        median_time_per_pair = np.median(mismatch_times)

        print(f"\nReference computation time statistics:")
        print(f"  Mean time per (mcz, td) pair: {avg_time_per_pair:.2f}s")
        print(f"  Median time per (mcz, td) pair: {median_time_per_pair:.2f}s")
        print(f"  Sample size: {len(mismatch_times)}")

        # Resolution scaling factor (1.0x - no scaling needed for template grid)
        resolution_factor = (
            (omega_pts / omega_ref) * (theta_pts / theta_ref) * (gamma_pts / gamma_ref)
        )
        td_factor = n_td / td_ref

        print(
            f"\nResolution scaling: {resolution_factor:.2f}x (no scaling - matches reference)"
        )
        print(f"TD points scaling: {td_factor:.2f}x")

        # Scale time per (mcz, td) pair
        # Note: avg_time_per_pair already includes internal parallelization over (theta, omega)
        scaled_time_per_pair = avg_time_per_pair * resolution_factor * td_factor

        print(f"\nNOTE: Using compute_mismatch_cubes.py parallelization structure:")
        print(
            f"  - Internal: Multiprocessing Pool over (theta, omega) pairs within each (mcz, td)"
        )
        print(
            f"  - External: SLURM array jobs to chunk over mcz or (mcz, I) combinations"
        )
        print(
            f"\nEstimated WALL time per (mcz, td) pair (with internal parallelization): {scaled_time_per_pair:.2f}s"
        )

        # Total computation structure:
        # The script takes one I value per run, so we need separate runs for each I
        # Each run processes: one I value × one mcz × all td values
        # But we can parallelize externally over (mcz, I) pairs using SLURM arrays

        # Wall time per (mcz, I) job = time_per_(mcz,td) * n_td
        wall_time_per_mcz_I_job = scaled_time_per_pair * n_td

        # Total wall time if we parallelize across all (mcz, I) pairs
        total_jobs = n_mcz * n_I  # 161 * 81 = 13,041 jobs
        total_wall_time_parallel = wall_time_per_mcz_I_job

        # Total CPU time (sum of all work)
        total_cpu_time = scaled_time_per_pair * n_mcz * n_td * n_I

        print(f"\n{'='*80}")
        print(f"PARALLELIZATION STRATEGY (using compute_mismatch_cubes.py)")
        print(f"{'='*80}")
        print(f"\nRun separate jobs for each (mcz, I) pair:")
        print(f"  Total jobs needed: {total_jobs:,} (one per (mcz, I) combination)")
        print(f"  Each job processes: 1 mcz × 1 I × {n_td} td values")
        print(
            f"\nWall time per job: {wall_time_per_mcz_I_job:.1f}s ({timedelta(seconds=int(wall_time_per_mcz_I_job))})"
        )
        print(f"  = {wall_time_per_mcz_I_job/60:.2f} minutes")
        print(f"  = {wall_time_per_mcz_I_job/3600:.2f} hours")

        print(f"\nIf all jobs run in parallel (requires {total_jobs:,} nodes/cores):")
        print(
            f"  Total wall time: {total_wall_time_parallel:.1f}s ({timedelta(seconds=int(total_wall_time_parallel))})"
        )
        print(f"  = {total_wall_time_parallel/60:.2f} minutes")
        print(f"  = {total_wall_time_parallel/3600:.2f} hours")

        print(f"\nTotal CPU time (sum of all work across all jobs):")
        print(f"  = {total_cpu_time:.1f}s ({timedelta(seconds=int(total_cpu_time))})")
        print(f"  = {total_cpu_time/3600:.2f} CPU hours")
        print(f"  = {total_cpu_time/86400:.2f} CPU days")
        print(f"  = {total_cpu_time/(86400*365):.2f} CPU years")

        # Alternative: chunk over mcz only (fewer jobs, longer per job)
        print(f"\n{'='*80}")
        print(f"ALTERNATIVE: Parallelize only over mcz (fewer jobs, longer runtime)")
        print(f"{'='*80}")
        wall_time_per_mcz_job = scaled_time_per_pair * n_I * n_td
        print(f"  Total jobs: {n_mcz} (one per mcz)")
        print(f"  Each job processes: 1 mcz × {n_I} I × {n_td} td values")
        print(
            f"  Wall time per job: {wall_time_per_mcz_job:.1f}s ({timedelta(seconds=int(wall_time_per_mcz_job))})"
        )
        print(f"  = {wall_time_per_mcz_job/3600:.2f} hours")
        print(
            f"  Total wall time (if parallelized): {wall_time_per_mcz_job:.1f}s ({timedelta(seconds=int(wall_time_per_mcz_job))})"
        )
    else:
        print("No suitable mismatch computation data found in CSV")
        print("Note: Current script computes per-mcz cubes (one I value per run)")
        print("To vary I, you would need to run separate jobs for each I value")


if __name__ == "__main__":
    estimate_template_bank_time()
    estimate_mismatch_time()
