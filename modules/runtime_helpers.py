"""Shared runtime helpers used across scripts.

These helpers are intentionally separate from waveform physics utilities.
"""

import os
import pickle
import time
from datetime import datetime
from typing import Optional


def pickle_data(data, dir: str, filename: str) -> str:
    """Serialize data to timestamped pickle under dir."""
    now = datetime.now()
    filename = filename + "_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
    filepath = os.path.join(dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
        print("Pickle saved as", filepath)
    return filepath


def timer_decorator(func):
    """Print wall-clock runtime for wrapped call."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            total_time = end_time - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"Total time to run the script: {int(hours)}:{int(minutes)}:{round(seconds, 2)} (h:m:s)"
            )

    return wrapper


def _env_int(name: str) -> Optional[int]:
    """Best-effort parse of integer environment variables."""
    try:
        val = os.environ.get(name)
        return None if val is None or val == "" else int(val)
    except Exception:
        return None


def effective_worker_count(total_jobs: int, requested: Optional[int] = None) -> int:
    """Return an HPC-aware worker count bounded by total_jobs.

    Priority order:
    1) explicit requested value
    2) scheduler-provided allocation env vars
    3) os.cpu_count()
    """
    jobs = max(1, int(total_jobs))
    if requested is not None:
        return max(1, min(int(requested), jobs))

    env_candidates = (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "PBS_NP",
        "NSLOTS",
        "OMP_NUM_THREADS",
    )
    for name in env_candidates:
        n = _env_int(name)
        if n is not None and n > 0:
            return max(1, min(n, jobs))

    return max(1, min(int(os.cpu_count() or 1), jobs))


def recommended_chunksize(total_jobs: int, workers: int, target_chunks_per_worker: int = 8) -> int:
    """Heuristic chunksize for multiprocessing.imap/map style dispatch."""
    t = max(1, int(total_jobs))
    w = max(1, int(workers))
    target = max(1, int(target_chunks_per_worker))
    return max(1, t // (w * target))
