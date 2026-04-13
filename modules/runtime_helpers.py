"""Shared runtime helpers used across scripts.

These helpers are intentionally separate from waveform physics utilities.
"""

import os
import pickle
import time
from datetime import datetime


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
