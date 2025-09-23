import os
from typing import Optional, Tuple


def get_env_int(name: str) -> Optional[int]:
    """Return env var as int or None if unset/empty/invalid."""
    try:
        val = os.environ.get(name)
        return None if val is None or val == "" else int(val)
    except Exception:
        return None


def chunk_bounds(total: int, num_chunks: int, chunk_index: int) -> Tuple[int, int]:
    """Return (start, end) indices splitting range(total) into num_chunks blocks."""
    if num_chunks <= 1:
        return 0, total
    base = total // num_chunks
    rem = total % num_chunks
    start = chunk_index * base + min(chunk_index, rem)
    end = start + base + (1 if chunk_index < rem else 0)
    return start, end
