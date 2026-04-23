import os


def get_env_int(name: str) -> int | None:
    """Return env var as int or None if unset/empty/invalid."""
    val = os.environ.get(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def chunk_bounds(total: int, num_chunks: int, chunk_index: int) -> tuple[int, int]:
    """Return (start, end) indices splitting range(total) into num_chunks blocks."""
    if num_chunks <= 1:
        return 0, total
    base = total // num_chunks
    rem = total % num_chunks
    start = chunk_index * base + min(chunk_index, rem)
    end = start + base + (1 if chunk_index < rem else 0)
    return start, end
