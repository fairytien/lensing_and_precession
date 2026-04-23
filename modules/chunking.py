"""Chunking heuristics for HDF5 datasets and in-memory iteration.

choose_bank_chunks(...) returns per-axis chunk sizes for the 4D bank dataset
optimized for access pattern bank[r, c, :, :].

choose_gamma_chunk(n_gamma) returns an in-memory gamma tile size for iteration
balanced between I/O and RAM.
"""


def choose_bank_chunks(
    theta_pts: int, omega_pts: int, gamma_pts: int, n_freq: int
) -> tuple:
    """
    Heuristic chunk sizes for HDF5 bank dataset optimized for access pattern bank[r, c, :, :].

    Returns a tuple (theta_chunk, omega_chunk, gamma_chunk, freq_chunk).
    """
    # We typically read all gamma and a slice of frequency for a fixed (r, c)
    theta_chunk = 1
    omega_chunk = 1
    gamma_chunk = min(8, max(1, gamma_pts))
    freq_chunk = min(4096, max(1, n_freq))
    return theta_chunk, omega_chunk, gamma_chunk, freq_chunk


def choose_gamma_chunk(n_gamma: int) -> int:
    """
    Heuristic gamma chunk size for iterating bank[r, c, k0:k1, :].

    Keeps chunks modest to reduce peak memory while allowing reasonable I/O throughput.
    """
    if n_gamma <= 0:
        return 1
    return int(max(1, min(32, n_gamma)))
