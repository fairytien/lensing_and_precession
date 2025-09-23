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
    gamma_chunk = int(min(8, max(1, gamma_pts)))
    # Keep frequency chunks moderate to balance I/O and memory
    freq_chunk = int(min(4096, max(1, n_freq)))
    return theta_chunk, omega_chunk, gamma_chunk, freq_chunk
