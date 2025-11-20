# Lindblom Pipeline

This pipeline computes the Lindblom criterion (distinguishability) between lensed source waveforms and precessing template waveforms.

## Workflow

1.  **Compute Cubes**: This is usually done via batch jobs (e.g., SLURM arrays) that run `compute_lindblom_contours.py` (or similar) to generate individual mismatch cubes.
2.  **Check Progress**: Run `check_lindblom_progress.sh` to verify that all expected cubes have been generated.
3.  **Aggregate & Plot**: Run `complete_lindblom_pipeline.sh`. This script orchestrates:
    -   `aggregate_lindblom_best_match.py`: Consolidates individual cubes into a single best-match HDF5 file.
    -   `create_contour_mcz_td_from_lindblom.py`: Generates the Lindblom criterion contour plots.
    -   `create_contour_mcz_td_from_snr.py`: Generates the SNR contour plots.

## Scripts

-   `complete_lindblom_pipeline.sh`: Main orchestrator for aggregation and plotting.
-   `check_lindblom_progress.sh`: Checks if all required data cubes exist.
-   `aggregate_lindblom_best_match.py`: Aggregates per-mcz/td results.
-   `create_contour_mcz_td_from_lindblom.py`: Plots Lindblom criterion contours.
-   `create_contour_mcz_td_from_snr.py`: Plots SNR contours.
-   `compute_source_snr_contour.py` & `create_contour_mcz_td_from_source_snr.py`: Tools for analyzing source-only SNR.

## Usage Example

```bash
# After batch jobs have finished:
./complete_lindblom_pipeline.sh
```

