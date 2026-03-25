# Mismatch MCZ-TD Pipeline

This pipeline computes generic mismatch cubes (and derived contours) between lensed sources and templates, varying `mcz` and `td`.

Redshift support:
- All three core stages support `--z`:
    - `compute_mismatch_cubes.py`
    - `aggregate_best_match.py`
    - `plot_contour_mcz_td_from_best_match.py`
- Source and template waveforms use detector-frame chirp mass
    `mcz_det = mcz * (1 + z)`.
- Output filenames include `_z...` when `z != 0`.

## Workflow

1.  **Compute Mismatch Cubes**: Run `compute_mismatch_cubes.py`. This is typically parallelized over `mcz` chunks.
2.  **Aggregate Results**: Run `aggregate_best_match.py` to combine the per-mcz cubes into a single summary file.
3.  **Visualize**: 
    - Run `plot_contour_mcz_td_from_best_match.py` to plot the global mismatch landscape.
    - Run `visualize_mcz_sweep_at_td.py` to see how the mismatch surface (over $\Omega-\theta$) changes with $m_{cz}$ at a fixed time delay.
    - Run `visualize_mismatch_cube.py` to interactively explore a specific cube.

## Scripts

-   `compute_mismatch_cubes.py`: Computes mismatch grids for given parameters.
-   `aggregate_best_match.py`: Combines outputs from `compute_mismatch_cubes.py`.
-   `plot_contour_mcz_td_from_best_match.py`: Plots the resulting mismatch landscape.
-   `visualize_mismatch_cube.py`: Helper to visualize specific parts of a mismatch cube.
-   `visualize_mcz_sweep_at_td.py`: Generates movies/sliders sweeping $m_{cz}$ at constant $t_d$.
-   `plot_omega_theta_from_cube.py`: Plots the $\Omega-\theta$ mismatch slice at a requested $t_d$ from a per-mcz cube.

## Naming And Discovery

-   Canonical file naming is centralized in `modules/filenames.py`.
-   This folder's core scripts write and find files through those helpers rather than hardcoded filename strings.
-   Helper visualizers now discover cubes by parsing valid mismatch-cube filenames in `results_dir/mismatch_cubes/`, so they remain stable if filename details evolve.
