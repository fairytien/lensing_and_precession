# Contour (td, mcz) Pipeline Guide

This document describes the three-stage modular pipeline for computing mismatch maps between lensed gravitational-wave sources and precessing template banks, then plotting contours across the (time delay, chirp mass) parameter space.

## Pipeline Overview

**Three stages with single responsibilities:**

1. **Build per-mcz template banks** (`python -m scripts.template_banks.build_template_banks`)
2. **Compute per-mcz mismatch cubes** (`python -m scripts.mismatch_mcz_td.compute_mismatch_cubes`)
3. **Aggregate cubes into best-match file** (`python -m scripts.mismatch_mcz_td.aggregate_best_match`)
4. **Plot mismatch contour** (`python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match`)

## Class Version Policy (Production)

- The `(td, mcz)` production pipeline is pinned to `modules/Classes_v2.py` for waveform physics (`LensingGeo` and `Precessing`).
- This policy is intended to carry forward to future production pipelines in this repository.
- Legacy/testing class implementations (`Classes_v3.py`, `Classes_v4.py`, `Classes_v5.py`) are retained under `modules/legacy/` only and are not treated as numerically reliable for production results.
- Do not mix outputs produced with non-`Classes_v2` class implementations into production contour datasets.

## Default Production Configuration

The production batch scripts are configured for the following default run:

- Flux ratio: `I = 0.5`
- Redshift: `z = 1e-8`
- Orientation preset: `Taman_edgeon`
- Chirp mass grid: `mcz = 10..90 Msun` with `81` points
- Time delay grid: `td = 20..70 ms` with `51` points
- Omega grid: `omega = 0..6` with `61` points
- Theta grid: `theta = 0..15` with `151` points
- Gamma grid: `[0, 2pi)` with `51` points

## Quick Start

```bash
# Stage 0: Build banks first (can be chunked across mcz with SLURM arrays)
sbatch batch_scripts/build_template_banks.sbatch

# Stage 1: Compute mismatch cubes
sbatch batch_scripts/compute_mismatch_cubes.sbatch

# Stage 2: Aggregate (run once after all chunks complete)
python -m scripts.mismatch_mcz_td.aggregate_best_match \
  --run_dir ./data/mismatch \
  --I 0.5 \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 90 \
  --orientation_tag Taman_edgeon \
  --z 1e-8

# Stage 3: Plot (can be run multiple times with different settings)
python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match \
  --input_path ./data/mismatch_I0p5_z1e-08_mcz10-90_td20-70_Taman_edgeon/best_match/<best_match_file>.h5 \
  --output_dir ./figures/mismatch
```

## HDF5 File Structure

### Per-mcz Mismatch Cube (`run_dir/mismatch_cubes/*.h5`)

**Datasets:**
- `mcz`: Scalar chirp mass value (Msun)
- `td`: Time delay array (seconds)
- `theta`, `omega`, `gamma`: Template parameter arrays
- `epsilon_min_grid`: (td, theta, omega) - minimum mismatch over gamma
- `gamma_best_grid`: (td, theta, omega) - gamma value achieving minimum
- `mismatch` (optional): (td, theta, omega, gamma) - full mismatch array if `--save_full_mismatch`

**File Attributes:**
- `I`: Flux ratio
- `theta_J`, `phi_J`: Detector orientation angles (or NaN if using preset)
- `theta_S`, `phi_S`: Source orientation angles (or NaN if using preset)
- `mcz_min`, `mcz_max`, `mcz_pts`: Intended mcz grid from Stage 1 compute settings

### Best-Match File (`run_dir/best_match/*.h5`)

**Datasets:**
- `mcz`: Expected chirp mass grid for plotting (missing internal rows are kept)
- `td`: Time delay array (seconds)
- `epsilon_min`: (mcz, td) - global minimum mismatch
- `omega_best`, `theta_best`, `gamma_best`: (mcz, td) - best-fit template parameters
- `expected_mcz`: Expected mcz grid used by Stage 2
- `missing_mcz` (optional): Missing internal mcz values detected during aggregation

**File Attributes:**
- `I`, `theta_J`, `phi_J`, `theta_S`, `phi_S`: Propagated from cubes
- `orientation_tag`, `z`: Used by Stage 3 for automatic figure naming
- `missing_mcz_count`: Number of missing internal mcz rows detected by Stage 2

## Stage 0: Build Template Banks

**Script:** `python -m scripts.template_banks.build_template_banks`

Builds one HDF5 template bank per `mcz` value. The mismatch stage streams from these banks and expects the same orientation tag and grid definition.

**Example:**
```bash
python -m scripts.template_banks.build_template_banks \
  --orient_preset Taman_edgeon \
  --mcz_min 10 --mcz_max 90 --mcz_pts 81 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --z 1e-8 \
  --bank_dir ./data/template_banks
```

## Stage 1: Compute Mismatch Cubes

**Script:** `python -m scripts.mismatch_mcz_td.compute_mismatch_cubes`

Streams templates from prebuilt banks and evaluates mismatches across (td, theta, omega, gamma) parameter space. Supports SLURM array job chunking for parallelization.

**Example:**
```bash
python -m scripts.mismatch_mcz_td.compute_mismatch_cubes \
  --I 0.5 \
  --orient_preset Taman_edgeon \
  --mcz_min 10 --mcz_max 90 --mcz_pts 81 \
  --td_min_ms 20 --td_max_ms 70 --td_pts 51 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --z 1e-8 \
  --n_workers 8 \
  --use_opt_match \
  --bank_dir ./data/template_banks \
  --run_dir ./data/mismatch
```

**Key arguments:**
- `--I`: Flux ratio (0 < I < 1)
- `--orient_preset`: Orientation preset (e.g., `Taman_edgeon`)
- `--mcz_min/max/pts`: Chirp mass grid
- `--td_min_ms/max_ms/pts`: Time delay grid (milliseconds)
- `--omega/theta/gamma`: Template parameter grids
- `--mcz_chunk_index/count`: For SLURM array job chunking
- `--run_dir` is a base root; final run directory is auto-derived.

## Stage 2: Aggregate Best-Match Data

**Script:** `python -m scripts.mismatch_mcz_td.aggregate_best_match`

Finds global minimum across (theta, omega) for each (mcz, td) and consolidates into a single best-match HDF5 file.
If internal mcz rows are missing, Stage 2 keeps those rows as NaNs so the contour plot shows explicit gaps.

**Example:**
```bash
python -m scripts.mismatch_mcz_td.aggregate_best_match \
  --run_dir ./data/mismatch \
  --I 0.5 \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 90 \
  --orientation_tag Taman_edgeon \
  --z 1e-8
```

## Stage 3: Plot Contour

**Script:** `python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match`

Generates publication-quality contour plot of mismatch vs (td, mcz) from the best-match file.
This stage now requires an exact best-match `--input_path`; all run tokens are inferred from file metadata.

**Example:**
```bash
python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match \
  --input_path ./data/mismatch_I0p5_z1e-08_mcz10-90_td20-70_Taman_edgeon/best_match/<best_match_file>.h5 \
  --output_dir ./figures/mismatch

```

## Batch Script Configuration

The batch pipeline shares defaults via:

- `batch_scripts/_contour_mcz_td_config.sh`

Important exported variables used by Stage 0/1 and Lindblom batch jobs:

- `FLUX_RATIO` (default `0.5`)
- `ORIENT_PRESET` (default `Taman_edgeon`)
- `MCZ_MIN`, `MCZ_MAX`, `MCZ_PTS` (defaults `10`, `90`, `81`)
- `TD_MIN_MS`, `TD_MAX_MS`, `TD_PTS` (defaults `20`, `70`, `51`)
- `OMEGA_MIN`, `OMEGA_MAX`, `OMEGA_PTS` (defaults `0`, `6`, `61`)
- `THETA_MIN`, `THETA_MAX`, `THETA_PTS` (defaults `0`, `15`, `151`)
- `GAMMA_PTS` (default `51`, interpreted as `[0, 2pi)`)
- `BANK_DIR` (default `./data/template_banks`)
- `RUN_DIR` (default `./data/mismatch`)
- `Z` (default `1e-8`, propagated to Python scripts as `--z`)

Lindblom batch jobs also resolve canonical cube/bank paths from these settings,
so avoid hardcoding file names in local wrappers.

## Directory and File Naming Conventions

### Run directories (resolved from base dirs)

- **Template banks:** `{bank_dir_base}_z{z}`
  - Example: `./data/template_banks_z0p2`
- **Contour results (cubes + best_match):** `{run_dir_base}_I{I}_z{z}_mcz{mcz_min}-{mcz_max}_td{td_min}-{td_max}`
  - Example: `./data/mismatch_I0p5_z0p2_mcz16-25_td20-70`
- **Contour figures:** `{fig_dir_base}_I{I}_z{z}_mcz{mcz_min}-{mcz_max}_td{td_min}-{td_max}`
  - Example: `./figures/mismatch_I0p5_z0p2_mcz16-25_td20-70`

### Canonical file names

- **Template banks:**
  - `rp_bank_z{z}_mcz{mcz}_omega{omin}-{omax}x{omega_pts}_theta{tmin}-{tmax}x{theta_pts}_gamma0-2pix{gamma_pts}_{tag}.h5`
- **Mismatch cubes (under `mismatch_cubes/`):**
  - `mismatch_cubes_z{z}_mcz{mcz}_I{I}_td{tdmin}-{tdmax}x{td_pts}_omega{omin}-{omax}x{omega_pts}_theta{tmin}-{tmax}x{theta_pts}_gamma0-2pix{gamma_pts}_{tag}.h5`
- **Best-match (under `best_match/`):**
  - `best_match_I{I}_z{z}_mcz{mczmin}-{mczmax}x{mcz_pts}_td{tdmin}-{tdmax}x{td_pts}_omega{omin}-{omax}x{omega_pts}_theta{tmin}-{tmax}x{theta_pts}_gamma0-2pix{gamma_pts}_{tag}.h5`
- **Contour figure:**
  - `contour_I{I}_z{z}_mcz{mczmin}-{mczmax}x{mcz_pts}_td{tdmin}-{tdmax}x{td_pts}_min_mismatch_{tag}.pdf`

Notes:
- Numeric tokens use minimal precision with `p` as decimal separator (e.g., `0p2`).
- Gamma naming is fixed to radians over `[0, 2pi]` and encoded as `gamma0-2pix{gamma_pts}`.

## Key Benefits

1. **Single Plotting Script** - No duplicate plotting code
2. **Full Metadata Chain** - Source parameters preserved throughout
3. **Clear Responsibilities** - Each script does exactly one thing
4. **Safer Parallelization** - No race conditions when chunking
5. **Reproducible** - Best-match files contain all necessary metadata

## Testing Checklist

Verify the pipeline works correctly:

- [ ] Template banks exist for the requested `mcz` range and orientation tag
- [ ] Mismatch cubes contain source attributes (`I`, orientation angles)
- [ ] Best-match file contains propagated attributes
- [ ] Plotting script can read and plot from best-match file
- [ ] Batch scripts call correct Python scripts
- [ ] File naming conventions allow automatic file discovery

## Prerequisites

Build template banks before running this pipeline:

```bash
python -m scripts.template_banks.build_template_banks \
  --orient_preset Taman_edgeon \
  --mcz_min 10 --mcz_max 90 --mcz_pts 81 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --z 1e-8 \
  --bank_dir ./data/template_banks
```

Or use the SLURM batch script:
```bash
sbatch batch_scripts/build_template_banks.sbatch
```
