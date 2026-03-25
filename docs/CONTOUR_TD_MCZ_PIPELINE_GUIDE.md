# Contour (td, mcz) Pipeline Guide

This document describes the three-stage modular pipeline for computing mismatch maps between lensed gravitational-wave sources and precessing template banks, then plotting contours across the (time delay, chirp mass) parameter space.

## Pipeline Overview

**Three stages with single responsibilities:**

1. **Build per-mcz template banks** (`python -m scripts.template_banks.build_template_banks`)
2. **Compute per-mcz mismatch cubes** (`python -m scripts.mismatch_mcz_td.compute_mismatch_cubes`)
3. **Aggregate cubes into best-match file** (`python -m scripts.mismatch_mcz_td.aggregate_best_match`)
4. **Plot mismatch contour** (`python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match`)

## Quick Start

```bash
# Stage 0: Build banks first (can be chunked across mcz with SLURM arrays)
sbatch batch_scripts/build_template_banks.sbatch

# Stage 1: Compute mismatch cubes
sbatch batch_scripts/compute_mismatch_cubes.sbatch

# Stage 2: Aggregate (run once after all chunks complete)
python -m scripts.mismatch_mcz_td.aggregate_best_match \
  --results_dir ./data/contours_td_mcz \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 16 --mcz_max 25 \
  --orientation_tag Taman_edgeon

# Stage 3: Plot (can be run multiple times with different settings)
python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match \
  --results_dir ./data/contours_td_mcz \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 16 --mcz_max 25 \
  --orientation_tag Taman_edgeon
```

## HDF5 File Structure

### Per-mcz Mismatch Cube (`results_dir/mismatch_cubes/*.h5`)

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

### Best-Match File (`results_dir/best_match/*.h5`)

**Datasets:**
- `mcz`: Expected chirp mass grid for plotting (missing internal rows are kept)
- `td`: Time delay array (seconds)
- `epsilon_min`: (mcz, td) - global minimum mismatch
- `omega_best`, `theta_best`, `gamma_best`: (mcz, td) - best-fit template parameters
- `expected_mcz`: Expected mcz grid used by Stage 2
- `missing_mcz` (optional): Missing internal mcz values detected during aggregation

**File Attributes:**
- `I`, `theta_J`, `phi_J`, `theta_S`, `phi_S`: Propagated from cubes
- `missing_mcz_count`: Number of missing internal mcz rows detected by Stage 2

## Stage 0: Build Template Banks

**Script:** `python -m scripts.template_banks.build_template_banks`

Builds one HDF5 template bank per `mcz` value. The mismatch stage streams from these banks and expects the same orientation tag and grid definition.

**Example:**
```bash
python -m scripts.template_banks.build_template_banks \
  --orient_preset Taman_edgeon \
  --mcz_min 16 --mcz_max 25 --mcz_pts 10 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
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
  --mcz_min 10 --mcz_max 80 --mcz_pts 71 \
  --td_min_ms 20 --td_max_ms 70 --td_pts 51 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --n_workers 8 \
  --use_opt_match \
  --bank_dir ./data/template_banks \
  --results_dir ./data/contours_td_mcz
```

**Key arguments:**
- `--I`: Flux ratio (0 < I < 1)
- `--orient_preset`: Orientation preset (e.g., `Taman_edgeon`)
- `--mcz_min/max/pts`: Chirp mass grid
- `--td_min_ms/max_ms/pts`: Time delay grid (milliseconds)
- `--omega/theta/gamma`: Template parameter grids
- `--mcz_chunk_index/count`: For SLURM array job chunking

## Stage 2: Aggregate Best-Match Data

**Script:** `python -m scripts.mismatch_mcz_td.aggregate_best_match`

Finds global minimum across (theta, omega) for each (mcz, td) and consolidates into a single best-match HDF5 file.
If internal mcz rows are missing, Stage 2 keeps those rows as NaNs so the contour plot shows explicit gaps.

**Example:**
```bash
python -m scripts.mismatch_mcz_td.aggregate_best_match \
  --results_dir ./data/contours_td_mcz \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 80 \
  --orientation_tag Taman_edgeon
```

## Stage 3: Plot Contour

**Script:** `python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match`

Generates publication-quality contour plot of mismatch vs (td, mcz) from the best-match file.

**Example:**
```bash
python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match \
  --results_dir ./data/contours_td_mcz \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 80 \
  --orientation_tag Taman_edgeon \
  --output_dir ./figures/mismatch_mcz_td
```

## File Naming Conventions

- **Template banks:** `rp_bank_mcz{mcz}_omega{min}-{max}x{omega_pts}_theta{min}-{max}x{theta_pts}_g{gamma_pts}_{tag}.h5`
- **Mismatch cubes:** `mismatch_cubes/mismatch_cubes_mcz{mcz}_I{I}_td{min}-{max}x{td_pts}_o{omega_pts}-t{theta_pts}-g{gamma_pts}_{tag}.h5`
- **Best-match:** `best_match/best_match_I{I}_mcz{min}-{max}x{mcz_pts}_td{min}-{max}x{td_pts}_o{omega_pts}-t{theta_pts}-g{gamma_pts}_{tag}.h5`
- **Figure:** `contour_I{I}_mcz{min}-{max}_td{min}-{max}_min_mismatch_{tag}.pdf`

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
  --mcz_min 10 --mcz_max 80 --mcz_pts 71 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --bank_dir ./data/template_banks
```

Or use the SLURM batch script:
```bash
sbatch batch_scripts/build_template_banks.sbatch
```
