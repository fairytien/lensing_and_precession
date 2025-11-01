# Contour (td, mcz) Pipeline Guide

This document describes the three-stage modular pipeline for computing mismatch maps between lensed gravitational-wave sources and precessing template banks, then plotting contours across the (time delay, chirp mass) parameter space.

## Pipeline Overview

**Three stages with single responsibilities:**

1. **Compute per-mcz mismatch cubes** (`scripts/compute_mismatch_cubes.py`)
2. **Aggregate cubes into best-match file** (`scripts/aggregate_best_match.py`)
3. **Plot mismatch contour** (`scripts/create_contour_mcz_td_from_best_match.py`)

## Quick Start

```bash
# Stage 1: Compute (can be chunked across mcz with SLURM arrays)
sbatch batch_scripts/compute_mismatch_cubes.sbatch

# Stage 2: Aggregate (run once after all chunks complete)
python scripts/aggregate_best_match.py \
  --results_dir ./data/contours \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 16 --mcz_max 25 \
  --orientation_tag Taman_edgeon

# Stage 3: Plot (can be run multiple times with different settings)
python scripts/create_contour_mcz_td_from_best_match.py \
  --results_dir ./data/contours \
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

### Best-Match File (`results_dir/best_match_*.h5`)

**Datasets:**
- `mcz`: Full chirp mass array (may include NaN rows for missing mcz values)
- `td`: Time delay array (seconds)
- `epsilon_min`: (mcz, td) - global minimum mismatch
- `omega_best`, `theta_best`, `gamma_best`: (mcz, td) - best-fit template parameters

**File Attributes:**
- `I`, `theta_J`, `phi_J`, `theta_S`, `phi_S`: Propagated from cubes

## Stage 1: Compute Mismatch Cubes

**Script:** `scripts/compute_mismatch_cubes.py`

Streams templates from prebuilt banks and evaluates mismatches across (td, theta, omega, gamma) parameter space. Supports SLURM array job chunking for parallelization.

**Example:**
```bash
python scripts/compute_mismatch_cubes.py \
  --I 0.5 \
  --orient_preset Taman_edgeon \
  --mcz_min 10 --mcz_max 80 --mcz_pts 71 \
  --td_min_ms 20 --td_max_ms 70 --td_pts 51 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --n_workers 8 \
  --use_opt_match \
  --results_dir ./data/contours
```

**Key arguments:**
- `--I`: Flux ratio (0 < I < 1)
- `--orient_preset`: Orientation preset (e.g., `Taman_edgeon`)
- `--mcz_min/max/pts`: Chirp mass grid
- `--td_min_ms/max_ms/pts`: Time delay grid (milliseconds)
- `--omega/theta/gamma`: Template parameter grids
- `--mcz_chunk_index/count`: For SLURM array job chunking

## Stage 2: Aggregate Best-Match Data

**Script:** `scripts/aggregate_best_match.py`

Finds global minimum across (theta, omega) for each (mcz, td) and consolidates into a single best-match HDF5 file.

**Example:**
```bash
python scripts/aggregate_best_match.py \
  --results_dir ./data/contours \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 80 \
  --orientation_tag Taman_edgeon
```

## Stage 3: Plot Contour

**Script:** `scripts/create_contour_mcz_td_from_best_match.py`

Generates publication-quality contour plot of mismatch vs (td, mcz) from the best-match file.

**Example:**
```bash
python scripts/create_contour_mcz_td_from_best_match.py \
  --results_dir ./data/contours \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 80 \
  --orientation_tag Taman_edgeon \
  --output_dir ./figures
```

## File Naming Conventions

- **Mismatch cubes:** `mismatch_cubes/mismatch_cubes_mcz{mcz}Msun_td{min}-{max}ms_{tag}.h5`
- **Best-match:** `best_match_td{min}-{max}ms_mcz{min}-{max}Msun_{tag}.h5`
- **Figure:** `contour_td_mcz_td{min}-{max}ms_mcz{min}-{max}Msun_{tag}.pdf`

## Key Benefits

1. **Single Plotting Script** - No duplicate plotting code
2. **Full Metadata Chain** - Source parameters preserved throughout
3. **Clear Responsibilities** - Each script does exactly one thing
4. **Safer Parallelization** - No race conditions when chunking
5. **Reproducible** - Best-match files contain all necessary metadata

## Testing Checklist

Verify the pipeline works correctly:

- [ ] Mismatch cubes contain source attributes (`I`, orientation angles)
- [ ] Best-match file contains propagated attributes
- [ ] Plotting script can read and plot from best-match file
- [ ] Batch scripts call correct Python scripts
- [ ] File naming conventions allow automatic file discovery

## Prerequisites

Build template banks before running this pipeline:

```bash
python scripts/build_template_banks.py \
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
