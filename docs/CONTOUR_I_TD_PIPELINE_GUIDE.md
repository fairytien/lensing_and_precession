# Contour (td, I) Pipeline Guide

This document describes the three-stage modular pipeline for computing mismatch maps between lensed gravitational-wave sources and precessing template banks, then plotting contours across the (time delay, flux ratio) parameter space.

## Pipeline Overview

**Key Difference from mcz_td Pipeline:**
- The **mcz_td pipeline** varies chirp mass `mcz` at fixed flux ratio `I`. Each `mcz` requires a different template bank.
- The **I_td pipeline** varies flux ratio `I` at fixed chirp mass `mcz`. Since template banks depend only on `mcz` (not `I`), **all I values share the SAME template bank**.

**Three stages with single responsibilities:**

1. **Build template bank for fixed mcz** (`python -m scripts.template_banks.build_template_banks`)
2. **Compute per-I mismatch cubes** (`python -m scripts.mismatch_I_td.compute_mismatch_cubes`)
3. **Aggregate cubes into best-match file** (`python -m scripts.mismatch_I_td.aggregate_best_match`)
4. **Plot mismatch contour** (`python -m scripts.mismatch_I_td.plot_contour_I_td_from_best_match`)

> **Class Version Policy:** See "Common Runtime Notes" in [SCRIPTS_PIPELINES_GUIDE.md](SCRIPTS_PIPELINES_GUIDE.md#common-runtime-notes) for module import conventions and class version policies.

## Default Production Configuration

The production batch scripts are configured for the following default run:

- Chirp mass (fixed): `mcz = 20 Msun`
- Redshift: `z = 1`
- Orientation preset: `Taman_edgeon`
- Flux ratio grid: `I = 0.1..0.9` with `41` points
- Time delay grid: `td = 20..70 ms` with `51` points
- Omega grid: `omega = 0..6` with `61` points
- Theta grid: `theta = 0..15` with `151` points
- Gamma grid: `[0, 2pi)` with `51` points

## Quick Start

```bash
# Stage 0: Build ONE template bank for the fixed mcz value
python -m scripts.template_banks.build_template_banks \
  --orient_preset Taman_edgeon \
  --mcz_min 20 --mcz_max 20 --mcz_pts 1 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --z 1 \
  --bank_dir ./data/template_banks

# Stage 1: Compute mismatch cubes (array job across I values)
sbatch batch_scripts/compute_mismatch_I_cubes.sbatch

# Stage 2: Aggregate (run once after all chunks complete)
python -m scripts.mismatch_I_td.aggregate_best_match \
  --run_dir ./data/mismatch \
  --mcz 20 \
  --td_min_ms 20 --td_max_ms 70 \
  --I_min 0.1 --I_max 0.9 \
  --orientation_tag Taman_edgeon \
  --z 1

# Stage 3: Plot (can be run multiple times with different settings)
python -m scripts.mismatch_I_td.plot_contour_I_td_from_best_match \
  --input_path ./data/mismatch_mcz20_I0p1-0p9_z1_td20-70_Taman_edgeon/best_match/<best_match_file>.h5 \
  --output_dir ./figures/mismatch
```

## HDF5 File Structure

### Per-I Mismatch Cube (`run_dir/mismatch_cubes/*.h5`)

**Datasets:**
- `I`: Scalar flux ratio value (dimensionless)
- `mcz`: Scalar chirp mass value (Msun)
- `td`: Time delay array (seconds)
- `theta`, `omega`, `gamma`: Template parameter arrays
- `epsilon_min_grid`: (td, theta, omega) - minimum mismatch over gamma
- `gamma_best_grid`: (td, theta, omega) - gamma value achieving minimum
- `mismatch` (optional): (td, theta, omega, gamma) - full mismatch array if `--save_full_mismatch`

**File Attributes:**
- `I_min`, `I_max`, `I_pts`: Intended I grid from Stage 1 compute settings
- `theta_J`, `phi_J`: Detector orientation angles (or NaN if using preset)
- `theta_S`, `phi_S`: Source orientation angles (or NaN if using preset)
- `orientation_tag`: Orientation preset used

### Best-Match File (`run_dir/best_match/*.h5`)

**Datasets:**
- `I`: Expected flux ratio grid for plotting (missing internal rows are kept)
- `mcz`: Scalar chirp mass value (Msun)
- `td`: Time delay array (seconds)
- `epsilon_min`: (I, td) - global minimum mismatch
- `omega_best`, `theta_best`, `gamma_best`: (I, td) - best-fit template parameters
- `expected_I`: Expected I grid used by Stage 2
- `missing_I` (optional): Missing internal I values detected during aggregation

**File Attributes:**
- `theta_J`, `phi_J`, `theta_S`, `phi_S`: Propagated from cubes
- `orientation_tag`, `z`: Used by Stage 3 for automatic figure naming
- `missing_I_count`: Number of missing internal I rows detected by Stage 2

## Stage 0: Build Template Bank

**Script:** `python -m scripts.template_banks.build_template_banks`

Builds ONE HDF5 template bank at the fixed `mcz` value. Unlike the mcz_td pipeline which needs multiple banks, the I_td pipeline reuses this single bank for all I values.

**Example:**
```bash
python -m scripts.template_banks.build_template_banks \
  --orient_preset Taman_edgeon \
  --mcz_min 20 --mcz_max 20 --mcz_pts 1 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --z 1 \
  --bank_dir ./data/template_banks
```

## Stage 1: Compute Mismatch Cubes

**Script:** `python -m scripts.mismatch_I_td.compute_mismatch_cubes`

Streams templates from the prebuilt bank and evaluates mismatches across (td, theta, omega, gamma) parameter space for each I value. Supports SLURM array job chunking for parallelization across I values.

**Example:**
```bash
python -m scripts.mismatch_I_td.compute_mismatch_cubes \
  --mcz 20 \
  --orient_preset Taman_edgeon \
  --I_min 0.1 --I_max 0.9 --I_pts 41 \
  --td_min_ms 20 --td_max_ms 70 --td_pts 51 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --z 1 \
  --n_workers 8 \
  --use_opt_match \
  --bank_dir ./data/template_banks \
  --run_dir ./data/mismatch
```

**Key arguments:**
- `--mcz`: Fixed chirp mass in solar masses
- `--orient_preset`: Orientation preset (e.g., `Taman_edgeon`)
- `--I_min/max/pts`: Flux ratio grid
- `--td_min_ms/max_ms/pts`: Time delay grid (milliseconds)
- `--omega/theta/gamma`: Template parameter grids
- `--I_chunk_index/count`: For SLURM array job chunking
- `--run_dir` is a base root; final run directory is auto-derived

## Stage 2: Aggregate Best-Match Data

**Script:** `python -m scripts.mismatch_I_td.aggregate_best_match`

Finds global minimum across (theta, omega) for each (I, td) and consolidates into a single best-match HDF5 file.
If internal I rows are missing, Stage 2 keeps those rows as NaNs so the contour plot shows explicit gaps.

**Example:**
```bash
python -m scripts.mismatch_I_td.aggregate_best_match \
  --run_dir ./data/mismatch \
  --mcz 20 \
  --td_min_ms 20 --td_max_ms 70 \
  --I_min 0.1 --I_max 0.9 \
  --orientation_tag Taman_edgeon \
  --z 1
```

## Stage 3: Plot Contour

**Script:** `python -m scripts.mismatch_I_td.plot_contour_I_td_from_best_match`

Generates publication-quality contour plot of mismatch vs (td, I) from the best-match file.
This stage requires an exact best-match `--input_path`; all run tokens are inferred from file metadata.

**Example:**
```bash
python -m scripts.mismatch_I_td.plot_contour_I_td_from_best_match \
  --input_path ./data/mismatch_mcz20_I0p1-0p9_z1_td20-70_Taman_edgeon/best_match/<best_match_file>.h5 \
  --output_dir ./figures/mismatch
```

## Batch Script Configuration

The batch pipeline shares defaults via:

- `batch_scripts/_contour_I_td_config.sh`

Important exported variables used by Stage 1:

- `MCZ` (default `20`)
- `ORIENT_PRESET` (default `Taman_edgeon`)
- `I_MIN`, `I_MAX`, `I_PTS` (defaults `0.1`, `0.9`, `41`)
- `TD_MIN_MS`, `TD_MAX_MS`, `TD_PTS` (defaults `20`, `70`, `51`)
- `OMEGA_MIN`, `OMEGA_MAX`, `OMEGA_PTS` (defaults `0`, `6`, `61`)
- `THETA_MIN`, `THETA_MAX`, `THETA_PTS` (defaults `0`, `15`, `151`)
- `GAMMA_PTS` (default `51`, interpreted as `[0, 2pi)`)
- `BANK_DIR` (default `./data/template_banks`)
- `RUN_DIR` (default `./data/mismatch`)
- `Z` (default `1`, propagated to Python scripts as `--z`)

## Directory and File Naming Conventions

### Run directories (resolved from base dirs)

- **Template banks:** `{bank_dir_base}_z{z}`
  - Example: `./data/template_banks_z1`
- **Contour results (cubes + best_match):** `{run_dir_base}_mcz{mcz}_I{I_min}-{I_max}_z{z}_td{td_min}-{td_max}`
  - Example: `./data/mismatch_mcz20_I0p1-0p9_z1_td20-70`
- **Contour figures:** `{fig_dir_base}_mcz{mcz}_I{I_min}-{I_max}_z{z}_td{td_min}-{td_max}`
  - Example: `./figures/mismatch_mcz20_I0p1-0p9_z1_td20-70`

### Canonical file names

- **Template banks:**
  - `rp_bank_z{z}_mcz{mcz}_omega{omin}-{omax}x{omega_pts}_theta{tmin}-{tmax}x{theta_pts}_gamma0-2pix{gamma_pts}_{tag}.h5`
- **Mismatch cubes (under `mismatch_cubes/`):**
  - `mismatch_cubes_z{z}_I{I}_mcz{mcz}_td{tdmin}-{tdmax}x{td_pts}_omega{omin}-{omax}x{omega_pts}_theta{tmin}-{tmax}x{theta_pts}_gamma0-2pix{gamma_pts}_{tag}.h5`
- **Best-match (under `best_match/`):**
  - `best_match_mcz{mcz}_I{Imin}-{Imax}x{I_pts}_z{z}_td{tdmin}-{tdmax}x{td_pts}_omega{omin}-{omax}x{omega_pts}_theta{tmin}-{tmax}x{theta_pts}_gamma0-2pix{gamma_pts}_{tag}.h5`
- **Contour figure:**
  - `contour_mcz{mcz}_I{Imin}-{Imax}x{I_pts}_z{z}_td{tdmin}-{tdmax}x{td_pts}_min_mismatch_{tag}.pdf`

Notes:
- Numeric tokens use minimal precision with `p` as decimal separator (e.g., `0p2`).
- Gamma naming is fixed to radians over `[0, 2pi]` and encoded as `gamma0-2pix{gamma_pts}`.

## Function Reference

### filenames.py (I_td pipeline functions)

| Function | Description |
|----------|-------------|
| `contour_I_td_run_dir()` | Return run directory tagged by mcz, I range, z, td range |
| `mismatch_I_cube_filename()` | Build HDF5 path for per-I mismatch cube outputs |
| `parse_I_from_mismatch_I_cube_path()` | Extract I value from canonical mismatch I-cube filenames |
| `best_match_I_td_filename()` | Build HDF5 path for aggregated best-match outputs across all I |
| `contour_I_td_filename()` | Build figure path for the final mismatch contour over (td, I) |
| `find_mismatch_I_cube_files()` | Return mismatch I-cube files matching the requested I-td contour run |
| `parse_I_range_from_best_match_I_td_path()` | Extract (I_min, I_max) from canonical I-td best-match filenames |
| `find_best_match_I_td_file()` | Return the newest best-match file for the requested I-td contour run |

### bank_io.py (I_td pipeline functions)

| Function | Description |
|----------|-------------|
| `write_I_grid_attrs()` | Write intended Stage 1 I grid metadata to an open HDF5 file |
| `read_I_grid_attrs()` | Read I grid metadata from an open HDF5 file if present |
| `I_grid_meta_consistent()` | Return True when two I grid metadata dicts are numerically consistent |
| `write_missing_I_metadata()` | Write aggregation completeness metadata for I-td pipeline |
| `read_missing_I_metadata()` | Read aggregation completeness metadata for I-td pipeline |
| `create_mismatch_I_cube()` | Create HDF5 file with per-I mismatch cube datasets |
| `read_best_match_I_td_contour_data()` | Load one best-match I-td contour dataset and infer plotting metadata |

### cli_utils.py (I_td pipeline argument helpers)

| Function | Description |
|----------|-------------|
| `add_I_grid_args()` | Add --I_min/max/pts/step arguments for flux ratio grid |
| `add_I_chunking_args()` | Add --I_chunk_index/count arguments for array job chunking |

## Comparison: mcz_td vs I_td Pipelines

| Aspect | mcz_td Pipeline | I_td Pipeline |
|--------|-----------------|---------------|
| **Varying parameter** | Chirp mass `mcz` | Flux ratio `I` |
| **Fixed parameter** | Flux ratio `I = 0.5` | Chirp mass `mcz = 20` |
| **Template banks needed** | One per mcz value (81 banks) | ONE bank (fixed mcz) |
| **Outer loop** | mcz values | I values |
| **Output shape** | (mcz, td) | (I, td) |
| **SLURM chunking** | `--mcz_chunk_index/count` | `--I_chunk_index/count` |
| **Run directory** | `_I{I}_z{z}_mcz{min}-{max}_td{min}-{max}` | `_mcz{mcz}_I{min}-{max}_z{z}_td{min}-{max}` |

## Key Benefits

1. **Efficient Bank Reuse** - Single template bank for all I values (no per-I bank generation)
2. **Single Plotting Script** - No duplicate plotting code
3. **Full Metadata Chain** - Source parameters preserved throughout
4. **Clear Responsibilities** - Each script does exactly one thing
5. **Safer Parallelization** - No race conditions when chunking
6. **Reproducible** - Best-match files contain all necessary metadata

## Testing Checklist

Verify the pipeline works correctly:

- [ ] Template bank exists for the requested `mcz` value and orientation tag
- [ ] Mismatch cubes contain source attributes (`I_min`, `I_max`, orientation angles)
- [ ] Best-match file contains propagated attributes
- [ ] Plotting script can read and plot from best-match file
- [ ] Batch scripts call correct Python scripts
- [ ] File naming conventions allow automatic file discovery

## Prerequisites

Build ONE template bank for the fixed mcz before running this pipeline:

```bash
python -m scripts.template_banks.build_template_banks \
  --orient_preset Taman_edgeon \
  --mcz_min 20 --mcz_max 20 --mcz_pts 1 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --z 1 \
  --bank_dir ./data/template_banks
```
