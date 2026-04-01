# HDF5 Metadata Schema v1.0 (Draft)

This document defines a practical metadata contract for the lensing/precession pipeline HDF5 outputs.

Goals:
- Make files self-describing and robust to filename drift.
- Enable reproducibility and lightweight validation.
- Keep compatibility with current pipeline outputs.

## 1) Common Global Attributes (all pipeline HDF5 outputs)

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| `schema_name` | string | yes | `lensing.pipeline.hdf5` | Constant across all files in this schema family. |
| `schema_version` | string | yes | `1.0.0` | Semantic version for reader compatibility checks. |
| `pipeline_stage` | string | yes | `stage0_bank`, `stage1_mismatch_cube`, `stage2_best_match` | Drives stage-specific validation. |
| `created_utc` | string (ISO-8601 UTC) | yes | `2026-03-24T22:05:11Z` | File creation timestamp. |
| `writer_module` | string | yes | `scripts.mismatch_mcz_td.compute_mismatch_cubes` | Python module path used to write file. |
| `command_line` | string | recommended | `--I 0.5 --mcz_min 10 ...` | Full CLI args for reproducibility. |
| `git_commit` | string | recommended | `a1b2c3d4` | Commit used to generate file. |
| `orientation_tag` | string | yes | `Taman_edgeon` | Must be explicit attribute, not filename-only. |

## 2) Stage 0: Template Bank File Schema (`stage0_bank`)

### Required datasets

| Dataset | Type | Shape | Required | Notes |
|---|---|---|---|---|
| `omega` | float64 | `(omega_pts,)` | yes | Coordinate axis. |
| `theta` | float64 | `(theta_pts,)` | yes | Coordinate axis. |
| `gamma` | float64 | `(gamma_pts,)` | yes | Coordinate axis. |
| `bank` | complex64/complex128 | `(theta_pts, omega_pts, gamma_pts, n_freq)` | yes | Main template bank. |

### Required file attributes

| Field | Type | Example |
|---|---|---|
| `omega_pts` | int | `61` |
| `theta_pts` | int | `151` |
| `gamma_pts` | int | `51` |
| `omega_min` | float | `0.0` |
| `omega_max` | float | `6.0` |
| `theta_min` | float | `0.0` |
| `theta_max` | float | `15.0` |

### Required `bank` dataset attributes

| Field | Type | Example | Notes |
|---|---|---|---|
| `f_min` | float | `20.0` | Frequency lower bound. |
| `delta_f` | float | `0.25` | Frequency spacing. |
| `mcz_msun` | float | `25.0` | Physical mcz represented by this bank. |
| `axis_order` | string | `theta,omega,gamma,freq` | Required to avoid dimension-order ambiguity. |

### Recommended coordinate dataset attributes

| Dataset | Field | Type | Example |
|---|---|---|---|
| `omega` | `units` | string | `dimensionless` |
| `theta` | `units` | string | `dimensionless` |
| `gamma` | `units` | string | `dimensionless` |

## 3) Stage 1: Per-mcz Mismatch Cube Schema (`stage1_mismatch_cube`)

### Required datasets

| Dataset | Type | Shape | Required | Notes |
|---|---|---|---|---|
| `mcz` | float64 | `(1,)` | yes | Scalar mass value for this file. |
| `td` | float64 | `(td_pts,)` | yes | Time-delay axis (seconds). |
| `theta` | float64 | `(theta_pts,)` | yes | Axis. |
| `omega` | float64 | `(omega_pts,)` | yes | Axis. |
| `gamma` | float64 | `(gamma_pts,)` | yes | Axis. |
| `epsilon_min_grid` | float32 | `(td_pts, theta_pts, omega_pts)` | yes | Min mismatch over gamma. |
| `gamma_best_grid` | float32 | `(td_pts, theta_pts, omega_pts)` | yes | Argmin gamma at each cell. |
| `mismatch` | float32 | `(td_pts, theta_pts, omega_pts, gamma_pts)` | optional | Present only when full mismatch is saved. |

### Required file attributes

| Field | Type | Example | Notes |
|---|---|---|---|
| `I` | float | `0.5` | Flux ratio. |
| `theta_J` | float or NaN | `NaN` | Source/lens orientation metadata. |
| `phi_J` | float or NaN | `NaN` | Source/lens orientation metadata. |
| `theta_S` | float or NaN | `NaN` | Source/lens orientation metadata. |
| `phi_S` | float or NaN | `NaN` | Source/lens orientation metadata. |
| `mcz_min` | float | `10.0` | Intended Stage 1 grid min. |
| `mcz_max` | float | `90.0` | Intended Stage 1 grid max. |
| `mcz_pts` | int | `81` | Intended Stage 1 grid size. |

### Required dataset attributes

| Dataset | Field | Type | Example |
|---|---|---|---|
| `td` | `units` | string | `s` |
| `mcz` | `units` | string | `Msun` |
| `theta` | `units` | string | `dimensionless` |
| `omega` | `units` | string | `dimensionless` |
| `gamma` | `units` | string | `dimensionless` |
| `epsilon_min_grid` | `axis_order` | string | `td,theta,omega` |
| `gamma_best_grid` | `axis_order` | string | `td,theta,omega` |
| `mismatch` (if present) | `axis_order` | string | `td,theta,omega,gamma` |

## 4) Stage 2: Best-Match Aggregate Schema (`stage2_best_match`)

### Required datasets

| Dataset | Type | Shape | Required | Notes |
|---|---|---|---|---|
| `mcz` | float64 | `(mcz_pts_out,)` | yes | Output mcz axis. |
| `td` | float64 | `(td_pts,)` | yes | Time-delay axis. |
| `epsilon_min` | float32 | `(mcz_pts_out, td_pts)` | yes | Global min mismatch over theta/omega. |
| `omega_best` | float32 | `(mcz_pts_out, td_pts)` | yes | Best omega at min mismatch. |
| `theta_best` | float32 | `(mcz_pts_out, td_pts)` | yes | Best theta at min mismatch. |
| `gamma_best` | float32 | `(mcz_pts_out, td_pts)` | yes | Best gamma at min mismatch. |
| `expected_mcz` | float64 | `(mcz_pts_expected,)` | yes | Declared complete expected grid. |
| `missing_mcz` | float64 | `(k,)` | optional | Missing internal rows if any. |

### Required file attributes

| Field | Type | Example |
|---|---|---|
| `I` | float | `0.5` |
| `theta_J` | float or NaN | `NaN` |
| `phi_J` | float or NaN | `NaN` |
| `theta_S` | float or NaN | `NaN` |
| `phi_S` | float or NaN | `NaN` |
| `missing_mcz_count` | int | `2` |

### Required dataset attributes

| Dataset | Field | Type | Example |
|---|---|---|---|
| `mcz` | `units` | string | `Msun` |
| `td` | `units` | string | `s` |
| `epsilon_min` | `axis_order` | string | `mcz,td` |
| `omega_best` | `axis_order` | string | `mcz,td` |
| `theta_best` | `axis_order` | string | `mcz,td` |
| `gamma_best` | `axis_order` | string | `mcz,td` |

## 5) Compatibility Policy

- Readers should require `schema_name` and parse `schema_version`.
- Minor versions (1.x) can add optional fields without breaking existing readers.
- Major version bumps are required for breaking changes to required fields or shapes.

## 6) Practical Validation Rules

A file is valid if:
1. Common global attributes exist and have correct types.
2. Stage-specific required datasets/attrs exist.
3. Coordinate dataset lengths match result tensor dimensions.
4. `missing_mcz_count == len(missing_mcz)` when `missing_mcz` exists.
5. Required `axis_order` and `units` attrs are present.

## 7) Why This Schema Is Useful

- Prevents silent mistakes from filename parsing assumptions.
- Makes each file self-describing for future analyses.
- Enables simple automated pre-plot/pre-aggregation checks.
- Improves reproducibility with script/CLI/commit provenance.
- Allows safe schema evolution through versioned compatibility.
