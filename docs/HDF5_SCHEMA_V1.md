# HDF5 Artifact Schema v1

This document describes the HDF5 files currently written by the production
`mcz_td` and `I_td` pipelines.

It is descriptive rather than aspirational: the tables below reflect the files
produced by the current writers in:

- `modules/template_bank.py`
- `scripts/mismatch_mcz_td/compute_mismatch_cubes.py`
- `scripts/mismatch_mcz_td/aggregate_best_match.py`
- `scripts/mismatch_I_td/compute_mismatch_cubes.py`
- `scripts/mismatch_I_td/aggregate_best_match.py`

For workflow, CLI, and naming conventions, see:

- [SCRIPTS_PIPELINES_GUIDE.md](SCRIPTS_PIPELINES_GUIDE.md)
- [CONTOUR_MCZ_TD_PIPELINE_GUIDE.md](CONTOUR_MCZ_TD_PIPELINE_GUIDE.md)
- [CONTOUR_I_TD_PIPELINE_GUIDE.md](CONTOUR_I_TD_PIPELINE_GUIDE.md)

## 1) Common Conventions

- Sweep axes live in datasets. If a file also carries a copied scalar attribute
  for the same quantity, treat the dataset as authoritative.
- Coordinate datasets are stored as `float64`.
- Mismatch result grids are stored as `float32`.
- Template-bank strain data are stored as `complex64` or `complex128`.
- Dataset units are stored in dataset attribute `units` when available.
- Result tensors carry an `axis_order` dataset attribute.
- Scalar metadata written through shared helpers may also carry companion
  `unit_<name>` file attributes.
- Parameter snapshots are stored as prefixed file attributes:
  - `source_param_*` and optional `source_param_unit_*`
  - `template_param_*` and optional `template_param_unit_*`
- Parameter snapshot units for the canonical physics parameters follow [modules/default_params.py](/Users/fairytien/Documents/TEXAS_Bridge_2324/code/lensing_and_precession/modules/default_params.py):
  - `theta_S`, `phi_S`, `theta_J`, `phi_J`, `phi_c`, `gamma_P`: `rad`
  - `mcz`, `dist`, `t_c`, `MLz`: `s`
  - `eta`, `y`, `I`, `theta_tilde`, `omega_tilde`: `dimensionless`
- Additional writer-added convenience metadata use the shared unit table in `modules/bank_io.py`, for example:
  - `mcz_msun`, `mcz_det_msun`, `mcz_source_msun`, `mcz_detector_msun`: `Msun`
- The template-bank coordinate datasets `theta` and `omega` are the swept
  dimensionless bank axes for `theta_tilde` and `omega_tilde` respectively.
- The template-bank coordinate dataset `gamma` is the swept phase axis for
  `gamma_P` and is stored in `rad`.
- Those bank coordinates are not the sky/binary angle parameters `theta_S`,
  `phi_S`, `theta_J`, or `phi_J`.
- Current files do not write global schema/provenance fields such as
  `schema_name`, `schema_version`, `pipeline_stage`, `created_utc`,
  `writer_module`, `command_line`, or `git_commit`.

## 2) Stage 0: Template Bank File

This file format is shared by both pipelines.

### Datasets written

| Dataset | Type | Shape | Notes |
|---|---|---|---|
| `omega` | float64 | `(omega_pts,)` | Template-bank sweep axis for `omega_tilde`. |
| `theta` | float64 | `(theta_pts,)` | Template-bank sweep axis for `theta_tilde`. |
| `gamma` | float64 | `(gamma_pts,)` | Template-bank sweep axis for `gamma_P` over `[0, 2pi)` in radians. |
| `bank` | complex64 or complex128 | `(theta_pts, omega_pts, gamma_pts, n_freq)` | Main template bank. |

### File attributes written

| Field | Type | Notes |
|---|---|---|
| `orientation_tag` | string | Required by downstream readers. |
| `z` | float or NaN | File-level redshift metadata, dimensionless. |
| `unit_z` | string | Present when `z` is written through the shared scalar helper. |
| `omega_pts` | int | Axis size. |
| `theta_pts` | int | Axis size. |
| `gamma_pts` | int | Axis size. |
| `omega_min` | float | Axis bounds written from the actual axis array, in dimensionless `omega_tilde` units. |
| `omega_max` | float | Axis bounds written from the actual axis array, in dimensionless `omega_tilde` units. |
| `theta_min` | float | Axis bounds written from the actual axis array, in dimensionless `theta_tilde` units. |
| `theta_max` | float | Axis bounds written from the actual axis array, in dimensionless `theta_tilde` units. |
| `template_param_*` | scalar attrs | Template parameter snapshot. |
| `template_param_unit_*` | string attrs | Units for known template params. |

### Dataset attributes written

| Dataset | Field | Value / meaning |
|---|---|---|
| `omega` | `units` | `dimensionless` |
| `theta` | `units` | `dimensionless` |
| `gamma` | `units` | `rad` |
| `bank` | `axis_order` | `theta,omega,gamma,freq` |
| `bank` | `f_min` | Lower frequency cutoff. |
| `bank` | `delta_f` | Frequency spacing. |
| `bank` | `mcz_msun` | Source-frame chirp mass in Msun. |
| `bank` | `mcz_det_msun` | Detector-frame chirp mass in Msun. |
| `bank` | `z` | Redshift written alongside the bank. |
| `bank` | `unit_f_min`, `unit_delta_f`, `unit_mcz_msun`, `unit_mcz_det_msun`, `unit_z` | Present for known physical units. |

## 3) Stage 1: Mismatch Cube Files

Both pipelines write the same core mismatch-cube layout. The only differences
are which scalar datasets define the per-file sweep point and which grid
metadata attrs are attached.

### Shared datasets written by both pipelines

| Dataset | Type | Shape | Notes |
|---|---|---|---|
| `td` | float64 | `(td_pts,)` | Time-delay axis in seconds. |
| `theta` | float64 | `(theta_pts,)` | Template-bank sweep axis for `theta_tilde`. |
| `omega` | float64 | `(omega_pts,)` | Template-bank sweep axis for `omega_tilde`. |
| `gamma` | float64 | `(gamma_pts,)` | Template-bank phase axis in radians. |
| `MLz` | float64 | `(td_pts,)` | Lens mass corresponding to each `td`, in seconds. |
| `epsilon_min_grid` | float32 | `(td_pts, theta_pts, omega_pts)` | Minimum mismatch over gamma. |
| `gamma_best_grid` | float32 | `(td_pts, theta_pts, omega_pts)` | Gamma value achieving that minimum. |
| `mismatch` | float32 | `(td_pts, theta_pts, omega_pts, gamma_pts)` | Optional full mismatch tensor, only when `--save_full_mismatch` is used. |

### Pipeline-specific scalar datasets

| Pipeline | Dataset | Type | Shape | Notes |
|---|---|---|---|---|
| `mcz_td` | `mcz` | float64 | `(1,)` | The source-frame chirp mass for this cube, in Msun. |
| `I_td` | `I` | float64 | `(1,)` | The flux-ratio value for this cube. |
| `I_td` | `mcz` | float64 | `(1,)` | The fixed source-frame chirp mass for this run, in Msun. |

### Shared file attributes written by both pipelines

| Field | Type | Notes |
|---|---|---|
| `orientation_tag` | string | Required by downstream readers. |
| `z` | float or NaN | Redshift used for the source metadata, dimensionless. |
| `unit_z` | string | Unit companion for `z`. |
| `I` | float | Source flux ratio written by `write_source_attrs`; dimensionless. In `I_td` cubes this mirrors dataset `I`. |
| `theta_J` | float or NaN | Orientation metadata in `rad`. |
| `phi_J` | float or NaN | Orientation metadata in `rad`. |
| `theta_S` | float or NaN | Orientation metadata in `rad`. |
| `phi_S` | float or NaN | Orientation metadata in `rad`. |
| `match_method` | string | Provenance for the match path used. |
| `minimizer` | string | Provenance for the optimizer used by that match path. |
| `source_param_*` | scalar attrs | Source parameter snapshot. |
| `source_param_unit_*` | string attrs | Units for known source params. |
| `template_param_*` | scalar attrs | Template-generation snapshot copied from the bank file. |
| `template_param_unit_*` | string attrs | Units for known template params. |

Current provenance values written by the mismatch scripts:

- `match_method`: `match`, `optimized_match_bounded`, or `compare_both`
- `minimizer`: `none`, `bounded`, or `bounded_and_discrete`

### Pipeline-specific file attributes

| Pipeline | Field | Type | Notes |
|---|---|---|---|
| `mcz_td` | `mcz_min` | float | Intended mcz grid minimum for the full run, in Msun. |
| `mcz_td` | `mcz_max` | float | Intended mcz grid maximum for the full run, in Msun. |
| `mcz_td` | `mcz_pts` | int | Intended mcz grid size for the full run. |
| `I_td` | `I_min` | float | Intended I-grid minimum for the full run, dimensionless. |
| `I_td` | `I_max` | float | Intended I-grid maximum for the full run, dimensionless. |
| `I_td` | `I_pts` | int | Intended I-grid size for the full run. |
| `I_td` | `mcz_source_msun` | float | Fixed source-frame chirp mass for the run, in Msun. |
| `I_td` | `unit_mcz_source_msun` | string | Unit companion for `mcz_source_msun` (`Msun`). |

### Dataset attributes written

| Dataset | Field | Value / meaning |
|---|---|---|
| `mcz` | `units` | `Msun` when dataset `mcz` is present. |
| `I` | `units` | `dimensionless` when dataset `I` is present. |
| `td` | `units` | `s` |
| `omega` | `units` | `dimensionless` |
| `theta` | `units` | `dimensionless` |
| `gamma` | `units` | `rad` |
| `MLz` | `units` | `s` |
| `epsilon_min_grid` | `axis_order` | `td,theta,omega` |
| `gamma_best_grid` | `axis_order` | `td,theta,omega` |
| `mismatch` | `axis_order` | `td,theta,omega,gamma` when dataset is present. |

## 4) Stage 2: Best-Match Aggregate Files

Both aggregation scripts reduce each Stage 1 cube over `(theta, omega)` and
write one best-match HDF5 file.

### Shared datasets written by both pipelines

| Dataset | Type | Shape | Notes |
|---|---|---|---|
| `td` | float64 | `(td_pts,)` | Time-delay axis. |
| `MLz` | float64 | `(td_pts,)` | Optional; written when available in the input cubes. |
| `epsilon_min` | float32 | `(axis_pts, td_pts)` | Global minimum mismatch over `(theta, omega)`. |
| `omega_best` | float32 | `(axis_pts, td_pts)` | Best template-bank `omega_tilde` coordinate at the minimum. |
| `theta_best` | float32 | `(axis_pts, td_pts)` | Best template-bank `theta_tilde` coordinate at the minimum. |
| `gamma_best` | float32 | `(axis_pts, td_pts)` | Best gamma at the minimum. |

### Pipeline-specific sweep datasets

| Pipeline | Dataset | Type | Shape | Notes |
|---|---|---|---|---|
| `mcz_td` | `mcz` | float64 | `(mcz_pts_out,)` | Output chirp-mass axis used for plotting. |
| `mcz_td` | `expected_mcz` | float64 | `(mcz_pts_expected,)` | Expected full mcz grid used to place rows. |
| `mcz_td` | `missing_mcz` | float64 | `(k,)` | Optional; present only when rows are missing. |
| `I_td` | `I` | float64 | `(I_pts_out,)` | Output flux-ratio axis used for plotting. |
| `I_td` | `mcz` | float64 | `(1,)` | Fixed source-frame chirp mass for the run. |
| `I_td` | `expected_I` | float64 | `(I_pts_expected,)` | Expected full I grid used to place rows. |
| `I_td` | `missing_I` | float64 | `(k,)` | Optional; present only when rows are missing. |

### Shared file attributes written by both pipelines

| Field | Type | Notes |
|---|---|---|
| `orientation_tag` | string | Required by downstream readers. |
| `z` | float or NaN | Redshift metadata written explicitly by the aggregator. |
| `unit_z` | string | Unit companion for `z`. |
| `match_method` | string | Propagated from Stage 1 through `read_source_attrs`. |
| `minimizer` | string | Propagated from Stage 1 through `read_source_attrs`. |
| `theta_J` | float or NaN | Propagated source metadata. |
| `phi_J` | float or NaN | Propagated source metadata. |
| `theta_S` | float or NaN | Propagated source metadata. |
| `phi_S` | float or NaN | Propagated source metadata. |
| `source_param_*` | scalar attrs | Propagated source parameter snapshot. |
| `source_param_unit_*` | string attrs | Units for known source params. |
| `template_param_*` | scalar attrs | Propagated template parameter snapshot. |
| `template_param_unit_*` | string attrs | Units for known template params. |

### Pipeline-specific file attributes

| Pipeline | Field | Type | Notes |
|---|---|---|---|
| `mcz_td` | `I` | float | Fixed source flux ratio for the sweep. |
| `mcz_td` | `missing_mcz_count` | int | Count corresponding to optional dataset `missing_mcz`. |
| `I_td` | `mcz_source_msun` | float | Fixed source-frame chirp mass for the sweep. |
| `I_td` | `unit_mcz_source_msun` | string | Unit companion for `mcz_source_msun`. |
| `I_td` | `missing_I_count` | int | Count corresponding to optional dataset `missing_I`. |

Notes:

- In aggregated `I_td` files written by the current writer, dataset `I` is the
  only flux-ratio sweep axis and no scalar file attribute `I` is written.
- Older aggregated `I_td` files may still contain a scalar file attribute `I`
  copied from the first Stage 1 cube; if present, treat it as historical
  baggage and ignore it in favor of dataset `I`.

### Dataset attributes written

| Dataset | Field | Value / meaning |
|---|---|---|
| `mcz` | `units` | `Msun` |
| `I` | `units` | `dimensionless` |
| `td` | `units` | `s` |
| `MLz` | `units` | `s` when dataset is present. |
| `omega_best` | `units` | `dimensionless` |
| `theta_best` | `units` | `dimensionless` |
| `gamma_best` | `units` | `rad` |
| `epsilon_min` | `axis_order` | `mcz,td` for `mcz_td`; `I,td` for `I_td`. |
| `omega_best` | `axis_order` | `mcz,td` for `mcz_td`; `I,td` for `I_td`. |
| `theta_best` | `axis_order` | `mcz,td` for `mcz_td`; `I,td` for `I_td`. |
| `gamma_best` | `axis_order` | `mcz,td` for `mcz_td`; `I,td` for `I_td`. |

## 5) Practical Checks

Use these checks when validating new files or updating readers:

1. Sweep axes should be read from datasets, not inferred from filenames.
2. `orientation_tag` must be present on all downstream-consumed artifacts.
3. Stage 1 cubes should have `epsilon_min_grid`, `gamma_best_grid`, and the
   expected grid metadata for the relevant pipeline (`mcz_*` or `I_*`).
4. Stage 2 best-match files should have `expected_mcz` or `expected_I`, plus
   the matching `missing_*_count` attribute.
5. If provenance matters, read `match_method` and `minimizer` from file attrs.