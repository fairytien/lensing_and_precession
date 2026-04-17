# Scripts and Pipelines Guide

This document is the canonical reference for everything under `scripts/`.
It replaces the previous per-folder README files and groups usage by workflow goals.

## Scope

Use this guide when you need to:

- build template banks
- compute mismatch cubes and contours in `(td, mcz)` space
- run Lindblom distinguishability post-processing
- generate super contours and parameter sweeps
- run one-off or test contours
- use utility scripts for conversion, diagnostics, and plotting

For full details of the modular contour pipeline, see `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md`.

## Directory Overview

| Path | Purpose |
|---|---|
| `scripts/template_banks/` | Build and manage precessing template banks. |
| `scripts/mismatch_mcz_td/` | Compute mismatch cubes, aggregate best match, and plot `(td, mcz)` contours. |
| `scripts/lindblom/` | Run Lindblom criterion and related SNR contour workflows. |
| `scripts/contour_mcz_td/` | Contour sweeps over (mcz, td) parameter space. |
| `scripts/contour_omega_theta/` | Single-case contour runs over (omega, theta) parameter space. |
| `scripts/utils/` | Conversion, inspection, timing, and plotting helpers. |
| `scripts/plot_bestmatch_waveform_overlays.py` | Overlay source and best-match template waveforms from best-match outputs. |

## Choose the Right Workflow

| Goal | Start Here |
|---|---|
| Build reusable banks for batch runs | `scripts/template_banks/build_template_banks.py` |
| Run the production mismatch contour pipeline | `scripts/mismatch_mcz_td/compute_mismatch_cubes.py` -> `aggregate_best_match.py` -> `plot_contour_mcz_td_from_best_match.py` |
| Produce Lindblom distinguishability contours | `scripts/lindblom/compute_lindblom_contours.py` and `scripts/lindblom/complete_lindblom_pipeline.sh` |
| Sweep across `mcz`, `td`, or `I` quickly | `legacy/scripts/contour_mcz_td/contours_mcz.py`, `contours_td.py`, `contours_I.py` (legacy) |
| Generate a one-off contour for debugging | `scripts/contour_omega_theta/v4_indiv_contour_otf.py` |
| Convert or inspect stored outputs | scripts in `scripts/utils/` |

## Common Runtime Notes

- Run Python modules from repository root with `python -m ...` to avoid path issues.
- Keep naming and discovery canonical via helpers in `modules/filenames.py`.
- Main-pipeline scripts should import canonical modules: `modules.Classes`, `modules.default_params`, `modules.functions`, and `modules.plot_utils`.
- Canonical modules are the source of truth for production code.
- Versioned compatibility modules (`Classes_v2`, `default_params_v3`, `functions_v3`, `plot_utils_v3`) are wrappers that re-export canonical modules.
- The old monolithic `functions_v3` implementation is split; source-of-truth function logic now lives in `modules.waveform`, `modules.numerics`, `modules.geometry`, and `modules.snr`.
- Matching/mismatch logic has one source of truth in `modules.match_utils`.
- `modules.functions` is a compatibility facade over these modules, and `modules.functions_v3` is a legacy wrapper.
- Production waveform physics remains pinned to the canonical `modules.Classes` implementation (originated from `Classes_v2`); treat `Classes_v3+` as testing-only (not numerically reliable for production outputs).
- Legacy/versioned module implementations are now kept under `legacy/modules/`.
- Legacy scripts under `legacy/scripts/` should import legacy helpers from `legacy.modules.*`.
- If historical files use legacy names, migrate them once with:

```bash
python -m scripts.utils.rename_legacy_filenames
python -m scripts.utils.rename_legacy_filenames --apply
```

- Redshift-aware scripts use detector-frame mass scaling:

  `mcz_det = mcz * (1 + z)`

## Template Banks (`scripts/template_banks/`)

### Primary script

- `build_template_banks.py`

### Secondary/legacy utility

- `template_bank_npz.py` (legacy NPZ format utility)

### Typical usage

```bash
python -m scripts.template_banks.build_template_banks \
  --mcz_min 10 --mcz_max 90 --mcz_pts 81 \
  --omega_min 0 --omega_max 6 --omega_pts 61 \
  --theta_min 0 --theta_max 15 --theta_pts 151 \
  --gamma_pts 51 \
  --orient_preset Taman_edgeon \
  --z 1 \
  --bank_dir ./data/template_banks
```

## Mismatch Pipeline (`scripts/mismatch_mcz_td/`)

### Core stage scripts

- `compute_mismatch_cubes.py` — Stage 1: compute per-mcz mismatch cubes
- `aggregate_best_match.py` — Stage 2: aggregate across mcz into best-match file
- `plot_contour_mcz_td_from_best_match.py` — Stage 3: plot contour from best-match

### Visualization helpers

- `visualize_mismatch_cube.py` — animate mismatch across td slices
- `visualize_mcz_sweep_at_td.py` — animate mismatch across mcz at fixed td
- `plot_omega_theta_from_cube.py` — plot omega-theta slices

```bash
python -m scripts.mismatch_mcz_td.visualize_mismatch_cube \
  --input_path ./data/mismatch/mismatch_cubes/<cube>.h5 \
  --output_dir ./figures/mismatch_cubes --gif

python -m scripts.mismatch_mcz_td.visualize_mcz_sweep_at_td \
  --input_dir ./data/mismatch/mismatch_cubes \
  --output_dir ./figures/mismatch_cubes_mcz_sweep --td_ms 40 --gif
```

### Maintenance helpers

- `convert_best_match_redshift.py`

> **For detailed stage examples, HDF5 schema, and batch configuration:** see [CONTOUR_TD_MCZ_PIPELINE_GUIDE.md](CONTOUR_TD_MCZ_PIPELINE_GUIDE.md).

## Lindblom Pipeline (`scripts/lindblom/`)

### Core scripts

- `compute_lindblom_contours.py`
- `aggregate_lindblom_best_match.py`
- `create_contour_mcz_td_from_lindblom.py`
- `create_contour_mcz_td_from_snr.py`

### Related scripts

- `compute_source_snr_contour.py`
- `create_contour_mcz_td_from_source_snr.py`
- `compute_lindblom_from_source_snr.py`
- `create_contour_mcz_td_from_lindblom_source_snr.py`
- `compute_lindblom_from_pickle.py`
- `create_contour_from_lindblom_pickle.py`

### Orchestration helpers

- `check_lindblom_progress.sh`
- `complete_lindblom_pipeline.sh`

### Typical usage

```bash
# Usually after batch cube generation completes
bash scripts/lindblom/check_lindblom_progress.sh
bash scripts/lindblom/complete_lindblom_pipeline.sh
```

## Contours over (mcz, td) (`scripts/contour_mcz_td/`)

### Active scripts

- `contour_L_NP_mcz_td.py` — main active script

### Legacy scripts (`legacy/scripts/contour_mcz_td/`)

- `super_contour_cli.py`
- `super_contour_L_NP.py`
- `contour_L_RP_mcz_td_otf.py`
- `contours_mcz.py`, `contours_td.py`, `contours_I.py` — parameter sweep scripts
- `v3_super_contour_from_bank.py`, `v3_contours_mcz_from_bank.py` — bank-based scripts

Use this folder when you want broad parameter sweeps rather than strict stage-by-stage production runs.

## Contours over (omega, theta) (`scripts/contour_omega_theta/`)

### Active scripts

- `v4_indiv_contour_otf.py` — latest OTF path (uses `Classes_v4` for solve_ivp)
- `v3_indiv_contour_otf.py`
- `v3_indiv_contour_otf_v2prec.py`

### Legacy scripts (`legacy/scripts/contour_omega_theta/`)

- `v2_indiv_contour_otf.py`
- `v3_indiv_contour_from_bank.py`
- `indiv_contour.py`
- `indiv_contour_test.py`

Use this folder for one-off experiments, spot checks, and debugging a specific source/template configuration without running full cube generation.

## Utilities (`scripts/utils/`)

### Conversion and compression

- `pickle_to_hdf5.py`
- `compress_pickle.py`

### Plotting and comparison

- `plot_contour_from_dataset.py` (supports both pickle and hdf5 contour inputs)
- `plot_cycles_and_extrema_mcz.py`
- `compare_contours.py`
- `rename_legacy_filenames.py`

### Diagnostics and metadata inspection

- `inspect_hdf5_metadata.py`
- `estimate_cpu_time.py`
- `plot_dist_vs_z.py`
- `plot_dist_vs_z_broken_axis.py`

## Batch Scripts Mapping

Production cluster jobs live in `batch_scripts/`.

- Template banks: `batch_scripts/build_template_banks.sbatch`
- Mismatch cubes (mcz-td): `batch_scripts/compute_mismatch_mcz_cubes.sbatch`
- Lindblom cubes: `batch_scripts/compute_lindblom_cubes.sbatch`

Shared contour config defaults are centralized in `batch_scripts/_contour_mcz_td_config.sh`.

Deprecated batch scripts (for older workflows) are in `legacy/batch_scripts/`.

## Recommended Collaboration Practices

- Keep new script entry points under an existing workflow folder whenever possible.
- Prefer adding examples in this guide and workflow docs under `docs/` instead of creating new nested README files.
- When adding new outputs, document filename/discovery behavior in `modules/filenames.py`-adjacent docs to prevent naming drift.
- For data layout, LFS, and STOCKYARD workflow see `docs/DATA_LFS.md` and `docs/STOCKYARD.md`.