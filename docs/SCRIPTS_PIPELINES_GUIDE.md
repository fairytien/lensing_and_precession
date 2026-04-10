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
| `scripts/super_contours/` | Multi-dimensional contour sweeps and super contour generation. |
| `scripts/indiv_contours/` | Single-case and on-the-fly contour runs. |
| `scripts/utils/` | Conversion, inspection, timing, and plotting helpers. |
| `scripts/plot_bestmatch_waveform_overlays.py` | Overlay source and best-match template waveforms from best-match outputs. |

## Choose the Right Workflow

| Goal | Start Here |
|---|---|
| Build reusable banks for batch runs | `scripts/template_banks/build_template_banks.py` |
| Run the production mismatch contour pipeline | `scripts/mismatch_mcz_td/compute_mismatch_cubes.py` -> `aggregate_best_match.py` -> `plot_contour_mcz_td_from_best_match.py` |
| Produce Lindblom distinguishability contours | `scripts/lindblom/compute_lindblom_contours.py` and `scripts/lindblom/complete_lindblom_pipeline.sh` |
| Sweep across `mcz`, `td`, or `I` quickly | `scripts/super_contours/contours_mcz.py`, `contours_td.py`, `contours_I.py` |
| Generate a one-off contour for debugging | `scripts/indiv_contours/v4_indiv_contour_otf.py` |
| Convert or inspect stored outputs | scripts in `scripts/utils/` |

## Common Runtime Notes

- Run Python modules from repository root with `python -m ...` to avoid path issues.
- Keep naming and discovery canonical via helpers in `modules/filenames.py`.
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
  --z 1e-8 \
  --bank_dir ./data/template_banks
```

## Mismatch Pipeline (`scripts/mismatch_mcz_td/`)

### Core stage scripts

- `compute_mismatch_cubes.py`
- `aggregate_best_match.py`
- `plot_contour_mcz_td_from_best_match.py`

### Visualization and inspection helpers

- `visualize_mismatch_cube.py`
- `visualize_mcz_sweep_at_td.py`
- `plot_omega_theta_from_cube.py`

Quick helper usage examples:

```bash
python -m scripts.mismatch_mcz_td.visualize_mismatch_cube \
  --input_path ./data/mismatch/mismatch_cubes/<cube>.h5 \
  --output_dir ./figures/mismatch_cubes \
  --gif

python -m scripts.mismatch_mcz_td.visualize_mcz_sweep_at_td \
  --input_dir ./data/mismatch/mismatch_cubes \
  --output_dir ./figures/mismatch_cubes_mcz_sweep \
  --td_ms 40 \
  --gif
```

### Maintenance helpers

- `convert_best_match_redshift.py`

Legacy filename migration now lives in `scripts/utils/rename_legacy_filenames.py`.

### Recommended stage order

1. Build banks (template bank stage).
2. Compute mismatch cubes.
3. Aggregate best-match results.
4. Plot contours.

### Typical usage

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
  --bank_dir ./data/template_banks \
  --run_dir ./data/mismatch

python -m scripts.mismatch_mcz_td.aggregate_best_match \
  --run_dir ./data/mismatch \
  --I 0.5 \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 90 \
  --orientation_tag Taman_edgeon \
  --z 1e-8

python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match \
  --input_path ./data/mismatch_I0p5_z1e-08_mcz10-90_td20-70_Taman_edgeon/best_match/<best_match_file>.h5 \
  --output_dir ./figures/mismatch
```

For stage-specific argument details and HDF5 schema notes, see `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md`.

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

## Super Contours (`scripts/super_contours/`)

### Main scripts

- `super_contour_cli.py`
- `super_contour_L_NP.py`
- `contour_L_NP_mcz_td.py`
- `contour_L_RP_mcz_td_otf.py`

### Sweep scripts

- `contours_mcz.py`
- `contours_td.py`
- `contours_I.py`

### Versioned bank-based scripts

- `v3_super_contour_from_bank.py`
- `v3_contours_mcz_from_bank.py`

Use this folder when you want broad parameter sweeps rather than strict stage-by-stage production runs.

## Individual and OTF Contours (`scripts/indiv_contours/`)

### Main scripts

- `v4_indiv_contour_otf.py` (latest OTF path)
- `v3_indiv_contour_otf.py`
- `v3_indiv_contour_otf_v2prec.py`
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
- Mismatch cubes: `batch_scripts/compute_mismatch_cubes.sbatch`
- Lindblom cubes: `batch_scripts/compute_lindblom_cubes.sbatch`

Shared contour config defaults are centralized in `batch_scripts/_contour_mcz_td_config.sh`.

## Recommended Collaboration Practices

- Keep new script entry points under an existing workflow folder whenever possible.
- Prefer adding examples in this guide and workflow docs under `docs/` instead of creating new nested README files.
- When adding new outputs, document filename/discovery behavior in `modules/filenames.py`-adjacent docs to prevent naming drift.