# Scripts and Pipelines Guide

This document is the canonical index for everything under `scripts/`.
Use it to answer two questions quickly:

- which workflow folder owns the task you want to run
- which document to open next for step-by-step instructions

For the two production mismatch pipelines, use the comparison table below and then open the matching runbook:

- `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md` for the production `(td, mcz)` pipeline
- `docs/CONTOUR_I_TD_PIPELINE_GUIDE.md` for the production `(td, I)` pipeline

## Scope

Use this guide when you need to:

- build template banks
- compute mismatch cubes and contours in `(td, mcz)` space
- compute mismatch cubes and contours in `(td, I)` space
- run Lindblom distinguishability post-processing
- generate super contours and parameter sweeps
- run one-off or test contours
- use utility scripts for conversion, diagnostics, and plotting

## Directory Overview

| Path | Purpose |
|---|---|
| `scripts/template_banks/` | Build and manage precessing template banks. |
| `scripts/mismatch_mcz_td/` | Compute mismatch cubes, aggregate best match, and plot `(td, mcz)` contours. |
| `scripts/mismatch_I_td/` | Compute mismatch cubes, aggregate best match, and plot `(td, I)` contours at fixed `mcz`. |
| `scripts/lindblom/` | Run Lindblom criterion and related SNR contour workflows. |
| `scripts/contour_mcz_td/` | Contour sweeps over (mcz, td) parameter space. |
| `scripts/contour_omega_theta/` | Single-case contour runs over (omega, theta) parameter space. |
| `scripts/analysis/` | Analysis helpers that are not tied to a single production pipeline. |
| `scripts/utils/` | Conversion, inspection, timing, and plotting helpers. |

## Choose the Right Workflow

| Goal | Start Here |
|---|---|
| Build reusable banks for batch runs | `scripts/template_banks/build_template_banks.py` |
| Run the production `(td, mcz)` mismatch contour pipeline | `scripts/mismatch_mcz_td/compute_mismatch_cubes.py` -> `aggregate_best_match.py` -> `plot_contour_mcz_td_from_best_match.py`, then `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md` |
| Run the production `(td, I)` mismatch contour pipeline | `scripts/mismatch_I_td/compute_mismatch_cubes.py` -> `aggregate_best_match.py` -> `plot_contour_I_td_from_best_match.py`, then `docs/CONTOUR_I_TD_PIPELINE_GUIDE.md` |
| Produce Lindblom distinguishability contours | `scripts/lindblom/compute_lindblom_contours.py` and `scripts/lindblom/complete_lindblom_pipeline.sh` |
| Sweep across `mcz`, `td`, or `I` quickly | `legacy/scripts/contour_mcz_td/contours_mcz.py`, `contours_td.py`, `contours_I.py` (legacy) |
| Generate a one-off contour for debugging | `scripts/contour_omega_theta/v4_indiv_contour_otf.py` |
| Convert or inspect stored outputs | scripts in `scripts/utils/` |

## Production Pipeline Comparison

Use this table when you already know you want the production mismatch workflow and just need to choose the sweep.
High-level pipeline choice belongs here; detailed commands, HDF5 layouts, and naming rules stay in the pipeline runbooks.

| Question | `(td, mcz)` pipeline | `(td, I)` pipeline |
|---|---|---|
| When should I use it? | When you want mismatch trends across chirp mass at a fixed flux ratio. | When you want mismatch trends across flux ratio at a fixed chirp mass. |
| Sweep variable | `mcz` | `I` |
| Fixed parameter | `I` | `mcz` |
| Template bank strategy | Build one bank per `mcz` value in the sweep. | Build one shared bank for the fixed `mcz` value. |
| Natural array-job split | `--mcz_chunk_index/count` | `--I_chunk_index/count` |
| Final aggregate grid | `(mcz, td)` | `(I, td)` |
| Main batch entry point | `batch_scripts/compute_mismatch_mcz_td_cubes.sbatch` | `batch_scripts/compute_mismatch_I_td_cubes.sbatch` |
| Runbook | `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md` | `docs/CONTOUR_I_TD_PIPELINE_GUIDE.md` |

## Common Runtime Notes

- Run Python modules from repository root with `python -m ...` to avoid path issues.
- Keep naming and discovery canonical via helpers in `modules/filenames.py`.
- Follow the repo naming grammar in `README.md` for the rule, token order, and examples.
- Existing production entry points use stable, versionless imports such as `modules.Classes`, `modules.default_params`, `modules.functions`, and `modules.plot_utils`.
- Canonical modules remain the source of truth for production code.
- For new code, prefer the specialized modules directly when they expose the needed functionality, rather than adding new imports to `modules.functions`.
- Versioned compatibility modules (`Classes_v2`, `default_params_v3`, `functions_v3`, `plot_utils_v3`) are wrappers that re-export canonical modules.
- The old monolithic `functions_v3` implementation is split; source-of-truth function logic now lives in `modules.waveform`, `modules.numerics`, `modules.geometry`, and `modules.snr`.
- Matching/mismatch logic has one source of truth in `modules.match_utils`.
- `modules.functions` is a compatibility facade over these modules, and `modules.functions_v3` is a legacy wrapper.
- Production waveform physics remains pinned to the canonical `modules.Classes` implementation (originated from `Classes_v2`); treat `Classes_v3+` as testing-only (not numerically reliable for production outputs).
- Legacy/versioned module implementations are kept under `legacy/modules/`.
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

- `legacy/scripts/template_banks/template_bank_npz.py` (legacy NPZ format utility)

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

## Mismatch Pipeline over `(td, mcz)` (`scripts/mismatch_mcz_td/`)

Use this folder for the production workflow that varies `mcz` across a fixed `I` grid.
For the full stage-by-stage runbook, see `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md`.

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

## Mismatch Pipeline over `(td, I)` (`scripts/mismatch_I_td/`)

Use this folder for the production workflow that varies `I` across a fixed `mcz` value.
Unlike the `(td, mcz)` pipeline, all `I` values reuse the same template bank.
For the full stage-by-stage runbook, see `docs/CONTOUR_I_TD_PIPELINE_GUIDE.md`.

### Core stage scripts

- `compute_mismatch_cubes.py` — Stage 1: compute per-`I` mismatch cubes
- `aggregate_best_match.py` — Stage 2: aggregate across `I` into best-match file
- `plot_contour_I_td_from_best_match.py` — Stage 3: plot contour from best-match output

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

- `v4_indiv_contour_otf.py` — latest on-the-fly contour path
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

- `plot_bestmatch_waveform_overlays.py`
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
- Mismatch cubes (mcz-td): `batch_scripts/compute_mismatch_mcz_td_cubes.sbatch`
- Mismatch cubes (I-td): `batch_scripts/compute_mismatch_I_td_cubes.sbatch`
- Lindblom cubes: `batch_scripts/compute_lindblom_cubes.sbatch`

Shared contour config defaults are centralized in `batch_scripts/_contour_mcz_td_config.sh`.

Deprecated batch scripts (for older workflows) are in `legacy/batch_scripts/`.

## Recommended Collaboration Practices

- Keep new script entry points under an existing workflow folder whenever possible.
- Prefer updating this guide plus the relevant pipeline guide under `docs/` instead of creating new nested README files.
- When adding new outputs, document filename/discovery behavior in `modules/filenames.py`-adjacent docs to prevent naming drift.
- For data layout, LFS, and STOCKYARD workflow see `docs/DATA_LFS.md` and `docs/STOCKYARD.md`.

## Main Pipeline Naming Strategy

Use this section when naming new scripts, outputs, or filename helpers. You can ignore it for routine runs.

Existing main-pipeline scripts use stable, versionless import targets:

- `modules.Classes`
- `modules.default_params`
- `modules.functions`
- `modules.plot_utils`

Treat these as compatibility-stable targets for existing production code.
For new code, prefer the more specific modules below when they expose the functionality directly.

Versioned compatibility modules (`Classes_v2`, `default_params_v3`, `functions_v3`, `plot_utils_v3`) are wrappers that re-export from canonical modules.

`modules.functions` is now a facade over specialized source modules:

- `modules.waveform`
- `modules.numerics`
- `modules.geometry`
- `modules.snr`

Matching and mismatch optimization now use `modules.match_utils` as the single source of truth.

`modules.functions_v3` remains a compatibility wrapper to preserve legacy imports.

Legacy/versioned implementations are kept under `legacy/modules/` and should be imported explicitly from `legacy.modules.*` when needed.

### Repo Naming Style

- Use artifact names for helpers that read, write, parse, or validate one concrete file or schema.
- Use pipeline names for helpers that encode sweep metadata, run directories, aggregated outputs, final contour products, or batch entrypoints.
- Keep shared low-level helpers pipeline-neutral.

Choose the name by scope:

- If the symbol names one concrete file, dataset layout, or schema object, name the artifact.
- If the symbol names a sweep, run configuration, aggregated product, plotting stage, or batch entrypoint, name the pipeline.
- If the symbol is reusable across pipelines and artifacts, keep it neutral and avoid pipeline tokens.

Patterns:

- Concrete artifact: `<qualifier>_<artifact>`
- Artifact helper: `<verb>_<qualifier>_<artifact>`
- Pipeline family or product: `<family>_<pipeline>`
- Pipeline helper or entrypoint: `<verb>_<family>_<pipeline>[_artifact]` or `<verb>_<pipeline>_<concept>`

Order:

- Put local qualifiers such as `mcz`, `I`, and `td` before the artifact noun.
- Put workflow families such as `mismatch`, `best_match`, and `contour` before pipeline tokens such as `mcz_td` and `I_td`.

Examples:

- Artifact names: `create_mcz_mismatch_cube`, `create_I_mismatch_cube`, `mismatch_mcz_cube_filename`
- Pipeline names: `write_mcz_td_grid_attrs`, `write_I_td_grid_attrs`, `best_match_mcz_td_filename`, `compute_mismatch_mcz_td_cubes.sbatch`

When in doubt:

- If the symbol could be reused in another workflow without changing what it names, use an artifact name.
- If reusing it outside the current sweep would make the name misleading, use a pipeline name.
- Prefer existing stable filename-builder names in `modules/filenames.py` unless you are doing a coordinated rename of the whole filename/discovery surface.