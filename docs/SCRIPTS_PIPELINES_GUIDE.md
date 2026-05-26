# Scripts and Pipelines Guide

This document is the routing guide for everything under [`scripts/`](../scripts/).
Use it to identify the workflow folder that owns a task and the next runbook or entry point to open.

## Start Here

| If you want to... | Open... |
|---|---|
| Build reusable template banks | [`scripts/template_banks/build_template_banks.py`](../scripts/template_banks/build_template_banks.py) or [`batch_scripts/build_template_banks.sbatch`](../batch_scripts/build_template_banks.sbatch) |
| Run the production `(td, mcz)` mismatch pipeline | [CONTOUR_MCZ_TD_PIPELINE_GUIDE.md](CONTOUR_MCZ_TD_PIPELINE_GUIDE.md) |
| Run the production `(td, I)` mismatch pipeline | [CONTOUR_I_TD_PIPELINE_GUIDE.md](CONTOUR_I_TD_PIPELINE_GUIDE.md) |
| Produce Lindblom distinguishability contours | [`scripts/lindblom/`](../scripts/lindblom/) |
| Run broad parameter sweeps in `(td, mcz)` | [`scripts/contour_mcz_td/`](../scripts/contour_mcz_td/) |
| Generate a one-off contour for debugging | [`scripts/contour_omega_theta/v4_indiv_contour_otf.py`](../scripts/contour_omega_theta/v4_indiv_contour_otf.py) |
| Convert or inspect stored outputs | [`scripts/utils/`](../scripts/utils/) |
| Migrate legacy filenames or file formats | [`scripts/migrate/`](../scripts/migrate/) |

## Production Pipeline Comparison

Use this table only to choose between the two production mismatch runbooks.
Detailed commands, outputs, and naming rules stay in the pipeline docs.
In both pipelines, source waveforms are evaluated over a two-parameter grid. The rows below separate the full source-waveform grid from the parameter that distinguishes the two workflows.

| Question | `(td, mcz)` pipeline | `(td, I)` pipeline |
|---|---|---|
| When should I use it? | When you want mismatch trends across chirp mass at a fixed flux ratio. | When you want mismatch trends across flux ratio at a fixed chirp mass. |
| Source-waveform grid | `(td, mcz)` | `(td, I)` |
| Workflow-distinguishing sweep | `mcz` | `I` |
| Fixed source parameter | `I` | `mcz` |
| Template bank strategy | Build one bank per `mcz` value in the sweep. | Build one shared bank for the fixed `mcz` value. |
| Natural array-job split | `--mcz_chunk_index/count` | `--I_chunk_index/count` |
| Final aggregate grid | `(mcz, td)` | `(I, td)` |
| Runbook | [CONTOUR_MCZ_TD_PIPELINE_GUIDE.md](CONTOUR_MCZ_TD_PIPELINE_GUIDE.md) | [CONTOUR_I_TD_PIPELINE_GUIDE.md](CONTOUR_I_TD_PIPELINE_GUIDE.md) |

## Parameter Order Convention

See [AGENTS.md — Parameter Order Convention](../AGENTS.md#parameter-order-convention) for the full rules on pipeline identifiers, human-facing contour plane notation, and HDF5/array axis order.

## Template Banks ([`scripts/template_banks/`](../scripts/template_banks/))

- Build and manage precessing template banks.
- Start with [`build_template_banks.py`](../scripts/template_banks/build_template_banks.py) for local runs or [`batch_scripts/build_template_banks.sbatch`](../batch_scripts/build_template_banks.sbatch) for cluster runs.
- For production bank settings tied to the mismatch pipelines, use the matching pipeline runbook instead of this guide.
- Legacy NPZ tooling remains under [`legacy/scripts/template_banks/template_bank_npz.py`](../legacy/scripts/template_banks/template_bank_npz.py).

## Mismatch Pipeline over `(td, mcz)` ([`scripts/mismatch_mcz_td/`](../scripts/mismatch_mcz_td/))

- Workflow token: `mcz_td`.
- Use this folder for the production workflow that varies `(td, mcz)` at fixed `I`.
- For all stage-by-stage commands, batch defaults, outputs, and naming, use [CONTOUR_MCZ_TD_PIPELINE_GUIDE.md](CONTOUR_MCZ_TD_PIPELINE_GUIDE.md).
- Folder-local helpers for existing outputs:
  - [`compare_contours_mcz_td.py`](../scripts/mismatch_mcz_td/compare_contours_mcz_td.py)
  - [`visualize_mcz_sweep_at_td.py`](../scripts/mismatch_mcz_td/visualize_mcz_sweep_at_td.py)

## Mismatch Pipeline over `(td, I)` ([`scripts/mismatch_I_td/`](../scripts/mismatch_I_td/))

- Workflow token: `I_td`.
- Use this folder for the production workflow that varies `(td, I)` at fixed `mcz`.
- For all stage-by-stage commands, batch defaults, outputs, and naming, use [CONTOUR_I_TD_PIPELINE_GUIDE.md](CONTOUR_I_TD_PIPELINE_GUIDE.md).

## Analysis ([`scripts/analysis/`](../scripts/analysis/))

- Use this folder for analysis helpers that are not tied to a single production pipeline.
- Prefer the more specific workflow folders when a script clearly belongs to `mismatch_*`, `lindblom`, `contour_*`, or `utils`.
- [`plot_dist_vs_z.py`](../scripts/analysis/plot_dist_vs_z.py) — luminosity distance vs. redshift reference figure
- [`plot_dist_vs_z_broken_axis.py`](../scripts/analysis/plot_dist_vs_z_broken_axis.py) — same with broken y-axis

## Lindblom Pipeline ([`scripts/lindblom/`](../scripts/lindblom/))

### Core scripts

- [`compute_lindblom_contours.py`](../scripts/lindblom/compute_lindblom_contours.py)
- [`aggregate_lindblom_best_match.py`](../scripts/lindblom/aggregate_lindblom_best_match.py)
- [`create_contour_mcz_td_from_lindblom.py`](../scripts/lindblom/create_contour_mcz_td_from_lindblom.py)
- [`create_contour_mcz_td_from_snr.py`](../scripts/lindblom/create_contour_mcz_td_from_snr.py)

### Related scripts

- [`compute_source_snr_contour.py`](../scripts/lindblom/compute_source_snr_contour.py)
- [`create_contour_mcz_td_from_source_snr.py`](../scripts/lindblom/create_contour_mcz_td_from_source_snr.py)
- [`compute_lindblom_from_source_snr.py`](../scripts/lindblom/compute_lindblom_from_source_snr.py)
- [`create_contour_mcz_td_from_lindblom_source_snr.py`](../scripts/lindblom/create_contour_mcz_td_from_lindblom_source_snr.py)
- [`compute_lindblom_from_pickle.py`](../scripts/lindblom/compute_lindblom_from_pickle.py)
- [`create_contour_from_lindblom_pickle.py`](../scripts/lindblom/create_contour_from_lindblom_pickle.py)

### Orchestration helpers

- [`check_lindblom_progress.sh`](../scripts/lindblom/check_lindblom_progress.sh)
- [`complete_lindblom_pipeline.sh`](../scripts/lindblom/complete_lindblom_pipeline.sh)

## Contours over `(td, mcz)` ([`scripts/contour_mcz_td/`](../scripts/contour_mcz_td/))

### Active scripts

- [`contour_L_NP_mcz_td.py`](../scripts/contour_mcz_td/contour_L_NP_mcz_td.py) — main active script

### Legacy scripts ([`legacy/scripts/contour_mcz_td/`](../legacy/scripts/contour_mcz_td/))

- `super_contour_cli.py`
- `super_contour_L_NP.py`
- `contour_L_RP_mcz_td_otf.py`
- `contours_mcz.py`, `contours_td.py`, `contours_I.py` — parameter sweep scripts
- `v3_super_contour_from_bank.py`, `v3_contours_mcz_from_bank.py` — bank-based scripts

Use this folder when you want broad parameter sweeps rather than strict stage-by-stage production runs.

## Contours over (omega, theta) ([`scripts/contour_omega_theta/`](../scripts/contour_omega_theta/))

### Active scripts

- [`v4_indiv_contour_otf.py`](../scripts/contour_omega_theta/v4_indiv_contour_otf.py) — latest on-the-fly contour path
- [`v3_indiv_contour_otf.py`](../scripts/contour_omega_theta/v3_indiv_contour_otf.py)
- [`v3_indiv_contour_otf_v2prec.py`](../scripts/contour_omega_theta/v3_indiv_contour_otf_v2prec.py)
- [`plot_bestmatch_waveform_overlays.py`](../scripts/contour_omega_theta/plot_bestmatch_waveform_overlays.py) — waveform overlay figures from contour pickle outputs

### Legacy scripts ([`legacy/scripts/contour_omega_theta/`](../legacy/scripts/contour_omega_theta/))

- `v2_indiv_contour_otf.py`
- `v3_indiv_contour_from_bank.py`
- `indiv_contour.py`
- `indiv_contour_test.py`

Use this folder for one-off experiments, spot checks, and debugging a specific source/template configuration without running full cube generation.

## Utilities ([`scripts/utils/`](../scripts/utils/))

### Conversion

- [`convert_best_match_mcz_td_redshift.py`](../scripts/utils/convert_best_match_mcz_td_redshift.py) — remap a `mcz_td` best-match HDF5 file from one redshift to another

### Plotting and comparison

- [`visualize_mismatch_cube.py`](../scripts/utils/visualize_mismatch_cube.py) (single-cube td sweep; `mcz_td` or `I_td` cubes)
- [`_cube_viz.py`](../scripts/utils/_cube_viz.py) — private helpers for cube movies/sliders (used by `visualize_mismatch_cube` and `visualize_mcz_sweep_at_td`)
- [`_best_match_plot.py`](../scripts/utils/_best_match_plot.py) — private shared renderer and `VARIABLE_MAPPING` for both `plot_contour_*_from_best_match.py` entry points
- [`plot_contour_from_dataset.py`](../scripts/utils/plot_contour_from_dataset.py) (supports pickle and HDF5 contour inputs, including mismatch cube schema with `--td_ms`)
- [`plot_cycles_and_extrema.py`](../scripts/utils/plot_cycles_and_extrema.py)
- [`compare_contours.py`](../scripts/utils/compare_contours.py)

### Diagnostics and metadata inspection

- [`inspect_hdf5_metadata.py`](../scripts/utils/inspect_hdf5_metadata.py)
- [`estimate_cpu_time.py`](../scripts/utils/estimate_cpu_time.py)


## Legacy Migration ([`scripts/migrate/`](../scripts/migrate/))

- [`compress_pickle.py`](../scripts/migrate/compress_pickle.py) — compress legacy pickle files
- [`pickle_to_hdf5.py`](../scripts/migrate/pickle_to_hdf5.py) — convert legacy pickle outputs to HDF5
- [`rename_legacy_filenames.py`](../scripts/migrate/rename_legacy_filenames.py) — rename legacy mismatch-pipeline filenames to the canonical naming convention

## Batch Scripts Mapping

Production cluster jobs live in [`batch_scripts/`](../batch_scripts/).

- Template banks: [`batch_scripts/build_template_banks.sbatch`](../batch_scripts/build_template_banks.sbatch)
- Mismatch cubes (`mcz_td`): [`batch_scripts/compute_mismatch_mcz_td_cubes.sbatch`](../batch_scripts/compute_mismatch_mcz_td_cubes.sbatch)
- Mismatch cubes (`I_td`): [`batch_scripts/compute_mismatch_I_td_cubes.sbatch`](../batch_scripts/compute_mismatch_I_td_cubes.sbatch)
- Lindblom cubes: [`batch_scripts/compute_lindblom_cubes.sbatch`](../batch_scripts/compute_lindblom_cubes.sbatch)

Shared mismatch-pipeline config defaults live in:

- [`batch_scripts/_contour_mcz_td_config.sh`](../batch_scripts/_contour_mcz_td_config.sh) for `mcz_td`
- [`batch_scripts/_contour_I_td_config.sh`](../batch_scripts/_contour_I_td_config.sh) for `I_td`

Deprecated batch scripts (for older workflows) are in [`legacy/batch_scripts/`](../legacy/batch_scripts/).

## Folder Ownership Criteria

Use this ranking to decide where a new script belongs.

1. **`scripts/mismatch_mcz_td/`** or **`scripts/mismatch_I_td/`** — the four production stages (`compute_mismatch_cubes`, `aggregate_best_match`, `plot_contour_*_from_best_match`, `compare_contours_*_td`) and any post-production helper that is inherently pipeline-specific (i.e., depends on which axis is swept).
2. **`scripts/contour_mcz_td/`** and **`scripts/contour_omega_theta/`** — broad parameter sweeps and one-off / debugging contour generators. Do **not** place paper-figure scripts that consume `best_match_*.h5` here; those belong with the pipeline that produced them.
3. **`scripts/lindblom/`** — Lindblom workflow.
4. **`scripts/template_banks/`** — bank builders.
5. **`scripts/analysis/`** — cross-pipeline physics analyses (e.g., `modality_nlens`, `plot_bestfit_prec_params`).
6. **`scripts/utils/`** — generic helpers and conversion scripts that operate across pipelines or on a common artifact schema (cube, generic dataset, HDF5 metadata).
7. **`scripts/migrate/`** — one-off tools for legacy-filename migration and format conversion (`pickle → HDF5`, redshift rescaling history). Run once; not part of the active research workflow.

`_underscore_helpers.py` private modules under a folder are imported only within that folder's siblings. For `scripts/utils/_*.py`, the consumers may be sibling scripts in any `scripts/` subfolder.

## Contributor Notes

- Keep new script entry points under an existing workflow folder whenever possible.
- Prefer updating this guide plus the relevant pipeline guide under [`docs/`](../docs/) instead of creating new nested README files.
- When adding new outputs, document filename and discovery behavior near [`modules/filenames.py`](../modules/filenames.py) to prevent naming drift.
- For data layout, LFS, and STOCKYARD workflow, see [DATA_LFS.md](DATA_LFS.md) and [STOCKYARD.md](STOCKYARD.md).
- For runtime rules, naming conventions, output filename order, parameter order notation, and figure typography, see [AGENTS.md](../AGENTS.md).
