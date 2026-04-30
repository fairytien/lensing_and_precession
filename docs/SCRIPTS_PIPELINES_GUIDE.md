# Scripts and Pipelines Guide

This document is the routing guide for everything under `scripts/`.
Use it to identify the workflow folder that owns a task and the next runbook or entry point to open.

## Start Here

| If you want to... | Open... |
|---|---|
| Build reusable template banks | `scripts/template_banks/build_template_banks.py` or `batch_scripts/build_template_banks.sbatch` |
| Run the production `(td, mcz)` mismatch pipeline | [CONTOUR_MCZ_TD_PIPELINE_GUIDE.md](CONTOUR_MCZ_TD_PIPELINE_GUIDE.md) |
| Run the production `(td, I)` mismatch pipeline | [CONTOUR_I_TD_PIPELINE_GUIDE.md](CONTOUR_I_TD_PIPELINE_GUIDE.md) |
| Produce Lindblom distinguishability contours | `scripts/lindblom/` |
| Run broad parameter sweeps in `(td, mcz)` | `scripts/contour_mcz_td/` |
| Generate a one-off contour for debugging | `scripts/contour_omega_theta/v4_indiv_contour_otf.py` |
| Convert or inspect stored outputs | `scripts/utils/` |

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

Use the following order rules consistently throughout the docs.

- Pipeline and workflow identifiers use `mcz_td` and `I_td`. Use these in folder names, script names, batch scripts, helper names, and runbook filenames because those tokens already match the repo's canonical code and filename surface.
- Human-facing contour planes use `(td, mcz)` and `(td, I)`. Use these in titles, prose, tables, and plot descriptions because they describe the source-waveform parameter plane as readers encounter it.
- HDF5, array, and aggregate-grid order use `(mcz, td)` and `(I, td)`. Use these only when describing dataset shapes, matrix layouts, or stored outputs because they should match stored axis order rather than plot wording.
- Do not introduce new pipeline tokens such as `td_mcz` or `td_I` in new docs, helpers, or filenames because extra aliases make links, filenames, and helper names harder to scan and maintain.

## Template Banks (`scripts/template_banks/`)

- Build and manage precessing template banks.
- Start with `build_template_banks.py` for local runs or `batch_scripts/build_template_banks.sbatch` for cluster runs.
- For production bank settings tied to the mismatch pipelines, use the matching pipeline runbook instead of this guide.
- Legacy NPZ tooling remains under `legacy/scripts/template_banks/template_bank_npz.py`.

## Mismatch Pipeline over `(td, mcz)` (`scripts/mismatch_mcz_td/`)

- Workflow token: `mcz_td`.
- Use this folder for the production workflow that varies `(td, mcz)` at fixed `I`.
- For all stage-by-stage commands, batch defaults, outputs, and naming, use [CONTOUR_MCZ_TD_PIPELINE_GUIDE.md](CONTOUR_MCZ_TD_PIPELINE_GUIDE.md).
- Folder-local helpers for existing outputs:
  - `visualize_mismatch_cube.py`
  - `visualize_mcz_sweep_at_td.py`
  - `plot_omega_theta_from_cube.py`
  - `convert_best_match_redshift.py`

## Mismatch Pipeline over `(td, I)` (`scripts/mismatch_I_td/`)

- Workflow token: `I_td`.
- Use this folder for the production workflow that varies `(td, I)` at fixed `mcz`.
- For all stage-by-stage commands, batch defaults, outputs, and naming, use [CONTOUR_I_TD_PIPELINE_GUIDE.md](CONTOUR_I_TD_PIPELINE_GUIDE.md).

## Analysis (`scripts/analysis/`)

- Use this folder for analysis helpers that are not tied to a single production pipeline.
- Prefer the more specific workflow folders when a script clearly belongs to `mismatch_*`, `lindblom`, `contour_*`, or `utils`.

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

## Contours over `(td, mcz)` (`scripts/contour_mcz_td/`)

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
- Mismatch cubes (`mcz_td`): `batch_scripts/compute_mismatch_mcz_td_cubes.sbatch`
- Mismatch cubes (`I_td`): `batch_scripts/compute_mismatch_I_td_cubes.sbatch`
- Lindblom cubes: `batch_scripts/compute_lindblom_cubes.sbatch`

Shared mismatch-pipeline config defaults live in:

- `batch_scripts/_contour_mcz_td_config.sh` for `mcz_td`
- `batch_scripts/_contour_I_td_config.sh` for `I_td`

Deprecated batch scripts (for older workflows) are in `legacy/batch_scripts/`.

## Common Runtime Notes

- Run Python modules from repository root with `python -m ...`.
- Keep naming and discovery canonical via helpers in `modules/filenames.py`.
- Follow the repo naming grammar in `README.md` for the rule, token order, and examples.
- Canonical modules remain the source of truth for production code.
- In new code, prefer specialized modules over adding new imports to `modules.functions`.
- `modules.functions` and `modules.functions_v3` are compatibility facades, not new dependencies.
- Matching and mismatch logic has one source of truth in `modules.match_utils`.
- Production waveform physics remains pinned to `modules.Classes`; treat `Classes_v3+` as testing-only.
- Legacy/versioned implementations live under `legacy/modules/`, and legacy scripts should import from `legacy.modules.*`.
- If historical files use legacy names, migrate them with `python -m scripts.utils.rename_legacy_filenames` and then `python -m scripts.utils.rename_legacy_filenames --apply`.
- Redshift-aware scripts use detector-frame mass scaling: `mcz_det = mcz * (1 + z)`.

## Figure Typography

Publication-facing figures should follow APS/REVTeX math typography.

- Set physical variables in italic and descriptive labels in upright roman.
- In this repo, italic variables include `$I$`, `$z$`, `$\theta$`, `$\omega$`, `$\mathcal{M}$`, and `$t$` in `$\Delta t_{\mathrm{d}}$`. Upright descriptors include `$\mathrm{s}$`, `$\mathrm{t}$`, `$\mathrm{NP}$`, `$\mathrm{RP}$`, `$\mathrm{P}$`, `$\mathrm{L}$`, `$\mathrm{UL}$`, `$\mathrm{d}$`, and units such as `$\mathrm{Hz}$` and `$\mathrm{ms}$`.
- Keep true running indices italic. In this repo, most figure subscripts are descriptive labels, so prefer forms such as `$\Phi_{\mathrm{s}} - \Phi_{\mathrm{t}}$`, `$\mathcal{M}_{\mathrm{s}}$`, `$\gamma_{\mathrm{P}}$`, `$\theta_{\mathrm{S}}$`, `$\phi_{\mathrm{J}}$`, `$\Delta t_{\mathrm{d}}$`, and `$\epsilon(\~h_{\mathrm{L}}, \~h_{\mathrm{P}})$`.
- In matplotlib mathtext, prefer `\mathrm{...}` over legacy `\rm`. `modules.plot_utils.apply_physics_paper_style()` sets defaults only; plotting scripts must still format labels, titles, legends, and colorbar labels explicitly, and touched figures should be normalized to this convention.

## Contributor Notes

### Maintaining This Guide

- Keep new script entry points under an existing workflow folder whenever possible.
- Prefer updating this guide plus the relevant pipeline guide under `docs/` instead of creating new nested README files.
- When adding new outputs, document filename and discovery behavior near `modules/filenames.py` to prevent naming drift.
- For data layout, LFS, and STOCKYARD workflow, see `docs/DATA_LFS.md` and `docs/STOCKYARD.md`.

### Naming Strategy

Use this section when naming new scripts, outputs, or filename helpers. Routine runs can ignore it.
For import and source-of-truth module rules, see `Common Runtime Notes` above.

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

### Output Filename Order

For canonical run directories and output filenames, list fixed-value tokens before swept or range tokens.

- Keep `z` immediately before `mcz` when both appear.
- Put the `td` range token after the other source-parameter tokens rather than mixing it into the fixed-value block.
- When filenames also include template-grid tokens such as `omega`, `theta`, and `gamma`, place those grid tokens after the `td` range token.
- Use the same token order in filename builders and discovery helpers in `modules/filenames.py`.

In the production mismatch pipelines, this means:

- `mcz_td`: fixed `I`, then fixed `z`, then swept `mcz`, then `td`.
- `I_td`: fixed `z`, then fixed `mcz`, then swept `I`, then `td`.
- For per-sweep mismatch cube filenames, both pipelines use the same token order: `z`, then `mcz`, then `I`, then `td`.

Examples:

- `mcz_td` run directory: `mismatch_I0p5_z1_mcz5-45_td20-70_Taman_faceon`
- `I_td` run directory: `mismatch_z1_mcz15_I0p1-0p9_td20-70_Taman_edgeon`

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