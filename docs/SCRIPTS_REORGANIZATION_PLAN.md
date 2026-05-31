---
name: scripts reorganization
overview: Reorganize `scripts/` to fix inconsistent placement and reduce duplication across the `mcz_td` and `I_td` pipelines, applying moderate consolidation (private shared helpers, separate per-pipeline entry points) with split-by-generality routing (generalizable code → `scripts/utils/`, pipeline-specific code → `scripts/mismatch_*_td/`).
todos:
  - id: criteria_doc
    content: Document the folder-ownership criteria in docs/SCRIPTS_PIPELINES_GUIDE.md and update file references for the moves below.
    status: completed
  - id: compare_move
    content: Move scripts/contour_mcz_td/compare_contours_mcz_td.py to scripts/mismatch_mcz_td/compare_contours_mcz_td.py to mirror compare_contours_I_td.py.
    status: completed
  - id: cube_viz_relocate
    content: Move scripts/mismatch_mcz_td/_viz_utils.py to scripts/utils/_cube_viz.py and move scripts/mismatch_mcz_td/visualize_mismatch_cube.py to scripts/utils/visualize_mismatch_cube.py, generalizing the mcz/I filename-token parsing. Update imports in scripts/mismatch_mcz_td/visualize_mcz_sweep_at_td.py.
    status: completed
  - id: extend_plot_contour
    content: Extend scripts/utils/plot_contour_from_dataset.py to accept the cube schema (td, theta, omega, epsilon_min_grid) with --td_ms slicing, then delete scripts/mismatch_mcz_td/plot_omega_theta_from_cube.py.
    status: completed
  - id: best_match_helper
    content: Create scripts/utils/_best_match_plot.py with the shared VARIABLE_MAPPING, contour renderer, and filename-suffix builder; refactor the two plot_contour_*_from_best_match.py entry points to call it (keeping cycle/peak/trough overlays only on the mcz_td side). While refactoring the mcz_td entry point, also swap its inlined overlay args for modules.cli_utils.add_cycle_extrema_overlay_args.
    status: completed
  - id: convert_redshift_move
    content: Move and rename scripts/mismatch_mcz_td/convert_best_match_redshift.py to scripts/utils/convert_best_match_mcz_td_redshift.py.
    status: completed
  - id: callsite_audit
    content: Audit Python files and batch_scripts/ for references to the moved/renamed scripts and update them. Run get_errors on all touched files.
    status: completed
  - id: compare_mcz_td_filename_helper
    content: Optional follow-up — add compare_mcz_td_figure_filename to modules/filenames.py and refactor the moved compare_contours_mcz_td.py to use it (mirrors the recent compare_I_td_figure_filename addition).
    status: completed
isProject: false
---

# scripts/ Reorganization Plan

## 0. Update notes (May 7)

Since this plan was first drafted, three commits landed that affect Issues 1 and 4 but do not change the plan's structure:

- [`scripts/mismatch_I_td/compare_contours_I_td.py`](../scripts/mismatch_I_td/compare_contours_I_td.py) was rewritten to accept 1+ paths and now consumes two new shared helpers in `modules/`.
- New filename helpers in [`modules/filenames.py`](../modules/filenames.py): `compare_I_td_figure_filename` and `bestfit_prec_params_I_td_figure_filename`.
- New CLI helper in [`modules/cli_utils.py`](../modules/cli_utils.py): `add_cycle_extrema_overlay_args` (already consumed by `compare_contours_I_td.py` and `scripts/analysis/plot_bestfit_prec_params.py`).

Two consequences:

- The repo's pattern for cross-script helpers is now `modules/` (`filenames.py`, `cli_utils.py`). Our `scripts/utils/_cube_viz.py` and `scripts/utils/_best_match_plot.py` placements remain correct because each helper is private to exactly two scripts; they would be over-promoted by living in `modules/`.
- Issues 1 and 4 gain small follow-ups noted inline below. The original moves and consolidations are unchanged.

## 1. Classification criteria (Issue 6)

Update [`docs/SCRIPTS_PIPELINES_GUIDE.md`](SCRIPTS_PIPELINES_GUIDE.md) to make folder ownership explicit:

- [`scripts/mismatch_*_td/`](../scripts/mismatch_mcz_td/): the four production stages (compute / aggregate / Stage-3 plot from best_match) and any post-production helper that is **inherently pipeline-specific** (depends on which axis is swept).
- [`scripts/contour_mcz_td/`](../scripts/contour_mcz_td/) and [`scripts/contour_omega_theta/`](../scripts/contour_omega_theta/): broader parameter sweeps and one-off / debugging contour generators. **No paper-figure scripts that consume `best_match_*.h5`** — those belong with the pipeline that produced them.
- [`scripts/lindblom/`](../scripts/lindblom/): Lindblom workflow.
- [`scripts/template_banks/`](../scripts/template_banks/): bank builders.
- [`scripts/analysis/`](../scripts/analysis/): cross-pipeline physics analyses (e.g. `modality_nlens`, `plot_bestfit_prec_params`).
- [`scripts/utils/`](../scripts/utils/): generic helpers and conversion scripts that operate across pipelines or on a common artifact schema (cube, generic dataset, HDF5 metadata).
- `_underscore_helpers.py` private modules under a folder are imported only within that folder's siblings (or a sibling folder for `scripts/utils/_*.py`).

## 2. File moves and consolidations

### Issue 1 — `compare_contours_*_td.py` placement — **done**

Moved [`scripts/mismatch_mcz_td/compare_contours_mcz_td.py`](../scripts/mismatch_mcz_td/compare_contours_mcz_td.py) (formerly under `contour_mcz_td/`) so it sits next to its `I_td` counterpart in [`scripts/mismatch_I_td/compare_contours_I_td.py`](../scripts/mismatch_I_td/compare_contours_I_td.py). The script consumes `best_match_*.h5` outputs, so it belongs with the production pipeline. The single `contour_L_NP_*.h5` reference panel does not change ownership — `contour_mcz_td/` keeps the generator, but the comparison figure is a pipeline-output consumer.

Optional follow-up (recommended for symmetry with the recent I_td refactor): add `compare_mcz_td_figure_filename` to [`modules/filenames.py`](../modules/filenames.py) — the `I_td` side now uses `compare_I_td_figure_filename` instead of a hardcoded `DEFAULT_OUTPUT`, but `compare_contours_mcz_td.py` still hardcodes `DEFAULT_OUTPUT`. Refactor it to use the new helper when ready.

### Issue 2 — cube visualization scripts — **done**

Per "split by generality": the cube schema `(td, theta, omega, epsilon_min_grid)` is identical across both pipelines, so single-cube visualization is generalizable; the `mcz`-sweep variant is pipeline-specific.

- Moved to [`scripts/utils/_cube_viz.py`](../scripts/utils/_cube_viz.py) (formerly `_viz_utils.py`). Shared plotting plumbing (`save_contour_movie`, `save_html_slider`, `find_td_index`, `format_resolution_suffix`, `global_min_max`, `infer_orientation_tag_from_filename`).
- Moved to [`scripts/utils/visualize_mismatch_cube.py`](../scripts/utils/visualize_mismatch_cube.py). Output basename uses dataset `I` when present (I_td cubes), else `mcz` (mcz_td cubes), with filename-parser fallbacks.
- [`scripts/mismatch_mcz_td/visualize_mcz_sweep_at_td.py`](../scripts/mismatch_mcz_td/visualize_mcz_sweep_at_td.py) kept in place; imports `scripts.utils._cube_viz`. Adding a symmetric `visualize_I_sweep_at_td.py` is **out of scope** unless explicitly requested later.

### Issue 3 — `plot_omega_theta_from_cube.py` overlap with `plot_contour_from_dataset.py`

[`scripts/utils/plot_contour_from_dataset.py`](../scripts/utils/plot_contour_from_dataset.py) already handles the `(omega_matrix, theta_matrix, epsilon_matrix)` and `(mcz, td, epsilon_min)` schemas. Extend it to also accept the cube schema `(td, theta, omega, epsilon_min_grid)`:

- Add a `--td_ms` arg used only when the input is a cube; pick the nearest td slice (mirroring [`plot_omega_theta_from_cube.py`](../scripts/mismatch_mcz_td/plot_omega_theta_from_cube.py) logic).
- Reuse the existing `_load_contour_from_h5` dispatcher and add a third branch for the cube schema.

Then delete [`scripts/mismatch_mcz_td/plot_omega_theta_from_cube.py`](../scripts/mismatch_mcz_td/plot_omega_theta_from_cube.py). The auto-discovery convenience (find cube via `find_mismatch_mcz_cube_files` from `--mcz`, `--td_ms`, `--orientation_tag`) is dropped; users pass `--input <cube_path>` directly. If the discovery convenience proves needed, a thin wrapper can be re-added later.

### Issue 4 — `plot_contour_*_from_best_match.py` shared core

Per "moderate": extract the shared rendering core; keep separate entry points.

- Create `scripts/utils/_best_match_plot.py` with helpers consumed by both Stage-3 plot scripts:
  - A renderer taking `(x_arr, y_arr, Zmap, x_label, y_label, cbar_label, title, output_path, overlay_kwargs)`.
  - The `VARIABLE_MAPPING` dict (currently duplicated verbatim in both scripts in [`plot_contour_mcz_td_from_best_match.py`](../scripts/mismatch_mcz_td/plot_contour_mcz_td_from_best_match.py) and [`plot_contour_I_td_from_best_match.py`](../scripts/mismatch_I_td/plot_contour_I_td_from_best_match.py)).
  - The filename-suffix builder (variable suffix + optional `overlayed` suffix).
- Refactor each entry point to read the appropriate `best_match_*` data via [`modules/bank_io.py`](../modules/bank_io.py), build the canonical run-dir / filename via the pipeline-specific `contour_*_filename` helpers, then delegate rendering to the shared helper.
- The cycle/peak/trough overlays stay opt-in via the same `overlay_*` flags. They remain available only on the `mcz_td` entry point because varying `mcz` is what makes those overlays meaningful — `I_td` keeps `mcz` fixed.
- While refactoring [`plot_contour_mcz_td_from_best_match.py`](../scripts/mismatch_mcz_td/plot_contour_mcz_td_from_best_match.py), replace its inlined `--overlay-cycles` / `--overlay-peaks` / `--overlay-troughs` / `--show-legend` / `--eta` / `--f_min` arg block with `add_cycle_extrema_overlay_args` from [`modules/cli_utils.py`](../modules/cli_utils.py). This matches the recent `compare_contours_I_td.py` and `plot_bestfit_prec_params.py` migrations.

### Issue 5 — `convert_best_match_redshift.py` placement

Move [`scripts/mismatch_mcz_td/convert_best_match_redshift.py`](../scripts/mismatch_mcz_td/convert_best_match_redshift.py) → `scripts/utils/convert_best_match_mcz_td_redshift.py`. Rename adds the `mcz_td` token to make the schema dependency explicit (the script reads the `mcz` axis dataset and the `mcz_td` template-grid filename tokens, so it does not work on `I_td` outputs as written). No behavior change. Generalizing it to also handle the `I_td` schema (where `mcz` is a scalar dataset) is **out of scope** for this refactor.

## 3. Documentation updates

Update [`docs/SCRIPTS_PIPELINES_GUIDE.md`](SCRIPTS_PIPELINES_GUIDE.md):

- Add the classification criteria above as a short, ranked subsection near "Contributor Notes".
- Update the file references in the `mismatch_mcz_td/` section to drop `_viz_utils.py`, `visualize_mismatch_cube.py`, `plot_omega_theta_from_cube.py`, `convert_best_match_redshift.py` and add `compare_contours_mcz_td.py`.
- Update the `contour_mcz_td/` section to drop `compare_contours_mcz_td.py`.
- Update the `utils/` section to add `visualize_mismatch_cube.py`, `convert_best_match_mcz_td_redshift.py`, and `_cube_viz.py` / `_best_match_plot.py` (private helpers).

No edits required to [`docs/CONTOUR_MCZ_TD_PIPELINE_GUIDE.md`](CONTOUR_MCZ_TD_PIPELINE_GUIDE.md) or [`docs/CONTOUR_I_TD_PIPELINE_GUIDE.md`](CONTOUR_I_TD_PIPELINE_GUIDE.md) since they only reference the four production-stage scripts (`compute_mismatch_cubes`, `aggregate_best_match`, `plot_contour_*_from_best_match`), all of which keep their current locations and module paths.

## 4. Cross-cutting checks

- After each move, search the repo (Python files and `batch_scripts/*.sbatch`) for the old module path and update imports / `python -m` invocations. Confirmed sites today: only `compare_contours_mcz_td.py` and the moved visualize scripts have outside callers (the `batch_scripts/` only invoke production-stage modules).
- Update [`scripts/utils/__init__.py`](../scripts/utils/__init__.py) and [`scripts/mismatch_mcz_td/__init__.py`](../scripts/mismatch_mcz_td/__init__.py) only if they currently re-export the moved symbols (they do not).
- Run `get_errors` on every touched file after edits.

## 5. Out of scope

- Adding `visualize_I_sweep_at_td.py` for symmetry.
- Generalizing `convert_best_match_*_redshift.py` to also handle the `I_td` schema.
- Merging the two `plot_contour_*_from_best_match.py` entry points into a single auto-detecting script (this is the "aggressive" path; user picked "moderate").
- Touching [`scripts/contour_omega_theta/`](../scripts/contour_omega_theta/), [`scripts/lindblom/`](../scripts/lindblom/), [`scripts/analysis/`](../scripts/analysis/), or [`scripts/template_banks/`](../scripts/template_banks/) — they already follow the criteria.
- The `compare_mcz_td_figure_filename` helper is listed as an optional follow-up under Issue 1 but is not required for the move itself; skip it if the goal is only to consolidate placement.
