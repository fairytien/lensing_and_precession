# Project Guidelines

Instructions for AI coding agents working in this repository. These guidelines apply to code, docs, and notebook edits in this repo.

## Environment

- Use the `gw` environment for this repository.

## Architecture and Imports

- Run Python from the repository root with `python -m <module>`.
- [`modules/`](modules/) contains the canonical source of truth:
  - [`Classes.py`](modules/Classes.py) — `LensingGeo`, `Precessing`
  - [`default_params.py`](modules/default_params.py) — physical constants such as `SOLMASS2SEC`
  - [`waveform.py`](modules/waveform.py) — waveform generation helpers
  - [`snr.py`](modules/snr.py) — detector noise PSD
  - [`match_utils.py`](modules/match_utils.py) — matching, mismatch computation, multiprocessing workers
  - [`filenames.py`](modules/filenames.py) — canonical filename builders
  - [`geometry.py`](modules/geometry.py) — orientation geometry helpers
  - [`plot_utils.py`](modules/plot_utils.py) — shared plotting utilities
- In new code, import from the specialized modules directly. [`functions.py`](modules/functions.py) and versioned shims such as `functions_v3` are compatibility layers, not new dependencies.
- Use [`modules/filenames.py`](modules/filenames.py) for output paths. Do not hand-roll naming logic.
- In [`match_utils.py`](modules/match_utils.py), use `_resolve_deps(**overrides)` for dependency injection. Do not import canonical modules at the top of that file.
- Production waveform physics remains pinned to `modules.Classes`. Treat `Classes_v3+` as testing-only unless the task explicitly says otherwise.
- Legacy/versioned implementations live under [`legacy/modules/`](legacy/modules/), and legacy scripts should import from `legacy.modules.*`.
- If historical files use legacy names, migrate them with `python -m scripts.utils.rename_legacy_filenames` and then `python -m scripts.utils.rename_legacy_filenames --apply`.

## Code Style

- **Lean and direct.** Implement the straightforward solution. Do not add abstractions, helpers, or indirection unless they reduce real complexity.
- **DRY, but not over-DRY.** Extract a shared helper only when the alternative duplicates non-trivial logic. Thin wrappers that only forward arguments do not help.
- **No speculative features.** Do not add error handling for impossible cases, docstrings for untouched code, or extra "improvements" beyond the request.
- **Typing is annotation, not armor.** Add types for correctness. Do not wrap ordinary expressions in `cast(...)` unless the checker cannot narrow them.
- **Guard helpers must earn their keep.** If a validation helper does not materially simplify the call site, inline it.
- **Sections over classes.** Prefer module-level sections over adding classes or extra modules purely for organization.

## Working Style

- Read before writing. Never assume function signatures, return shapes, or call conventions.
- Keep scope minimal. Report unrelated problems instead of fixing them silently.
- Prefer updating existing docs over creating new ones. Keep shared workflow-selection guidance in [`docs/SCRIPTS_PIPELINES_GUIDE.md`](docs/SCRIPTS_PIPELINES_GUIDE.md) and pipeline-specific execution details in the pipeline runbooks.
- Follow the naming grammar in the **Naming** section below for token order and examples.
- Before and after non-trivial edits, run `get_errors` on the touched files.
- Prefer the narrowest meaningful validation: `get_errors`, a targeted runtime check, or the smallest relevant command. Use `git diff` to confirm intent, not as the only validation when a narrower executable check exists.
- When refactoring, changed lines should preserve behavior unless the task explicitly calls for a logic change.
- Before renaming or deleting a module, search Python files and legacy notebooks for references. Update all references in the same change or stop and ask.
- Parallelize only truly independent edits.
- Worker globals such as `_S_STRAIN` and `_PSD` are initialized by `init_mismatch_worker`. Do not restructure them into dataclasses or separate modules unless asked.
- Be brief in explanations and summaries.
- Do not create new files unless they are necessary.
- Do not add comments or docstrings to code you did not change.
- Do not refactor code that was not part of the request.
- Do not introduce new abstractions unless explicitly requested.
- Do not rewrite working logic to match a style preference alone.

## Physics Notes

- Redshift-aware scripts use detector-frame mass scaling: `mcz_det = mcz_src * (1 + z)`.

## Parameter Order Convention

Use the following order rules consistently throughout the codebase and docs.

- Pipeline and workflow identifiers use `mcz_td` and `I_td`. Use these in folder names, script names, batch scripts, helper names, and runbook filenames because those tokens already match the repo's canonical code and filename surface.
- Human-facing contour planes use `(td, mcz)` and `(td, I)`. Use these in titles, prose, tables, and plot descriptions because they describe the source-waveform parameter plane as readers encounter it.
- HDF5, array, and aggregate-grid order use `(mcz, td)` and `(I, td)`. Use these only when describing dataset shapes, matrix layouts, or stored outputs because they should match stored axis order rather than plot wording.
- Do not introduce new pipeline tokens such as `td_mcz` or `td_I` in new docs, helpers, or filenames because extra aliases make links, filenames, and helper names harder to scan and maintain.

## Naming

Use this section when naming new scripts, outputs, or filename helpers. Routine runs can ignore it.

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
- Use the same token order in filename builders and discovery helpers in [`modules/filenames.py`](modules/filenames.py).

In the production mismatch pipelines, this means:

- `mcz_td`: fixed `I`, then fixed `z`, then swept `mcz`, then `td`.
- `I_td`: fixed `z`, then fixed `mcz`, then swept `I`, then `td`.
- For per-sweep mismatch cube filenames, both pipelines use the same token order: `z`, then `mcz`, then `I`, then `td`.

Examples:

- `mcz_td` run directory: `mismatch_I0p5_z1_mcz5-45_td20-70_Taman_faceon`
- `I_td` run directory: `mismatch_z1_mcz15_I0p1-0p9_td20-70_Taman_edgeon`

### Symbol Name Order

- Put local qualifiers such as `mcz`, `I`, and `td` before the artifact noun.
- Put workflow families such as `mismatch`, `best_match`, and `contour` before pipeline tokens such as `mcz_td` and `I_td`.

Examples:

- Artifact names: `create_mcz_mismatch_cube`, `create_I_mismatch_cube`, `mismatch_mcz_cube_filename`
- Pipeline names: `write_mcz_td_grid_attrs`, `write_I_td_grid_attrs`, `best_match_mcz_td_filename`, `compute_mismatch_mcz_td_cubes.sbatch`

When in doubt:

- If the symbol could be reused in another workflow without changing what it names, use an artifact name.
- If reusing it outside the current sweep would make the name misleading, use a pipeline name.
- Prefer existing stable filename-builder names in [`modules/filenames.py`](modules/filenames.py) unless you are doing a coordinated rename of the whole filename/discovery surface.

## Figure Typography

Publication-facing figures should follow APS/REVTeX math typography.

- Set physical variables in italic and descriptive labels in upright roman.
- In this repo, italic variables include `$I$`, `$z$`, `$\theta$`, `$\omega$`, `$\mathcal{M}$`, and `$t$` in `$\Delta t_{\mathrm{d}}$`. Upright descriptors include `$\mathrm{s}$`, `$\mathrm{t}$`, `$\mathrm{NP}$`, `$\mathrm{RP}$`, `$\mathrm{P}$`, `$\mathrm{L}$`, `$\mathrm{UL}$`, `$\mathrm{d}$`, and units such as `$\mathrm{Hz}$` and `$\mathrm{ms}$`.
- Keep true running indices italic. In this repo, most figure subscripts are descriptive labels, so prefer forms such as `$\Phi_{\mathrm{s}} - \Phi_{\mathrm{t}}$`, `$\mathcal{M}_{\mathrm{s}}$`, `$\gamma_{\mathrm{P}}$`, `$\theta_{\mathrm{S}}$`, `$\phi_{\mathrm{J}}$`, `$\Delta t_{\mathrm{d}}$`, and `$\epsilon(\~h_{\mathrm{L}}, \~h_{\mathrm{P}})$`.
- In matplotlib mathtext, prefer `\mathrm{...}` over legacy `\rm`. `apply_physics_paper_style()` in [`modules/plot_utils.py`](modules/plot_utils.py) sets defaults only; plotting scripts must still format labels, titles, legends, and colorbar labels explicitly, and touched figures should be normalized to this convention.
