# Project Guidelines

Instructions for AI coding agents working in this repository. These guidelines apply to code, docs, and notebook edits in this repo.

## Code Style

- **Lean and direct.** Implement the straightforward solution. Do not add abstractions, helpers, or indirection unless they reduce real complexity.
- **DRY, but not over-DRY.** Extract a shared helper only when the alternative duplicates non-trivial logic. Thin wrappers that only forward arguments do not help.
- **No speculative features.** Do not add error handling for impossible cases, docstrings for untouched code, or extra "improvements" beyond the request.
- **Typing is annotation, not armor.** Add types for correctness. Do not wrap ordinary expressions in `cast(...)` unless the checker cannot narrow them.
- **Guard helpers must earn their keep.** If a validation helper does not materially simplify the call site, inline it.
- **Sections over classes.** Prefer module-level sections over adding classes or extra modules purely for organization.

## Architecture and Imports

- Run Python from the repository root with `python -m <module>`.
- `modules/` contains the canonical source of truth:
  - `Classes.py` — `LensingGeo`, `Precessing`
  - `default_params.py` — physical constants such as `SOLMASS2SEC`
  - `waveform.py` — waveform generation helpers
  - `snr.py` — detector noise PSD
  - `match_utils.py` — matching, mismatch computation, multiprocessing workers
  - `filenames.py` — canonical filename builders
  - `geometry.py` — orientation geometry helpers
  - `plot_utils.py` — shared plotting utilities
- In new code, import from the specialized modules directly. `functions.py` and versioned shims such as `functions_v3` are compatibility layers, not new dependencies.
- Use `modules/filenames.py` for output paths. Do not hand-roll naming logic.
- In `match_utils.py`, use `_resolve_deps(**overrides)` for dependency injection. Do not import canonical modules at the top of that file.
- Production waveform physics remains pinned to `modules.Classes`. Treat `Classes_v3+` as testing-only unless the task explicitly says otherwise.

## Environment and Validation

- Use the `gw` environment for this repository.
- Follow the naming grammar in `docs/SCRIPTS_PIPELINES_GUIDE.md`.
- Before and after non-trivial edits, run `get_errors` on the touched files.
- Prefer the narrowest meaningful validation: `get_errors`, a targeted runtime check, or the smallest relevant command. Use `git diff` to confirm intent, not as the only validation when a narrower executable check exists.
- When refactoring, changed lines should preserve behavior unless the task explicitly calls for a logic change.
- Before renaming or deleting a module, search Python files and legacy notebooks for references. Update all references in the same change or stop and ask.

## Working Style

- Read before writing. Never assume function signatures, return shapes, or call conventions.
- Keep scope minimal. Report unrelated problems instead of fixing them silently.
- Prefer updating existing docs over creating new ones. Keep shared workflow-selection guidance in `docs/SCRIPTS_PIPELINES_GUIDE.md` and pipeline-specific execution details in the pipeline runbooks.
- Parallelize only truly independent edits.
- Worker globals such as `_S_STRAIN` and `_PSD` are initialized by `init_mismatch_worker`. Do not restructure them into dataclasses or separate modules unless asked.
- Be brief in explanations and summaries.
- Do not create new files unless they are necessary.
- Do not add comments or docstrings to code you did not change.
- Do not refactor code that was not part of the request.
- Do not introduce new abstractions unless explicitly requested.
- Do not rewrite working logic to match a style preference alone.