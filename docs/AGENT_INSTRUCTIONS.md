# Agent Instructions

Guidelines for AI agents working in this repository.

## Code Style

- **Lean and direct.** Implement the straightforward solution. Do not add
  abstractions, helpers, or indirection unless they are used in three or more
  places with a real reduction in complexity.
- **DRY, but not over-DRY.** Extract a shared helper only when the alternative
  is duplicating non-trivial logic. A thin wrapper that merely forwards
  arguments with the same signature provides no net value — leave it inline.
- **No speculative features.** Do not add error handling for scenarios that
  cannot happen, docstrings for code you did not change, or "improvements"
  beyond what was asked.
- **Typing is annotation, not armor.** Add `Optional[T]` and type aliases
  (`PsdLike = Union[...]`) for correctness. Do not wrap every expression in
  `cast(...)` unless the static checker cannot narrow the type on its own.
- **Guard helpers must earn their keep.** A validation guard that returns a
  verbose typed tuple is over-engineered. If a guard does not materially
  simplify the call site, inline it.
- **Sections over classes.** Module-level sections separated by `# ===` banners
  are preferred to creating classes or extra modules purely for organization.

## Architecture and Modules

- Run Python from the repository root with `python -m <module>` to avoid path
  issues.
- `modules/` contains the canonical source of truth:
  - `Classes.py` — `LensingGeo`, `Precessing`
  - `default_params.py` — physical constants (e.g. `SOLMASS2SEC`)
  - `waveform.py` — `get_gw`, `set_to_params`, `get_fcut_from_mcz`
  - `snr.py` — `Sn` (noise PSD)
  - `match_utils.py` — matched filtering, mismatch computation, multiprocessing workers
  - `filenames.py` — canonical filename builders (use these; do not hand-roll paths)
  - `functions.py`, `plot_utils.py` — shared physics helpers and plotting
- Versioned compatibility shims (`Classes_v2`, `functions_v3`, etc.) re-export
  the canonical modules. Do not import from them in new code.
- Dependency injection in `match_utils` uses `_resolve_deps(**overrides)`.
  Pass `None` to use the canonical default; pass an explicit value to override.
  Do not import canonical modules at the top of `match_utils.py` to avoid
  circular imports.

## Naming Conventions

- Follow the naming grammar in `docs/SCRIPTS_PIPELINES_GUIDE.md`.
- Filename helpers live in `modules/filenames.py`. Use them for all output paths.
- Pipeline-neutral modules (e.g. `match_utils.py`) must not embed
  pipeline-specific tokens or naming logic.

## Behavior Preservation

- When refactoring, verify via `git diff` that every changed line is either a
  type annotation, a comment, a renamed local, or a DRY extraction — not a
  logic change.
- Before and after any non-trivial edit, run a static check:
  `get_errors` should report zero errors.
- Worker-process globals (`_S_STRAIN`, `_PSD`, etc.) are set once by
  `init_mismatch_worker`. Do not restructure them into a dataclass or a
  separate module unless asked.

## What Not To Do

- Do not create new files unless absolutely necessary.
- Do not add comments or docstrings to code you did not change.
- Do not refactor code that was not part of the request.
- Do not introduce new abstractions (base classes, protocols, config objects)
  unless explicitly requested.
- Do not rewrite working logic to match a different style preference.
