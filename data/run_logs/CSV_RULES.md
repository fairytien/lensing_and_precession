# Run Log CSV Rules

This document is the source of truth for parsing/writing CSV files in `data/run_logs`.

## Scope

- Keep a single CSV per run log context (do not split by stage).
- One row = one stage execution event.

## Required columns

Use these canonical names:

- `log_file`
- `pipeline` (`mcz_td` or `I_td`)
- `stage` (`build` or `mismatch`)
- existing parameter/grid columns already used by this repo
- `orientation`
- `date`
- `status`
- `Note` (if present in that file)

Column tail order must be:

- `orientation` -> `date` -> `status` -> `Note` (if `Note` exists)

## Field conventions

- `date`: `YYYY-MM-DD` only (no time-of-day).
- `orientation`: for example, `Taman_faceon`, `Taman_edgeon`, `Taman_random`, `Tien_faceon`, `Tien_edgeon`, `Tien_random`, `Ben_random`, etc. (or blank if unknown).
- `status`: one of `ok`, `partial`, `failed`.
- `pipeline`: one of `mcz_td`, `I_td`.
- `stage`: one of `build`, `mismatch`.

## Status rules (apply in this exact order)

1. If `processed >= 1` in `Note`, set `status=partial`.
2. If `saved >= 1` in `Note`, set `status=partial`.
3. If template-bank open/unavailable errors appear in `Note` (for example `template bank unavailable` or `Unable to synchronously open file`), set `status=failed`.
4. If cancelled/time-limit appears in `Note` and rules 1-2 did not trigger, set `status=failed`.
5. Else if `total_time` is present and no partial/failure rule triggered, set `status=ok` (warning-only notes such as `ODEintWarning` are still `ok`).
6. Else if `total_time` is missing and no rule above triggered:
   - if `Note` has error text, set `status=failed`;
   - otherwise set `status=partial`.

## Notes

- Do not infer failure from missing `total_time` alone.
- Do not introduce extra operational metadata columns (for example job IDs) unless explicitly requested.
