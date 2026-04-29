# Run Log CSV Rules

This document is the source of truth for parsing/writing CSV files in `data/run_logs`.

## File naming

One CSV file per `{z}_{orientation}_{pipeline}` slice, for example:

- `runlog_z1_Taman_faceon_mcz_td.csv`
- `runlog_z1_Taman_edgeon_I_td.csv`

Use `runlog_mcz_td.csv` only for mixed slices that do not fit a single `z`/orientation bucket.

## Scope

- Keep a single CSV per run log context (do not split by stage).
- One row = one stage execution event.

## Required columns

Use these canonical names:

- `log_file`
- `pipeline` (`mcz_td` or `I_td`)
- `stage` (`build` or `mismatch`)
- existing parameter/grid columns already used by this repo (for example `mcz`, `I`, `td_*`, `omega_*`, `theta_*`, `gamma_pts`)
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

1. If `processed >= 1` and `skipped=0`, set `status=ok` and leave `Note` blank.
2. If `processed >= 1` and `skipped > 0`, set `status=partial`.
3. If `processed=0` and `skipped > 0`, set `status=failed`.
4. If `saved=m/n` appears in `Note`:
   - if `m=0`, set `status=failed`;
   - if `0 < m < n`, set `status=partial`;
   - if `m=n`, set `status=ok` and leave `Note` blank unless there are extra diagnostics.
5. If template-bank open/unavailable errors appear in `Note` (for example `template bank unavailable` or `Unable to synchronously open file`), set `status=failed`.
6. If cancelled/time-limit appears in `Note` and rules 1-4 did not trigger, set `status=failed`.
7. Else if `total_time` is present and no partial/failure rule triggered, set `status=ok` (warning-only notes such as `ODEintWarning` are still `ok`).
8. Else if `total_time` is missing and no rule above triggered:
   - if `Note` has error text, set `status=failed`;
   - otherwise set `status=partial`.

## Notes

- Do not infer failure from missing `total_time` alone.
- Do not introduce extra operational metadata columns (for example job IDs) unless explicitly requested.
