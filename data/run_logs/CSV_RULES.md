# Run Log CSV Rules

Source of truth for parsing/writing CSVs in `data/run_logs`.

## File naming

One CSV per `{z}_{orientation}_{pipeline}` slice (e.g. `runlog_z1_Taman_edgeon_I_td.csv`). Use `runlog_mcz_td.csv` only for mixed slices that do not fit a single `z`/orientation bucket.

One row = one stage execution event. Do not split a run log by stage.

## Columns

All canonical columns, in order. Columns marked *(I_td only)* are absent from `mcz_td` files; columns marked *(optional)* may be absent from any file.

- `log_file`
- `pipeline`: `mcz_td` or `I_td`.
- `stage`: `build` or `mismatch`.
- `total_time`
- `z` *(absent from `runlog_mcz_td.csv`)*
- `mcz`, `mcz_min`, `mcz_max`, `mcz_pts`
- `I`: `0.5` for `mcz_td` mismatch rows; swept value for `I_td` rows; blank for `build` stage rows.
- `I_min`, `I_max`, `I_pts` *(I_td only)*
- `td_min_ms`, `td_max_ms`, `td_pts`
- `omega_min`, `omega_max`, `omega_pts`
- `theta_min`, `theta_max`, `theta_pts`
- `gamma_pts`
- `dtype`
- `orientation`: e.g. `Taman_faceon`, `Taman_edgeon`, `Taman_random`, `Tien_faceon`, `Tien_edgeon`, `Tien_random`, `Ben_random` (blank if unknown).
- `save_dir`: path to the output directory relative to `/work/10000/fairytien33/gw_shared_data/`. Use the nearest enclosing directory that groups the artifacts. Blank when no artifact exists or context is insufficient.
- `save_file`: basename of the saved HDF5 file. Blank when the row represents multiple files or no file exists.
- `size`: total on-disk byte size for the row's artifact(s). Single non-negative integer. When `save_file` is set, the size of that file. When one job wrote multiple HDF5 files, the sum of those files only (not unrelated files in the same `save_dir`). Blank when no artifacts exist or sizes are unknown.
- `date`: `YYYY-MM-DD` only.
- `status`: `ok`, `partial`, or `failed`.
- `note` *(optional)*: run diagnostics, errors, and `saved=m/n` summaries.

Do not introduce extra operational metadata columns (e.g. job IDs) unless explicitly requested.

## Status rules (apply in this exact order)

1. If `processed >= 1` and `skipped=0`, set `status=ok` and leave `note` blank.
2. If `processed >= 1` and `skipped > 0`, set `status=partial`.
3. If `processed=0` and `skipped > 0`, set `status=failed`.
4. If `saved=m/n` appears in `note`:
   - `m=0` → `status=failed`
   - `0 < m < n` → `status=partial`
   - `m=n` → `status=ok`; clear `note` unless extra diagnostics remain.
5. If template-bank open/unavailable errors appear in `note` (e.g. `template bank unavailable`, `Unable to synchronously open file`), set `status=failed`.
6. If cancelled/time-limit appears in `note` and rules 1–4 did not trigger, set `status=failed`.
7. Else if `total_time` is present and no partial/failure rule triggered, set `status=ok` (warning-only notes such as `ODEintWarning` are still `ok`).
8. Else if `total_time` is missing and no rule above triggered: error text in `note` → `status=failed`; otherwise `status=partial`.

Do not infer failure from missing `total_time` alone.
