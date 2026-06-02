# Add mcz-Optimized L-vs-NP Mismatch to `contour_L_NP_mcz_td` and `plot_np_rp_mcz_slice`

## Goal

Extend `contour_L_NP_mcz_td.py` to produce a **second HDF5 output** containing the L-vs-NP mismatch **optimized over template mcz**, alongside its existing fixed-mcz contour output. Then extend `plot_np_rp_mcz_slice.py` to accept **3 data inputs** and plot three curves.

## Background

Currently:
- [contour_L_NP_mcz_td.py](../scripts/np_fast/contour_L_NP_mcz_td.py) computes ε(h\_L, h\_NP) on a (mcz\_s, td) grid where source and template share the same mcz. Uses `MatchMethod.OPTIMIZED_BRENT`.
- [plot_np_rp_mcz_slice.py](../scripts/np_fast/plot_np_rp_mcz_slice.py) reads 2 inputs (L-vs-NP contour, RP best\_match) and plots two curves at a fixed td slice. Uses a broken import (`read_best_match_mcz_td_contour_data` which does not exist in `bank_io.py`).

## Key Design Decisions

### Inline mcz scan instead of `optimize_mismatch_mcz`

The existing [optimize_mismatch_mcz](../modules/match_utils.py#L409-L494) in `match_utils.py` rebuilds full parameter dicts via `set_to_params` / `_resolve_deps` / `Classes` on every call. For the simpler NP case, an inline scan that directly calls `get_gw` + `mismatch_from_strains` avoids that overhead.

### NP templates are td-independent

NP template waveforms depend only on mcz (and fixed orientation/eta), **not on td** (a lensing parameter). For each source mcz row, the 51 NP templates are generated **once** and reused across all td values. Only the lensed source strain changes per td (through MLz).

### Cross-row template caching

With source mcz step < 1.0 M☉ and a ±0.5 M☉ template window, consecutive rows share template mcz values. A dict cache keyed by template mcz avoids regenerating overlapping templates. Source mcz values are processed in sorted order; after each row, cache entries outside any future row's window are evicted.

> [!NOTE]
> With multiprocessing across rows, each worker has its own memory space, so cross-row caching only works in serial mode. The script uses multiprocessing for the outer loop. Two approaches:
> 1. **Parallelize across rows** (current pattern): each worker self-contained, no cross-row cache. Simple, HPC-friendly.
> 2. **Serial outer loop with parallel td inner loop**: enables cross-row cache but changes the parallelism axis.
>
> I'll go with **option 1** (parallel across rows) since HPC parallelism across rows is the bigger win. Each worker generates its 51 templates independently. Cross-row caching is a constant-factor saving (at most 51 `get_gw` calls per row) that doesn't justify the complexity of option 2.

## Proposed Changes

### Computation: `contour_L_NP_mcz_td.py`

#### [MODIFY] [contour_L_NP_mcz_td.py](../scripts/np_fast/contour_L_NP_mcz_td.py)

**1. Switch `MatchMethod` from `OPTIMIZED_BRENT` to `OPTIMIZED_BOUNDED`** in `_compute_mismatch_row` (line 100).

**2. Add `_compute_opt_mcz_row` function.** For a given source mcz\_s:
   - Build NP parameter dict, apply redshift.
   - Compute PSD from source f\_cut (once, reused for all td and all template mcz).
   - Generate 51 NP template strains at `linspace(mcz_s - 0.5, mcz_s + 0.5, 51)`. Each is a single `get_gw` call. Pad all templates to source-strain length and stack into a 2D `template_block`.
   - Loop over td values: generate lensed source strain via `get_gw`, then call `mismatch_gamma_block_serial(template_block, mcz_t_arr, s_strain, psd, f_min, delta_f, OPTIMIZED_BOUNDED)`. This returns `(ep_vec, ep_min, best_mcz_t)`.
   - Return arrays: `ep_min_arr[n_td]` and `mcz_best_arr[n_td]`.

   This delegates the inner min-over-templates loop to [mismatch_gamma_block_serial](../scripts/np_fast/match_utils_np.py) — the same function already used for the fixed-mcz pass. Its `gamma_arr` parameter is repurposed as the template mcz array (it's just a label array for identifying the best template).

**3. Add CLI arguments:**
   - `--compute_opt_mcz` (flag, default False): whether to run the optimized pass.
   - `--opt_mcz_run_dir` (str, default `data/mismatch_L_NP_opt_mcz`): base output directory for optimized output.
   - `--opt_mcz_window` (float, default 0.5): half-width of template mcz window in M☉.
   - `--opt_mcz_pts` (int, default 51): number of template mcz grid points.

**4. In `main()`**, after the existing fixed-mcz computation:
   - If `--compute_opt_mcz`, run a second parallel map with `_compute_opt_mcz_row`.
   - Write a second HDF5 with datasets and schema matching `best_match_mcz_td`:
     - `mcz` (1D, source mcz), `td` (1D), `MLz` (1D)
     - `epsilon_min` (2D, mcz×td): min mismatch over template mcz
     - `mcz_best` (2D, mcz×td): best-fit template mcz — extra dataset
     - `omega_best`, `theta_best`, `gamma_best` (2D, zeros): schema compatibility
   - Attrs: `I`, `z`, `template_family=NP`, `orientation_tag`, `optimized_over=mcz`, `opt_mcz_window`, `opt_mcz_pts`, `match_method=optimized_bounded`.
   - Output path via `best_match_mcz_td_filename` with `run_dir` resolved from `contour_run_dir(opt_mcz_run_dir, ...)`.

**5. Import `mismatch_from_strains` from `modules.match_utils`** (used inside `mismatch_gamma_block_serial` already; no new module dependency). Import `ensure_same_length`, `cast_to_match_precision` from `modules.match_utils` for pre-padding template block.

---

### Plotting: `plot_np_rp_mcz_slice.py`

#### [MODIFY] [plot_np_rp_mcz_slice.py](../scripts/np_fast/plot_np_rp_mcz_slice.py)

**1. Fix import**: `read_best_match_mcz_td_contour_data` → `read_best_match_mcz_td_data` (the actual function in [bank_io.py](../modules/bank_io.py#L497-L531)).

**2. Add `--l-np-opt-contour` CLI argument** for the mcz-optimized HDF5 path.

**3. Update `_build_curves`** to accept and load the third input, extracting its td slice.

**4. Update `_plot`** to draw three curves:

| Curve | Color | Style | Legend |
|-------|-------|-------|--------|
| L-vs-NP (fixed mcz) | red | solid | $\epsilon(\tilde{h}\_{\mathrm{L}}, \tilde{h}\_{\mathrm{NP}})$ |
| L-vs-NP (opt mcz) | green | dashed | $\min\_{\mathcal{M}\_{\mathrm{t}}}\,\epsilon(\tilde{h}\_{\mathrm{L}}, \tilde{h}\_{\mathrm{NP}})$ |
| L-vs-RP | blue | solid | $\epsilon(\tilde{h}\_{\mathrm{L}}, \tilde{h}\_{\mathrm{RP}})$ |

**5. Visual changes:**
   - Plot background: **white** instead of grey (`#d9d9d9`).
   - Trough lines: **teal/cyan dotted** instead of white dotted.

---

## Output Schema (optimized HDF5)

Consistent with `best_match_mcz_td` format so `read_best_match_mcz_td_data(path, "epsilon_min")` works unchanged:

```
Datasets:
  mcz          (n_mcz,)       float64   [Msun]  — source mcz array
  td           (n_td,)        float64   [s]     — time delay array
  MLz          (n_td,)        float64   [s]     — lens mass array
  epsilon_min  (n_mcz, n_td)  float32           — min mismatch over template mcz
  mcz_best     (n_mcz, n_td)  float32   [Msun]  — best-fit template mcz
  omega_best   (n_mcz, n_td)  float32           — zeros (schema compat)
  theta_best   (n_mcz, n_td)  float32           — zeros (schema compat)
  gamma_best   (n_mcz, n_td)  float32           — zeros (schema compat)

Attributes:
  I, z, template_family="NP", orientation_tag
  optimized_over="mcz", opt_mcz_window, opt_mcz_pts
  match_method="optimized_bounded", minimizer="bounded"
```

## File Summary

| File | Action | Description |
|------|--------|-------------|
| [contour_L_NP_mcz_td.py](../scripts/np_fast/contour_L_NP_mcz_td.py) | MODIFY | Add optimized-mcz pass, switch to OPTIMIZED_BOUNDED, write second HDF5 |
| [plot_np_rp_mcz_slice.py](../scripts/np_fast/plot_np_rp_mcz_slice.py) | MODIFY | Fix import, add 3rd input, 3 curves, white bg, teal troughs |

No new files. No changes to modules — the script imports existing functions from [match_utils.py](../modules/match_utils.py), [match_utils_np.py](../scripts/np_fast/match_utils_np.py), [bank_io.py](../modules/bank_io.py), and [filenames.py](../modules/filenames.py).

## Verification Plan

### Automated Tests
- `python -c "from scripts.np_fast.contour_L_NP_mcz_td import main"` — verify import chain.
- `python -c "from scripts.np_fast.plot_np_rp_mcz_slice import main"` — verify fixed import.
- `get_errors` on both modified files.

### Manual Verification
- Run `contour_L_NP_mcz_td.py` with a small grid:
  ```
  python -m scripts.np_fast.contour_L_NP_mcz_td \
    --I 0.5 --mcz_min 15 --mcz_max 20 --mcz_points 3 \
    --td_min_ms 30 --td_max_ms 40 --td_points 3 -z 1 \
    --compute_opt_mcz
  ```
- Inspect both HDF5 outputs with `h5dump -H` to confirm datasets, shapes, attributes.
- Verify `read_best_match_mcz_td_data(opt_path, "epsilon_min")` returns correct dict.
- On HPC: full-grid run with array jobs for production.
