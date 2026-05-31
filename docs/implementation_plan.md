# Combine Figures 4 & 5 into a 3×2 Contour+Waveform Panel

Merge the current two separate figures (mismatch contours and best-match waveform overlays) into a single 6-panel figure. Each row represents one chirp mass; the left column shows the mismatch contour over $(\tilde{\Omega}, \tilde{\theta})$, and the right column shows the best-match waveform overlay (amplitude + phase sub-rows).

## Data Source

Read pre-computed mismatch cube HDF5 files from the `mcz_td` pipeline on the cluster. The expected files are:

```
/work/10000/fairytien33/gw_shared_data/mismatch_I0p5_z1_mcz5-45_td20-70_Taman_edgeon/mismatch_cubes/
├── mismatch_cubes_z1_mcz5_I0p5_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5
├── mismatch_cubes_z1_mcz15_I0p5_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5
└── mismatch_cubes_z1_mcz25_I0p5_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5
```

Each cube contains (per [HDF5_SCHEMA.md](HDF5_SCHEMA.md)):
- `epsilon_min_grid`: shape `(51, 151, 61)` — axis order `td, theta, omega`
- `gamma_best_grid`: shape `(51, 151, 61)` — best γ_P at each point
- `td`, `omega`, `theta` axis datasets
- `source_param_*` and `template_param_*` file attributes

For a fixed `td` (default 30 ms), the script slices out a 2D `(theta, omega)` contour from each cube.

## Confirmed Requirements & Decisions

- **X markers placement:** Placing an X marker at the global minimum of $\epsilon_{\mathrm{RP}}$ on each contour and showing its $(\tilde\Omega, \tilde\theta)$ coordinates in the legend is correct.
- **Time delay value:** The default is `td = 30 ms`, but the script accepts it as a CLI argument `--td_ms`.
- **Execution Environment:** The script will be run on the computing cluster where the HDF5 mismatch cubes are stored (`/work/10000/fairytien33/gw_shared_data/mismatch_I0p5_z1_mcz5-45_td20-70_Taman_edgeon/mismatch_cubes/`), so copying the files locally is not necessary.

## Proposed Changes

### New plotting script

#### [IN PROGRESS] [plot_combined_contour_waveform.py](../scripts/contour_omega_theta/plot_combined_contour_waveform.py)

A single script that:

1. **Reads mismatch cube HDF5 files** for 3 chirp masses. For each cube at a given `td`:
   - Finds the `td` index closest to the requested value.
   - Extracts the 2D slice `epsilon_min_grid[td_idx, :, :]` → contour data.
   - Extracts `gamma_best_grid[td_idx, :, :]` → best γ_P at each grid point.
   - Reconstructs `source_params` and `template_params` from `source_param_*` / `template_param_*` file attributes via `extract_prefixed_params()`.

2. **Builds a 3-row × 2-column figure** using `matplotlib.gridspec`:
   - **Left column (contour):** `contourf` of $\epsilon_{\mathrm{RP}}(\tilde\Omega, \tilde\theta)$ for each mcz. Colorbar mode selectable via `--colorbar {shared,individual}` (default: `shared`). In `shared` mode a single colorbar spans all 3 rows via `add_colorbar_axes()`; in `individual` mode each panel gets its own colorbar with its own `vmin`/`vmax`. X marker at global min with coordinates in legend.
   - **Right column (waveform):** Each cell has 2 sub-rows (amplitude top, phase bottom), using `plot_best_match_overlay_from_contour()` from [waveform_plotting.py](../modules/waveform_plotting.py). The contour_data dict is built from the HDF5 data to match the expected interface.

3. **Follows AGENTS.md conventions:**
   - Imports from `modules.*`, uses `apply_physics_paper_style()`, APS/REVTeX math typography.
   - Uses $\epsilon_{\mathrm{RP}}$ (not $\epsilon_{\min}$) in all labels.
   - All axis and colorbar labels are drawn from `LBL_*` constants in `modules/plot_utils.py` (e.g. `LBL_OMEGA`, `LBL_THETA`, `LBL_F`, `LBL_BRATIO_TS`, `LBL_PHASE_TS`). The colorbar label uses a `LBL_EPS_RP` constant — **add `LBL_EPS_RP = r"$\epsilon_{\mathrm{RP}}$"` to `plot_utils.py`** if it does not yet exist, and import it in the script instead of writing the LaTeX inline.
   - Uses `save_figure()` from `plot_utils`.
   - CLI via argparse; defaults match the production config.

Key layout decisions:
- `GridSpec(nrows, 2)` outer grid; right column uses `GridSpecFromSubplotSpec(2, 1)` for amplitude/phase sub-rows.
- Colorbar controlled by `--colorbar {shared,individual}` (default `shared`). Shared: single colorbar via `add_colorbar_axes()` spanning all contour panels with a unified `vmin`/`vmax`. Individual: one colorbar per row, each with its own range.
- Row labels show $\mathcal{M}_{\mathrm{s}}$ values.
- Output: single PDF in `figures/contour_omega_theta/`.

---

### ε_min → ε_RP rename in existing code

#### [MODIFY] [plot_bestmatch_waveform_overlays.py](../scripts/contour_omega_theta/plot_bestmatch_waveform_overlays.py)

- Line 57: Change `$\epsilon_{\mathrm{min}}$` → `$\epsilon_{\mathrm{RP}}$` in `_row_parameter_box_text()`.

---

### Paper LaTeX update

#### [MODIFY] [results.tex](../../../paper_lens_prec/sections/results.tex)

- **Lines 12–22** (Figure 4 — contours): Replace with new combined figure.
- **Lines 56–70** (Figure 5 — waveforms): Remove; content merged into combined figure.
- New caption describes the 3-row layout with $\mathcal{M} = 5, 15, 25\,M_\odot$.
- Rename `\epsilon_{\min}` → `\epsilon_{\mathrm{RP}}` in caption and surrounding prose (note: use `\epsilon`, not `\varepsilon`).
- Merge labels into `\label{fig: sys2 contour waveform combined}`.
- Update all `\ref{fig: sys2 contours mcz}` and `\ref{fig: sys2 waveforms mcz}` references.

## Verification Plan

### Automated Tests
1. Run `get_errors` on the new script file.
2. Run the script with `--help` to verify CLI parses.
3. When HDF5 files are available locally, run the script and verify output PDF exists and is non-empty.

### Manual Verification
- Visual inspection of the generated PDF for correct layout, X markers, ε_RP labeling (using `\epsilon`, not `\varepsilon`), and that shared vs individual colorbar modes both render correctly.
- Verify LaTeX compiles after label/reference changes.
