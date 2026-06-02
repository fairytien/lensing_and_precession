# Walkthrough: Add mcz-Optimized L-vs-NP Mismatch to `contour_L_NP_mcz_td` and `plot_np_rp_mcz_slice`

We have implemented the entire plan successfully. Here is a summary of the changes:

## Changes Made

### 1. Computation: [contour_L_NP_mcz_td.py](../scripts/np_fast/contour_L_NP_mcz_td.py)
- Switched default `MatchMethod` in `_compute_mismatch_row` to `OPTIMIZED_BOUNDED`.
- Implemented `_compute_opt_mcz_row` which:
  - Scans over template mcz values `linspace(mcz_s - 0.5, mcz_s + 0.5, 51)`.
  - Stacks and pre-pads the template waveforms into a 2D block.
  - Algebraically constructs the lensed source strain using cached unlensed waveform factors.
  - Scans over all time-delay (`td`) bins in the inner loop using `mismatch_block_serial` with `MatchMethod.OPTIMIZED_BOUNDED`.
- Added command line options (`--compute_opt_mcz`, `--opt_mcz_run_dir`, `--opt_mcz_window`, and `--opt_mcz_pts`).
- Extended the `main()` function to run a second pass for template-mcz optimization when `--compute_opt_mcz` is specified.
- Wrote the second optimized HDF5 file with the dataset `mcz_best` (representing the best fit template mcz) along with the min mismatch dataset `epsilon_min` and compatibility fields.

### 2. Plotting: [plot_np_rp_mcz_slice.py](../scripts/np_fast/plot_np_rp_mcz_slice.py)
- Fixed imports to load `read_best_match_mcz_td_data` instead of the non-existent `read_best_match_mcz_td_contour_data`.
- Imported label constants `LBL_EPS_LNP`, `LBL_MIN_MCZ_EPS_LNP`, and `LBL_EPS_LRP` from `modules.plot_utils`.
- Added the `--l-np-opt-contour` CLI option for the mcz-optimized NP mismatch file.
- Added the `--one-column` CLI option to allow toggling between single-column and two-column legend layouts.
- Updated `_build_curves` to load and extract the td slice from three files: fixed L-vs-NP, optimized L-vs-NP, and L-vs-RP.
- Updated `_plot` to plot three curves (Red Solid, Green Dashed, Blue Solid) using the standard label constants, and drawing the curves above helper lines (`zorder=4`).
- Updated visual styling: background color is set to white, and trough lines are set to dotted cyan/teal.
- Added custom handles to the legend for cycle lines ($N_{\mathrm{lensed}} = 1, 2, 3$), extrema vertical lines (peak/trough), and the horizontal analytical limit line.
- Changed the horizontal limit line style from dotted to dashed-dotted, with `zorder=2` and legend label $1 - (1 + I)^{-1/2}$.
- Structured the legend into 2 columns laid out column-by-column: Column 1 contains the 3 mismatch curves and the analytical limit line (with an empty spacing dummy handle at the bottom), and Column 2 contains the cycle/extrema vertical overlay lines. If `--one-column` is passed, it falls back to a single vertical column containing all elements. The legend font size is set to `14` (reduced from `16`).
- Set default output path filename format to `compare_LvsNP_RP_{I}_{td}_{z}_mcz{min}-{max}_{orientation_preset}.pdf` based on grid metadata. This leverages `_canonical_token` and `_range_token` from `modules.filenames` to maintain consistent naming logic.

## Verification & Validation

1. **Compilation Check**: Compiled both updated python scripts using `py_compile` under conda `gw` environment:
   - `python -m py_compile scripts/np_fast/contour_L_NP_mcz_td.py`
   - `python -m py_compile scripts/np_fast/plot_np_rp_mcz_slice.py`
   Both compiled without errors.

2. **Full Sweep Test Grid Run**: Ran a full sweep grid locally using:
   ```bash
   conda run -n gw python -m scripts.np_fast.contour_L_NP_mcz_td --I 0.5 --mcz_min 5 --mcz_max 45 --mcz_points 81 --td_min_ms 20 --td_max_ms 70 --td_points 51 -z 1 --compute_opt_mcz
   ```
   Both fixed-mcz and optimized-mcz passes ran successfully, writing the output files:
   - Fixed: `data/mismatch_L_NP_I0p5_z1_mcz5-45_td20-70_Taman_edgeon/best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-0x1_theta0-0x1_gamma0-2pix1_Taman_edgeon.h5`
   - Optimized: `data/mismatch_L_NP_opt_mcz_I0p5_z1_mcz5-45_td20-70_Taman_edgeon/best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-0x1_theta0-0x1_gamma0-2pix1_Taman_edgeon.h5`

3. **Output Inspection**: Inspected the HDF5 datasets and attributes, verifying:
   - Dataset shapes are exactly `(81, 51)` corresponding to `mcz` and `td` grids.
   - `mcz_best` is correctly populated with the best-fitting template chirp mass values.
   - Epsilon mismatch values in the optimized file are strictly less than or equal to those in the fixed file, showing the optimization worked.

4. **Plot Execution**: Executed the slice plotting script using the three datasets at `td = 30 ms`:
   ```bash
   conda run -n gw python -m scripts.np_fast.plot_np_rp_mcz_slice \
     --l-np-contour data/mismatch_L_NP_I0p5_z1_mcz5-45_td20-70_Taman_edgeon/best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-0x1_theta0-0x1_gamma0-2pix1_Taman_edgeon.h5 \
     --l-np-opt-contour data/mismatch_L_NP_opt_mcz_I0p5_z1_mcz5-45_td20-70_Taman_edgeon/best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-0x1_theta0-0x1_gamma0-2pix1_Taman_edgeon.h5 \
     --rp-best-match data/mismatch_I0p5_z1e-08_mcz10-90_td20-70_Taman_edgeon/best_match/best_match_I0p5_z1_mcz5-45x81_td20-70x51_omega0-6x61_theta0-15x151_gamma0-2pix51_Taman_edgeon.h5 \
     --td-ms 30 --z 1
   ```
   The figure successfully generated and saved:
   - Saved PDF: [compare_LvsNP_RP_I0p5_td30_z1_mcz5-45_Taman_edgeon.pdf](file:///Users/fairytien/Documents/TEXAS_Bridge_2324/code/lensing_and_precession/figures/contour_mcz_td/compare_LvsNP_RP_I0p5_td30_z1_mcz5-45_Taman_edgeon.pdf)
