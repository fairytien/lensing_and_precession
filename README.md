# Distinguishing regularly precessing and lensed gravitational waveforms

## Description
Gravitational waves (GWs) from binary black hole (BBH) inspirals are affected by the black hole spins and orbital angular momentum, which, when misaligned, cause precession and nutation and introduce modulations in GW amplitudes and phases. In regular precession (without transitional precession or nutation), the total angular momentum has nearly constant direction and the orbital angular momentum precesses on a cone whose opening angle and frequency slowly increase on the radiation-reaction timescale. Regularly precessing BBH systems include those with a single spin, equal masses, or those trapped in spin-orbit resonances.

On the other hand, GWs can also be lensed by massive objects along the line of sight, resulting in amplification, potentially multiple images, and modulation of GWs. GWs are analyzed in the wave-optics regime and geometrical-optics regime depending on the mass of the lens and the wavelength. In axisymmetric lens models such as the point mass and singular isothermal sphere, the gravitational waveform can be described by the lens mass and the source position relative to the optic axis.

We investigate various parameters governing regular precession, including the precession amplitude, frequency, and the initial precessing phase, and lensing parameters, such as the lens mass and source position, to identify scenarios where the resulting waveforms may appear indistinguishable. The source’s chirp mass inversely correlates with the innermost stable circular orbit frequency cutoff and the inspiral waveform duration in the frequency band. At high chirp masses, waveforms may lack distinctive features, thus simplifying waveform matching. Through parameter tuning, a parameter space can be identified where the secular, oscillatory regularly precessing waveform aligns with the purely oscillatory lensed one. In addition, analytical approximations can predict the mismatch behavior between the lensed source and the regularly precessing template, as a function of the source’s chirp mass, which further elucidates the contribution of BBHs’ regular precession to waveform ambiguity.

Employing match-filtering analysis and various `PyCBC` packages, we quantify the mismatch and apply the Lindblom criterion to establish discernibility conditions for waveforms. Our study explores the parameter space to understand waveform distinguishability between regular precession and lensing, offering insights into the signal-to-noise requirement for GW detectors to effectively discern these waveforms.

## Getting Started
```bash
conda create -n gw python=3.10 -y
conda activate gw
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib h5py astropy lalsuite pycbc
```

Core Python libraries used by the main pipeline:

- `numpy`
- `scipy`
- `matplotlib`
- `h5py`
- `astropy`
- `lalsuite`
- `pycbc`

## Pipeline Workflow

This pipeline calculates **minimum mismatch contours** for lensed gravitational wave sources across a 2D parameter space of time delay (x-axis) and chirp mass (y-axis). For each (time delay, chirp mass) point, it finds the best-matching template from a 4D RP template bank and records the minimum mismatch.

### Template Bank Construction
1. **Build template banks**: Run `python -m scripts.template_banks.build_template_banks` to generate one HDF5 RP bank per chirp mass.
2. **For cluster/array jobs**: Use `batch_scripts/build_template_banks.sbatch`.

### Mismatch Computation
1. **Compute mismatch cubes**: Run `python -m scripts.mismatch_mcz_td.compute_mismatch_cubes` to compare lensed sources against the prebuilt banks.
2. **For cluster/array jobs**: Use `batch_scripts/compute_mismatch_mcz_cubes.sbatch`.
3. **Output shape**: Each run writes one per-`mcz` mismatch cube under `data/mismatch/mismatch_cubes/`.

### Aggregation
After all requested `mcz` values finish, run `python -m scripts.mismatch_mcz_td.aggregate_best_match` once to combine the per-`mcz` cubes into a single best-match file.

**Example workflow for array jobs:**
```bash
# 1. Build template banks
sbatch batch_scripts/build_template_banks.sbatch

# 2. Submit array job for mismatch computation
sbatch batch_scripts/compute_mismatch_mcz_cubes.sbatch

# 3. After all array tasks complete, aggregate results
python -m scripts.mismatch_mcz_td.aggregate_best_match \
  --run_dir ./data/mismatch \
  --I 0.5 \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 90 \
  --orientation_tag Taman_edgeon \
  --z 1

# 4. Plot the final contour
python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match \
  --input_path ./data/mismatch_I0p5_z1_mcz10-90_td20-70_Taman_edgeon/best_match/<best_match_file>.h5 \
  --output_dir ./figures/mismatch
```

### Plotting
- **Final contour**: Use `python -m scripts.mismatch_mcz_td.plot_contour_mcz_td_from_best_match`
- **Per-cube inspection**: Use the helper scripts under `scripts/mismatch_mcz_td/` for slices, sweeps, and interactive cube inspection
- **Workflow guide**: See `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md` for the current stage-by-stage reference

## Documentation Guide

Detailed usage documentation is centralized under `docs/`.

- `docs/SCRIPTS_PIPELINES_GUIDE.md`: Script inventory and task-based usage across all `scripts/` workflows.
- `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md`: Stage-by-stage `(td, mcz)` mismatch pipeline details.
- `docs/DATA_LFS.md`: Data organization, Git LFS workflow, checksums, size guard, and optional history cleanup.
- `docs/STOCKYARD.md`: Shared STOCKYARD workflow and symlink patterns on TACC.
- `docs/HDF5_SCHEMA_V1.md`: HDF5 metadata schema conventions.

## Main Pipeline Naming Strategy

Main-pipeline scripts now use stable, versionless import targets:

- `modules.Classes`
- `modules.default_params`
- `modules.functions`
- `modules.plot_utils`

These canonical modules are now the source of truth for production code.
Versioned compatibility modules (`Classes_v2`, `default_params_v3`, `functions_v3`, `plot_utils_v3`) are wrappers that re-export from canonical modules.

`modules.functions` is now a facade over specialized source modules:

- `modules.waveform`
- `modules.numerics`
- `modules.geometry`
- `modules.snr`

Matching and mismatch optimization now use `modules.match_utils` as the single source of truth.

`modules.functions_v3` remains a compatibility wrapper to preserve legacy imports.

Legacy/versioned implementations are kept under `modules/legacy/` and should be imported explicitly as `modules.legacy.*` when needed.

## GitHub Landing Page Note

Keep `README.md` at the repository root if you want this introduction to appear by default on the main GitHub repository page.

If detailed docs are moved under `docs/`, keep a root `README.md` (full or short) that links to those pages.

## Authors
* Tien Nguyen
* Tamanjyot Singh
* Michael Kesden
* Lindsay King

## Acknowledgement
This work is supported by the TEXAS Bridge Program 2023-2024 as a collaboration between the University of Texas at Dallas and the University of North Texas and funded by the NSF PAARE grant AST-2219128.
