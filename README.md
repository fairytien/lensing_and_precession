# Distinguishing regularly precessing and lensed gravitational waveforms

## Description
Gravitational waves (GWs) from binary black hole (BBH) inspirals are affected by the black hole spins and orbital angular momentum, which, when misaligned, cause precession and nutation and introduce modulations in GW amplitudes and phases. In regular precession (without transitional precession or nutation), the total angular momentum has nearly constant direction and the orbital angular momentum precesses on a cone whose opening angle and frequency slowly increase on the radiation-reaction timescale. Regularly precessing BBH systems include those with a single spin, equal masses, or those trapped in spin-orbit resonances.

On the other hand, GWs can also be lensed by massive objects along the line of sight, resulting in amplification, potentially multiple images, and modulation of GWs. GWs are analyzed in the wave-optics regime and geometrical-optics regime depending on the mass of the lens and the wavelength. In axisymmetric lens models such as the point mass and singular isothermal sphere, the gravitational waveform can be described by the lens mass and the source position relative to the optic axis.

We investigate various parameters governing regular precession, including the precession amplitude, frequency, and the initial precessing phase, and lensing parameters, such as the lens mass and source position, to identify scenarios where the resulting waveforms may appear indistinguishable. The source’s chirp mass inversely correlates with the innermost stable circular orbit frequency cutoff and the inspiral waveform duration in the frequency band. At high chirp masses, waveforms may lack distinctive features, thus simplifying waveform matching. Through parameter tuning, a parameter space can be identified where the secular, oscillatory regularly precessing waveform aligns with the purely oscillatory lensed one. In addition, analytical approximations can predict the mismatch behavior between the lensed source and the regularly precessing template, as a function of the source’s chirp mass, which further elucidates the contribution of BBHs’ regular precession to waveform ambiguity.

Employing match-filtering analysis and various `PyCBC` packages, we quantify the mismatch and apply the Lindblom criterion to establish discernibility conditions for waveforms. Our study explores the parameter space to understand waveform distinguishability between regular precession and lensing, offering insights into the signal-to-noise requirement for GW detectors to effectively discern these waveforms.

## Getting started
This project requires the installation of [`lalsuite`](https://pypi.org/project/lalsuite/) and [`PyCBC`](https://pycbc.org):
```
python -m pip install lalsuite PyCBC
```

## Pipeline Workflow

This pipeline calculates **minimum mismatch contours** for lensed gravitational wave sources across a 2D parameter space of time delay (x-axis) and chirp mass (y-axis). For each (time delay, chirp mass) point, it finds the best-matching template from a 4D RP template bank and records the minimum mismatch.

### Template Bank Construction
1. **Build template banks**: Run `python -m scripts.template_banks.build_template_banks` to generate one HDF5 RP bank per chirp mass.
2. **For cluster/array jobs**: Use `batch_scripts/build_template_banks.sbatch`.

### Mismatch Computation
1. **Compute mismatch cubes**: Run `python -m scripts.mismatch_mcz_td.compute_mismatch_cubes` to compare lensed sources against the prebuilt banks.
2. **For cluster/array jobs**: Use `batch_scripts/compute_mismatch_cubes.sbatch`.
3. **Output shape**: Each run writes one per-`mcz` mismatch cube under `data/contours_td_mcz/mismatch_cubes/`.

### Aggregation
After all requested `mcz` values finish, run `python -m scripts.mismatch_mcz_td.aggregate_best_match` once to combine the per-`mcz` cubes into a single best-match file.

**Example workflow for array jobs:**
```bash
# 1. Build template banks
sbatch batch_scripts/build_template_banks.sbatch

# 2. Submit array job for mismatch computation
sbatch batch_scripts/compute_mismatch_cubes.sbatch

# 3. After all array tasks complete, aggregate results
python -m scripts.mismatch_mcz_td.aggregate_best_match \
  --results_dir ./data/contours_td_mcz \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 16 --mcz_max 25 \
  --orientation_tag Taman_edgeon

# 4. Plot the final contour
python -m scripts.mismatch_mcz_td.create_contour_mcz_td_from_best_match \
  --results_dir ./data/contours_td_mcz \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 16 --mcz_max 25 \
  --orientation_tag Taman_edgeon
```

### Plotting
- **Final contour**: Use `python -m scripts.mismatch_mcz_td.create_contour_mcz_td_from_best_match`
- **Per-cube inspection**: Use the helper scripts under `scripts/mismatch_mcz_td/` for slices, sweeps, and interactive cube inspection
- **Workflow guide**: See `docs/CONTOUR_TD_MCZ_PIPELINE_GUIDE.md` for the current stage-by-stage reference

## Authors
* Tien Nguyen
* Tamanjyot Singh
* Michael Kesden
* Lindsay King

## Acknowledgement
This work is supported by the TEXAS Bridge Program 2023-2024 as a collaboration between the University of Texas at Dallas and the University of North Texas and funded by the NSF PAARE grant AST-2219128.
