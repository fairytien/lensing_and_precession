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
1. **Build template banks**: Run `scripts/build_template_banks.py` to generate 4D RP template banks for each chirp mass
2. **For cluster/array jobs**: Use `batch_scripts/build_template_banks.sbatch` with SLURM array jobs

### Mismatch Computation
1. **Compute mismatches**: Run `scripts/contour_td_mcz_from_banks.py` to compute mismatches between lensed sources and template banks
2. **For cluster/array jobs**: Use `batch_scripts/contour_td_mcz_from_banks.sbatch` with SLURM array jobs
3. **⚠️ Important**: Each chunk writes a partial `best_match` file (only its slice of mcz values). These are NOT the final results.

### Aggregation (Required for Array Jobs/Multiple Chunks)
**For single chunk runs**: The `best_match` file from `contour_td_mcz_from_banks.py` is the final result.

**For multiple chunks/array jobs**: After all chunks complete, run `scripts/aggregate_best_match.py` once to:
- Combine all partial mismatch cubes into a single final `best_match` file
- Generate the final, complete contour plot across all mcz values

**Example workflow for array jobs/multiple chunks:**
```bash
# 1. Submit array job for mismatch computation (with --no_plot)
sbatch batch_scripts/contour_td_mcz_from_banks.sbatch

# 2. After all array tasks complete, aggregate results
python scripts/aggregate_best_match.py \
  --results_dir ./data/contours \
  --td_min_ms 20 --td_max_ms 70 \
  --mcz_min 10 --mcz_max 80 \
  --orientation_tag Taman_edgeon
```

### Plotting
- **Individual plots**: Use `scripts/plot_from_hdf5.py` on any `best_match_*.h5` file
- **Final plots**: Generated automatically by `aggregate_best_match.py` (unless `--no_plot` is used)

## Authors
* Tien Nguyen
* Tamanjyot Singh
* Michael Kesden
* Lindsay King

## Acknowledgement
This work is supported by the TEXAS Bridge Program 2023-2024 as a collaboration between the University of Texas at Dallas and the University of North Texas and funded by the NSF PAARE grant AST-2219128.
