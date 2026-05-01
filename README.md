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

## Documentation Guide

Repository documentation is centralized under `docs/`.
If you are new to the repository, read in this order:

- [AGENTS.md](AGENTS.md): Code style, architecture, naming conventions, runtime rules, and figure typography for human contributors and AI agents.
- [docs/SCRIPTS_PIPELINES_GUIDE.md](docs/SCRIPTS_PIPELINES_GUIDE.md): Canonical index of workflow folders and entry points.
- [docs/CONTOUR_MCZ_TD_PIPELINE_GUIDE.md](docs/CONTOUR_MCZ_TD_PIPELINE_GUIDE.md): Runbook for the production `(td, mcz)` mismatch pipeline.
- [docs/CONTOUR_I_TD_PIPELINE_GUIDE.md](docs/CONTOUR_I_TD_PIPELINE_GUIDE.md): Runbook for the production `(td, I)` mismatch pipeline at fixed `mcz`.
- [docs/HDF5_SCHEMA.md](docs/HDF5_SCHEMA.md): HDF5 metadata and dataset conventions.
- [docs/DATA_LFS.md](docs/DATA_LFS.md): Data layout, Git LFS workflow, checksums, and large-file safeguards.
- [docs/STOCKYARD.md](docs/STOCKYARD.md): Shared-storage workflow and symlink patterns on TACC.

## Authors
* Tien Nguyen
* Tamanjyot Singh
* Benjamin McKallip
* Michael Kesden
* Lindsay King

## Acknowledgement
This work is supported by the TEXAS Bridge Program 2023-2024 as a collaboration between the University of Texas at Dallas and the University of North Texas and funded by the NSF PAARE grant AST-2219128.
