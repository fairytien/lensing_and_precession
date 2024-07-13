# Distinguishing regularly precessing and lensed gravitational waveforms

## Description
Gravitational waves (GWs) from binary black hole (BBH) inspirals are affected by the black hole spins and orbital angular momentum, which, when misaligned, cause precession and nutation and introduce modulations in GW amplitudes and phases. In regular precession (without transitional precession or nutation), the total angular momentum has nearly constant direction and the orbital angular momentum precesses on a cone whose opening angle and frequency slowly increase on the radiation-reaction timescale. Regularly precessing BBH systems include those with a single spin, equal masses, or those trapped in spin-orbit resonances.

On the other hand, GWs can also be lensed by massive objects along the line of sight, resulting in amplification, potentially multiple images, and modulation of GWs. GWs are analyzed in the wave-optics regime and geometrical-optics regime depending on the mass of the lens and the wavelength. In axisymmetric lens models such as the point mass and singular isothermal sphere, the gravitational waveform can be described by the lens mass and the source position relative to the optic axis.

We investigate various parameters governing regular precession, including the precession amplitude, frequency, and the initial precessing phase, and lensing parameters, such as the lens mass and source position, to identify scenarios where the resulting waveforms may appear indistinguishable. The source’s chirp mass inversely correlates with the innermost stable circular orbit frequency cutoff and the inspiral waveform duration in the frequency band. At high chirp masses, waveforms may lack distinctive features, thus simplifying waveform matching. Through parameter tuning, a parameter space can be identified where the secular, oscillatory regularly precessing waveform aligns with the purely oscillatory lensed one. In addition, analytical approximations can predict the mismatch behavior between the lensed source and the regularly precessing template, as a function of the source’s chirp mass, which further elucidates the contribution of BBHs’ regular precession to waveform ambiguity.

Employing match-filtering analysis and various `PyCBC` packages, we quantify the mismatch and apply the Lindblom criterion to establish discernibility conditions for waveforms. Our study explores the parameter space to understand waveform distinguishability between regular precession and lensing, offering insights into the signal-to-noise requirement for GW detectors to effectively discern these waveforms.

## Getting started
This project requires the installation of [`lalsuite`](https://pypi.org/project/lalsuite/)
```
python -m pip install lalsuite
```
and [`PyCBC`](https://pycbc.org)
```
python -m pip install PyCBC
```

## Authors
* Tien Nguyen
* Tamanjyot Singh
* Michael Kesden
* Lindsay King

## Acknowledgement
This work is supported by the TEXAS Bridge Program 2023-2024 as a collaboration between the University of Texas at Dallas and the University of North Texas and funded by the NSF PAARE grant AST-2219128.
