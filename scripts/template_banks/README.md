# Template Banks

Scripts for generating and managing precessing template banks.

Redshift support:
- `--z` is supported in `build_template_banks.py`.
- Effective detector-frame chirp mass used for waveform generation is
	`mcz_det = mcz * (1 + z)`.
- Bank filenames include `_z...` when `z != 0`.

## Scripts

-   `build_template_banks.py`: The main script to build template banks across an `mcz` grid. Saves results as HDF5. Supports SLURM array chunking.
-   `template_bank_npz.py`: A deprecated utility for handling NPZ format banks (legacy or specific use case).

## Usage Example

```bash
python build_template_banks.py --mcz_min 10 --mcz_max 90 --z 0.5 --bank_dir ../../data/template_banks
```

