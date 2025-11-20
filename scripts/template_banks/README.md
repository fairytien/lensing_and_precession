# Template Banks

Scripts for generating and managing precessing template banks.

## Scripts

-   `build_template_banks.py`: The main script to build template banks across an `mcz` grid. Saves results as HDF5. Supports SLURM array chunking.
-   `template_bank_npz.py`: Likely a utility for handling NPZ format banks (legacy or specific use case).

## Usage Example

```bash
python build_template_banks.py --mcz_min 10 --mcz_max 90 --bank_dir ../../data/template_banks
```

