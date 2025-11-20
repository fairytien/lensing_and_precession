# Super Contours Pipeline

This pipeline contains scripts for generating "super contours" and multi-dimensional contour sweeps. These scripts compute mismatch contours across multiple parameter dimensions (e.g., sweeping over `mcz`, `td`, `I`, or combinations thereof).

## Scripts

-   `super_contour_L_NP.py`: Generates super contours for Lensed sources vs Non-Precessing templates.
-   `super_contour_cli.py`: CLI interface for generating super contours.
-   `v3_super_contour_from_bank.py`: Version 3 super contour generation using precomputed template banks.
-   `contour_L_NP_mcz_td.py`: Mismatch contour (Lensed vs NP templates) over `mcz` and time delay.
-   `contour_L_RP_mcz_td_otf.py`: Mismatch contour (Lensed vs RP templates) over `mcz` and time delay, computed on-the-fly.
-   `contours_mcz.py`: Sweeps over `mcz` values, generating a contour for each.
-   `contours_td.py`: Sweeps over time delay values, generating a contour for each.
-   `contours_I.py`: Sweeps over flux ratio `I` values, generating a contour for each.
-   `v3_contours_mcz_from_bank.py`: Version 3 `mcz` sweep using precomputed template banks.

## Usage

Most scripts take command line arguments for parameter ranges. Use `--help` on specific scripts for details.

