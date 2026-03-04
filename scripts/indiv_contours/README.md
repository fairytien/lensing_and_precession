# Individual & OTF Contours

This directory contains scripts for computing individual contours "on-the-fly" (OTF) without generating full template banks or mismatch cubes first. These are useful for testing specific parameters or one-off investigations.

## Key Scripts

-   `v4_indiv_contour_otf.py`: Latest version for computing an individual mismatch contour for a specific set of parameters.
-   `v3_indiv_contour_otf.py`: Version 3 OTF contour computation.
-   `v3_indiv_contour_otf_v2prec.py`: Version 3 OTF with v2 precession class.
-   `v2_indiv_contour_otf.py`: Version 2 OTF contour computation.
-   `v3_indiv_contour_from_bank.py`: Version 3 individual contour using precomputed banks.
-   `indiv_contour.py`: General script for individual contours.
-   `indiv_contour_test.py`: Test script for individual contours.

## Usage

Most scripts here take command line arguments for source parameters (`mcz`, `td`, angles). Use `--help` on specific scripts for details.
