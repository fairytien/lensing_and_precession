# Scripts

This folder contains the analysis pipelines for the Lensing and Precession project. The scripts have been organized into the following subdirectories:

-   **`lindblom/`**: Scripts for the full Lindblom distinguishability study (Generation -> Aggregation -> Plotting).
-   **`mismatch_mcz_td/`**: Generic mismatch cube computation and visualization for varying $m_{cz}$ and $t_d$.
-   **`super_contours/`**: Scripts for generating super contours and multi-dimensional contour sweeps (e.g., over `mcz`, `td`, `I`).
-   **`template_banks/`**: Tools for building and managing precessing template banks.
-   **`individual_contours/`**: "On-the-fly" contour generation for specific parameter sets (no precomputed banks).
-   **`utils/`**: General utilities (file conversion, plotting helpers, comparison tools).

Please refer to the `README.md` in each subdirectory for detailed usage instructions.
