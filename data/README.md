# Data Directory

This directory stores generated analysis artifacts (pickle files) grouped into three main categories plus a placeholder:

- `TACC/` : Results produced on TACC (naming contains `TACC_`). These often include large super contour aggregates.
- `super_contours/` : Any file whose name includes `super_contour`, `mismatch_contour`, `mismatch_contours`, plain `contours` plural, or `mismatch_` dictionary style outputs not tagged as indiv.
- `indiv_contours/` : Files with `indiv_contour` or `indiv_mismatch` (single target/system runs) and related versioned variants (`v2_`, `v3_`, `v4_` etc.).
- `contours/` : Currently empty placeholder (kept per user request; can later store derived visualizations or moved/archived raw contour grids).

## File Naming Conventions
Typical components (underscore-delimited):

`[prefix]_[type]_[system]_[mczXX]_[extra modifiers]_[timestamp].pkl`

Common tokens:
- `mcz20`, `mcz30`, `mcz40`: Mass configuration index.
- `L_NP`, `L_RP`: Lens / precession model tags.
- `td22ms`, `td10_100`, `td0.03`: Time delay configuration.
- `I0.5`, `I0.6`: Flux ratio configuration.
- `res_omega101_theta401`, `res_51x51`: Grid resolutions.
- Version experiments: `v2_`, `v3_`, `v4_` prefix.

## Adding New Outputs
Generate into the top-level `data/` first or directly into correct subfolder. If unsure, drop into root and run the helper script to reclassify safely without wildcard collisions:

```bash
python3 lfs/organize_data.py --dry-run   # preview
python3 lfs/organize_data.py            # execute
```

### Organizing Script Classification
Priority order (first match wins):
1. `TACC_`
2. `indiv_contour` or `indiv_mismatch`
3. `super_contour`
4. `mismatch_contour` / `mismatch_contours`
5. plural `contours` (but not if already matched above)
6. `mismatch_` dictionaries (e.g. `mismatch_I_dict`)

The script skips any file already inside a categorized directory, and avoids moving destination folders themselves.

## Reproducibility Notes
Pickle files are Python-version sensitive. For long-term archival, consider exporting lightweight metadata (JSON/CSV) or HDF5 arrays alongside pickles.

## Version Control
All `*.pkl` files are tracked using Git LFS. For LFS setup, workflow, size guards, and history cleanup options, see:
- `../docs/WORKFLOW_DATA_LFS.md` - Complete data & LFS workflow guide
- `../lfs/README.md` - LFS toolkit documentation

---
Last updated: 2025-10-10
