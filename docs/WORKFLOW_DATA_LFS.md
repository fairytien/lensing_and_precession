# Data and Large File Workflow Guide

This document is the canonical guide for managing generated outputs under `data/` and large binary artifacts tracked with Git LFS.

It consolidates the previous `data/README.md` and `lfs/README.md` material into one reference for collaborators.

## 1. Data Layout

Generated analysis artifacts are organized under `data/`.

### Canonical subdirectories

- `data/TACC/`
  - Outputs produced on TACC, commonly with `TACC_` in the filename.
- `data/super_contours/`
  - Super contour and multi-system aggregate outputs.
- `data/indiv_contours/`
  - Individual target/system contour outputs (`indiv_contour`, `indiv_mismatch`, and versioned variants).
- `data/contours/`
  - Reserved placeholder directory for future derived contour products.

### Typical filename tokens

Common underscore-delimited tokens include:

- `mcz20`, `mcz30`, `mcz40` for chirp-mass configurations
- `L_NP`, `L_RP` for model pair tags
- `td22ms`, `td10_100`, `td0.03` for time-delay settings
- `I0.5`, `I0.6` for flux-ratio settings
- `res_omega101_theta401`, `res_51x51` for grid resolution
- `v2_`, `v3_`, `v4_` for experiment/version series

## 2. LFS Toolkit Overview

Scripts in `lfs/` are the standard tooling for large-file workflow:

| Tool | Purpose |
|---|---|
| `lfs/setup_lfs.sh` | One-shot setup and pointer normalization for tracked files. |
| `lfs/install_precommit_hook.sh` | Installs the non-LFS size guard hook. |
| `lfs/precommit_size_guard.sh` | Prevents oversized files from being committed outside LFS rules. |
| `lfs/organize_data.py` | Classifies/moves output files into canonical `data/` folders. |
| `lfs/organize_data.sh` | Convenience wrapper for organizer execution from repo root. |
| `lfs/checksums.py` | Generate, verify, and diff SHA256 manifests for tracked outputs. |
| `lfs/history_cleanup.md` | Optional history rewrite procedures if repository bloat must be removed. |

## 3. First-Time Setup (Per Machine)

From repository root:

```bash
git lfs install
git lfs pull
bash lfs/install_precommit_hook.sh
```

Optional one-shot normalization:

```bash
bash lfs/setup_lfs.sh
```

## 4. Daily Workflow for New Results

1. Generate data files (usually `.pkl` and `.h5`) from pipeline scripts.
2. If outputs land in top-level `data/`, classify them:

```bash
python3 lfs/organize_data.py --dry-run
python3 lfs/organize_data.py
```

3. Stage files and confirm large pickle files are LFS pointers:

```bash
git add data/path/to/new_file.pkl
head -n 3 data/path/to/new_file.pkl
```

The first line should be:

`version https://git-lfs.github.com/spec/v1`

4. Update checksum manifest when adding or replacing authoritative outputs:

```bash
python3 lfs/checksums.py generate
git add lfs/checksums/manifest.sha256
```

5. Commit and push normally.

## 5. Organizer Classification Rules

`lfs/organize_data.py` uses first-match priority:

1. `TACC_`
2. `indiv_contour` or `indiv_mismatch`
3. `super_contour`
4. `mismatch_contour` or `mismatch_contours`
5. plural `contours`
6. `mismatch_` dictionary-style outputs

Notes:

- The organizer skips files already inside classified destination folders.
- Destination folders are never moved by the organizer.

## 6. LFS Tracking Rules and Checks

This repository tracks pickle files with:

```gitattributes
*.pkl filter=lfs diff=lfs merge=lfs -text
```

Useful checks:

```bash
git lfs ls-files
du -h data/**/*.pkl 2>/dev/null | sort -h | tail
```

To add another large extension in the future:

```bash
git lfs track "*.ipynb"
git add .gitattributes
git commit -m "chore: track notebooks with LFS"
```

## 7. Integrity and Drift Detection

Checksum commands:

```bash
python3 lfs/checksums.py generate
python3 lfs/checksums.py verify
python3 lfs/checksums.py changed
```

Guidance:

- Run `git lfs pull` before trusting verification on a fresh clone.
- Commit manifest updates only when data changes are intentional and part of the contribution.

## 8. Size Guard for Non-LFS Files

Install once:

```bash
bash lfs/install_precommit_hook.sh
```

Default non-LFS size threshold is 5 MB. Override for the current shell session if needed:

```bash
export MAX_SIZE_MB=15
```

## 9. Reproducibility and Portability Notes

- Pickle files can be Python-version sensitive.
- For long-term portability, consider exporting summary metadata alongside pickles (CSV, JSON, or HDF5 metadata tables).
- Keep HDF5 schema expectations aligned with `docs/HDF5_SCHEMA_V1.md`.

## 10. Shared Filesystem Workflow (TACC)

For shared canonical storage and symlink-based access on STOCKYARD, follow:

- `docs/WORKFLOW_STOCKYARD.md`

## 11. Troubleshooting

If push size unexpectedly grows:

1. Confirm large files are pointers (`head -n 3` check).
2. Re-run `bash lfs/setup_lfs.sh`.
3. Ensure new large formats are tracked in `.gitattributes`.

If a pointer appears corrupted locally:

```bash
git checkout -- data/path/to/file.pkl
git lfs pull
```

## 12. Optional History Rewrite

If historical repository bloat must be reduced, use:

- `lfs/history_cleanup.md`

History rewrites are disruptive and should be coordinated with all collaborators before execution.

---
Last updated: 2026-04-03
