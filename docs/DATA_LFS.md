# Data and Large File Workflow

Use this guide for generated outputs under `data/` and large binary artifacts tracked with Git LFS.

## 1. Data Layout

Generated artifacts are organized under `data/`.

### Per-run artifacts

Per-run artifacts live in canonical run directories at the top of `data/`, named with the full sweep tokens. Examples:

- `data/mismatch_I0p5_z1_mcz5-45_td20-70_Taman_faceon/` (`mcz_td` pipeline)
- `data/mismatch_z1_mcz15_I0p1-0p9_td20-70_Taman_edgeon/` (`I_td` pipeline)

See [AGENTS.md — Output Filename Order](../AGENTS.md#output-filename-order) for the token grammar.

### Cross-run aggregates and grouped subdirectories

| Directory | Contents |
|---|---|
| `data/contour_mcz_td/` | Final `(td, mcz)` contour HDF5/PDF artifacts and super-contour aggregates |
| `data/contour_I_td/` | Final `(td, I)` contour HDF5 artifacts |
| `data/contour_omega_theta/` | Individual omega-theta contour outputs |
| `data/run_logs/` | Per-pipeline run-log CSVs (see [`data/run_logs/CSV_RULES.md`](../data/run_logs/CSV_RULES.md)) |

`figures/` mirrors the same shape: top-level run-dirs (`figures/mismatch_*`) for per-run plots, plus pipeline-grouped subdirs (`contour_mcz_td/`, `contour_I_td/`, `contour_omega_theta/`, `lindblom/`, `mismatch_cubes/`, `mismatch_cubes_mcz_sweep/`, `modality_nlens/`, `waveforms/`, `utils/`).

### Legacy artifacts

Pre-canonical data outputs and figures retained for reference live under `legacy/`:

- `legacy/data/` — pre-canonical data outputs (including `legacy/data/TACC/` for TACC-produced pickle outputs)
- `legacy/figures/` — pre-canonical figures

### Common filename tokens

`mcz20`, `mcz30`, `mcz40` · `L_NP`, `L_RP` · `td22ms`, `td10_100` · `I0.5`, `I0.6` · `res_omega101_theta401` · `v2_`, `v3_`, `v4_`

## 2. LFS Toolkit

Scripts in `lfs/` are the standard tooling for large-file workflow:

| Tool | Purpose |
|---|---|
| `lfs/setup_lfs.sh` | One-shot setup and pointer normalization for tracked files |
| `lfs/install_precommit_hook.sh` | Installs the non-LFS size guard hook |
| `lfs/precommit_size_guard.sh` | Prevents oversized files from being committed outside LFS rules |
| `lfs/organize_data.py` | Classifies/moves output files into canonical `data/` folders |
| `lfs/organize_data.sh` | Convenience wrapper for organizer execution from repo root |
| `lfs/checksums.py` | Generate, verify, and diff SHA256 manifests for tracked outputs |

## 3. First-Time Setup (Per Machine)

```bash
git lfs install
git lfs pull
bash lfs/install_precommit_hook.sh
bash lfs/setup_lfs.sh    # optional one-shot normalization
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
# First line should be: version https://git-lfs.github.com/spec/v1
```

4. Update checksum manifest when adding or replacing authoritative outputs:

```bash
python3 lfs/checksums.py generate
git add lfs/checksums/manifest.sha256
```

5. Commit and push normally.

## 5. Organizer Classification Rules

`lfs/organize_data.py` uses first-match priority:

1. `TACC_` → `legacy/data/TACC/`
2. `indiv_contour` or `indiv_mismatch` → `data/contour_omega_theta/`
3. `super_contour` → `data/contour_mcz_td/`
4. `mismatch_contour` or `mismatch_contours` → `data/contour_mcz_td/`
5. Remaining filenames containing plural `contours` → `data/contour_mcz_td/`
6. `mismatch_*.pkl` dictionary-style outputs → `data/contour_mcz_td/`

The organizer skips files already inside classified destination folders and never moves destination folders themselves.

## 6. LFS Tracking Rules and Checks

```gitattributes
*.pkl filter=lfs diff=lfs merge=lfs -text
```

```bash
git lfs ls-files
du -h data/**/*.pkl 2>/dev/null | sort -h | tail
```

To track a new large extension:

```bash
git lfs track "*.ipynb"
git add .gitattributes
git commit -m "chore: track notebooks with LFS"
```

## 7. Integrity and Drift Detection

```bash
python3 lfs/checksums.py generate
python3 lfs/checksums.py verify
python3 lfs/checksums.py changed
```

Run `git lfs pull` before verifying on a fresh clone. Commit manifest updates only when data changes are intentional.

## 8. Size Guard for Non-LFS Files

```bash
bash lfs/install_precommit_hook.sh
```

Default non-LFS size threshold is 5 MB. Override for the current shell session:

```bash
export MAX_SIZE_MB=15
```

## 9. Reproducibility Notes

- Pickle files can be Python-version sensitive.
- For long-term portability, export summary metadata alongside pickles (CSV, JSON, or HDF5).
- Keep HDF5 schema expectations aligned with `docs/HDF5_SCHEMA.md`.

## 10. Shared Filesystem Workflow (TACC)

For shared canonical storage on STOCKYARD and symlink-based access, see [STOCKYARD.md](STOCKYARD.md).

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

## 12. Repository History Cleanup (Optional)

Purge already-committed large binary blobs from git history only if:
- Clone/fetch size is problematic, or
- Large binaries were committed before LFS and you want a smaller repo footprint.

Skip if others have cloned/forked — rebasing them is disruptive. Coordinate with all collaborators first.

### Approaches

| Tool | Pros | Cons |
|---|---|---|
| `git filter-repo` | Fast, robust, preserves tags | Requires local install |
| BFG Repo-Cleaner | Very simple for pattern deletion | Less flexible |

### Before you start

Create a recovery tag:

```bash
git tag pre-history-cleanup-$(date +%Y%m%d)
git push origin pre-history-cleanup-$(date +%Y%m%d)
```

### Using `git filter-repo` (Recommended)

```bash
brew install git-filter-repo
```

Dry run — list largest historical blobs:

```bash
git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| awk '$1=="blob" {print $3, $4}' \
| sort -nr | head -20
```

Remove all `*.pkl` history (commit LFS pointers first):

```bash
git filter-repo --path-glob '*.pkl' --invert-paths
git push --force origin main
```

Remove specific large files only:

```bash
git filter-repo \
  --path legacy/data/TACC/TACC_sys3_super_contour_mcz40_2024-07-29.pkl \
  --path legacy/data/TACC/TACC_sys2_super_contour_mcz40_2024-08-03.pkl \
  --invert-paths
git push --force origin main
```

### Using BFG Repo-Cleaner

```bash
brew install bfg
git clone --mirror <repo-url>
cd <repo>.git
bfg --delete-files '*.pkl'
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```

### Post-rewrite sanity checks

```bash
git fsck --full
git lfs fsck
git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| awk '$1=="blob" {print $3, $4}' | sort -nr | head -10
```

### Notify collaborators

> History rewritten at `<new commit sha>`. Please re-clone or:
> `git fetch origin && git reset --hard origin/main && git clean -fd`
