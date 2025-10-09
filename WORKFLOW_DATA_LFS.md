# Data & Large File Workflow Guide

This guide explains how to use the repository's data organization, Git LFS integration, checksum integrity tooling, and size guard to manage large analysis artifacts.

## Overview

| Task | Tool / File | Purpose |
|------|-------------|---------|
| Track large pickle outputs | `.gitattributes` (`*.pkl` rule) | Store large binary content via Git LFS (lightweight pointers in Git) |
| Organize new result files | `scripts/organize_data.py` | Classify raw output files into `TACC/`, `super_contours/`, `indiv_contours/` deterministically |
| Automate LFS enablement | `scripts/setup_lfs.sh` | Ensure LFS installed, track `*.pkl`, convert staged pickles to LFS pointers |
| Prevent oversized mistakes | `scripts/precommit_size_guard.sh` + install script | Block accidental commit of large non-LFS blobs |
| Integrity & drift detection | `scripts/checksums.py` | Generate / verify SHA256 manifest of data pickles |
| Optional deep cleanup | `scripts/history_cleanup.md` | Instructions for rewriting history to expunge historical large blobs |

## 1. Clone / Initial Setup
```bash
# Standard clone
git clone https://github.com/fairytien/lensing_and_precession.git
cd lensing_and_precession

# Install Git LFS once per machine
git lfs install

# Pull any large objects (usually automatic after checkout; explicit for safety)
git lfs pull
```

## 2. Producing New Data Files
Drop raw newly generated `.pkl` result files into `data/` (root) or let scripts write them there. Then run the organizer:
```bash
python3 scripts/organize_data.py --dry-run   # Preview moves
python3 scripts/organize_data.py            # Apply moves
```
Classification priority (first match wins):
1. `TACC_`
2. `indiv_contour`
3. `super_contour`
4. `mismatch_contour` / `mismatch_contours`
5. plural `contours`
6. `mismatch_` dictionaries

## 3. Ensuring Files Are Tracked by LFS
The repo already includes a wildcard rule:
```
*.pkl filter=lfs diff=lfs merge=lfs -text
```
To be sure new large pickles are pointers:
```bash
git add path/to/new.pkl
git commit -m "feat: add new contour result"
# Inspect pointer
head -n 3 path/to/new.pkl
```
If it is a proper pointer, the first line starts with `version https://git-lfs.github.com/spec/v1`.

If you ever add a *new* large type (e.g. notebooks):
```bash
git lfs track "*.ipynb"
git add .gitattributes
git commit -m "chore: track notebooks in LFS"
```

## 4. Automating (If LFS Not Yet Active Locally)
Run the helper script (idempotent):
```bash
bash scripts/setup_lfs.sh
```
This installs (if possible), ensures the pattern, re-adds pickles to convert them to pointers, and commits if needed.

## 5. Preventing Accidental Large Commits (Size Guard)
Install the pre-commit hook:
```bash
bash scripts/install_precommit_hook.sh
```
Default limit: 5 MB for non-LFS files. Override temporarily:
```bash
export MAX_SIZE_MB=15  # for this shell session
```
A failed commit prints which files exceeded the limit and suggests tracking them with LFS or reducing size.

## 6. Integrity: Checksums
Generate (or refresh) a baseline manifest:
```bash
python3 scripts/checksums.py generate
# Commit if you intentionally updated data set
```
Verify integrity later:
```bash
python3 scripts/checksums.py verify
```
See what changed relative to the manifest:
```bash
python3 scripts/checksums.py changed
```
The manifest stores relative path + SHA256 digest. (Hashes reflect current on-disk content; ensure `git lfs pull` before trusting them on a fresh clone.)

## 7. Typical Contribution Flow
```bash
# Produce data locally
python3 some_pipeline_script.py

# Organize
python3 scripts/organize_data.py --dry-run
python3 scripts/organize_data.py

# (Optional) Update checksum baseline if adding authoritative results
python3 scripts/checksums.py generate

git add data/updated.pkl checksums/manifest.sha256

git commit -m "feat: add updated mcz40 results"
git push origin main
```

## 8. Listing Large Objects & LFS Status
```bash
git lfs ls-files          # list pointer files
# Quick largest working-tree pickle sizes
du -h data/**/*.pkl 2>/dev/null | sort -h | tail
```

## 9. If Push Size Explodes Again
- Confirm new large file actually became a pointer (inspect first line). If not, re-add after ensuring rule exists.
- Avoid committing raw large binaries outside configured patterns.
- Run `bash scripts/setup_lfs.sh` to re-normalize.

## 10. (Optional) Full History Cleanup
If future repository bloat becomes problematic, see:
```
scripts/history_cleanup.md
```
You decided *not* to rewrite history now. This doc contains filter-repo & BFG recipes should the decision change.

## 11. Recovery & Safety
- Pre-clean tag (if one created) allows comparing old vs new state.
- LFS pointers are safe to merge; conflicts are rare unless editing pointer text.
- If a pointer file becomes corrupted locally, checkout from git and re-run `git lfs pull`.

## 12. FAQ
**Q: Why are pointer files tiny but checksum manifest lists big sizes?**  
Checksums hash the actual binary content present locally after LFS smudge. If you generated manifest before pulling real content, you'd hash pointer text instead—always ensure `git lfs pull` first.

**Q: Can I exclude experimental huge runs?**  
Yes—add them to `.gitignore` or store externally; do not track via Git if not needed for reproducibility.

**Q: How to raise default size guard permanently?**  
Edit `MAX_SIZE_MB` default inside `scripts/precommit_size_guard.sh` or export it in your shell profile.

---
_Last updated: 2025-10-09_
