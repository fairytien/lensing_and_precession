# Data Directory

This directory stores generated analysis artifacts (pickle files) grouped into three main categories plus a placeholder:

- `TACC/` : Results produced on TACC (naming contains `TACC_`). These often include large super contour aggregates.
- `super_contours/` : Any file whose name includes `super_contour`, `mismatch_contour`, `mismatch_contours`, plain `contours` plural, or `mismatch_` dictionary style outputs not tagged as indiv.
- `indiv_contours/` : Files with `indiv_contour` (single target/system runs) and related versioned variants (`v2_`, `v3_`, `v4_` etc.).
- `contours/` : Currently empty placeholder (kept per user request; can later store derived visualizations or moved/archived raw contour grids).

## File Naming Conventions
Typical components (underscore-delimited):

`[prefix]_[type]_[system]_[mczXX]_[extra modifiers]_[timestamp].pkl`

Common tokens:
- `mcz20`, `mcz30`, `mcz40`: Mass configuration index.
- `L_NP`, `L_RP`: Lens / precession model tags.
- `td22ms`, `td10_100`, `td0.03`: Time delay configuration.
- `I0.5`, `I0.6`: Inclination or intensity parameter.
- `res_omega101_theta401`, `res_51x51`: Grid resolutions.
- Version experiments: `v2_`, `v3_`, `v4_` prefix.

## Size & Version Control Strategy
Total pickle footprint ~1.5 GB (top four files individually ~300 MB). All `*.pkl` are now declared in `.gitattributes` for Git LFS tracking. To fully activate:

```bash
# One-time (per machine):
git lfs install
# Ensure pattern registered (already in repo):
cat .gitattributes
# Migrate existing history (optional, if large objects already committed):
#   Install git-filter-repo then run slimming commands (see below)
```

If you decide NOT to store binaries in Git at all, remove them and add `data/**/*.pkl` to `.gitignore` instead. (Currently we assume you want them tracked via LFS.)

## History Slimming (Optional)
If large pickle blobs already sit in prior commits, you can rewrite history:
```
# Install git-filter-repo (recommended over filter-branch)
brew install git-filter-repo  # macOS
# Example: prune historical large .pkl (after pushing LFS-tracked versions)
git filter-repo --path-glob '*.pkl' --invert-paths
# Force push rewritten main (DANGEROUS for collaborators)
git push --force origin main
```
Consider instead leaving history intact if collaborators have clones.

## Adding New Outputs
Generate into the top-level `data/` first or directly into correct subfolder. If unsure, drop into root and run the helper script below to reclassify safely without wildcard collisions.

```bash
python3 scripts/organize_data.py --dry-run   # preview
python3 scripts/organize_data.py            # execute
```

## Helper Script Behavior Summary
Priority order (first match wins):
1. `TACC_`
2. `indiv_contour`
3. `super_contour`
4. `mismatch_contour` / `mismatch_contours`
5. plural `contours` (but not if already matched above)
6. `mismatch_` dictionaries (e.g. `mismatch_I_dict`)

The script skips any file already inside a categorized directory, and avoids moving destination folders themselves.

## Reproducibility Notes
Pickle files are Python-version sensitive. For long-term archival, consider exporting lightweight metadata (JSON/CSV) or HDF5 arrays alongside pickles.

---
Last updated: 2025-10-09
