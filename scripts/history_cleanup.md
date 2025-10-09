# Repository History Cleanup (Optional)

This document outlines two approaches to purge already-committed large binary pickle blobs from git history after migrating to Git LFS pointers.

## When to Perform a Rewrite
Do a history cleanup only if:
- Clone/fetch size is problematic; or
- You accidentally committed large binaries before LFS and want a smaller repo footprint.

Skip rewriting history if others have forked/cloned and rebasing them would be painful.

## Summary of Approaches
| Tool | Pros | Cons |
|------|------|------|
| `git filter-repo` | Fast, robust, single pass, preserves tags by default | Requires local install (not bundled with Git) |
| BFG Repo-Cleaner | Very simple for pattern deletion | Less flexible for complex transformations |

## 1. Using `git filter-repo` (Recommended)

Install (macOS):
```bash
brew install git-filter-repo
```

Dry run (list size of largest paths first):
```bash
git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| awk '$1=="blob" {print $3, $4}' \
| sort -nr | head -20
```

Rewrite history to remove *all* previous pickle blobs while keeping current working tree (ensure working tree has LFS pointers first):
```bash
# Commit current state with LFS pointers BEFORE running this.
git filter-repo --path-glob '*.pkl' --invert-paths
```
This deletes every historical non-pointer pickle object. Your *current* commit (after the rewrite) will still have the pointer files because they are present in the working tree snapshot that filter-repo keeps.

Force push (DANGEROUS):
```bash
git push --force origin main
```

### Keeping Only the Largest Files Removed
If you want to purge only big (>100MB) legacy picks but leave small ones:
```bash
# Example: remove four specific large historical files
git filter-repo --path data/TACC/TACC_sys3_super_contour_mcz40_2024-07-29.pkl \
                --path data/TACC/TACC_sys2_super_contour_mcz40_2024-08-03.pkl \
                --path data/TACC/TACC_sys2_super_contour_mcz30_2024-08-03.pkl \
                --path data/TACC/TACC_sys2_super_contour_mcz20_2024-08-04.pkl \
                --invert-paths
```
(Repeat with any additional large paths.)

## 2. Using BFG Repo-Cleaner
Install:
```bash
brew install bfg
```
Remove all pickle history (but keep current pointers):
```bash
# Create a bare mirror clone (safer)
git clone --mirror <repo-url>
cd <repo>.git
bfg --delete-files '*.pkl'
# Cleanup and push
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```
Selective removal (only >100MB files) requires creating a file list and passing to BFG with `--delete-files` patterns; BFG works pattern-based so per-file granularity is clunkier.

## Post-Rewrite Sanity Checks
After either method:
```bash
git fsck --full
git lfs fsck
# Confirm largest non-LFS blobs are small now
git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| awk '$1=="blob" {print $3, $4}' | sort -nr | head -10
```

## Communication
Notify collaborators:
> History rewritten at <new commit sha>. Please re-clone or:
> git fetch origin && git reset --hard origin/main && git clean -fd

## Rollback Plan
Before rewrite create a tag:
```bash
git tag pre-history-cleanup-$(date +%Y%m%d)
git push origin pre-history-cleanup-$(date +%Y%m%d)
```
You can later recover any deleted object from that tag if needed.

---
Last updated: 2025-10-09
