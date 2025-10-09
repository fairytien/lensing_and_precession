#!/usr/bin/env bash
set -euo pipefail

# Automate Git LFS enablement and conversion for *.pkl files.
# Safe to re-run; idempotent except for creating new commits when needed.

if ! command -v git >/dev/null 2>&1; then
	echo "git not found" >&2; exit 1; fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	echo "Not inside a git repository" >&2; exit 1; fi

echo "==> Installing git lfs (if not already)"
if command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1; then
	echo "git-lfs already installed"
else
	if command -v brew >/dev/null 2>&1; then
		brew install git-lfs
	else
		echo "Please install git-lfs manually (no Homebrew detected)." >&2; exit 1;
	fi
fi

echo "==> Initializing Git LFS"
git lfs install --skip-repo || true

echo "==> Ensuring .gitattributes has '*.pkl' tracking rule"
if ! grep -q "^\*\.pkl" .gitattributes 2>/dev/null; then
	echo "*.pkl filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
	git add .gitattributes
fi

echo "==> Staging pickle files for LFS pointer replacement"
# Re-add all *.pkl so LFS smudge/clean filters process them.
find data -type f -name '*.pkl' -print0 2>/dev/null | xargs -0 git add || true

changed=$(git diff --cached --name-only)
if [ -z "$changed" ]; then
	echo "No staged changes (pickle files already LFS pointers)."
else
	echo "==> Committing LFS pointer updates"
	git commit -m "chore: convert *.pkl to Git LFS pointers"
fi

echo "==> Summary of LFS tracked patterns"
git lfs track

echo "==> Largest remaining unstaged pickle files (if any)"
# Show sizes for awareness
if command -v python3 >/dev/null 2>&1; then
python3 - <<'PY'
import os, pathlib
sizes=[]
for root,_,files in os.walk('data'):
	for f in files:
		if f.endswith('.pkl'):
			p=pathlib.Path(root,f)
			sizes.append((p.stat().st_size,p))
if not sizes:
	print('No pickle files found.')
else:
	sizes.sort(reverse=True)
	for sz,p in sizes[:10]:
		print(f"{sz/1024/1024:8.2f} MB\t{p}")
PY
fi

echo "==> Done. Use 'git push' to upload (LFS files go to LFS storage)."
