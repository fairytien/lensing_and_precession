#!/usr/bin/env bash
# Pre-commit size guard
# Blocks committing large files that are not tracked by Git LFS.
# Intended threshold defaults:
#   MAX_SIZE_MB: max allowed size for non-LFS files (default 5 MB)
#   LFS_PATTERNS: glob patterns considered OK if large (default '*.pkl')
# Install by running: bash lfs/install_precommit_hook.sh
set -euo pipefail

MAX_SIZE_MB=${MAX_SIZE_MB:-5}
# space-separated patterns
LFS_PATTERNS=${LFS_PATTERNS:-"*.pkl"}

max_bytes=$((MAX_SIZE_MB*1024*1024))

# Collect staged files (excluding deletions) that are added or modified
staged=$(git diff --cached --name-only --diff-filter=ACMRT | sed '/^$/d')
[ -z "$staged" ] && exit 0

fail=0

is_lfs_tracked() {
	local f="$1"
	# If file has an LFS pointer in index it starts with version https://git-lfs.github.com
	if git show :"$f" 2>/dev/null | head -n 1 | grep -q "^version https://git-lfs.github.com"; then
		return 0
	fi
	# Otherwise match against declared LFS_PATTERNS glob list
	for pat in $LFS_PATTERNS; do
		if [[ $f == $pat ]]; then
			return 0
		fi
	done
	return 1
}

while IFS= read -r f; do
	[ ! -f "$f" ] && continue
	size=$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f") || true
	if [ -n "$size" ] && [ "$size" -gt $max_bytes ]; then
		if ! is_lfs_tracked "$f"; then
			echo "[size-guard] Rejected: $f is $(printf '%.2f' "$(echo "$size/1048576" | bc -l)") MB (> ${MAX_SIZE_MB}MB) and not LFS-tracked." >&2
			fail=1
		fi
	fi
done <<<"$staged"

if [ $fail -eq 1 ]; then
	echo "[size-guard] Commit aborted. Either reduce file size, add to LFS (git lfs track), or raise MAX_SIZE_MB." >&2
	exit 1
fi
exit 0
