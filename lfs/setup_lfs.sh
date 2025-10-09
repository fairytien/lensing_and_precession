#!/usr/bin/env bash
set -euo pipefail

# Wrapper shims to keep backward compatibility if users still call scripts/setup_lfs.sh
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$DIR"/../scripts/setup_lfs.sh "$@"
