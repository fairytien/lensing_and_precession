# LFS Toolkit

This directory provides the canonical entry points for Git LFS and data workflow tools used in this repo.

Contents:
- `setup_lfs.sh` — LFS setup and pointer normalization
- `install_precommit_hook.sh` — installs size guard pre-commit hook
- `precommit_size_guard.sh` — the size guard itself
- `checksums.py` — generate/verify/changed checksum manifest
- `organize_data.py` — organize new outputs into data taxonomy
- `organize_data.sh` — convenience wrapper to run the organizer from repo root

See also:
- `history_cleanup.md` for optional full history rewrite guidance
- `../WORKFLOW_DATA_LFS.md` for end-to-end usage guide
