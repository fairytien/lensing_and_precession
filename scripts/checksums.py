#!/usr/bin/env python3
"""Generate or verify a checksum manifest for data pickle files.

Usage:
  python3 scripts/checksums.py generate   # writes checksums/manifest.sha256
  python3 scripts/checksums.py verify     # verifies against existing manifest
  python3 scripts/checksums.py changed    # lists files whose current hash differs from manifest

Notes:
  * Only includes *.pkl by default. Use --glob to override (e.g. '--glob "*.pkl"').
  * Manifest records relative paths from repo root plus SHA256 digest.
  * Designed to run after LFS smudge; it hashes local content (pointer files are tiny; real content is in LFS). For integrity of large binary content post-clone, ensure `git lfs pull` executed; otherwise pointer text hashes will appear.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / 'checksums'
MANIFEST_PATH = MANIFEST_DIR / 'manifest.sha256'


def iter_files(glob: str):
    for p in REPO_ROOT.glob(f'data/**/{glob}'):
        if p.is_file():
            yield p

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_manifest():
    if not MANIFEST_PATH.exists():
        return {}
    data = {}
    with MANIFEST_PATH.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            digest, rel = line.split(maxsplit=1)
            data[rel] = digest
    return data

def write_manifest(records: dict[str,str]):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open('w') as f:
        for rel,d in sorted(records.items()):
            f.write(f"{d} {rel}\n")
    print(f"Wrote {len(records)} entries -> {MANIFEST_PATH}")

def generate(glob: str):
    records = {}
    for p in iter_files(glob):
        rel = p.relative_to(REPO_ROOT).as_posix()
        records[rel] = sha256_file(p)
    write_manifest(records)

def verify(glob: str):
    manifest = load_manifest()
    missing = []
    mismatched = []
    checked = 0
    for p in iter_files(glob):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel not in manifest:
            missing.append(rel)
            continue
        digest = sha256_file(p)
        if digest != manifest[rel]:
            mismatched.append(rel)
        checked += 1
    extra = sorted(set(manifest) - {p.relative_to(REPO_ROOT).as_posix() for p in iter_files(glob)})
    print(f"Checked entries: {checked}")
    if missing:
        print(f"Missing in manifest ({len(missing)}):")
        for m in missing: print('  +', m)
    if mismatched:
        print(f"Hash mismatches ({len(mismatched)}):")
        for m in mismatched: print('  !', m)
    if extra:
        print(f"Manifest lists files no longer present ({len(extra)}):")
        for e in extra: print('  -', e)
    if not (missing or mismatched or extra):
        print("All files match manifest.")

def changed(glob: str):
    manifest = load_manifest()
    current = {}
    for p in iter_files(glob):
        rel = p.relative_to(REPO_ROOT).as_posix()
        current[rel] = sha256_file(p)
    changes = []
    for rel,d in current.items():
        if rel not in manifest:
            changes.append((rel,'NEW'))
        elif manifest[rel] != d:
            changes.append((rel,'MODIFIED'))
    removed = sorted(set(manifest)-set(current))
    for r in removed:
        changes.append((r,'REMOVED'))
    if not changes:
        print('No changes detected relative to manifest.')
    else:
        for rel,status in changes:
            print(f"{status}: {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['generate','verify','changed'])
    ap.add_argument('--glob', default='*.pkl', help='Filename glob under data/** to include (default: *.pkl)')
    args = ap.parse_args()
    if args.command == 'generate':
        generate(args.glob)
    elif args.command == 'verify':
        verify(args.glob)
    elif args.command == 'changed':
        changed(args.glob)

if __name__ == '__main__':
    main()
