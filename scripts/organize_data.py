#!/usr/bin/env python3
"""Safely classify new pickle outputs into data subfolders.

Usage:
  python3 scripts/organize_data.py --dry-run
  python3 scripts/organize_data.py

Logic priority (first rule that matches a filename wins):
  1. TACC_ -> data/TACC/
  2. indiv_contour -> data/indiv_contours/
  3. super_contour -> data/super_contours/
  4. mismatch_contour / mismatch_contours -> data/super_contours/
  5. (remaining) filenames containing 'contours' (plural) -> data/super_contours/
  6. mismatch_ dictionaries (mismatch_*_dict) -> data/super_contours/

Safeguards:
  * Skips files already in a categorized directory.
  * Ignores non-.pkl files by default (override with --all).
  * Dry-run mode prints planned moves only.
  * Avoids descending into version control dirs.

"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGETS = {
    "TACC": DATA_DIR / "TACC",
    "super": DATA_DIR / "super_contours",
    "indiv": DATA_DIR / "indiv_contours",
}

PRIORITY_RULES = [
    (lambda n: "TACC_" in n, TARGETS["TACC"]),
    (lambda n: "indiv_contour" in n, TARGETS["indiv"]),
    (lambda n: "super_contour" in n, TARGETS["super"]),
    (lambda n: "mismatch_contour" in n, TARGETS["super"]),
    (lambda n: "contours" in n, TARGETS["super"]),  # plural catch-all
    (lambda n: n.startswith("mismatch_") and n.endswith(".pkl"), TARGETS["super"]),
]

CATEGORIZED_DIR_NAMES = {p.name for p in TARGETS.values()} | {"contours"}


def classify(path: Path) -> Path | None:
    name = path.name
    for predicate, dest in PRIORITY_RULES:
        if predicate(name):
            return dest
    return None


def is_categorized(path: Path) -> bool:
    # Already inside one of the target dirs or the placeholder contours directory
    try:
        rel = path.relative_to(DATA_DIR)
    except ValueError:
        return True  # outside data dir
    return rel.parts[0] in CATEGORIZED_DIR_NAMES


def scan_files(include_all: bool) -> list[Path]:
    files = []
    for child in DATA_DIR.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            # Only inspect top-level loose files; skip subdirs (they are already organized)
            continue
        if not include_all and child.suffix != ".pkl":
            continue
        files.append(child)
    return files


def plan_moves(files: list[Path]):
    plan = []
    for f in files:
        if is_categorized(f):
            continue
        dest_dir = classify(f)
        if dest_dir and dest_dir != f.parent:
            plan.append((f, dest_dir / f.name))
    return plan


def ensure_dirs():
    for d in TARGETS.values():
        d.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Organize data pickle files into taxonomy folders."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be moved."
    )
    parser.add_argument(
        "--all", action="store_true", help="Consider non-.pkl files as well."
    )
    args = parser.parse_args()

    ensure_dirs()
    files = scan_files(include_all=args.all)
    moves = plan_moves(files)

    if not moves:
        print("No files need moving.")
        return

    for src, dst in moves:
        print(f"MOVE: {src.relative_to(DATA_DIR)} -> {dst.relative_to(DATA_DIR)}")
    if args.dry_run:
        print(f"Planned {len(moves)} move(s). Dry-run only.")
        return

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"Executed {len(moves)} move(s).")


if __name__ == "__main__":
    main()
