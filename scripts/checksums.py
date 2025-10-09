#!/usr/bin/env python3
"""Compatibility wrapper: forward to lfs/checksums.py.

Usage remains the same; please prefer:
  python3 lfs/checksums.py <command>
"""
from __future__ import annotations
import runpy
from pathlib import Path

HERE = Path(__file__).resolve()
TARGET = HERE.parent.parent / "lfs" / "checksums.py"
runpy.run_path(str(TARGET), run_name="__main__")
