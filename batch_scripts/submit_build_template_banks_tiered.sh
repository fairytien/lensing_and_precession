#!/bin/bash
#SBATCH -J submit_build_banks_tiered
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 00:05:00
#SBATCH -o batch_outputs/submit_build_template_banks_tiered_%j.out
set -euo pipefail

SCRIPT_PATH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_PATH_DIR}"

# Under sbatch, this script runs from a spool path; prefer the original submit dir.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  if [ -d "${SLURM_SUBMIT_DIR}/batch_scripts" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/batch_scripts"
  elif [ -f "${SLURM_SUBMIT_DIR}/build_template_banks.sbatch" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  fi
fi

JOB_SCRIPT="${JOB_SCRIPT:-$SCRIPT_DIR/build_template_banks.sbatch}"

if [ ! -f "$JOB_SCRIPT" ]; then
  echo "Error: build job script not found: $JOB_SCRIPT" >&2
  exit 2
fi

# Load default grid configuration.
source "$SCRIPT_DIR/_contour_mcz_td_config.sh"

# Data-driven tier configuration (override via env vars when needed).
JOB_SUMMARY="${JOB_SUMMARY:-$SCRIPT_DIR/../data/run_logs/job_summary_Taman_edgeon.csv}"
TIER1_UPPER_MCZ="${TIER1_UPPER_MCZ:-30}"
TIER2_UPPER_MCZ="${TIER2_UPPER_MCZ:-50}"
SAFETY_FACTOR="${SAFETY_FACTOR:-1.25}"
SAFETY_PAD_SEC="${SAFETY_PAD_SEC:-30}"
MIN_WALLTIME="${MIN_WALLTIME:-00:02:00}"
MAX_WALLTIME="${MAX_WALLTIME:-00:12:00}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_PARTITION="${SUBMIT_PARTITION:-${SLURM_JOB_PARTITION:-normal}}"

export JOB_SUMMARY TIER1_UPPER_MCZ TIER2_UPPER_MCZ SAFETY_FACTOR SAFETY_PAD_SEC MIN_WALLTIME MAX_WALLTIME

PLAN="$(python - <<'PY'
import csv
import math
import os
import numpy as np
from pathlib import Path

mcz_min = float(os.environ["MCZ_MIN"])
mcz_max = float(os.environ["MCZ_MAX"])
mcz_pts = int(os.environ["MCZ_PTS"])
if mcz_pts <= 0:
    raise SystemExit("MCZ_PTS must be > 0")

summary_path = Path(os.environ["JOB_SUMMARY"])
tier1_upper = float(os.environ["TIER1_UPPER_MCZ"])
tier2_upper = float(os.environ["TIER2_UPPER_MCZ"])
if tier2_upper <= tier1_upper:
  raise SystemExit("TIER2_UPPER_MCZ must be greater than TIER1_UPPER_MCZ")

safety_factor = float(os.environ["SAFETY_FACTOR"])
safety_pad_sec = float(os.environ["SAFETY_PAD_SEC"])
min_wall = os.environ["MIN_WALLTIME"]
max_wall = os.environ["MAX_WALLTIME"]

mcz_arr = np.linspace(mcz_min, mcz_max, mcz_pts, dtype=float)


def parse_hms_to_sec(value: str) -> float:
  parts = value.strip().split(":")
  if len(parts) != 3:
    raise ValueError(f"Invalid H:M:S format: {value}")
  h, m, s = parts
  return int(h) * 3600 + int(m) * 60 + float(s)


def sec_to_hms(seconds: int) -> str:
  h = seconds // 3600
  rem = seconds % 3600
  m = rem // 60
  s = rem % 60
  return f"{h:02d}:{m:02d}:{s:02d}"


def minutes_from_hms(value: str) -> int:
  sec = parse_hms_to_sec(value)
  return int(math.ceil(sec / 60.0))


def per_mcz_samples_from_row(row):
  total = (row.get("total_time") or "").strip()
  if not total:
    return []
  try:
    total_sec = parse_hms_to_sec(total)
  except Exception:
    return []

  mmin_s = (row.get("mcz_min") or "").strip()
  mmax_s = (row.get("mcz_max") or "").strip()
  m_s = (row.get("mcz") or "").strip()

  if mmin_s and mmax_s:
    mmin = float(mmin_s)
    mmax = float(mmax_s)
  elif m_s:
    mmin = mmax = float(m_s)
  else:
    return []

  if mmax < mmin:
    mmin, mmax = mmax, mmin

  span = mmax - mmin
  # Build runs historically used integer mcz grids. If this row covers a range,
  # distribute total time evenly across that covered integer grid.
  if span < 1e-9:
    return [(mmin, total_sec)]

  n = int(round(span)) + 1
  if n <= 1:
    return [(mmin, total_sec)]

  grid = np.linspace(mmin, mmax, n, dtype=float)
  per = total_sec / float(n)
  return [(float(x), per) for x in grid]


samples_by_mcz = {}
build_rows = 0
if summary_path.is_file():
  with summary_path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      if (row.get("type") or "").strip() != "build":
        continue
      build_rows += 1
      for m, sec in per_mcz_samples_from_row(row):
        key = round(float(m), 6)
        samples_by_mcz.setdefault(key, []).append(float(sec))

if not samples_by_mcz:
  raise SystemExit(
    f"No usable build runtime samples found in {summary_path}. "
    "Populate that file with build rows or set JOB_SUMMARY to a valid summary CSV."
  )

obs_mcz = np.array(sorted(samples_by_mcz.keys()), dtype=float)
all_samples = np.array(
  [sec for vals in samples_by_mcz.values() for sec in vals],
  dtype=float,
)


def p90_for_mask(mask: np.ndarray) -> float:
  vals = []
  for m in mcz_arr[mask]:
    key = round(float(m), 6)
    if key in samples_by_mcz:
      vals.extend(samples_by_mcz[key])

  if not vals:
    # Fallback to nearest observed mcz for each target point in the tier.
    for m in mcz_arr[mask]:
      idx = int(np.argmin(np.abs(obs_mcz - float(m))))
      nearest = float(obs_mcz[idx])
      vals.extend(samples_by_mcz.get(nearest, []))

  if not vals:
    vals = all_samples.tolist()

  return float(np.quantile(np.array(vals, dtype=float), 0.9))


def to_wall_minutes(p90_sec: float, min_wall_min: int, max_wall_min: int) -> int:
  rec_sec = p90_sec * safety_factor + safety_pad_sec
  rec_min = int(math.ceil(rec_sec / 60.0))
  return int(np.clip(rec_min, min_wall_min, max_wall_min))

min_wall_min = minutes_from_hms(min_wall)
max_wall_min = minutes_from_hms(max_wall)

tier_masks = [
  mcz_arr < tier1_upper,
  (mcz_arr >= tier1_upper) & (mcz_arr < tier2_upper),
  mcz_arr >= tier2_upper,
]

tier_segments = []
for mask in tier_masks:
  idx = np.where(mask)[0]
  if idx.size == 0:
    continue
  p90 = p90_for_mask(mask)
  wall_min = to_wall_minutes(p90, min_wall_min, max_wall_min)
  tier_segments.append((int(idx[0]), int(idx[-1]), wall_min))

# Higher mcz should not request longer walltime than lower mcz tiers.
for i in range(1, len(tier_segments)):
  prev_min = tier_segments[i - 1][2]
  cur = tier_segments[i]
  tier_segments[i] = (cur[0], cur[1], min(cur[2], prev_min))

total_samples = int(sum(len(v) for v in samples_by_mcz.values()))
wall_mins = [seg[2] for seg in tier_segments] if tier_segments else [min_wall_min]
print(
  f"META|{summary_path}|{build_rows}|{total_samples}|"
  f"{obs_mcz[0]:g}|{obs_mcz[-1]:g}|{max(wall_mins)}|{min(wall_mins)}"
)

for start, end, wall_min in tier_segments:
  wall = sec_to_hms(int(wall_min * 60))
  print(f"TIER|{wall}|{start}-{end}|{mcz_arr[start]:g}|{mcz_arr[end]:g}")
PY
)"

echo "Submitting tiered template-bank jobs from: $JOB_SCRIPT"
echo "mcz grid: ${MCZ_MIN}..${MCZ_MAX} (${MCZ_PTS} points)"
echo "tier source: ${JOB_SUMMARY}"
echo "submit partition: ${SUBMIT_PARTITION}"
echo "tier split: mcz<${TIER1_UPPER_MCZ}, ${TIER1_UPPER_MCZ}<=mcz<${TIER2_UPPER_MCZ}, mcz>=${TIER2_UPPER_MCZ}"
echo "tier controls: SAFETY_FACTOR=${SAFETY_FACTOR}, SAFETY_PAD_SEC=${SAFETY_PAD_SEC}, MIN_WALLTIME=${MIN_WALLTIME}, MAX_WALLTIME=${MAX_WALLTIME}"

while IFS='|' read -r kind a b c d e f g; do
  [ -z "$kind" ] && continue

  if [ "$kind" = "META" ]; then
  echo "history coverage: build_rows=${b}, samples=${c}, mcz=${d}..${e}, recommended wall range=${f}..${g} min"
  continue
  fi

  if [ "$kind" != "TIER" ]; then
  continue
  fi

  wall="$a"
  array_range="$b"
  mcz_lo="$c"
  mcz_hi="$d"

  echo "  - mcz ${mcz_lo}..${mcz_hi}: sbatch --partition=${SUBMIT_PARTITION} --array=${array_range} --time=${wall} ${JOB_SCRIPT}"

  if [ "$DRY_RUN" = "1" ]; then
    continue
  fi

  sbatch --partition="${SUBMIT_PARTITION}" --array="${array_range}" --time="${wall}" "${JOB_SCRIPT}"
done <<< "$PLAN"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete. Set DRY_RUN=0 (or unset) to submit."
fi
