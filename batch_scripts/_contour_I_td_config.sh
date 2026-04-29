#!/bin/bash
# Configuration for the I-td mismatch pipeline (fixed mcz, varying I and td)

export MCZ="${MCZ:-10}"
export ORIENT_PRESET="${ORIENT_PRESET:-Taman_edgeon}"
export Z="${Z:-1}"

export I_MIN="${I_MIN:-0.1}"
export I_MAX="${I_MAX:-0.9}"
export I_PTS="${I_PTS:-81}"
export I_STEP="${I_STEP:-}"  # If set, overrides I_PTS (arange-style)

export TD_MIN_MS="${TD_MIN_MS:-20}"
export TD_MAX_MS="${TD_MAX_MS:-70}"
export TD_PTS="${TD_PTS:-51}"
export TD_STEP_MS="${TD_STEP_MS:-}"  # If set, overrides TD_PTS (arange-style)

export OMEGA_MIN="${OMEGA_MIN:-0}"
export OMEGA_MAX="${OMEGA_MAX:-6}"
export OMEGA_PTS="${OMEGA_PTS:-61}"

export THETA_MIN="${THETA_MIN:-0}"
export THETA_MAX="${THETA_MAX:-15}"
export THETA_PTS="${THETA_PTS:-151}"

export GAMMA_PTS="${GAMMA_PTS:-51}"

export SHARED_DATA_ROOT="${SHARED_DATA_ROOT:-/work/10000/fairytien33/gw_shared_data}"
export BANK_DIR="${BANK_DIR:-${SHARED_DATA_ROOT}/template_banks}"
export RUN_DIR="${RUN_DIR:-${SHARED_DATA_ROOT}/mismatch}"
