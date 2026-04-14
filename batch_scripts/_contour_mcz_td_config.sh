#!/bin/bash

export FLUX_RATIO="${FLUX_RATIO:-0.5}"
export ORIENT_PRESET="${ORIENT_PRESET:-Taman_random}"
export Z="${Z:-1}"

export MCZ_MIN="${MCZ_MIN:-5}"
export MCZ_MAX="${MCZ_MAX:-45}"
export MCZ_PTS="${MCZ_PTS:-81}"
export MCZ_STEP="${MCZ_STEP:-}"  # If set, overrides MCZ_PTS (arange-style)

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

export BANK_DIR="${BANK_DIR:-./data/template_banks}"
export RUN_DIR="${RUN_DIR:-./data/mismatch}"