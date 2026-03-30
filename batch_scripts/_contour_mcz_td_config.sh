#!/bin/bash

export FLUX_RATIO="${FLUX_RATIO:-0.5}"
export ORIENT_PRESET="${ORIENT_PRESET:-Taman_edgeon}"
export Z="${Z:-1e-8}"

export MCZ_MIN="${MCZ_MIN:-10}"
export MCZ_MAX="${MCZ_MAX:-90}"
export MCZ_PTS="${MCZ_PTS:-81}"

export TD_MIN_MS="${TD_MIN_MS:-20}"
export TD_MAX_MS="${TD_MAX_MS:-70}"
export TD_PTS="${TD_PTS:-51}"

export OMEGA_MIN="${OMEGA_MIN:-0}"
export OMEGA_MAX="${OMEGA_MAX:-6}"
export OMEGA_PTS="${OMEGA_PTS:-61}"

export THETA_MIN="${THETA_MIN:-0}"
export THETA_MAX="${THETA_MAX:-15}"
export THETA_PTS="${THETA_PTS:-151}"

export GAMMA_PTS="${GAMMA_PTS:-51}"

export BANK_DIR="${BANK_DIR:-./data/template_banks}"
export RESULTS_DIR="${RESULTS_DIR:-./data/mismatch}"