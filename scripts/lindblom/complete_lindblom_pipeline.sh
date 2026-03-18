#!/bin/bash
# Complete the Lindblom pipeline: aggregate and create plots
# Run this after all batch jobs complete

set -e

source /home1/10000/fairytien33/miniconda3/etc/profile.d/conda.sh
conda activate fairytien_gw

RESULTS_DIR="data/contours_td_mcz"
TD_MIN_MS=20
TD_MAX_MS=70
MCZ_MIN=10
MCZ_MAX=90
ORIENTATION_TAG="Taman_edgeon"
OUTPUT_DIR="figures"

echo "=========================================="
echo "Lindblom Pipeline: Aggregate and Plot"
echo "=========================================="
echo ""

# Step 1: Aggregate Lindblom cubes
echo "Step 1: Aggregating Lindblom cubes..."
python -m scripts.lindblom.aggregate_lindblom_best_match \
    --results_dir "$RESULTS_DIR" \
    --td_min_ms "$TD_MIN_MS" \
    --td_max_ms "$TD_MAX_MS" \
    --mcz_min "$MCZ_MIN" \
    --mcz_max "$MCZ_MAX" \
    --orientation_tag "$ORIENTATION_TAG"

echo ""

# Step 2: Create Lindblom contour plots
echo "Step 2: Creating Lindblom contour plots..."
python -m scripts.lindblom.create_contour_mcz_td_from_lindblom \
    --results_dir "$RESULTS_DIR" \
    --td_min_ms "$TD_MIN_MS" \
    --td_max_ms "$TD_MAX_MS" \
    --mcz_min "$MCZ_MIN" \
    --mcz_max "$MCZ_MAX" \
    --orientation_tag "$ORIENTATION_TAG" \
    --output_dir "$OUTPUT_DIR" \
    --overlay-cycles \
    --overlay-peaks \
    --overlay-troughs \
    --cmap jet \
    --no-zero-contour

python -m scripts.lindblom.create_contour_mcz_td_from_lindblom \
    --results_dir "$RESULTS_DIR" \
    --td_min_ms "$TD_MIN_MS" \
    --td_max_ms "$TD_MAX_MS" \
    --mcz_min "$MCZ_MIN" \
    --mcz_max "$MCZ_MAX" \
    --orientation_tag "$ORIENTATION_TAG" \
    --output_dir "$OUTPUT_DIR" \
    --cmap jet \
    --no-zero-contour

echo ""

# Step 3: Create SNR contour plots
echo "Step 3: Creating SNR contour plots..."
python -m scripts.lindblom.create_contour_mcz_td_from_snr \
    --results_dir "$RESULTS_DIR" \
    --td_min_ms "$TD_MIN_MS" \
    --td_max_ms "$TD_MAX_MS" \
    --mcz_min "$MCZ_MIN" \
    --mcz_max "$MCZ_MAX" \
    --orientation_tag "$ORIENTATION_TAG" \
    --output_dir "$OUTPUT_DIR" \
    --overlay-cycles \
    --overlay-peaks \
    --overlay-troughs \
    --cmap jet

python -m scripts.lindblom.create_contour_mcz_td_from_snr \
    --results_dir "$RESULTS_DIR" \
    --td_min_ms "$TD_MIN_MS" \
    --td_max_ms "$TD_MAX_MS" \
    --mcz_min "$MCZ_MIN" \
    --mcz_max "$MCZ_MAX" \
    --orientation_tag "$ORIENTATION_TAG" \
    --output_dir "$OUTPUT_DIR" \
    --cmap jet

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Lindblom plots: $OUTPUT_DIR/lindblom_contour_*.pdf"
echo "SNR plots: $OUTPUT_DIR/snr_contour_*.pdf"

