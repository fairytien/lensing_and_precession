#!/bin/bash
# Check progress of Lindblom cube computation

RESULTS_DIR="data/contours_td_mcz/mismatch_cubes"
TOTAL=81
COMPLETED=0
WITH_SNR=0

echo "Checking Lindblom cube computation progress..."
echo "=============================================="

for mcz in {10..90}; do
    cube="${RESULTS_DIR}/mismatch_cubes_mcz${mcz}Msun_td20-70ms_Taman_edgeon.h5"
    lindblom="${cube%.h5}_lindblom.h5"
    
    if [ -f "$lindblom" ]; then
        COMPLETED=$((COMPLETED + 1))
        # Check if it has SNR data
        python3 -c "import h5py; h5 = h5py.File('$lindblom', 'r'); has_snr = 'snr_cube' in h5; h5.close(); exit(0 if has_snr else 1)" 2>/dev/null
        if [ $? -eq 0 ]; then
            WITH_SNR=$((WITH_SNR + 1))
        fi
    fi
done

echo "Total cubes: $TOTAL"
echo "Completed: $COMPLETED"
echo "With SNR data: $WITH_SNR"
echo "Remaining: $((TOTAL - COMPLETED))"
echo "Progress: $((COMPLETED * 100 / TOTAL))%"

if [ $WITH_SNR -eq $TOTAL ] && [ $COMPLETED -eq $TOTAL ]; then
    echo ""
    echo "✓ All cubes computed with SNR data!"
    echo "Ready to aggregate and plot."
fi

