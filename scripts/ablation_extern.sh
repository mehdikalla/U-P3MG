#!/bin/bash
# Fichier : scripts/ablation_extern.sh
# Ablation a) : num_pd_layers (intern) fixe, num_layers (extern) varie.

CONFIG="config.yaml"
FIXED_NUM_PD_LAYERS=10
GPU_ID=0

RESULTS_DIR="Results/ablation/extern"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
mkdir -p "$RESULTS_DIR"
echo "num_layers,num_pd_layers,run_dir,mean,median,std,best,worst" > "$SUMMARY_CSV"

for NUM_LAYERS in $(seq 5 5 60); do
    echo "=== Ablation EXTERN : num_layers=$NUM_LAYERS | num_pd_layers=$FIXED_NUM_PD_LAYERS ==="

    BEFORE_RUNS=$(ls -1 runs/p3mg/unrolling/*/ 2>/dev/null)

    ./scripts/run.sh --config "$CONFIG" --gpu $GPU_ID --full \
        --num_layers $NUM_LAYERS \
        --num_pd_layers $FIXED_NUM_PD_LAYERS

    RUN_DIR=$(find runs/p3mg/unrolling -mindepth 2 -maxdepth 2 -type d -name "*_full" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    TABLE_FILE="$RUN_DIR/logs/test_results_table.txt"

    if [ -f "$TABLE_FILE" ]; then
        MEAN=$(grep "Mean" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
        MEDIAN=$(grep "Median" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
        STD=$(grep "Std" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
        BEST=$(grep "Min (Best)" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
        WORST=$(grep "Max (Worst)" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
    else
        MEAN=""; MEDIAN=""; STD=""; BEST=""; WORST=""
    fi

    echo "$NUM_LAYERS,$FIXED_NUM_PD_LAYERS,$RUN_DIR,$MEAN,$MEDIAN,$STD,$BEST,$WORST" >> "$SUMMARY_CSV"
done

echo "=== Ablation EXTERN terminee. Resultats consolides : $SUMMARY_CSV ==="

