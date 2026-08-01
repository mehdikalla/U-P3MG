#!/bin/bash
# Fichier : scripts/ablation_intern.sh
# Ablation b) : num_layers (extern) fixe, num_pd_layers (intern) varie.

CONFIG="config.yaml"
FIXED_NUM_LAYERS=30
GPU_ID=0
# Espace de noms isole (ne pollue pas 'data_1' utilise par run_all/compare).
RUN_TAG="ablation_intern"

RESULTS_DIR="Results/ablation/intern"

SUMMARY_CSV="$RESULTS_DIR/summary.csv"
mkdir -p "$RESULTS_DIR"
echo "num_layers,num_pd_layers,run_dir,mean,median,std,best,worst" > "$SUMMARY_CSV"

for NUM_PD_LAYERS in $(seq 5 5 60); do
    echo "=== Ablation INTERN : num_layers=$FIXED_NUM_LAYERS | num_pd_layers=$NUM_PD_LAYERS ==="

    ./scripts/run.sh --config "$CONFIG" --gpu $GPU_ID --full \
        --run_group ablation \
        --run_tag "$RUN_TAG" \
        --num_layers $FIXED_NUM_LAYERS \
        --num_pd_layers $NUM_PD_LAYERS

    RUN_DIR=$(find "runs/ablation/p3mg/unrolling/${RUN_TAG}" -mindepth 1 -maxdepth 1 -type d -name "*_full" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)


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

    echo "$FIXED_NUM_LAYERS,$NUM_PD_LAYERS,$RUN_DIR,$MEAN,$MEDIAN,$STD,$BEST,$WORST" >> "$SUMMARY_CSV"
done

echo "=== Ablation INTERN terminee. Resultats consolides : $SUMMARY_CSV ==="
