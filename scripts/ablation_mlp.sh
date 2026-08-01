#!/bin/bash
# Fichier : scripts/ablation_mlp.sh
# Ablation c) : architecture (nombre de couches) du MLP interne (lambda) du P3MG.
# num_layers et num_pd_layers restent fixes ; seule la profondeur du MLP varie.

CONFIG="config.yaml"
FIXED_NUM_LAYERS=30
FIXED_NUM_PD_LAYERS=10
GPU_ID=0
# Espace de noms isole (ne pollue pas 'data_1' utilise par run_all/compare).
RUN_TAG="ablation_mlp"


# Chaque entree decrit les tailles des couches cachees du MLP (entre l'entree 100 et la sortie 1)
MLP_CONFIGS=(
    "25"
    "50,25"
    "50,25,12"
    "100,50,25,12"
    "100,50,25,12,6"
)

RESULTS_DIR="Results/ablation/mlp"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
mkdir -p "$RESULTS_DIR"
echo "mlp_hidden,num_layers,num_pd_layers,run_dir,mean,median,std,best,worst" > "$SUMMARY_CSV"

for MLP_HIDDEN in "${MLP_CONFIGS[@]}"; do
    echo "=== Ablation MLP : mlp_hidden=$MLP_HIDDEN | num_layers=$FIXED_NUM_LAYERS | num_pd_layers=$FIXED_NUM_PD_LAYERS ==="

    ./scripts/run.sh --config "$CONFIG" --gpu $GPU_ID --full \
        --run_group ablation \
        --run_tag "$RUN_TAG" \
        --num_layers $FIXED_NUM_LAYERS \
        --num_pd_layers $FIXED_NUM_PD_LAYERS \
        --mlp_hidden "$MLP_HIDDEN"

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

    echo "\"$MLP_HIDDEN\",$FIXED_NUM_LAYERS,$FIXED_NUM_PD_LAYERS,$RUN_DIR,$MEAN,$MEDIAN,$STD,$BEST,$WORST" >> "$SUMMARY_CSV"
done

echo "=== Ablation MLP terminee. Resultats consolides : $SUMMARY_CSV ==="
