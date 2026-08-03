#!/bin/bash
# Fichier : scripts/ablation_mlp.sh
# Ablation c) : architecture (nombre de couches) du MLP interne (lambda) du P3MG.
# num_layers et num_pd_layers restent fixes ; seule la profondeur du MLP varie.
#
# Structure de sortie (une iteration du script = un "run_set") :
#   runs/ablation/p3mg/ablation_mlp/run_set_<i>/   -> tous les runs P3MG de cette iteration
#   runs/ablation/p3mg/ablation_mlp/results_<i>/   -> synthese (summary.csv + loss_vs_layers.png)

CONFIG="config.yaml"
FIXED_NUM_LAYERS=30
FIXED_NUM_PD_LAYERS=10
GPU_ID=0
STUDY_TAG="ablation_mlp"

# Chaque entree decrit les tailles des couches cachees du MLP (entre l'entree 100 et la sortie 1)
MLP_CONFIGS=(
    "25"
    "50,25"
    "50,25,12"
    "100,50,25,12"
    "100,50,25,12,6"
)

BASE_DIR="runs/ablation/p3mg/${STUDY_TAG}"
mkdir -p "$BASE_DIR"

# Determination du prochain indice de run_set disponible (iteration du script).
RUN_SET_IDX=1
while [ -d "$BASE_DIR/run_set_${RUN_SET_IDX}" ]; do
    RUN_SET_IDX=$((RUN_SET_IDX + 1))
done

RUN_SET_DIR="$BASE_DIR/run_set_${RUN_SET_IDX}"
RESULTS_DIR="$BASE_DIR/results_${RUN_SET_IDX}"
mkdir -p "$RUN_SET_DIR" "$RESULTS_DIR"

# Espace de noms isole (ne pollue pas 'data_1' utilise par run_all/compare).
RUN_TAG="${STUDY_TAG}/run_set_${RUN_SET_IDX}"

SUMMARY_CSV="$RESULTS_DIR/summary.csv"
echo "mlp_hidden,num_layers,num_pd_layers,run_dir,mean,median,std,best,worst" > "$SUMMARY_CSV"

echo "=== Ablation MLP : iteration run_set_${RUN_SET_IDX} ==="

for MLP_HIDDEN in "${MLP_CONFIGS[@]}"; do
    echo "=== Ablation MLP : mlp_hidden=$MLP_HIDDEN | num_layers=$FIXED_NUM_LAYERS | num_pd_layers=$FIXED_NUM_PD_LAYERS ==="

    ./scripts/run.sh --config "$CONFIG" --gpu $GPU_ID --full \
        --model p3mg \
        --strategy unrolling \
        --run_group ablation \
        --run_tag "$RUN_TAG" \
        --num_layers $FIXED_NUM_LAYERS \
        --num_pd_layers $FIXED_NUM_PD_LAYERS \
        --mlp_hidden "$MLP_HIDDEN"

    RUN_DIR=$(find "$RUN_SET_DIR" -mindepth 1 -maxdepth 1 -type d -name "*_full" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

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

python scripts/plot_ablation.py \
    --csv "$SUMMARY_CSV" \
    --x mlp_hidden \
    --xlabel "Profondeur du MLP interne" \
    --title "Ablation - Loss vs. profondeur du MLP interne" \
    --output "$RESULTS_DIR/loss_vs_layers.png" \
    --metric "Loss" || echo "[WARN] Generation du graphe d'ablation echouee."

echo "=== Ablation MLP terminee (run_set_${RUN_SET_IDX}). Runs : $RUN_SET_DIR | Resultats : $RESULTS_DIR ==="
