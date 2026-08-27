#!/bin/bash
# Fichier : scripts/ablation_extern.sh
# Ablation a) : num_pd_layers (intern) fixe, num_layers (extern) varie.
# Entrainement realise exclusivement sur le dataset 'data_0'. Runs executes
# sequentiellement sur un unique GPU ($GPU_ID) : au plus 1 GPU utilise a la fois.
#
# Structure de sortie (une iteration du script = un "run_set") :
#   runs/ablation/p3mg/ablation_extern/run_set_<i>/   -> tous les runs P3MG de cette iteration
#   runs/ablation/p3mg/ablation_extern/results_<i>/   -> synthese (summary.csv + loss_vs_layers.png)

CONFIG="config.yaml"
FIXED_NUM_PD_LAYERS=10
GPU_ID=0
STUDY_TAG="ablation_extern"

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

echo "=== Ablation EXTERN : iteration run_set_${RUN_SET_IDX} ==="

for NUM_LAYERS in $(seq 5 5 60); do
    echo "=== Ablation EXTERN : num_layers=$NUM_LAYERS | num_pd_layers=$FIXED_NUM_PD_LAYERS ==="

    ./scripts/run.sh --config "$CONFIG" --gpu $GPU_ID --full \
        --model p3mg \
        --strategy unrolling \
        --data_folder data_0 \
        --run_group ablation \
        --run_tag "$RUN_TAG" \
        --num_layers $NUM_LAYERS \
        --num_pd_layers $FIXED_NUM_PD_LAYERS || echo "[AVERTISSEMENT] Echec pour num_layers=$NUM_LAYERS, poursuite."
done

# Reconstruction de summary.csv directement depuis les sous-runs sur disque
# (logs/run_config.json + logs/test_results_table.txt), plutot qu'un
# parsing awk fragile ligne par ligne pendant la boucle (source de
# desynchronisation entre la config supposee et celle reellement executee).
python scripts/build_ablation_summary.py \
    --run_set_dir "$RUN_SET_DIR" \
    --study extern \
    --output "$SUMMARY_CSV"

python scripts/plot_ablation.py \
    --csv "$SUMMARY_CSV" \
    --x num_layers \
    --xlabel "Nombre de couches externes (num_layers)" \
    --title "Ablation - Loss vs. nombre de couches externes" \
    --output "$RESULTS_DIR/loss_vs_layers.png" \
    --metric "Loss" || echo "[WARN] Generation du graphe d'ablation echouee."

echo "=== Ablation EXTERN terminee (run_set_${RUN_SET_IDX}). Runs : $RUN_SET_DIR | Resultats : $RESULTS_DIR ==="
