#!/bin/bash
# =============================================================================
# Configuration SLURM (ETAPE 1/3 : CALIBRATION) - PD/HQ/ISTA/P3MG
# (random_search) + FCUN/FCAE (deep learning) + U-HQ/U-ISTA/U-P3MG
# (unrolling), data_1 uniquement
# =============================================================================
# Ce script est le job d'ENTRAINEMENT/CALIBRATION uniquement (mode 'full').
# Il ne produit AUCUN rapport de comparaison : il se contente d'ecrire les
# checkpoints ('best_model.pt') et parametres ('best_params.json') dans
# 'runs/<model>/<strategy>/data_1/'. Le rapport comparatif final
# (compare_report.csv/.json/.txt + graphes, via src.strategies.compare) est
# produit par l'ETAPE 2 (scripts/run_compare_rs_dl_unroll_data1_report.slurm),
# elle-meme suivie de l'ETAPE 3 de consolidation
# (scripts/build_compare_summary.slurm). Ne pas soumettre ce script isolement
# si un rapport 'compare' est attendu : utiliser
# scripts/submit_compare_rs_dl_unroll_data1.sh qui enchaine les 3 etapes.
#
# Lance, via un unique job array {modele x strategie}, l'etude comparative
# demandee sur le seul jeu de donnees 'data_1' :
#   - Strategie 'random_search'  : PD, HQ, ISTA, P3MG (4 modeles)
#   - Strategie 'unrolling'      : HQ, ISTA, P3MG -> "U-HQ", "U-ISTA", "U-P3MG"
#                                   (PD est volontairement exclu de l'unrolling,
#                                   cf. demande : seul PD en random_search)
#   - Deep learning pur (strategie 'unrolling' sans algo.py) : FCUN, FCAE
#
# Usage (etape isolee, sans rapport) :
#   sbatch scripts/run_compare_rs_dl_unroll_data1.slurm
# Usage recommande (3 etapes enchainees + rapport) :
#   ./scripts/submit_compare_rs_dl_unroll_data1.sh
#
# Pour ne lancer qu'un sous-ensemble des taches :
#   sbatch --array=0-3 scripts/run_compare_rs_dl_unroll_data1.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_COMPARE_D1_TRAIN
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-compare-d1-%A_%a.out
#SBATCH --error=logs/slurm-compare-d1-%A_%a.err
#SBATCH --array=0-3%4
# ^ Borne exacte = nb modeles random_search (4: pd hq ista p3mg)
#   + nb modeles unrolling (3: hq ista p3mg) + nb modeles DL (2: fcun fcae)
#   = 4 + 3 + 2 = 9 taches (indices 0-8). A recalculer si RS_MODELS/
#   UNROLLING_MODELS/DL_MODELS changent ci-dessous (voir section 1).
#   %4 = 4 taches simultanees au maximum (a ajuster selon les GPUs
#   disponibles sur la partition).

set -euo pipefail
module purge
module load cuda/13.1
export MPLBACKEND=Agg   # Evite les erreurs liees a l'affichage graphique (X11)
export PYTHONUNBUFFERED=1

cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

mkdir -p logs

# -----------------------------------------------------------------------------
# 1. Definition de l'espace de recherche
# -----------------------------------------------------------------------------
# Modeles algorithmiques compares en strategie 'random_search' (PD HQ ISTA P3MG)
#RS_MODELS=(pd hq ista p3mg)

# Modeles algorithmiques compares en strategie 'unrolling' (U-HQ U-ISTA U-P3MG),
# PD volontairement exclu (cf. demande : PD uniquement en random_search).
UNROLLING_MODELS=(pd)

# Modeles purement deep-learning (supportent uniquement 'unrolling')
#DL_MODELS=(fcun fcae)

DATA_FOLDER=(data_0 data_1 data_2)
MODE="full"
CONFIG="config.yaml"

# -----------------------------------------------------------------------------
# 1bis. Hyperparametres specifiques par modele (cf. scripts/run_all.slurm)
# -----------------------------------------------------------------------------
declare -A NUM_LAYERS=(
    [p3mg]=30
    [ista]=50
    [hq]=30
    [pd]=50
)

declare -A ALGO_ITERS=(
    [p3mg]=5000
    [ista]=5000
    [hq]=5000
    [pd]=5000
)

# Nombre de sous-couches Primal-Dual internes (--num_pd_layers), utilise
# uniquement par P3MG et HQ en mode 'unrolling'.
declare -A NUM_PD_LAYERS=(
    [p3mg]=10
    [hq]=10
)

# -----------------------------------------------------------------------------
# 2. Construction de la liste exhaustive des combinaisons {modele x strategie}
# -----------------------------------------------------------------------------
COMBOS=()

for model in "${RS_MODELS[@]}"; do
    COMBOS+=("${model}|random_search")
done

for model in "${UNROLLING_MODELS[@]}"; do
    COMBOS+=("${model}|unrolling")
done

for model in "${DL_MODELS[@]}"; do
    COMBOS+=("${model}|unrolling")
done

TOTAL=${#COMBOS[@]}
echo "--- Nombre total de combinaisons definies : ${TOTAL} ---"

if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT}" -ne "${TOTAL}" ]; then
    echo "[AVERTISSEMENT] SLURM_ARRAY_TASK_COUNT (${SLURM_ARRAY_TASK_COUNT}) != TOTAL calcule (${TOTAL})." \
         "Mettez a jour '#SBATCH --array' en tete de ce script (ou relancez avec" \
         "'sbatch --array=0-$((TOTAL - 1))%4 scripts/run_compare_rs_dl_unroll_data1.slurm')." >&2
fi

# -----------------------------------------------------------------------------
# 3. Selection de la combinaison correspondant a cette tache du job array
# -----------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "[INFO] Tache ${TASK_ID} hors bornes (total=${TOTAL}). Rien a executer."
    exit 0
fi

IFS='|' read -r MODEL STRATEGY <<< "${COMBOS[$TASK_ID]}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Tache: ${TASK_ID}/${TOTAL} ---"
echo "--- Modele: ${MODEL} | Strategie: ${STRATEGY} | Dataset: ${DATA_FOLDER} | Mode: ${MODE} ---"

# -----------------------------------------------------------------------------
# 4. Execution via scripts/run.sh
# -----------------------------------------------------------------------------
EXTRA_ARGS=()

if [ -n "${NUM_LAYERS[$MODEL]+x}" ]; then
    EXTRA_ARGS+=(--num_layers "${NUM_LAYERS[$MODEL]}")
fi
if [ -n "${NUM_PD_LAYERS[$MODEL]+x}" ]; then
    EXTRA_ARGS+=(--num_pd_layers "${NUM_PD_LAYERS[$MODEL]}")
fi
if [ "${STRATEGY}" = "random_search" ] && [ -n "${ALGO_ITERS[$MODEL]+x}" ]; then
    EXTRA_ARGS+=(--algo_iters "${ALGO_ITERS[$MODEL]}")
fi
if [ "$STRATEGY" = "unrolling" ] && [ "${MODEL}" = "ista" ]; then
    EXTRA_ARGS+=(--epochs 50)
fi

echo "--- Arguments specifiques : ${EXTRA_ARGS[*]:-(aucun)} ---"

./scripts/run.sh \
    --config "${CONFIG}" \
    --gpu 0 \
    --mode "${MODE}" \
    --model "${MODEL}" \
    --strategy "${STRATEGY}" \
    --data_folder "${DATA_FOLDER}" \
    "${EXTRA_ARGS[@]}"

echo "--- Tache ${TASK_ID} (${MODEL}/${STRATEGY}/${DATA_FOLDER}) terminee ---"
