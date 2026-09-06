#!/bin/bash
# =============================================================================
# Configuration SLURM - Tous les modeles sauf PMMS, sur data_1 uniquement
# =============================================================================
# Lance le mode 'full' (train + test) pour tous les modeles disponibles a
# l'exception de PMMS, sur le seul jeu de donnees 'data_1', via un job array
# {modele x strategie}. Les modeles algorithmiques (p3mg, ista, hq, pd)
# supportent les strategies 'unrolling' ET 'random_search' ; les modeles
# purement deep-learning (fcae, fctn, fcun, resu) ne supportent que
# 'unrolling'. PMMS est volontairement exclu (cf. scripts/run_pmms_random_search_all.slurm
# qui lui est dedie sur les trois datasets).
#
# Usage:
#   sbatch scripts/run_all_except_pmms_data1.slurm
#
# Pour ne lancer qu'un sous-ensemble des taches :
#   sbatch --array=0-3 scripts/run_all_except_pmms_data1.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_ALL_NO_PMMS_D1
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-nopmms-d1-%A_%a.out
#SBATCH --error=logs/slurm-nopmms-d1-%A_%a.err
#SBATCH --array=0-11%4
# ^ Borne exacte = (nb modeles algo * nb strategies) + nb modeles DL - 1, soit
#   4 modeles algo (p3mg ista hq pd) * 2 strategies + 4 modeles DL (fcae fctn
#   fcun resu) * 1 strategie = 8 + 4 = 12 taches (indices 0-11). A recalculer
#   si ALGO_MODELS/DL_MODELS/STRATEGIES changent ci-dessous (voir section 1).
#   %4 = 4 taches simultanees au maximum (a ajuster selon les GPUs disponibles
#   sur la partition).

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
# Modeles algorithmiques (supportent 'unrolling' ET 'random_search'), PMMS
# volontairement exclu.
ALGO_MODELS=(p3mg ista hq pd)

# Modeles purement deep-learning (supportent uniquement 'unrolling')
DL_MODELS=(fcae fctn fcun resu)

STRATEGIES=(unrolling random_search)

DATA_FOLDER="data_1"
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

for model in "${ALGO_MODELS[@]}"; do
    for strat in "${STRATEGIES[@]}"; do
        COMBOS+=("${model}|${strat}")
    done
done

for model in "${DL_MODELS[@]}"; do
    COMBOS+=("${model}|unrolling")
done

TOTAL=${#COMBOS[@]}
echo "--- Nombre total de combinaisons definies : ${TOTAL} ---"

if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT}" -ne "${TOTAL}" ]; then
    echo "[AVERTISSEMENT] SLURM_ARRAY_TASK_COUNT (${SLURM_ARRAY_TASK_COUNT}) != TOTAL calcule (${TOTAL})." \
         "Mettez a jour '#SBATCH --array' en tete de ce script (ou relancez avec" \
         "'sbatch --array=0-$((TOTAL - 1))%4 scripts/run_all_except_pmms_data1.slurm')." >&2
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
