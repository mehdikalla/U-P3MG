#!/bin/bash
# =============================================================================
# Configuration SLURM - Lancement du modele PMMS uniquement
# =============================================================================
# Reprend les parametres definis pour PMMS dans run_all.slurm (num_layers=25,
# algo_iters=1000), en job array sur {strategie x dataset}.
#
# Usage:
#   sbatch scripts/run_algo.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_algo
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-algo-%A_%a.out
#SBATCH --error=logs/slurm-algo-%A_%a.err
#SBATCH --array=0-5

set -euo pipefail
module purge
module load cuda/13.1
export MPLBACKEND=Agg   # Evite les erreurs liees a l'affichage graphique (X11)

cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

mkdir -p logs

# -----------------------------------------------------------------------------
# 1. Definition de l'espace de recherche (PMMS uniquement)
# -----------------------------------------------------------------------------
MODEL="ista"
STRATEGIES=(unrolling random_search)
DATA_FOLDERS=(data_0 data_1 data_2)

MODE="full"
CONFIG="config.yaml"

# Hyperparametres specifiques a PMMS (cf. run_all.slurm)
NUM_LAYERS=50

# -----------------------------------------------------------------------------
# 2. Construction de la liste des combinaisons {strategie x dataset}
# -----------------------------------------------------------------------------
COMBOS=()
for data in "${DATA_FOLDERS[@]}"; do
    for strat in "${STRATEGIES[@]}"; do
        COMBOS+=("${strat}|${data}")
    done
done

TOTAL=${#COMBOS[@]}
echo "--- Nombre total de combinaisons definies : ${TOTAL} ---"

# -----------------------------------------------------------------------------
# 3. Selection de la combinaison correspondant a cette tache du job array
# -----------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "[INFO] Tache ${TASK_ID} hors bornes (total=${TOTAL}). Rien a executer."
    exit 0
fi

IFS='|' read -r STRATEGY DATA_FOLDER <<< "${COMBOS[$TASK_ID]}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Tache: ${TASK_ID}/${TOTAL} ---"
echo "--- Modele: ${MODEL} | Strategie: ${STRATEGY} | Dataset: ${DATA_FOLDER} | Mode: ${MODE} ---"

# -----------------------------------------------------------------------------
# 4. Execution via scripts/run.sh
# -----------------------------------------------------------------------------
EXTRA_ARGS=(--num_layers "${NUM_LAYERS}")

if [ "${STRATEGY}" = "random_search" ]; then
    EXTRA_ARGS+=(--algo_iters "${ALGO_ITERS}")
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
