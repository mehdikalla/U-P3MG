#!/bin/bash
# =============================================================================
# Configuration SLURM - Random Search sur tous les algorithmes compatibles
# =============================================================================
# Lance le mode 'full' (train + test) en strategie 'random_search' pour tous
# les modeles algorithmiques supportes (P3MG, PD, PMMS, ISTA, HQ), sur tous
# les jeux de donnees disponibles, via un job array {modele x dataset}.
#
# Le nombre de tirages d'hyperparametres (n_trials) et le nombre d'iterations
# de l'algorithme (algo_iters) sont fixes via CONFIG (config.yaml) et peuvent
# etre surcharges ci-dessous si necessaire.
#
# Usage:
#   sbatch scripts/run_random_search_all.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_RS_ALL
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-rs-%A_%a.out
#SBATCH --error=logs/slurm-rs-%A_%a.err
#SBATCH --array=0-14

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
# 1. Definition de l'espace de recherche (modeles x datasets)
# -----------------------------------------------------------------------------
MODELS=(p3mg pd pmms ista hq)
DATA_FOLDERS=(data_0 data_1 data_2)

STRATEGY="random_search"
MODE="full"
CONFIG="config.yaml"

# -----------------------------------------------------------------------------
# 2. Construction de la liste des combinaisons {modele x dataset}
# -----------------------------------------------------------------------------
COMBOS=()
for model in "${MODELS[@]}"; do
    for data in "${DATA_FOLDERS[@]}"; do
        COMBOS+=("${model}|${data}")
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

IFS='|' read -r MODEL DATA_FOLDER <<< "${COMBOS[$TASK_ID]}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Tache: ${TASK_ID}/${TOTAL} ---"
echo "--- Modele: ${MODEL} | Strategie: ${STRATEGY} | Dataset: ${DATA_FOLDER} | Mode: ${MODE} ---"

# -----------------------------------------------------------------------------
# 4. Execution via scripts/run.sh
# -----------------------------------------------------------------------------
./scripts/run.sh \
    --config "${CONFIG}" \
    --gpu 0 \
    --mode "${MODE}" \
    --model "${MODEL}" \
    --strategy "${STRATEGY}" \
    --data_folder "${DATA_FOLDER}"

echo "--- Tache ${TASK_ID} (${MODEL}/${STRATEGY}/${DATA_FOLDER}) terminee ---"
