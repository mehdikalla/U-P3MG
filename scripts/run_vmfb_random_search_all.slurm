#!/bin/bash
# =============================================================================
# Configuration SLURM - Random Search VMFB sur les trois datasets
# =============================================================================
# Lance le mode 'full' (train + test) en strategie 'random_search' pour le
# modele VMFB (Variable Metric Forward-Backward, cf. src/models/vmfb/algo.py),
# sur les trois jeux de donnees disponibles (data_0, data_1, data_2), via un
# job array {dataset}. VMFB n'ayant pas d'unrolling (pas de net.py), seule la
# strategie 'random_search' est pertinente pour ce modele.
#
# Le nombre de tirages d'hyperparametres (n_trials) est fixe via CONFIG
# (config.yaml) et peut etre surcharge ci-dessous si necessaire. ALGO_ITERS
# est aligne sur celui de P3MG (cf. scripts/run_all.slurm), pour rester
# comparable aux autres modeles de l'etude (memes couches externes, seule la
# sous-resolution interne differe).
#
# Usage:
#   sbatch scripts/run_vmfb_random_search_all.slurm
#
# Pour ne lancer qu'un sous-ensemble des taches (ex: uniquement data_1) :
#   sbatch --array=1-1 scripts/run_vmfb_random_search_all.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_VMFB_RS
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-vmfb-rs-%A_%a.out
#SBATCH --error=logs/slurm-vmfb-rs-%A_%a.err
#SBATCH --array=0-2

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
# 1. Definition de l'espace de recherche (un seul modele x datasets)
# -----------------------------------------------------------------------------
MODEL="vmfb"
DATA_FOLDERS=(data_0 data_1 data_2)

STRATEGY="random_search"
MODE="full"
CONFIG="config.yaml"

# ALGO_ITERS : nombre d'iterations de l'algorithme statique (--algo_iters),
# aligne sur P3MG (cf. scripts/run_all.slurm, declare -A ALGO_ITERS).
ALGO_ITERS=5000

# -----------------------------------------------------------------------------
# 2. Selection du dataset correspondant a cette tache du job array
# -----------------------------------------------------------------------------
TOTAL=${#DATA_FOLDERS[@]}
echo "--- Nombre total de datasets definis : ${TOTAL} ---"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "[INFO] Tache ${TASK_ID} hors bornes (total=${TOTAL}). Rien a executer."
    exit 0
fi

DATA_FOLDER="${DATA_FOLDERS[$TASK_ID]}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Tache: ${TASK_ID}/${TOTAL} ---"
echo "--- Modele: ${MODEL} | Strategie: ${STRATEGY} | Dataset: ${DATA_FOLDER} | Mode: ${MODE} ---"

# -----------------------------------------------------------------------------
# 3. Execution via scripts/run.sh
# -----------------------------------------------------------------------------
./scripts/run.sh \
    --config "${CONFIG}" \
    --gpu 0 \
    --mode "${MODE}" \
    --model "${MODEL}" \
    --strategy "${STRATEGY}" \
    --data_folder "${DATA_FOLDER}" \
    --algo_iters "${ALGO_ITERS}"

echo "--- Tache ${TASK_ID} (${MODEL}/${STRATEGY}/${DATA_FOLDER}) terminee ---"
