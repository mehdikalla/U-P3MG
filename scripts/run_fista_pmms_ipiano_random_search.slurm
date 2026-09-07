#!/bin/bash
# =============================================================================
# Configuration SLURM - Random Search FISTA / PMMS / iPiano (separement)
# =============================================================================
# Lance le mode 'full' (train + test) en strategie 'random_search' pour les
# trois algorithmes purement iteratifs FISTA, PMMS et iPiano (aucun n'a
# d'unrolling, cf. src/models/<model>/algo.py sans net.py associe), sur les
# trois jeux de donnees disponibles (data_0, data_1, data_2), via un job
# array {modele x dataset}. Chaque modele est traite separement (pas de
# combinaison croisee autre que modele x dataset).
#
# Usage:
#   sbatch scripts/run_fista_pmms_ipiano_random_search.slurm
#
# Pour ne lancer qu'un sous-ensemble des taches (ex: seulement FISTA, les
# 3 premiers indices) :
#   sbatch --array=0-2 scripts/run_fista_pmms_ipiano_random_search.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_FISTA_PMMS_IPIANO_RS
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-fista-pmms-ipiano-rs-%A_%a.out
#SBATCH --error=logs/slurm-fista-pmms-ipiano-rs-%A_%a.err
#SBATCH --array=0-8
# ^ Borne exacte = nb modeles (3: fista pmms ipiano) * nb datasets (3) - 1,
#   soit 9 taches (indices 0-8). A recalculer si MODELS/DATA_FOLDERS changent
#   ci-dessous (voir section 1).

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
MODELS=(fista pmms ipiano)
DATA_FOLDERS=(data_0 data_1 data_2)

STRATEGY="random_search"
MODE="full"
CONFIG="config.yaml"

# ALGO_ITERS : nombre d'iterations de l'algorithme statique (--algo_iters),
# aligne sur P3MG (cf. scripts/run_all.slurm, declare -A ALGO_ITERS), commun
# aux trois modeles pour rester comparables entre eux.
ALGO_ITERS=5000

# -----------------------------------------------------------------------------
# 2. Construction de la liste exhaustive des combinaisons {modele x dataset}
# -----------------------------------------------------------------------------
COMBOS=()
for model in "${MODELS[@]}"; do
    for data in "${DATA_FOLDERS[@]}"; do
        COMBOS+=("${model}|${data}")
    done
done

TOTAL=${#COMBOS[@]}
echo "--- Nombre total de combinaisons definies : ${TOTAL} ---"

if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT}" -ne "${TOTAL}" ]; then
    echo "[AVERTISSEMENT] SLURM_ARRAY_TASK_COUNT (${SLURM_ARRAY_TASK_COUNT}) != TOTAL calcule (${TOTAL})." \
         "Mettez a jour '#SBATCH --array' en tete de ce script (ou relancez avec" \
         "'sbatch --array=0-$((TOTAL - 1)) scripts/run_fista_pmms_ipiano_random_search.slurm')." >&2
fi

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
    --data_folder "${DATA_FOLDER}" \
    --algo_iters "${ALGO_ITERS}"

echo "--- Tache ${TASK_ID} (${MODEL}/${STRATEGY}/${DATA_FOLDER}) terminee ---"
