#!/bin/bash
# =============================================================================
# Configuration SLURM - Comparaison finale de tous les algorithmes (mode compare)
# =============================================================================
# Ce script execute le mode 'compare' de main.py pour chaque jeu de donnees
# (data_0, data_1, data_2), en inference uniquement (aucun entrainement),
# avec les poids/parametres deja calibres (checkpoints 'unrolling' et
# 'best_params.json' 'random_search' les plus recents disponibles dans
# 'runs/'). Le critere de selection du signal qualitatif est fixe a SNR,
# mais le rapport produit contient systematiquement la MSE ET le SNR
# (moyenne + ecart-type) pour chaque combinaison (modele, strategie).
#
# A la fin, les rapports CSV individuels (un par data_folder) sont
# consolides en un unique fichier CSV global.
#
# Usage:
#   sbatch scripts/run_compare.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_COMPARE
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/compare-%j.out
#SBATCH --error=logs/compare-%j.err

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
# 1. Parametres
# -----------------------------------------------------------------------------
CONFIG="config.yaml"
CRITERION="SNR"
DATA_FOLDERS=(data_0 data_1 data_2)

RESULTS_DIR="Results/compare"
FINAL_CSV="${RESULTS_DIR}/all_compare_report.csv"
mkdir -p "${RESULTS_DIR}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Mode: compare | Critere: ${CRITERION} ---"

# -----------------------------------------------------------------------------
# 2. Execution du mode 'compare' pour chaque jeu de donnees
# -----------------------------------------------------------------------------
PER_DATASET_CSVS=()

for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
    echo "=== [COMPARE] Dataset: ${DATA_FOLDER} ==="

    ./scripts/run.sh \
        --config "${CONFIG}" \
        --gpu 0 \
        --mode compare \
        --data_folder "${DATA_FOLDER}" \
        --criterion "${CRITERION}"

    RUN_DIR=$(find "runs/compare/${DATA_FOLDER}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "${RUN_DIR}" ]; then
        echo "[AVERTISSEMENT] Aucun run 'compare' trouve pour ${DATA_FOLDER}. Ignore."
        continue
    fi

    CSV_PATH="${RUN_DIR}/logs/compare_report.csv"
    if [ -f "${CSV_PATH}" ]; then
        PER_DATASET_CSVS+=("${CSV_PATH}")
        echo "[INFO] Rapport recupere : ${CSV_PATH}"
    else
        echo "[AVERTISSEMENT] Rapport CSV introuvable pour ${DATA_FOLDER} (${CSV_PATH})."
    fi
done

# -----------------------------------------------------------------------------
# 3. Consolidation des rapports individuels en un seul fichier CSV global
# -----------------------------------------------------------------------------
if [ "${#PER_DATASET_CSVS[@]}" -eq 0 ]; then
    echo "[ERREUR] Aucun rapport de comparaison n'a ete genere. Verifiez que des runs existent dans 'runs/'."
    exit 1
fi

HEADER_WRITTEN=0
: > "${FINAL_CSV}"
for CSV_PATH in "${PER_DATASET_CSVS[@]}"; do
    if [ "${HEADER_WRITTEN}" -eq 0 ]; then
        head -n 1 "${CSV_PATH}" > "${FINAL_CSV}"
        HEADER_WRITTEN=1
    fi
    tail -n +2 "${CSV_PATH}" >> "${FINAL_CSV}"
done

echo "--- Rapport de comparaison global consolide : ${FINAL_CSV} ---"
echo "--- Job de comparaison termine ---"
