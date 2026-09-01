#!/bin/bash
# =============================================================================
# Configuration SLURM - Comparaison qualitative rapide sur un signal unique
# =============================================================================
# Tire un signal au hasard dans le jeu de test (data_folder configurable),
# applique toutes les methodes disponibles (unrolling, random_search, deep
# learning pur) avec les derniers poids/parametres calibres trouves dans
# 'runs/', et sauvegarde un fichier .png par methode dans un dossier dedie
# sous 'quick_compare_results/'.
#
# Usage:
#   sbatch scripts/quick_signal_compare.slurm
#   sbatch scripts/quick_signal_compare.slurm data_2       # data_folder custom
#   sbatch scripts/quick_signal_compare.slurm data_1 0    # index de signal fixe
# =============================================================================
#SBATCH --job-name=UP3MG_QUICK_COMPARE
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/quick_compare-%j.out
#SBATCH --error=logs/quick_compare-%j.err

set -uo pipefail
module purge
module load cuda/13.1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

mkdir -p logs

DATA_FOLDER="${1:-data_0}"
SIGNAL_INDEX="${2:-63}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Quick Compare | Data: ${DATA_FOLDER} ---"

INDEX_ARG=()
if [ -n "${SIGNAL_INDEX}" ]; then
    INDEX_ARG=(--index "${SIGNAL_INDEX}")
fi

python scripts/quick_signal_compare.py \
    --config config.yaml \
    --data_folder "${DATA_FOLDER}" \
    --output_dir "quick_compare_results/job_${SLURM_JOB_ID:-manual}" \
    "${INDEX_ARG[@]}"

echo "--- Job termine ---"
