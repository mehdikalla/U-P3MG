#!/bin/bash
# =============================================================================
# Configuration SLURM (ETAPE 2/3 : RAPPORT COMPARE) - PD/HQ/ISTA/P3MG
# (random_search) + FCUN/FCAE (deep learning) + U-HQ/U-ISTA/U-P3MG
# (unrolling), data_1 uniquement
# =============================================================================
# Execute le mode 'compare' de main.py (src.strategies.compare.run) sur le
# jeu de donnees 'data_1', restreint EXACTEMENT au sous-ensemble de modeles
# demande via les nouveaux arguments --compare_unrolling_models /
# --compare_random_search_models / --compare_dl_models (cf. main.py) :
#   - random_search      : pd, hq, ista, p3mg
#   - unrolling           : hq, ista, p3mg (-> U-HQ, U-ISTA, U-P3MG)
#   - deep_learning (pur) : fcun, fcae
#
# Ce job DOIT etre soumis apres que les checkpoints/parametres de tous ces
# modeles aient ete calibres sur 'data_1' (cf.
# scripts/run_compare_rs_dl_unroll_data1.slurm). Il produit, en inference
# uniquement, le rapport comparatif complet (memes sorties que
# scripts/run_compare.slurm) :
#   runs/compare/run_set_<id>/data_1/<timestamp>_compare/logs/compare_report.{csv,json,txt}
#   runs/compare/run_set_<id>/data_1/<timestamp>_compare/plots/compare_summary_*.png, compare_timing.png
#
# Usage recommande (3 etapes enchainees) :
#   ./scripts/submit_compare_rs_dl_unroll_data1.sh
# =============================================================================
#SBATCH --job-name=UP3MG_COMPARE_D1_REPORT
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/compare-d1-report-%j.out
#SBATCH --error=logs/compare-d1-report-%j.err

set -uo pipefail
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
# 1. Parametres
# -----------------------------------------------------------------------------
CONFIG="config.yaml"
CRITERION="SNR"
DATA_FOLDER="data_2"

#RS_MODELS="pd,hq,ista,p3mg"
UNROLLING_MODELS="pd"
#DL_MODELS="fcun,fcae"

# Identifiant de l'iteration courante du script (run_set), base sur le job
# Slurm afin de garantir l'independance entre deux soumissions successives.
RUN_SET_TAG="run_set_${SLURM_JOB_ID:-manual}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Mode: compare (data_1, sous-ensemble cible) | Critere: ${CRITERION} | run_set: ${RUN_SET_TAG} ---"

# -----------------------------------------------------------------------------
# 2. Execution du mode 'compare' restreint au sous-ensemble de modeles demande
# -----------------------------------------------------------------------------
./scripts/run.sh \
    --config "${CONFIG}" \
    --gpu 0 \
    --mode compare \
    --data_folder "${DATA_FOLDER}" \
    --run_tag "${RUN_SET_TAG}/${DATA_FOLDER}" \
    --criterion "${CRITERION}" \
    --epochs 30 \
    --compare_unrolling_models "${UNROLLING_MODELS}" \

echo "--- Runs bruts consolides sous : runs/compare/${RUN_SET_TAG}/ ---"
echo "--- Job de comparaison (data_1, sous-ensemble cible) termine ---"
