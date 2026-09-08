#!/bin/bash
# =============================================================================
# Configuration SLURM (ETAPE 2/3 : RAPPORT COMPARE) - FISTA / PMMS / iPiano
# (random_search, separement), sur les trois datasets
# =============================================================================
# Execute le mode 'compare' de main.py (src.strategies.compare.run) pour
# chaque jeu de donnees (data_0, data_1, data_2), restreint EXACTEMENT au
# sous-ensemble {fista, pmms, ipiano} en strategie 'random_search', via les
# nouveaux arguments --compare_random_search_models / --compare_unrolling_models
# / --compare_dl_models (cf. main.py). Les strategies 'unrolling' et
# 'deep_learning' sont explicitement exclues (sentinel 'none'), puisque
# FISTA/PMMS/iPiano ne possedent aucune architecture deroulee (pas de
# net.py associe) et que la demande porte uniquement sur une comparaison
# iterative "random_search" separee entre ces trois algorithmes.
#
# Ce job DOIT etre soumis apres que les 'best_params.json' de FISTA, PMMS et
# iPiano aient ete calibres sur les trois datasets (cf.
# scripts/run_fista_pmms_ipiano_random_search.slurm). Il produit, en
# inference uniquement, le rapport comparatif complet (memes sorties que
# scripts/run_compare.slurm) pour chaque dataset :
#   runs/compare/run_set_<id>/<data_folder>/<timestamp>_compare/logs/compare_report.{csv,json,txt}
#   runs/compare/run_set_<id>/<data_folder>/<timestamp>_compare/plots/compare_summary_*.png, compare_timing.png
#
# Usage recommande (3 etapes enchainees) :
#   ./scripts/submit_fista_pmms_ipiano_random_search.sh
# =============================================================================
#SBATCH --job-name=UP3MG_FISTA_PMMS_IPIANO_RS_REPORT
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/compare-fista-pmms-ipiano-report-%j.out
#SBATCH --error=logs/compare-fista-pmms-ipiano-report-%j.err

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
DATA_FOLDERS=(data_0 data_1 data_2)
RS_MODELS="fista,pmms,ipiano"

# Identifiant de l'iteration courante du script (run_set), base sur le job
# Slurm afin de garantir l'independance entre deux soumissions successives.
RUN_SET_TAG="run_set_${SLURM_JOB_ID:-manual}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Mode: compare (FISTA/PMMS/iPiano, random_search) | Critere: ${CRITERION} | run_set: ${RUN_SET_TAG} ---"

# -----------------------------------------------------------------------------
# 2. Execution du mode 'compare' pour chaque jeu de donnees, restreint a
#    {fista, pmms, ipiano} en strategie 'random_search' uniquement.
# -----------------------------------------------------------------------------
for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
    echo "=== [COMPARE][FISTA/PMMS/iPiano] Dataset: ${DATA_FOLDER} ==="

    ./scripts/run.sh \
        --config "${CONFIG}" \
        --gpu 0 \
        --mode compare \
        --data_folder "${DATA_FOLDER}" \
        --run_tag "${RUN_SET_TAG}/${DATA_FOLDER}" \
        --criterion "${CRITERION}" \
        --epochs 30 \
        --compare_random_search_models "${RS_MODELS}" \
        --compare_unrolling_models none \
        --compare_dl_models none || echo "[AVERTISSEMENT] Echec de 'compare' pour ${DATA_FOLDER}, poursuite avec les datasets restants."
done

echo "--- Runs bruts consolides sous : runs/compare/${RUN_SET_TAG}/ ---"
echo "--- Job de comparaison (FISTA/PMMS/iPiano, random_search) termine ---"
