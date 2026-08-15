#!/bin/bash
# =============================================================================
# Job SLURM - Consolidation du rapport CSV global de comparaison, decouple
# de l'execution des runs 'compare' (meme architecture que
# scripts/plot_ablation.slurm pour l'etude d'ablation).
#
# Ce job est concu pour etre soumis avec une dependance `afterany` sur le
# job scripts/run_compare.slurm, ce qui garantit qu'il s'execute une seule
# fois APRES que tous les datasets aient ete traites, qu'ils aient reussi
# ou echoue individuellement.
#
# La consolidation est effectuee par scripts/build_compare_summary.py, qui
# lit directement chaque logs/compare_report.csv produit par
# src.strategies.compare._save_report, plutot qu'une concatenation bash
# (head/tail) fragile executee inline dans le job principal.
#
# Usage (voir aussi scripts/submit_compare.sh pour l'enchainement
# automatique) :
#   COMPARE_JOB_ID=$(sbatch --parsable scripts/run_compare.slurm)
#   sbatch --dependency=afterany:${COMPARE_JOB_ID} \
#          --export=ALL,COMPARE_JOB_ID=${COMPARE_JOB_ID} \
#          scripts/build_compare_summary.slurm
# =============================================================================
#SBATCH --job-name=Build_Compare_Summary
#SBATCH --partition=opis
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/build_compare_summary-%A.out
#SBATCH --error=logs/build_compare_summary-%A.err

set -uo pipefail
module purge
export MPLBACKEND=Agg

cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

mkdir -p logs

# COMPARE_JOB_ID doit correspondre au SLURM_JOB_ID du job run_compare.slurm
# dont on veut consolider les resultats (transmis via --export ou en argument).
COMPARE_JOB_ID="${COMPARE_JOB_ID:-${1:-}}"
if [ -z "$COMPARE_JOB_ID" ]; then
    echo "[ERREUR] COMPARE_JOB_ID non fourni (variable d'environnement ou argument positionnel)." >&2
    exit 1
fi

RUN_SET_DIR="runs/compare/run_set_${COMPARE_JOB_ID}"
RESULTS_DIR="Results/compare/results_${COMPARE_JOB_ID}"
FINAL_CSV="${RESULTS_DIR}/all_compare_report.csv"

echo "--- Consolidation du rapport de comparaison pour le job ${COMPARE_JOB_ID} ---"

python scripts/build_compare_summary.py \
    --run_set_dir "$RUN_SET_DIR" \
    --output "$FINAL_CSV"

STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "[ERREUR] Consolidation du rapport de comparaison echouee."
    exit $STATUS
fi

echo "--- Rapport de comparaison global consolide : ${FINAL_CSV} ---"
