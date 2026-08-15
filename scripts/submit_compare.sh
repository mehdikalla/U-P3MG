#!/bin/bash
# Fichier : scripts/submit_compare.sh
#
# Soumet la comparaison complete en 2 jobs SLURM enchaines (meme
# architecture que scripts/submit_ablation.sh) :
#   1) scripts/run_compare.slurm (job unique : mode 'compare' sur tous
#      les datasets)
#   2) scripts/build_compare_summary.slurm (job unique, dependance
#      --dependency=afterany sur le job 1), qui consolide les rapports
#      individuels compare_report.csv en un seul all_compare_report.csv
#      une fois TOUS les datasets traites, qu'ils aient reussi ou echoue
#      individuellement.
#
# Ceci corrige le bug ou la consolidation du CSV global etait executee a
# la fin du meme script sequentiel sous 'set -euo pipefail' : l'echec
# d'un seul dataset avant d'atteindre cette section empechait le CSV
# global d'etre jamais produit, meme si les autres datasets avaient
# reussi.
#
# Usage:
#   ./scripts/submit_compare.sh

set -euo pipefail

COMPARE_JOB_ID=$(sbatch --parsable scripts/run_compare.slurm)
echo "[INFO] Job de comparaison soumis : ${COMPARE_JOB_ID}"

SUMMARY_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:"${COMPARE_JOB_ID}" \
    --export=ALL,COMPARE_JOB_ID="${COMPARE_JOB_ID}" \
    scripts/build_compare_summary.slurm)
echo "[INFO] Job de consolidation du rapport soumis : ${SUMMARY_JOB_ID} (dependance: afterany:${COMPARE_JOB_ID})"

echo "[INFO] Suivi : squeue -j ${COMPARE_JOB_ID},${SUMMARY_JOB_ID}"
