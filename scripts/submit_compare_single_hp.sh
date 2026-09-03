#!/bin/bash
# Fichier : scripts/submit_compare_single_hp.sh
#
# Variante de scripts/submit_compare.sh : soumet la comparaison complete,
# ou chaque modele calibre par random search (P3MG, ISTA, HQ, PD) est
# recalibre avec un SEUL tirage d'hyperparametre (--n_trials 1) au lieu des
# 100 tirages habituels, afin d'evaluer l'impact d'une recherche
# d'hyperparametre degeneree sur les performances rapportees en mode
# 'compare'.
#
# Soumet la comparaison en 2 jobs SLURM enchaines (meme architecture que
# scripts/submit_compare.sh) :
#   1) scripts/run_compare_single_hp.slurm (job unique : recalibration a 1
#      tirage puis mode 'compare' sur tous les datasets)
#   2) scripts/build_compare_summary_single_hp.slurm (job unique, dependance
#      --dependency=afterany sur le job 1), qui consolide les rapports
#      individuels compare_report.csv en un seul all_compare_report.csv
#      une fois TOUS les datasets traites, qu'ils aient reussi ou echoue
#      individuellement.
#
# Usage:
#   ./scripts/submit_compare_single_hp.sh

set -euo pipefail

COMPARE_JOB_ID=$(sbatch --parsable scripts/run_compare_single_hp.slurm)
echo "[INFO] Job de comparaison (1 hyperparametre) soumis : ${COMPARE_JOB_ID}"

SUMMARY_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:"${COMPARE_JOB_ID}" \
    --export=ALL,COMPARE_JOB_ID="${COMPARE_JOB_ID}" \
    scripts/build_compare_summary_single_hp.slurm)
echo "[INFO] Job de consolidation du rapport soumis : ${SUMMARY_JOB_ID} (dependance: afterany:${COMPARE_JOB_ID})"

echo "[INFO] Suivi : squeue -j ${COMPARE_JOB_ID},${SUMMARY_JOB_ID}"
