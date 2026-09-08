#!/bin/bash
# Fichier : scripts/submit_fista_pmms_ipiano_random_search.sh
#
# Enchaine les 3 etapes SLURM necessaires a la comparaison FISTA / PMMS /
# iPiano en strategie 'random_search' (separement), sur les trois datasets,
# avec dependances afterany (meme architecture que scripts/submit_compare.sh) :
#   1) scripts/run_fista_pmms_ipiano_random_search.slurm         -> calibration
#      (job array {modele x dataset}, ecrit les best_params.json)
#   2) scripts/run_fista_pmms_ipiano_random_search_report.slurm  -> rapport
#      'compare' restreint a {fista, pmms, ipiano} (depend de l'etape 1)
#   3) scripts/build_compare_summary.slurm                        -> consolidation
#      CSV globale (depend de l'etape 2, cf. scripts/submit_compare.sh)
#
# Usage:
#   ./scripts/submit_fista_pmms_ipiano_random_search.sh

set -euo pipefail

TRAIN_JOB_ID=$(sbatch --parsable scripts/run_fista_pmms_ipiano_random_search.slurm)
echo "[INFO] Job de calibration (job array) soumis : ${TRAIN_JOB_ID}"

# afterany (et non afterok) : src.strategies.compare.run ignore gracieusement
# (avec un message [AVERTISSEMENT]) tout modele dont le best_params.json
# serait manquant, donc l'echec isole d'une tache du job array de calibration
# ne doit pas empecher la production du rapport pour les modeles ayant reussi.
REPORT_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:"${TRAIN_JOB_ID}" \
    scripts/run_fista_pmms_ipiano_random_search_report.slurm)
echo "[INFO] Job de rapport 'compare' soumis : ${REPORT_JOB_ID} (dependance: afterany:${TRAIN_JOB_ID})"

SUMMARY_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:"${REPORT_JOB_ID}" \
    --export=ALL,COMPARE_JOB_ID="${REPORT_JOB_ID}" \
    scripts/build_compare_summary.slurm)
echo "[INFO] Job de consolidation du rapport soumis : ${SUMMARY_JOB_ID} (dependance: afterany:${REPORT_JOB_ID})"

echo "[INFO] Suivi : squeue -j ${TRAIN_JOB_ID},${REPORT_JOB_ID},${SUMMARY_JOB_ID}"
