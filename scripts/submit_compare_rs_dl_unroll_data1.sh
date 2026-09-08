#!/bin/bash
# Fichier : scripts/submit_compare_rs_dl_unroll_data1.sh
#
# Enchaine les 3 etapes SLURM necessaires a l'etude comparative PD/HQ/ISTA/
# P3MG (random_search) + FCUN/FCAE (deep learning) + U-HQ/U-ISTA/U-P3MG
# (unrolling) sur 'data_1' uniquement, avec dependances afterok/afterany
# (meme architecture que scripts/submit_compare.sh) :
#   1) scripts/run_compare_rs_dl_unroll_data1.slurm         -> calibration
#      (job array, entraine/calibre chaque modele demande sur data_1)
#   2) scripts/run_compare_rs_dl_unroll_data1_report.slurm  -> rapport
#      'compare' (depend du succes complet de l'etape 1 : afterok)
#   3) scripts/build_compare_summary.slurm                  -> consolidation
#      CSV globale (depend de l'etape 2 : afterany, cf. scripts/submit_compare.sh)
#
# Usage:
#   ./scripts/submit_compare_rs_dl_unroll_data1.sh

set -euo pipefail

TRAIN_JOB_ID=$(sbatch --parsable scripts/run_compare_rs_dl_unroll_data1.slurm)
echo "[INFO] Job de calibration (job array) soumis : ${TRAIN_JOB_ID}"

# afterany (et non afterok) : src.strategies.compare.run ignore gracieusement
# (avec un message [AVERTISSEMENT]) tout modele dont le checkpoint/best_params
# serait manquant, donc l'echec isole d'une tache du job array de calibration
# ne doit pas empecher la production du rapport pour les modeles ayant reussi.
REPORT_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:"${TRAIN_JOB_ID}" \
    scripts/run_compare_rs_dl_unroll_data1_report.slurm)
echo "[INFO] Job de rapport 'compare' soumis : ${REPORT_JOB_ID} (dependance: afterany:${TRAIN_JOB_ID})"

SUMMARY_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:"${REPORT_JOB_ID}" \
    --export=ALL,COMPARE_JOB_ID="${REPORT_JOB_ID}" \
    scripts/build_compare_summary.slurm)
echo "[INFO] Job de consolidation du rapport soumis : ${SUMMARY_JOB_ID} (dependance: afterany:${REPORT_JOB_ID})"

echo "[INFO] Suivi : squeue -j ${TRAIN_JOB_ID},${REPORT_JOB_ID},${SUMMARY_JOB_ID}"
