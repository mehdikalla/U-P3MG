#!/bin/bash
# Fichier : scripts/submit_ablation.sh
#
# Soumet l'etude d'ablation complete en 2 jobs SLURM enchaines :
#   1) scripts/run_ablation.slurm (job array : tous les entrainements)
#   2) scripts/plot_ablation.slurm (job unique, dependance --dependency=afterany
#      sur le job array), qui genere les 3 graphes loss_vs_layers.png
#      (extern, intern, mlp) une fois TOUTES les taches terminees,
#      qu'elles aient reussi ou echoue individuellement.
#
# Ceci corrige le bug ou le graphe n'etait genere que si la tache
# "sentinelle" (derniere config, la plus lourde) du job array reussissait,
# et n'etait donc jamais produit en cas d'echec de cette seule tache.
#
# Usage:
#   ./scripts/submit_ablation.sh

set -euo pipefail

ARRAY_JOB_ID=$(sbatch --parsable scripts/run_ablation.slurm)
echo "[INFO] Job array d'ablation soumis : ${ARRAY_JOB_ID}"

PLOT_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:"${ARRAY_JOB_ID}" \
    --export=ALL,ARRAY_JOB_ID="${ARRAY_JOB_ID}" \
    scripts/plot_ablation.slurm)
echo "[INFO] Job de generation des graphes soumis : ${PLOT_JOB_ID} (dependance: afterany:${ARRAY_JOB_ID})"

echo "[INFO] Suivi : squeue -j ${ARRAY_JOB_ID},${PLOT_JOB_ID}"
