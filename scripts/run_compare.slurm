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
# Structure de sortie (une soumission sbatch = un "run_set", identifie par
# SLURM_JOB_ID, pour pouvoir relancer l'etude sans ecraser les resultats
# precedents) :
#   runs/compare/run_set_<id>/<data_folder>/<timestamp>_compare/  -> runs bruts
#   Results/compare/results_<id>/all_compare_report.csv           -> synthese
#
# NOTE IMPORTANTE (meme architecture que scripts/run_ablation.slurm) :
# la consolidation du CSV global N'EST PLUS effectuee ici. L'ancienne
# implementation concatenait les rapports individuels (head/tail) a la
# fin de CE MEME script sequentiel sous 'set -euo pipefail' : si le run
# 'compare' d'un dataset echouait avant la section de consolidation, le
# CSV global final risquait de ne jamais etre produit (meme defaut de
# conception que la tache "sentinelle" corrigee dans l'etude d'ablation).
#
# La consolidation est desormais deportee vers un job SLURM independant
# (scripts/build_compare_summary.slurm), soumis avec une dependance
# `--dependency=afterany:<job_id>` sur ce job, garantissant qu'elle
# s'execute une seule fois APRES que tous les datasets aient ete traites,
# qu'ils aient reussi ou echoue individuellement. Voir
# scripts/submit_compare.sh pour l'enchainement automatique des deux jobs.
#
# Usage:
#   ./scripts/submit_compare.sh
# =============================================================================
#SBATCH --job-name=UP3MG_COMPARE
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/compare-%j.out
#SBATCH --error=logs/compare-%j.err

set -uo pipefail
module purge
module load cuda/13.1
export MPLBACKEND=Agg   # Evite les erreurs liees a l'affichage graphique (X11)
export PYTHONUNBUFFERED=1   # Force le flush immediat des print() dans les logs Slurm


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

# Identifiant de l'iteration courante du script (run_set), base sur le job
# Slurm afin de garantir l'independance entre deux soumissions successives.
RUN_SET_TAG="run_set_${SLURM_JOB_ID:-manual}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Mode: compare | Critere: ${CRITERION} | run_set: ${RUN_SET_TAG} ---"

# -----------------------------------------------------------------------------
# 2. Execution du mode 'compare' pour chaque jeu de donnees
# -----------------------------------------------------------------------------
# L'echec d'un dataset (OOM, checkpoint manquant, crash) n'interrompt pas les
# suivants : chaque iteration est isolee via '|| echo [...]' pour que la
# consolidation ulterieure (job separe) dispose de tous les rapports
# effectivement produits, sans dependre de la reussite de chacun.
for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
    echo "=== [COMPARE] Dataset: ${DATA_FOLDER} ==="

    ./scripts/run.sh \
        --config "${CONFIG}" \
        --gpu 0 \
        --mode compare \
        --data_folder "${DATA_FOLDER}" \
        --run_tag "${RUN_SET_TAG}/${DATA_FOLDER}" \
        --criterion "${CRITERION}" \
        --epochs 30 || echo "[AVERTISSEMENT] Echec de 'compare' pour ${DATA_FOLDER}, poursuite avec les datasets restants."
done

echo "--- Runs bruts consolides sous : runs/compare/${RUN_SET_TAG}/ ---"
echo "--- Job de comparaison termine ---"
