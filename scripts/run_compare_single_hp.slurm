#!/bin/bash
# =============================================================================
# Configuration SLURM - Comparaison finale de tous les algorithmes (mode
# compare), variante "single hyperparameter" pour la strategie random_search.
# =============================================================================
# Variante de scripts/run_compare.slurm : avant d'executer le mode 'compare'
# de main.py pour chaque jeu de donnees, ce script recalibre chaque modele
# algorithmique compatible random_search (P3MG, ISTA, HQ, PD - cf.
# src.strategies.compare.RANDOM_SEARCH_MODELS) avec UN SEUL tirage
# d'hyperparametre (--n_trials 1) au lieu des 100 tirages habituels
# (config.yaml: n_trials). Le mode 'compare' s'appuie ensuite sur ce nouveau
# 'best_params.json' (le plus recent produit dans
# 'runs/<model>/random_search/<data_folder>/'), automatiquement selectionne
# par src.strategies.compare.find_latest_best_params, qui trie les runs par
# horodatage decroissant, sans aucune modification du code de comparaison.
#
# Les checkpoints 'unrolling' (P3MG, HQ) ne sont pas affectes par cette
# recalibration et restent ceux les plus recemment entraines dans 'runs/'.
#
# ATTENTION : la recalibration avec --n_trials 1 produit un nouveau run
# 'runs/<model>/random_search/<data_folder>/<timestamp>_train/', qui devient
# le plus recent pour ce (modele, data_folder). Toute execution ulterieure
# de scripts/run_compare.slurm (comparaison "normale") reutilisera donc ce
# calibrage a un seul tirage tant qu'un nouveau random search complet
# (n_trials=100) n'aura pas ete relance. Pour rejouer une comparaison
# "normale", relancer d'abord scripts/run_random_search_all.slurm.
#
# Structure de sortie (une soumission sbatch = un "run_set", identifie par
# SLURM_JOB_ID) :
#   runs/<model>/random_search/<data_folder>/<timestamp>_train/  -> calibrages a 1 tirage
#   runs/compare/run_set_<id>/<data_folder>/<timestamp>_compare/ -> runs bruts de comparaison
#   Results/compare_single_hp/results_<id>/all_compare_report.csv -> synthese
#
# Usage:
#   ./scripts/submit_compare_single_hp.sh
# =============================================================================
#SBATCH --job-name=UP3MG_COMPARE_1HP
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/compare_1hp-%j.out
#SBATCH --error=logs/compare_1hp-%j.err

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
# Modeles calibres par random search dans le mode 'compare' (doit rester
# synchronise avec src.strategies.compare.RANDOM_SEARCH_MODELS).
RANDOM_SEARCH_MODELS=(p3mg ista hq pd)
N_TRIALS_SINGLE=1

# Identifiant de l'iteration courante du script (run_set), base sur le job
# Slurm afin de garantir l'independance entre deux soumissions successives.
RUN_SET_TAG="run_set_${SLURM_JOB_ID:-manual}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Mode: compare (1 hyperparametre) | Critere: ${CRITERION} | run_set: ${RUN_SET_TAG} ---"

# -----------------------------------------------------------------------------
# 2. Recalibration des modeles random_search avec un seul tirage d'hyperparametre
# -----------------------------------------------------------------------------
# NOTE : --run_tag n'est volontairement PAS surcharge ici (il reste egal a
# --data_folder par defaut, cf. main.py::setup_paths) afin que le nouveau
# 'best_params.json' soit ecrit dans 'runs/<model>/random_search/<data_folder>/',
# le seul chemin explore par src.strategies.compare.find_latest_best_params.
for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
    for MODEL in "${RANDOM_SEARCH_MODELS[@]}"; do
        echo "=== [CALIBRATION 1-TRIAL] Modele: ${MODEL} | Dataset: ${DATA_FOLDER} ==="

        ./scripts/run.sh \
            --config "${CONFIG}" \
            --gpu 0 \
            --mode train \
            --model "${MODEL}" \
            --strategy random_search \
            --data_folder "${DATA_FOLDER}" \
            --n_trials "${N_TRIALS_SINGLE}" \
            --criterion "${CRITERION}" || echo "[AVERTISSEMENT] Echec de la calibration 1-trial pour ${MODEL}/${DATA_FOLDER}, poursuite avec les combinaisons restantes."
    done
done

# -----------------------------------------------------------------------------
# 3. Execution du mode 'compare' pour chaque jeu de donnees
# -----------------------------------------------------------------------------
# L'echec d'un dataset (OOM, checkpoint manquant, crash) n'interrompt pas les
# suivants : chaque iteration est isolee via '|| echo [...]' pour que la
# consolidation ulterieure (job separe) dispose de tous les rapports
# effectivement produits, sans dependre de la reussite de chacun.
for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
    echo "=== [COMPARE 1-HP] Dataset: ${DATA_FOLDER} ==="

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
echo "--- Job de comparaison (1 hyperparametre) termine ---"
