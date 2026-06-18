#!/bin/bash
# =============================================================
# Configuration SLURM - Test Modèle HQ
# =============================================================
#SBATCH --job-name=Test_HQ
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00  # Temps réduit pour un simple test
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# -------------------------------------------------------------
# 1. Environnement et Modules
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/13.1
export MPLBACKEND=Agg   # Évite les erreurs liées à l'affichage graphique (X11)

# Se placer dans le répertoire du projet
cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

# Création du dossier de logs (si inexistant)
mkdir -p logs

# -------------------------------------------------------------
# 2. Paramètres de l'expérience
# -------------------------------------------------------------
CONFIG="config.yaml"
MODE="test"
  # Mode de fonctionnement (train/test)
echo "--- Démarrage du Job Slurm ID: $SLURM_JOB_ID ---"
echo "--- Mode: $MODE | Config: $CONFIG ---"

# -------------------------------------------------------------
# 3. Exécution via le script run.sh
# -------------------------------------------------------------
# On passe l'argument --model pour que le main.py sélectionne HQ
./scripts/run.sh --config "$CONFIG" --gpu 0 --mode "$MODE"

echo "--- Job de test terminé ---"