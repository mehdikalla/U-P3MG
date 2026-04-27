#!/bin/bash
# =============================================================
# Configuration SLURM
# =============================================================
#SBATCH --job-name=P3MG_Unrolling
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# -------------------------------------------------------------
# 1. Environnement et Modules
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/12.5
export MPLBACKEND=Agg   # Évite les erreurs liées à l'affichage graphique (X11)

# Se placer dans le répertoire du projet
cd /scratch/mkalla/project_unrolling
source gpu_env/bin/activate

# Création du dossier de logs (si inexistant)
mkdir -p logs

# -------------------------------------------------------------
# 2. Paramètres de l'expérience
# -------------------------------------------------------------
# On définit ici le fichier de config et le mode
CONFIG="config.yaml"
MODE="--train" # ou --test ou --full

echo "--- Démarrage du Job Slurm ID: $SLURM_JOB_ID ---"
echo "--- Mode: $MODE | Config: $CONFIG ---"

# -------------------------------------------------------------
# 3. Exécution via le script run.sh
# -------------------------------------------------------------
# Note: On passe --gpu 0 car Slurm expose le GPU alloué comme le device 0 
# dans le contexte du job.
./scripts/run.sh --config "$CONFIG" --gpu 0 "$MODE"

echo "--- Job terminé avec succès ---"