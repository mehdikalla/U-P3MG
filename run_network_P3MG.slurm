#!/bin/bash
#
# =============================================================
# Configuration SLURM pour l'Entrainement P3MG (Unrolling)
# =============================================================
#SBATCH --job-name=Train_P3MG
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4            # Augmenté légèrement pour le chargement des données
#SBATCH --mem=16G                    # Augmenté par sécurité
#SBATCH --time=5:30:00
#SBATCH --output=logs/P3MG_unrolling/slurm-%j.out        # Log à la racine (ignoré par ton gitignore)
#SBATCH --error=logs/P3MG_unrolling/slurm-%j.err

# -------------------------------------------------------------
# Configuration ENVIRONNEMENT
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/12.5
export MPLBACKEND=Agg   

# Adapter le chemin si nécessaire
cd /scratch/mkalla/project_unrolling
source gpu_env/bin/activate

echo "--- Démarrage de l'entrainement P3MG (Unrolled) ---"

# -------------------------------------------------------------
# EXÉCUTION DU CODE PYTHON
# -------------------------------------------------------------

# Lancement de main.py avec les NOUVEAUX arguments
# On utilise le mode 'full' pour entrainer PUIS tester dans la foulée
python main.py \
    --model p3mg \
    --strategy unrolling \
    --mode full \
    --device cuda \
    --num_layers 25 \
    --num_pd_layers 10 \
    --epochs 50 \
    --lr 9e-3 \
    --batch_size 10 \
    --criterion TSNR

echo "--- Fin du job Slurm ---"