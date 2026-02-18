#!/bin/bash
#
# =============================================================
# Configuration SLURM pour le Random Search P3MG
# =============================================================
#SBATCH --job-name=Train_P3MG
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:30:00
#SBATCH --output=logs/P3MG_random//slurm-%j.out
#SBATCH --error=logs/P3MG_random/slurm-%j.err

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
# NOTE : Bornes Lambda 10^-8 à 1.0
python main.py \
    --model p3mg \
    --strategy random_search \
    --mode full \
    --device cuda \
    --batch_size 10 \
    --criterion TSNR \
    --n_samples 100 \
    --algo_iters 10000 \
    --lmbd_min 1e-8 \
    --lmbd_max 1.0 \
    --tau_min 0.01 \
    --tau_max 1.99

echo "--- Fin du job Slurm ---"