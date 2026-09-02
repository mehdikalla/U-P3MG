#!/bin/bash
# =============================================================
# Configuration SLURM - P3MG sans FC_block (lambda statique)
# Entrainement + test sur le dataset 'data_0'.
#
# Contrairement au P3MG standard, lambda n'est pas produit par un
# FC_block (MLP dependant de y) mais entraine directement comme un
# nn.Parameter scalaire par couche (--use_fc_lambda 0).
#
# Un run_tag dedie ("no_fc_block") isole les checkpoints et logs de
# cette etude dans un chemin particulier
# (runs/p3mg/unrolling/no_fc_block/), sans interference avec les
# runs standard bases sur 'data_0' (--data_folder data_0).
# =============================================================
#SBATCH --job-name=P3MG_noFC
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/no_fc_block-%j.out
#SBATCH --error=logs/no_fc_block-%j.err

# -------------------------------------------------------------
# 1. Environnement et Modules
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/13.1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

mkdir -p logs

# -------------------------------------------------------------
# 2. Parametres de l'experience
# -------------------------------------------------------------
CONFIG="config.yaml"
DATA_FOLDER="data_0"
RUN_TAG="no_fc_block"

echo "--- Demarrage du Job Slurm ID: $SLURM_JOB_ID ---"
echo "--- Modele: P3MG (sans FC_block) | Data: $DATA_FOLDER | Run tag: $RUN_TAG ---"

# -------------------------------------------------------------
# 3. Execution (train + test) via scripts/run.sh
# -------------------------------------------------------------
./scripts/run.sh --config "$CONFIG" --gpu 0 --mode full \
    --model p3mg \
    --strategy unrolling \
    --data_folder "$DATA_FOLDER" \
    --run_tag "$RUN_TAG" \
    --use_fc_lambda 0

echo "--- Job P3MG (sans FC_block) termine ---"
