#!/bin/bash
# =============================================================================
# Configuration SLURM - Lancement massif des modeles Deep Learning purs
# =============================================================================
# Ce script lance en parallele, via un SLURM job array, uniquement les
# modeles purement deep-learning (fcae, fctn, fcun, resu), qui ne dependent
# d'aucun algorithme d'optimisation deroule (pas de algo.py) et ne supportent
# donc que la strategie 'unrolling'. Chaque tache du tableau
# (SLURM_ARRAY_TASK_ID) correspond a une combinaison {modele x dataset} et
# s'execute sur son propre GPU alloue par le scheduler.
#
# Usage:
#   sbatch scripts/run_DL.slurm
#
# Pour ne lancer qu'un sous-ensemble des taches (ex: les 3 premieres) :
#   sbatch --array=0-2 scripts/run_DL.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_DL
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --array=0-999%16
# ^ Borne superieure large (recalculee/validee au runtime), %16 = 16 taches
#   simultanees au maximum (a ajuster selon les GPUs disponibles sur la
#   partition).

set -euo pipefail
module purge
module load cuda/13.1
export MPLBACKEND=Agg   # Evite les erreurs liees a l'affichage graphique (X11)

cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

mkdir -p logs

# -----------------------------------------------------------------------------
# 1. Definition de l'espace de recherche
# -----------------------------------------------------------------------------
# Modeles purement deep-learning : aucun n'expose de algo.py, ils n'existent
# donc qu'en strategie 'unrolling' (pas de random_search possible, faute
# d'algorithme iteratif statique associe).
DL_MODELS=(fcae fctn fcun resu)

STRATEGY="unrolling"

# Jeux de donnees a traiter (sous-dossiers de ./Dataset)
DATA_FOLDERS=(data_0 data_1 data_2)

MODE="full"
CONFIG="config.yaml"

# NB: ces modeles sont instancies dans main.py via ModelClass(N_dim=..., M_dim=...)
# sans l'argument num_layers : leur profondeur reelle (nombre de couches
# Transformer pour FCTN, nombre de niveaux d'encodeur/decodeur pour FCUN/ResU,
# etc.) est fixee en dur par la valeur par defaut du constructeur et n'est pas
# pilotable depuis la ligne de commande. Aucun hyperparametre supplementaire
# n'est donc a transmettre ici.

# -----------------------------------------------------------------------------
# 2. Construction de la liste exhaustive des combinaisons
# -----------------------------------------------------------------------------
COMBOS=()

for data in "${DATA_FOLDERS[@]}"; do
    for model in "${DL_MODELS[@]}"; do
        COMBOS+=("${model}|${data}")
    done
done

TOTAL=${#COMBOS[@]}
echo "--- Nombre total de combinaisons definies : ${TOTAL} ---"

# -----------------------------------------------------------------------------
# 3. Selection de la combinaison correspondant a cette tache du job array
# -----------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "[INFO] Tache ${TASK_ID} hors bornes (total=${TOTAL}). Rien a executer."
    exit 0
fi

IFS='|' read -r MODEL DATA_FOLDER <<< "${COMBOS[$TASK_ID]}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Tache: ${TASK_ID}/${TOTAL} ---"
echo "--- Modele: ${MODEL} | Strategie: ${STRATEGY} | Dataset: ${DATA_FOLDER} | Mode: ${MODE} ---"

# -----------------------------------------------------------------------------
# 4. Execution via scripts/run.sh
# -----------------------------------------------------------------------------
./scripts/run.sh \
    --config "${CONFIG}" \
    --gpu 0 \
    --mode "${MODE}" \
    --model "${MODEL}" \
    --strategy "${STRATEGY}" \
    --data_folder "${DATA_FOLDER}"

echo "--- Tache ${TASK_ID} (${MODEL}/${STRATEGY}/${DATA_FOLDER}) terminee ---"
