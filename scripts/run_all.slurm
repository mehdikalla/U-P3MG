#!/bin/bash
# =============================================================================
# Configuration SLURM - Lancement massif (Job Array)
# =============================================================================
# Ce script unique lance en parallele, via un SLURM job array, toutes les
# combinaisons {modele x strategie x dataset} definies ci-dessous. Chaque
# tache du tableau (SLURM_ARRAY_TASK_ID) correspond a une combinaison unique
# et s'execute sur son propre GPU alloue par le scheduler.
#
# Usage:
#   sbatch scripts/run_all.slurm
#
# Pour ne lancer qu'un sous-ensemble des taches (ex: les 10 premieres) :
#   sbatch --array=0-9 scripts/run_all.slurm
# =============================================================================
#SBATCH --job-name=UP3MG_ALL
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --array=0-29%4
# ^ Borne exacte = (nb modeles algo * nb strategies * nb datasets) - 1, soit
#   5 modeles * 2 strategies * 3 datasets = 30 taches (indices 0-29). A
#   recalculer si ALGO_MODELS/DL_MODELS/STRATEGIES/DATA_FOLDERS changent
#   ci-dessous (voir section 1). %4 = 4 taches simultanees au maximum
#   (a ajuster selon les GPUs disponibles sur la partition).

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
# Modeles algorithmiques (supportent 'unrolling' ET 'random_search')
ALGO_MODELS=(ista)

#p3mg pmms ista hq pd

# Modeles purement deep-learning (supportent uniquement 'unrolling')
#DL_MODELS=(fcae fctn fcun resu)


STRATEGIES=(unrolling random_search)

# Jeux de donnees a traiter (sous-dossiers de ./Dataset)
DATA_FOLDERS=(data_0 data_1 data_2)

MODE="full"
CONFIG="config.yaml"

# -----------------------------------------------------------------------------
# 1bis. Hyperparametres specifiques par modele
# -----------------------------------------------------------------------------
# Chaque modele possede sa propre profondeur de deroulement (unrolling) et son
# propre nombre d'iterations pour la boucle iterative statique (random_search).
# Ces valeurs surchargent celles de config.yaml car un argument passe en ligne
# de commande a toujours priorite sur le YAML (cf. main.py::parse_args).
#
# NUM_LAYERS   : profondeur de deroulement (--num_layers) utilisee en mode
#                'unrolling'. Ex: ISTA necessite un tres grand nombre de
#                couches pour converger (proche de son homologue iteratif),
#                alors que P3MG converge en beaucoup moins de couches grace a
#                sa majoration quadratique.
# ALGO_ITERS   : nombre d'iterations de l'algorithme statique (--algo_iters)
#                utilise en mode 'random_search'.
# NB: les modeles purement deep-learning (fcae, fctn, fcun, resu) sont
# instancies dans main.py via ModelClass(N_dim=..., M_dim=...) SANS l'argument
# num_layers : leur profondeur reelle (ex. nombre de couches Transformer pour
# FCTN, nombre de niveaux d'encodeur/decodeur pour FCUN/ResU) est fixee en dur
# par la valeur par defaut du constructeur et n'est pas pilotable depuis la
# ligne de commande. Ils sont donc absents de ce tableau ; leur entree
# --num_layers ne sera simplement jamais transmise (cf. section 4).

# NB: PMMS et P3MG doivent avoir la meme profondeur externe (num_layers /
# algo_iters), car PMMS constitue l'homologue "Majorant-Minimizer" direct de
# P3MG dans les comparaisons de l'etude (memes couches externes, seule la
# sous-resolution interne differe).
declare -A NUM_LAYERS=(
    [p3mg]=30
    [pmms]=30
    [ista]=50
    [hq]=30
    [pd]=50
)


declare -A ALGO_ITERS=(
    [p3mg]=5000
    [pmms]=5000
    [ista]=5000
    [hq]=5000
    [pd]=5000
)


# Nombre de sous-couches Primal-Dual internes (--num_pd_layers), utilise
# uniquement par P3MG et HQ en mode 'unrolling'.
declare -A NUM_PD_LAYERS=(
    [p3mg]=10
    [hq]=10
)


# -----------------------------------------------------------------------------
# 2. Construction de la liste exhaustive des combinaisons
# -----------------------------------------------------------------------------
COMBOS=()

for data in "${DATA_FOLDERS[@]}"; do
    for model in "${ALGO_MODELS[@]}"; do
        for strat in "${STRATEGIES[@]}"; do
            COMBOS+=("${model}|${strat}|${data}")
        done
    done
    # DL_MODELS est desactive (voir section 1 ci-dessus). On ne parcourt le
    # tableau que s'il est effectivement declare, afin d'eviter une erreur
    # de variable non liee sous "set -u" lorsqu'il est commente.
    if declare -p DL_MODELS &>/dev/null; then
        for model in "${DL_MODELS[@]}"; do
            COMBOS+=("${model}|unrolling|${data}")
        done
    fi
done

TOTAL=${#COMBOS[@]}
echo "--- Nombre total de combinaisons definies : ${TOTAL} ---"

# -----------------------------------------------------------------------------
# Avertissement si la borne SBATCH --array (statique, definie en en-tete) ne
# correspond plus au nombre reel de combinaisons calculees dynamiquement
# ci-dessus. Un simple avertissement suffit : le garde-fou de la section 3
# (TASK_ID >= TOTAL) empeche toute execution hors bornes.
# -----------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT}" -ne "${TOTAL}" ]; then
    echo "[AVERTISSEMENT] SLURM_ARRAY_TASK_COUNT (${SLURM_ARRAY_TASK_COUNT}) != TOTAL calcule (${TOTAL})." \
         "Mettez a jour '#SBATCH --array' en tete de ce script (ou relancez avec" \
         "'sbatch --array=0-$((TOTAL - 1))%4 scripts/run_all.slurm')." >&2
fi

# -----------------------------------------------------------------------------
# 3. Selection de la combinaison correspondant a cette tache du job array
# -----------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "[INFO] Tache ${TASK_ID} hors bornes (total=${TOTAL}). Rien a executer."
    exit 0
fi

IFS='|' read -r MODEL STRATEGY DATA_FOLDER <<< "${COMBOS[$TASK_ID]}"

echo "--- Job Slurm ID: ${SLURM_JOB_ID:-N/A} | Tache: ${TASK_ID}/${TOTAL} ---"
echo "--- Modele: ${MODEL} | Strategie: ${STRATEGY} | Dataset: ${DATA_FOLDER} | Mode: ${MODE} ---"

# -----------------------------------------------------------------------------
# 4. Execution via scripts/run.sh
# -----------------------------------------------------------------------------
# Les arguments --num_layers / --num_pd_layers / --algo_iters passes ici
# priment toujours sur config.yaml (cf. main.py::parse_args, ligne "passed_args").
# Ils ne sont ajoutes que si le modele possede une entree dans les tableaux
# correspondants : les modeles purement deep-learning (fcae, fctn, fcun, resu)
# ne recoivent aucun de ces trois arguments car ils ne sont pas exploites par
# main.py pour ces architectures (profondeur fixee en dur dans le constructeur).

EXTRA_ARGS=()

if [ -n "${NUM_LAYERS[$MODEL]+x}" ]; then
    EXTRA_ARGS+=(--num_layers "${NUM_LAYERS[$MODEL]}")
fi
if [ -n "${NUM_PD_LAYERS[$MODEL]+x}" ]; then
    EXTRA_ARGS+=(--num_pd_layers "${NUM_PD_LAYERS[$MODEL]}")
fi
if [ "${STRATEGY}" = "random_search" ] && [ -n "${ALGO_ITERS[$MODEL]+x}" ]; then
    EXTRA_ARGS+=(--algo_iters "${ALGO_ITERS[$MODEL]}")
fi
if [ "$STRATEGY" = "unrolling" ] && [ "${MODEL}" = "ista" ]; then
    EXTRA_ARGS+=(--epochs 50)
fi

echo "--- Arguments specifiques : ${EXTRA_ARGS[*]:-(aucun)} ---"

./scripts/run.sh \
    --config "${CONFIG}" \
    --gpu 0 \
    --mode "${MODE}" \
    --model "${MODEL}" \
    --strategy "${STRATEGY}" \
    --data_folder "${DATA_FOLDER}" \
    "${EXTRA_ARGS[@]}"

echo "--- Tache ${TASK_ID} (${MODEL}/${STRATEGY}/${DATA_FOLDER}) terminee ---"


