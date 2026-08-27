#!/bin/bash
# =============================================================
# Configuration SLURM - Etude d'ablation U-P3MG (job array)
# Lance en parallele toutes les configurations des 3 volets :
#   a) extern : num_layers  in {5,10,...,60} | num_pd_layers=10
#   b) intern : num_pd_layers in {5,10,...,60} | num_layers=30
#   c) mlp    : architectures du MLP interne | num_layers=30, num_pd_layers=10
#
# Contraintes imposees :
#   - Entrainement realise exclusivement sur le dataset 'data_0'
#     (--data_folder data_0 explicite sur chaque appel de run.sh).
#   - Au plus 4 GPU utilises simultanement : le job array est throttle
#     via le suffixe '%4' de --array, limitant a 4 le nombre de taches
#     (donc de GPU, --gres=gpu:1 par tache) executees en parallele.
# =============================================================
#SBATCH --job-name=Ablation_P3MG
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/ablation-%A_%a.out
#SBATCH --error=logs/ablation-%A_%a.err
#SBATCH --array=0-28%4

# -------------------------------------------------------------
# 1. Environnement et Modules
# -------------------------------------------------------------
set -euo pipefail
module purge
module load cuda/13.1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1   # Force le flush immediat des print() dans les logs Slurm


cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

mkdir -p logs

CONFIG="config.yaml"

# -------------------------------------------------------------
# 2. Liste exhaustive des configurations d'ablation
#    Format par entree : "<study>;<num_layers>;<num_pd_layers>;<mlp_hidden>"
#    (mlp_hidden vide => valeur par defaut du modele)
# -------------------------------------------------------------
CONFIGS=(
    # --- a) extern : num_pd_layers fixe = 10, num_layers varie ---
    "extern;5;10;"
    "extern;10;10;"
    "extern;15;10;"
    "extern;20;10;"
    "extern;25;10;"
    "extern;30;10;"
    "extern;35;10;"
    "extern;40;10;"
    "extern;45;10;"
    "extern;50;10;"
    "extern;55;10;"
    "extern;60;10;"
    # --- b) intern : num_layers fixe = 30, num_pd_layers varie ---
    "intern;30;5;"
    "intern;30;10;"
    "intern;30;15;"
    "intern;30;20;"
    "intern;30;25;"
    "intern;30;30;"
    "intern;30;35;"
    "intern;30;40;"
    "intern;30;45;"
    "intern;30;50;"
    "intern;30;55;"
    "intern;30;60;"
    # --- c) mlp : num_layers=30, num_pd_layers=10, architecture du MLP varie ---
    "mlp;30;10;25"
    "mlp;30;10;50,25"
    "mlp;30;10;50,25,12"
    "mlp;30;10;100,50,25,12"
    "mlp;30;10;100,50,25,12,6"
)

ENTRY="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS=';' read -r STUDY NUM_LAYERS NUM_PD_LAYERS MLP_HIDDEN <<< "$ENTRY"

# Espace de noms dedie a l'ablation, isole de 'data_1' (utilise par
# run_all.slurm et run_compare.slurm) afin qu'aucun run standard/compare ne
# soit pollue par les checkpoints d'ablation, et inversement.
#
# Chaque soumission sbatch (SLURM_ARRAY_JOB_ID) constitue une iteration
# independante de l'etude ("run_set"). Toutes les taches du meme volet
# (STUDY) issues d'une meme soumission partagent donc le meme run_set,
# sans risque de collision entre deux soumissions successives.
STUDY_TAG="ablation_${STUDY}"
RUN_SET_TAG="run_set_${SLURM_ARRAY_JOB_ID}"
RUN_TAG="${STUDY_TAG}/${RUN_SET_TAG}"


echo "--- Job Slurm ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ---"
echo "--- Volet: $STUDY | num_layers=$NUM_LAYERS | num_pd_layers=$NUM_PD_LAYERS | mlp_hidden=$MLP_HIDDEN | run_tag=$RUN_TAG ---"

# -------------------------------------------------------------
# 3. Execution
# -------------------------------------------------------------
EXTRA_ARGS=()
if [ -n "$MLP_HIDDEN" ]; then
    EXTRA_ARGS+=(--mlp_hidden "$MLP_HIDDEN")
fi

./scripts/run.sh --config "$CONFIG" --gpu 0 --mode full \
    --model p3mg \
    --strategy unrolling \
    --data_folder data_0 \
    --run_group ablation \
    --run_tag "$RUN_TAG" \
    --num_layers "$NUM_LAYERS" \
    --num_pd_layers "$NUM_PD_LAYERS" \
    --epochs 30 \
    "${EXTRA_ARGS[@]}"



echo "--- Tache ${SLURM_ARRAY_TASK_ID} terminee (volet: $STUDY) ---"

# -------------------------------------------------------------
# 4. Consolidation (summary.csv) et generation du graphe
# -------------------------------------------------------------
# NI L'UNE NI L'AUTRE ne sont plus effectuees ici. Deux defauts distincts
# de l'ancienne implementation ont ete corriges en deportant ces etapes
# vers un job SLURM independant (scripts/plot_ablation.slurm) :
#
#   1) Le graphe n'etait genere QUE par la derniere tache du volet
#      (SLURM_ARRAY_TASK_ID sentinelle, systematiquement la config la plus
#      lourde). Sous `set -euo pipefail`, l'echec de cette seule tache
#      empechait toute generation, meme si les autres avaient reussi.
#
#   2) Le CSV summary.csv etait reconstruit ligne par ligne par CHAQUE
#      tache via un parsing awk fragile du tableau test_results_table.txt,
#      avec ecriture concurrente protegee par flock. Ce mecanisme dupliquait
#      la configuration (NUM_LAYERS/NUM_PD_LAYERS/MLP_HIDDEN) depuis le
#      tableau bash CONFIGS plutot que depuis la configuration reellement
#      executee, risquant une desynchronisation silencieuse.
#
# scripts/build_ablation_summary.py reconstruit desormais summary.csv en
# relisant directement, pour chaque sous-run trouve sur disque, le fichier
# logs/run_config.json (configuration reelle, ecrite par
# src.strategies.network.save_config) et logs/test_results_table.txt
# (statistiques de test). Voir scripts/submit_ablation.sh pour
# l'enchainement automatique des deux jobs.



