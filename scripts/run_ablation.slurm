#!/bin/bash
# =============================================================
# Configuration SLURM - Etude d'ablation U-P3MG (job array)
# Lance en parallele toutes les configurations des 3 volets :
#   a) extern : num_layers  in {5,10,...,60} | num_pd_layers=10
#   b) intern : num_pd_layers in {5,10,...,60} | num_layers=30
#   c) mlp    : architectures du MLP interne | num_layers=30, num_pd_layers=10
# =============================================================
#SBATCH --job-name=Ablation_P3MG
#SBATCH --partition=opis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation-%A_%a.out
#SBATCH --error=logs/ablation-%A_%a.err
#SBATCH --array=0-28

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
    --run_group ablation \
    --run_tag "$RUN_TAG" \
    --num_layers "$NUM_LAYERS" \
    --num_pd_layers "$NUM_PD_LAYERS" \
    "${EXTRA_ARGS[@]}"



# -------------------------------------------------------------
# 4. Consolidation du resultat de ce run dans le CSV du volet
#    (une ligne par tache, ajout protege par flock pour eviter
#    les ecritures concurrentes entre taches du job array)
# -------------------------------------------------------------
# 'RUN_TAG' pointe vers le dossier des runs P3MG bruts (run_set_<id>).
# La synthese (CSV + graphe) est ecrite dans un dossier "results_<id>"
# distinct, cote a cote avec "run_set_<id>", au sein du meme volet.
RUN_SET_DIR="runs/ablation/p3mg/${RUN_TAG}"
RESULTS_DIR="runs/ablation/p3mg/${STUDY_TAG}/results_${SLURM_ARRAY_JOB_ID}"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
LOCK_FILE="$RESULTS_DIR/.summary.lock"
mkdir -p "$RESULTS_DIR"

RUN_DIR=$(find "$RUN_SET_DIR" -mindepth 1 -maxdepth 1 -type d -name "*_full" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)



TABLE_FILE="$RUN_DIR/logs/test_results_table.txt"

if [ -f "$TABLE_FILE" ]; then
    MEAN=$(grep "Mean" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
    MEDIAN=$(grep "Median" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
    STD=$(grep "Std" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
    BEST=$(grep "Min (Best)" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
    WORST=$(grep "Max (Worst)" "$TABLE_FILE" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
else
    MEAN=""; MEDIAN=""; STD=""; BEST=""; WORST=""
fi

(
    flock -x 200
    if [ ! -f "$SUMMARY_CSV" ]; then
        echo "study,num_layers,num_pd_layers,mlp_hidden,run_dir,mean,median,std,best,worst" > "$SUMMARY_CSV"
    fi
    echo "$STUDY,$NUM_LAYERS,$NUM_PD_LAYERS,\"$MLP_HIDDEN\",$RUN_DIR,$MEAN,$MEDIAN,$STD,$BEST,$WORST" >> "$SUMMARY_CSV"
) 200>"$LOCK_FILE"

echo "--- Tache ${SLURM_ARRAY_TASK_ID} terminee (volet: $STUDY) ---"

# -------------------------------------------------------------
# 5. Generation du graphe loss vs. layers pour ce volet
#    (uniquement effectuee par la derniere tache du volet dans le
#    job array, une fois que toutes les autres taches du meme volet
#    ont pu ecrire leur ligne dans le CSV)
# -------------------------------------------------------------
case "$STUDY" in
    extern) LAST_TASK_ID=11; X_COL="num_layers"; XLABEL="Nombre de couches externes (num_layers)"; TITLE="Ablation - Loss vs. nombre de couches externes" ;;
    intern) LAST_TASK_ID=23; X_COL="num_pd_layers"; XLABEL="Nombre de couches internes (num_pd_layers)"; TITLE="Ablation - Loss vs. nombre de couches internes" ;;
    mlp)    LAST_TASK_ID=28; X_COL="mlp_hidden"; XLABEL="Profondeur du MLP interne"; TITLE="Ablation - Loss vs. profondeur du MLP interne" ;;
    *) LAST_TASK_ID=-1 ;;
esac

if [ "${SLURM_ARRAY_TASK_ID}" -eq "${LAST_TASK_ID}" ]; then
    # Attente courte pour laisser les taches concurrentes finaliser leur ecriture CSV
    sleep 30
    python scripts/plot_ablation.py \
        --csv "$SUMMARY_CSV" \
        --x "$X_COL" \
        --xlabel "$XLABEL" \
        --title "$TITLE" \
        --output "$RESULTS_DIR/loss_vs_layers.png" \
        --metric "Loss" || echo "[WARN] Generation du graphe d'ablation echouee pour le volet $STUDY."

fi

