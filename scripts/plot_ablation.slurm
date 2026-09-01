#!/bin/bash
# =============================================================
# Job SLURM - Reconstruction du summary.csv et generation des
# graphes loss vs. layers pour l'etude d'ablation, decouple de
# l'execution des runs.
#
# Ce job est concu pour etre soumis avec une dependance `afterany`
# sur le job array de scripts/run_ablation.slurm, ce qui garantit
# qu'il s'execute une seule fois APRES que toutes les taches du
# tableau se soient terminees, qu'elles aient reussi ou echoue.
#
# Pour chaque volet (extern, intern, mlp), le summary.csv est
# reconstruit directement a partir des sous-runs presents sur disque
# (scripts/build_ablation_summary.py lit logs/run_config.json et
# logs/test_results_table.txt de chaque sous-run), plutot que d'etre
# assemble ligne par ligne pendant l'execution du job array (ancien
# mecanisme fragile a base de parsing awk + flock, cf. commentaires
# dans scripts/run_ablation.slurm).
#
# Usage (voir aussi scripts/submit_ablation.sh pour l'enchainement
# automatique) :
#   ARRAY_JOB_ID=$(sbatch --parsable scripts/run_ablation.slurm)
#   sbatch --dependency=afterany:${ARRAY_JOB_ID} \
#          --export=ALL,ARRAY_JOB_ID=${ARRAY_JOB_ID} \
#          scripts/plot_ablation.slurm
# =============================================================
#SBATCH --job-name=Plot_Ablation_P3MG
#SBATCH --partition=opis
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/plot_ablation-%A.out
#SBATCH --error=logs/plot_ablation-%A.err

set -uo pipefail
module purge
export MPLBACKEND=Agg

cd /scratch/mkalla/U-P3MG
eval "$(conda shell.bash hook)"
conda init
conda activate U-P3MG

mkdir -p logs

# ARRAY_JOB_ID doit correspondre au SLURM_ARRAY_JOB_ID du job run_ablation.slurm
# dont on veut consolider les resultats (transmis via --export ou en argument).
ARRAY_JOB_ID="${ARRAY_JOB_ID:-${1:-}}"
if [ -z "$ARRAY_JOB_ID" ]; then
    echo "[ERREUR] ARRAY_JOB_ID non fourni (variable d'environnement ou argument positionnel)." >&2
    exit 1
fi

echo "--- Consolidation et generation des graphes d'ablation pour le job array ${ARRAY_JOB_ID} ---"

declare -A X_COL=( [extern]="num_layers" [intern]="num_pd_layers" [mlp]="mlp_hidden" )
declare -A XLABEL=(
    [extern]="Number of external layers K"
    [intern]="Number of internal layers J"
    [mlp]="Internal MLP depth"
)
declare -A TITLE=(
    [extern]=" "
    [intern]=" "
    [mlp]=" "
)

STATUS=0

for STUDY in extern intern mlp; do
    STUDY_TAG="ablation_${STUDY}"
    RUN_SET_DIR="runs/ablation/p3mg/${STUDY_TAG}/run_set_${ARRAY_JOB_ID}"
    RESULTS_DIR="runs/ablation/p3mg/${STUDY_TAG}/results_${ARRAY_JOB_ID}"
    SUMMARY_CSV="$RESULTS_DIR/summary.csv"

    if [ ! -d "$RUN_SET_DIR" ]; then
        echo "[WARN] $RUN_SET_DIR introuvable, volet '$STUDY' ignore."
        STATUS=1
        continue
    fi

    echo "--- Volet $STUDY : reconstruction de summary.csv depuis $RUN_SET_DIR ---"
    python scripts/build_ablation_summary.py \
        --run_set_dir "$RUN_SET_DIR" \
        --study "$STUDY" \
        --output "$SUMMARY_CSV"

    if [ $? -ne 0 ] || [ ! -f "$SUMMARY_CSV" ]; then
        echo "[WARN] Reconstruction de summary.csv echouee pour le volet $STUDY."
        STATUS=1
        continue
    fi

    python scripts/plot_ablation.py \
        --csv "$SUMMARY_CSV" \
        --x "${X_COL[$STUDY]}" \
        --xlabel "${XLABEL[$STUDY]}" \
        --title "${TITLE[$STUDY]}" \
        --output "$RESULTS_DIR/loss_vs_layers.png" \
        --metric "Loss"

    if [ $? -ne 0 ]; then
        echo "[WARN] Generation du graphe echouee pour le volet $STUDY."
        STATUS=1
    fi
done

exit $STATUS
