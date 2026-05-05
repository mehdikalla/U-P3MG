#!/bin/bash
# Fichier : scripts/run.sh

GPU_ID=""
ARGS=()

# Extraction du GPU et conversion des raccourcis --train/--test en --mode
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --train)
            ARGS+=("--mode" "train")
            shift
            ;;
        --test)
            ARGS+=("--mode" "test")
            shift
            ;;
        --full)
            ARGS+=("--mode" "full")
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Assignation matérielle si demandée
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "[INFO] Exécution verrouillée sur le GPU : $GPU_ID"
fi

# Transfert des arguments formatés vers le script principal
python main.py "${ARGS[@]}"