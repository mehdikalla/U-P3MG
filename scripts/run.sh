#!/bin/bash
# Fichier : scripts/run.sh

GPU_ID=""
ARGS=()

# Extraction exclusive de l'argument --gpu, conservation des autres
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
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

# Transfert des arguments restants vers le script principal
python main.py "${ARGS[@]}"