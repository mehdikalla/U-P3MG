#!/bin/bash
# Fichier : scripts/generate_dataset_suite.sh
#
# Génère séquentiellement trois datasets RMN DOSY :
#   1. 2 pics -> data_0
#   2. 2 pics -> data_1
#   3. 3 pics -> data_2
#
# Usage:
#   scripts/generate_dataset_suite.sh [OPTIONS]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOTAL_SAMPLES=1000
NOISE=0.01
OUT_DIR="./Dataset"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --total_samples)
            TOTAL_SAMPLES="$2"
            shift 2
            ;;
        --noise)
            NOISE="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "[ERREUR] Option inconnue : $1"
            exit 1
            ;;
    esac
done


COMMON_ARGS=(--total_samples "$TOTAL_SAMPLES" --noise "$NOISE" --out_dir "$OUT_DIR")

echo "[SUITE] ================================================"
echo "[SUITE] Génération 1/3 : 2 pics (-> data_0)"
echo "[SUITE] ================================================"
"$SCRIPT_DIR/generate_dataset.sh" --number 2 "${COMMON_ARGS[@]}"

echo "[SUITE] ================================================"
echo "[SUITE] Génération 2/3 : 2 pics avec beta (skew) variable entre 1.5 et 3.0 (-> data_1)"
echo "[SUITE] ================================================"
"$SCRIPT_DIR/generate_dataset.sh" --number 2 --skew --beta_min 1.5 --beta_max 3.0 "${COMMON_ARGS[@]}"

echo "[SUITE] ================================================"
echo "[SUITE] Génération 3/3 : 3 pics (-> data_2)"
echo "[SUITE] ================================================"
"$SCRIPT_DIR/generate_dataset.sh" --number 3 "${COMMON_ARGS[@]}"

echo "[SUITE] ================================================"
echo "[SUITE] Les datasets data_0, data_1 et data_2 ont été générés dans $OUT_DIR."