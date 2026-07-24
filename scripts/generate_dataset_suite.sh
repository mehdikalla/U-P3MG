#!/bin/bash
# Fichier : scripts/generate_dataset_suite.sh
#
# Génère séquentiellement trois datasets RMN DOSY correspondant à des
# configurations croissantes en complexité :
#   1. 2 pics, sans asymétrie (skew désactivé)
#   2. 2 pics, avec asymétrie (skew activé)
#   3. 3 pics, avec asymétrie (skew activé)
#
# Chaque appel délègue la génération à scripts/generate_dataset.sh, qui
# crée automatiquement un nouveau dossier 'data_i' dans --out_dir.
#
# Usage:
#   scripts/generate_dataset_suite.sh [OPTIONS]
#
# Options:
#   --total_samples <int> Nombre total d'échantillons par dataset (défaut: 1000)
#   --noise <float>        Écart-type du bruit gaussien additif (défaut: 0.01)
#   --beta <float>         Paramètre de forme de la gaussienne généralisée (défaut: 2.0)
#   --out_dir <path>       Dossier de sauvegarde (défaut: ./Dataset)
#
# Exemple:
#   scripts/generate_dataset_suite.sh --total_samples 5000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOTAL_SAMPLES=1000
NOISE=0.01
BETA=2.0
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
        --beta)
            BETA="$2"
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

COMMON_ARGS=(--total_samples "$TOTAL_SAMPLES" --noise "$NOISE" --beta "$BETA" --out_dir "$OUT_DIR")

echo "[SUITE] ================================================"
echo "[SUITE] Génération 1/3 : 2 pics, sans skew"
echo "[SUITE] ================================================"
"$SCRIPT_DIR/generate_dataset.sh" --number 2 "${COMMON_ARGS[@]}"

echo "[SUITE] ================================================"
echo "[SUITE] Génération 2/3 : 2 pics, avec skew"
echo "[SUITE] ================================================"
"$SCRIPT_DIR/generate_dataset.sh" --number 2 --skew "${COMMON_ARGS[@]}"

echo "[SUITE] ================================================"
echo "[SUITE] Génération 3/3 : 3 pics, avec skew"
echo "[SUITE] ================================================"
"$SCRIPT_DIR/generate_dataset.sh" --number 3 --skew "${COMMON_ARGS[@]}"

echo "[SUITE] Les trois datasets ont été générés avec succès dans $OUT_DIR."
