#!/bin/bash
# Fichier : scripts/generate_dataset.sh
#
# Génère un dataset RMN DOSY en une seule commande.
#
# Usage:
#   scripts/generate_dataset.sh [OPTIONS]
#
# Options:
#   --skew                Active la gaussienne généralisée asymétrique (défaut: désactivé)
#   --number <int>        Nombre de gaussiennes fondamentales sommées (défaut: 2)
#   --beta <float>        Paramètre de forme de la gaussienne généralisée (défaut: 2.0)
#   --noise <float>       Écart-type du bruit gaussien additif (défaut: 0.01)
#   --total_samples <int> Nombre total d'échantillons à générer (défaut: 1000)
#   --out_dir <path>      Dossier de sauvegarde (défaut: ./Dataset)
#
# Exemple:
#   scripts/generate_dataset.sh --skew --number 3 --total_samples 5000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ARGS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skew)
            ARGS+=("--skew")
            shift
            ;;
        --number)
            ARGS+=("--number" "$2")
            shift 2
            ;;
        --beta)
            ARGS+=("--beta" "$2")
            shift 2
            ;;
        --noise)
            ARGS+=("--noise" "$2")
            shift 2
            ;;
        --total_samples)
            ARGS+=("--total_samples" "$2")
            shift 2
            ;;
        --out_dir)
            ARGS+=("--out_dir" "$2")
            shift 2
            ;;
        *)
            echo "[ERREUR] Option inconnue : $1"
            exit 1
            ;;
    esac
done

PYTHON_BIN="python"
if ! command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

echo "[INFO] Lancement de la génération du dataset..."
"$PYTHON_BIN" "$PROJECT_ROOT/Dataset/generate_dataset.py" "${ARGS[@]}"

