#!/usr/bin/env python
"""
Fichier : scripts/plot_ablation.py

Genere, pour un volet d'etude d'ablation donne, un graphe de la loss
(mean +/- std) en fonction du parametre etudie (num_layers, num_pd_layers
ou profondeur du MLP interne), a partir du fichier summary.csv produit par
run_ablation.slurm (ou les scripts ablation_*.sh).

Usage:
    python scripts/plot_ablation.py --csv Results/ablation/extern/summary.csv \
        --x num_layers --xlabel "Nombre de couches (num_layers)" \
        --title "Ablation - Nombre de couches externes" \
        --output Results/ablation/extern/loss_vs_layers.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mlp_depth(mlp_hidden_str):
    """Convertit une chaine '50,25,12' en profondeur (nombre de couches cachees)."""
    if not isinstance(mlp_hidden_str, str) or not mlp_hidden_str.strip():
        return np.nan
    return len([v for v in mlp_hidden_str.split(",") if v.strip() != ""])


def load_summary(csv_path, x_col):
    """Charge le CSV de synthese et retourne (x, mean, std) tries par x croissant."""
    df = pd.read_csv(csv_path)

    for col in ("mean", "std"):
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' absente de {csv_path}")

    if x_col == "mlp_hidden":
        if "mlp_hidden" not in df.columns:
            raise ValueError(f"Colonne 'mlp_hidden' absente de {csv_path}")
        df["_x"] = df["mlp_hidden"].apply(_mlp_depth)
    else:
        if x_col not in df.columns:
            raise ValueError(f"Colonne '{x_col}' absente de {csv_path}")
        df["_x"] = pd.to_numeric(df[x_col], errors="coerce")

    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")

    df = df.dropna(subset=["_x", "mean"]).sort_values("_x")
    if df.empty:
        raise ValueError(f"Aucune donnee exploitable dans {csv_path}")

    x = df["_x"].to_numpy()
    mean = df["mean"].to_numpy()
    std = df["std"].fillna(0.0).to_numpy()
    return x, mean, std


def plot_ablation(csv_path, x_col, xlabel, title, output_path, metric_name="Loss"):
    """Genere et sauvegarde le graphe loss = f(x), avec bande d'ecart type."""
    x, mean, std = load_summary(csv_path, x_col)

    plt.figure(figsize=(9, 6))
    plt.plot(x, mean, "o-", color="tab:blue", label=f"{metric_name} (moyenne)")
    plt.fill_between(
        x, mean - std, mean + std,
        color="tab:blue", alpha=0.2, label="+/- 1 ecart type"
    )
    plt.errorbar(x, mean, yerr=std, fmt="none", ecolor="tab:blue", elinewidth=1, capsize=3)

    plt.xlabel(xlabel)
    plt.ylabel(metric_name)
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot_ablation] Graphe sauvegarde : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Trace la loss en fonction du parametre d'ablation.")
    parser.add_argument("--csv", required=True, help="Chemin vers summary.csv")
    parser.add_argument("--x", required=True, choices=["num_layers", "num_pd_layers", "mlp_hidden"],
                         help="Colonne du CSV a utiliser comme abscisse")
    parser.add_argument("--xlabel", required=True, help="Libelle de l'axe des abscisses")
    parser.add_argument("--title", required=True, help="Titre du graphe")
    parser.add_argument("--output", required=True, help="Chemin du fichier PNG de sortie")
    parser.add_argument("--metric", default="Loss", help="Nom de la metrique (defaut: Loss)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[plot_ablation] Fichier introuvable : {args.csv}", file=sys.stderr)
        sys.exit(1)

    plot_ablation(args.csv, args.x, args.xlabel, args.title, args.output, args.metric)


if __name__ == "__main__":
    main()
