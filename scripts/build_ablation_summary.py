#!/usr/bin/env python
"""
Fichier : scripts/build_ablation_summary.py

Reconstruit le fichier summary.csv d'un volet d'ablation directement a
partir des sous-runs presents sur disque (runs/ablation/p3mg/<study_tag>/
run_set_<id>/*/), plutot que de s'appuyer sur une consolidation fragile
effectuee ligne par ligne pendant l'execution du job array SLURM (parsing
awk du tableau texte, ecriture concurrente protegee par flock, risque de
desynchronisation entre la configuration reellement executee et celle
supposee par le tableau CONFIGS du script bash).

Pour chaque sous-run trouve, ce script lit :
  - logs/run_config.json  : configuration reelle utilisee (num_layers,
    num_pd_layers, mlp_hidden), ecrite par src.strategies.network.save_config
    au lancement de l'entrainement. C'est la source de verite : elle reflete
    exactement ce qui a ete execute, independamment de tout tableau bash.
  - logs/test_results_table.txt : statistiques de la loss de test (mean,
    median, std, best, worst), ecrites par src.strategies.network.test.

Usage:
    python scripts/build_ablation_summary.py \
        --run_set_dir runs/ablation/p3mg/ablation_extern/run_set_1 \
        --output runs/ablation/p3mg/ablation_extern/results_1/summary.csv
"""
import argparse
import csv
import json
import os
import re
import sys


TABLE_FIELD_PATTERNS = {
    "mean": r"^\|\s*Mean\s*\|\s*([^\|]+?)\s*\|",
    "median": r"^\|\s*Median\s*\|\s*([^\|]+?)\s*\|",
    "std": r"^\|\s*Std\s*\|\s*([^\|]+?)\s*\|",
    "best": r"^\|\s*Min \(Best\)\s*\|\s*([^\|]+?)\s*\|",
    "worst": r"^\|\s*Max \(Worst\)\s*\|\s*([^\|]+?)\s*\|",
}


def parse_test_results_table(path):
    """Extrait mean/median/std/best/worst depuis test_results_table.txt.

    Retourne un dict avec les 5 cles, ou None manquant si absent/illisible.
    """
    result = {k: None for k in TABLE_FIELD_PATTERNS}
    if not os.path.isfile(path):
        return result

    with open(path, "r") as f:
        content = f.read()

    for key, pattern in TABLE_FIELD_PATTERNS.items():
        match = re.search(pattern, content, flags=re.MULTILINE)
        if match:
            result[key] = match.group(1).strip()

    return result


def find_full_run_dirs(run_set_dir):
    """Retourne la liste des sous-dossiers '*_full' directement sous run_set_dir.

    Chaque sous-dossier correspond a un run 'full' (train+test) unique d'une
    configuration d'ablation particuliere.
    """
    if not os.path.isdir(run_set_dir):
        return []
    return sorted(
        os.path.join(run_set_dir, d)
        for d in os.listdir(run_set_dir)
        if d.endswith("_full") and os.path.isdir(os.path.join(run_set_dir, d))
    )


def build_summary(run_set_dir, study, output_csv):
    """Reconstruit summary.csv a partir de tous les sous-runs de run_set_dir."""
    run_dirs = find_full_run_dirs(run_set_dir)
    if not run_dirs:
        print(f"[build_ablation_summary] Aucun run '*_full' trouve sous {run_set_dir}.", file=sys.stderr)
        return 1

    rows = []
    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, "logs", "run_config.json")
        table_path = os.path.join(run_dir, "logs", "test_results_table.txt")

        if not os.path.isfile(config_path):
            print(f"[build_ablation_summary] [WARN] run_config.json absent pour {run_dir}, ignore.")
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        num_layers = config.get("num_layers", "")
        num_pd_layers = config.get("num_pd_layers", "")
        mlp_hidden = config.get("mlp_hidden", "")
        if mlp_hidden in ("None", None):
            mlp_hidden = ""

        stats = parse_test_results_table(table_path)
        if stats["mean"] is None:
            print(f"[build_ablation_summary] [WARN] test_results_table.txt manquant/illisible pour {run_dir}, "
                  f"ligne exclue (run probablement incomplet ou echoue).")
            continue

        rows.append({
            "study": study,
            "num_layers": num_layers,
            "num_pd_layers": num_pd_layers,
            "mlp_hidden": mlp_hidden,
            "run_dir": run_dir,
            "mean": stats["mean"],
            "median": stats["median"],
            "std": stats["std"],
            "best": stats["best"],
            "worst": stats["worst"],
        })

    if not rows:
        print(f"[build_ablation_summary] Aucun run exploitable sous {run_set_dir}. summary.csv non genere.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = ["study", "num_layers", "num_pd_layers", "mlp_hidden", "run_dir",
                  "mean", "median", "std", "best", "worst"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[build_ablation_summary] {len(rows)} run(s) consolide(s) -> {output_csv}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruit summary.csv a partir des sous-runs d'un volet d'ablation."
    )
    parser.add_argument("--run_set_dir", required=True,
                         help="Dossier contenant les sous-runs '*_full' (ex: runs/ablation/p3mg/ablation_extern/run_set_1)")
    parser.add_argument("--study", required=True, choices=["extern", "intern", "mlp"],
                         help="Nom du volet d'ablation")
    parser.add_argument("--output", required=True, help="Chemin du summary.csv a generer")
    args = parser.parse_args()

    sys.exit(build_summary(args.run_set_dir, args.study, args.output))


if __name__ == "__main__":
    main()
