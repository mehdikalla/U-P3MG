#!/usr/bin/env python
"""
Fichier : scripts/build_compare_summary.py
Reconstruit le fichier compare_report.csv global d'un run_set de comparaison

Usage:
    python scripts/build_compare_summary.py \
        --run_set_dir runs/compare/run_set_12345 \
        --output Results/compare/results_12345/all_compare_report.csv
"""
import argparse
import csv
import os
import sys


FIELDNAMES = ['model', 'strategy', 'data_folder', 'n_samples',
              'mse_mean', 'mse_std', 'snr_mean', 'snr_std',
              'total_time_sec', 'avg_time_per_signal_sec', 'source']


def find_compare_reports(run_set_dir):
    """Retourne la liste des compare_report.csv trouves sous run_set_dir.

    Structure attendue : run_set_dir/<data_folder>/<timestamp>_compare/logs/compare_report.csv
    Pour chaque data_folder, seul le run le plus recent (timestamp le plus
    grand) est retenu, au cas ou plusieurs executions partielles existeraient.
    """
    reports = []
    if not os.path.isdir(run_set_dir):
        return reports

    for data_folder in sorted(os.listdir(run_set_dir)):
        data_folder_path = os.path.join(run_set_dir, data_folder)
        if not os.path.isdir(data_folder_path):
            continue

        run_dirs = sorted(
            (d for d in os.listdir(data_folder_path)
             if os.path.isdir(os.path.join(data_folder_path, d))),
            reverse=True
        )

        for run_name in run_dirs:
            candidate = os.path.join(data_folder_path, run_name, 'logs', 'compare_report.csv')
            if os.path.isfile(candidate):
                reports.append(candidate)
                break
        else:
            print(f"[build_compare_summary] [WARN] Aucun compare_report.csv trouve pour '{data_folder}'.")

    return reports


def build_summary(run_set_dir, output_csv):
    """Concatene tous les compare_report.csv trouves en un seul CSV global."""
    reports = find_compare_reports(run_set_dir)
    if not reports:
        print(f"[build_compare_summary] Aucun rapport trouve sous {run_set_dir}.", file=sys.stderr)
        return 1

    all_rows = []
    for report_path in reports:
        with open(report_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
        print(f"[build_compare_summary] Rapport integre : {report_path}")

    if not all_rows:
        print(f"[build_compare_summary] Tous les rapports trouves sont vides. Rien a consolider.", file=sys.stderr)
        return 1

    all_rows.sort(key=lambda r: (r.get('data_folder', ''), float(r.get('mse_mean', 'inf') or 'inf')))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})

    print(f"[build_compare_summary] {len(all_rows)} ligne(s) consolidee(s) -> {output_csv}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruit le rapport CSV global d'un run_set de comparaison."
    )
    parser.add_argument("--run_set_dir", required=True,
                         help="Dossier contenant les sous-runs par data_folder (ex: runs/compare/run_set_12345)")
    parser.add_argument("--output", required=True, help="Chemin du CSV global a generer")
    args = parser.parse_args()

    sys.exit(build_summary(args.run_set_dir, args.output))


if __name__ == "__main__":
    main()
