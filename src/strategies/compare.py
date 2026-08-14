"""
Mode de comparaison : applique tous les modeles algorithmiques (unrolling et
random_search), a l'exclusion des modeles purement deep learning (fcae, fcun,
fctn), sur l'integralite du jeu de test, en calculant a la fois la MSE et le
SNR (uniquement en inference, avec les poids/parametres deja calibres), puis
produit un rapport CSV comparatif (moyenne et ecart-type par modele/
strategie) ainsi qu'une visualisation sur un signal tire aleatoirement.

Les poids/parametres de chaque modele sont charges depuis le dernier run
disponible pour la combinaison (modele, strategie) dans le dossier 'runs/'.
"""

import os
import csv
import json
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.models import NET_ARCHITECTURES
from src.strategies.network import init_static_params
from src.strategies.random_search import get_algo_and_static, run_iterative_algo

# Modeles algorithmiques disponibles pour chaque strategie (DL exclus).
UNROLLING_MODELS = ['p3mg', 'ista', 'hq', 'pd', 'pmms']
RANDOM_SEARCH_MODELS = ['p3mg', 'ista', 'hq', 'pd', 'pmms']

# Metriques systematiquement calculees en inference, independamment du
# critere utilise a l'entrainement/a la calibration.
REPORT_METRICS = ['MSE', 'SNR']


def _list_run_dirs(model_name, strategy, data_folder):
    """Retourne (liste des dossiers de run tries du plus recent au plus ancien,
    chemin du dossier de strategie) ou (None, None) si absent.

    Le dossier est cherche sous 'runs/<model>/<strategy>/<data_folder>' afin
    de garantir que la comparaison ne porte que sur des runs entraines/
    calibres sur le meme jeu de donnees.
    """
    strategy_dir = os.path.join("runs", model_name, strategy, data_folder.strip().lower())
    if not os.path.isdir(strategy_dir):
        return None, None
    run_dirs = sorted(
        (d for d in os.listdir(strategy_dir) if os.path.isdir(os.path.join(strategy_dir, d))),
        reverse=True
    )
    return run_dirs, strategy_dir


def find_latest_checkpoint(model_name, strategy, data_folder):
    """Cherche le dernier 'best_model.pt' disponible pour (model_name, strategy, data_folder)."""
    run_dirs, strategy_dir = _list_run_dirs(model_name, strategy, data_folder)
    if not run_dirs:
        return None
    for run_name in run_dirs:
        candidate = os.path.join(strategy_dir, run_name, 'checkpoints', 'best_model.pt')
        if os.path.exists(candidate):
            return candidate
    return None


def find_latest_best_params(model_name, strategy, data_folder):
    """Cherche le dernier 'best_params.json' disponible pour (model_name, strategy, data_folder)."""
    run_dirs, strategy_dir = _list_run_dirs(model_name, strategy, data_folder)
    if not run_dirs:
        return None
    for run_name in run_dirs:
        candidate = os.path.join(strategy_dir, run_name, 'checkpoints', 'best_params.json')
        if os.path.exists(candidate):
            return candidate
    return None


def _build_unrolled_model(model_name, args, M_dim=None):
    """Instancie l'architecture unrolled correspondant a model_name.

    Le parametre M_dim est requis pour HQ afin de dimensionner correctement
    les couches lineaires internes (fc_cvx/fc_ncvx), sous peine de charger un
    state_dict incompatible en silence (strict=False).
    """
    ModelClass = NET_ARCHITECTURES[model_name]
    if model_name == 'hq':
        kwargs = {'num_layers': args.num_layers, 'num_pd_layers': args.num_pd_layers}
        if M_dim is not None:
            kwargs['M_dim'] = M_dim
        return ModelClass(**kwargs)
    if model_name == 'p3mg':
        return ModelClass(num_layers=args.num_layers, num_pd_layers=args.num_pd_layers)
    return ModelClass(num_layers=args.num_layers)



def _compute_all_metrics(xh, xt):
    """Calcule toutes les metriques de REPORT_METRICS entre xh et xt (batch=1).

    Retourne un dict {nom_metrique: valeur_scalaire}.
    """
    mse = torch.mean((xh - xt) ** 2).item()
    noise = torch.mean((xt - xh) ** 2)
    sig = torch.mean(xt ** 2)
    snr = (-10 * torch.log10(sig / (noise + 1e-12))).item()
    return {'MSE': mse, 'SNR': snr}


def _evaluate_unrolling_on_testset(model_name, args, dataset, device):
    """Evalue un modele 'unrolling' sur l'integralite du jeu de test.

    Retourne (metrics_per_sample, checkpoint_path) ou (None, None) si aucun
    checkpoint n'est disponible.
        metrics_per_sample : dict {nom_metrique: liste des valeurs par echantillon}
    """
    data_folder = getattr(args, 'data_folder', 'data_1').strip().lower()
    ckpt_path = find_latest_checkpoint(model_name, 'unrolling', data_folder)
    if ckpt_path is None:
        return None, None

    original_model_arg = args.model
    args.model = model_name
    try:
        model = _build_unrolled_model(model_name, args).to(device).double()
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()

        metrics_per_sample = {m: [] for m in REPORT_METRICS}
        static_params = None
        n_total = len(dataset)
        progress_step = max(1, n_total // 10)

        with torch.no_grad():
            for idx in range(n_total):
                sample = dataset[idx]
                xt, y = sample[0], sample[1]
                xt = xt.to(device).double().unsqueeze(0)
                y = y.to(device).double().unsqueeze(0)
                N_dim, M_dim = xt.shape[1], y.shape[1]

                if static_params is None:
                    static_params, _ = init_static_params(args, N_dim, M_dim, device)

                x0 = y.sum(1, keepdim=True).repeat(1, N_dim) / (M_dim * N_dim)
                current_static = static_params if model_name in ('p3mg', 'pmms') else None
                xp, _, _ = model(current_static, None, x0, y)

                sample_metrics = _compute_all_metrics(xp, xt)
                for m in REPORT_METRICS:
                    metrics_per_sample[m].append(sample_metrics[m])

                if (idx + 1) % progress_step == 0 or (idx + 1) == n_total:
                    print(f"[COMPARE][unrolling][{model_name}] Progression : {idx + 1}/{n_total} signaux evalues.")

        return metrics_per_sample, ckpt_path

    finally:
        args.model = original_model_arg


def _evaluate_random_search_on_testset(model_name, args, dataset, device):
    """Evalue un modele 'random_search' sur l'integralite du jeu de test.

    Retourne (metrics_per_sample, params_path) ou (None, None) si aucun
    'best_params.json' n'est disponible.
    """
    data_folder = getattr(args, 'data_folder', 'data_1').strip().lower()
    params_path = find_latest_best_params(model_name, 'random_search', data_folder)
    if params_path is None:
        return None, None

    original_model_arg = args.model
    args.model = model_name
    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)

        metrics_per_sample = {m: [] for m in REPORT_METRICS}
        algo, static = None, None
        n_total = len(dataset)
        progress_step = max(1, n_total // 10)

        print(f"[COMPARE][random_search][{model_name}] Debut evaluation : {n_total} signaux, "
              f"{args.algo_iters} iterations/signal.")

        with torch.no_grad():
            for idx in range(n_total):
                sample = dataset[idx]
                xt, y = sample[0], sample[1]
                xt = xt.to(device).double().unsqueeze(0)
                y = y.to(device).double().unsqueeze(0)
                N_dim, M_dim = xt.shape[1], y.shape[1]

                if algo is None:
                    algo, static = get_algo_and_static(args, N_dim, M_dim, device)

                x0 = y.sum(1, keepdim=True).repeat(1, N_dim) / (M_dim * N_dim)
                xh = run_iterative_algo(
                    model_name, algo, y, x0, static,
                    hp=best_params, max_iter=args.algo_iters
                )

                sample_metrics = _compute_all_metrics(xh, xt)
                for m in REPORT_METRICS:
                    metrics_per_sample[m].append(sample_metrics[m])

                if (idx + 1) % progress_step == 0 or (idx + 1) == n_total:
                    print(f"[COMPARE][random_search][{model_name}] Progression : {idx + 1}/{n_total} signaux evalues.")

        return metrics_per_sample, params_path

    finally:
        args.model = original_model_arg


def run(dataset, args, paths):
    """
    Point d'entree du mode 'compare'.

    Evalue tous les modeles algorithmiques (unrolling + random_search) sur
    l'integralite du jeu de test fourni, en calculant simultanement la MSE et
    le SNR (en inference uniquement, avec les poids/parametres deja
    calibres). Produit :
      - un rapport CSV/JSON avec moyenne et ecart-type par (modele, strategie)
      - une visualisation qualitative sur un signal tire aleatoirement.

    Args:
        dataset : instance de MyDataset (typiquement le split de test).
        args    : namespace argparse contenant la configuration.
        paths   : tuple (base_dir, path_checkpoints, path_plots, path_logs)
                   du run 'compare' en cours.
    """
    device = args.device
    data_folder = getattr(args, 'data_folder', 'data_1').strip().lower()
    _, _, path_plots, path_logs = paths

    if len(dataset) == 0:
        print("[COMPARE] Jeu de donnees vide, impossible d'evaluer.")
        return

    print(f"--- [COMPARE] Evaluation complete du jeu de test ({len(dataset)} signaux) | Data: {data_folder} ---")

    summary_rows = []   # lignes destinees au CSV final
    signal_results = {}  # cle -> (xh_numpy, loss_scalaire) pour le graphique qualitatif

    # Signal tire aleatoirement, reutilise pour la visualisation qualitative.
    plot_idx = random.randint(0, len(dataset) - 1)
    plot_criterion = getattr(args, 'criterion', 'MSE')

    def _record(model_name, strategy, metrics_per_sample, source_path):
        row = {'model': model_name, 'strategy': strategy, 'data_folder': data_folder,
               'n_samples': len(metrics_per_sample['MSE']), 'source': source_path}
        for m in REPORT_METRICS:
            values = np.array(metrics_per_sample[m])
            row[f'{m.lower()}_mean'] = float(np.mean(values))
            row[f'{m.lower()}_std'] = float(np.std(values))
        summary_rows.append(row)

    # --- Strategie 'unrolling' ---
    for model_name in UNROLLING_MODELS:
        metrics_per_sample, ckpt_path = _evaluate_unrolling_on_testset(model_name, args, dataset, device)
        if metrics_per_sample is None:
            print(f"[COMPARE][unrolling][{model_name}] Aucun checkpoint trouve, ignore.")
            continue
        _record(model_name, 'unrolling', metrics_per_sample, ckpt_path)
        print(f"[COMPARE][unrolling][{model_name}] MSE={summary_rows[-1]['mse_mean']:.4e} "
              f"SNR={summary_rows[-1]['snr_mean']:.4e} (poids: {ckpt_path})")

        idx_pos = metrics_per_sample[plot_criterion]  # reuse loop below for plot signal
    # Recalcule le signal unique pour la visualisation qualitative (unrolling)
    for model_name in UNROLLING_MODELS:
        if not any(r['model'] == model_name and r['strategy'] == 'unrolling' for r in summary_rows):
            continue
        try:
            ckpt_path = find_latest_checkpoint(model_name, 'unrolling', data_folder)
            original_model_arg = args.model
            args.model = model_name
            model = _build_unrolled_model(model_name, args).to(device).double()
            ckpt = torch.load(ckpt_path, map_location=device)
            sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            model.eval()

            sample = dataset[plot_idx]
            xt, y = sample[0], sample[1]
            xt = xt.to(device).double().unsqueeze(0)
            y = y.to(device).double().unsqueeze(0)
            N_dim, M_dim = xt.shape[1], y.shape[1]
            x0 = y.sum(1, keepdim=True).repeat(1, N_dim) / (M_dim * N_dim)

            static_params, _ = init_static_params(args, N_dim, M_dim, device)
            current_static = static_params if model_name in ('p3mg', 'pmms') else None
            with torch.no_grad():
                xp, _, _ = model(current_static, None, x0, y)
            loss = _compute_all_metrics(xp, xt)[plot_criterion]
            signal_results[f"{model_name}_unrolling"] = (xp.squeeze(0).cpu().numpy(), loss)
            args.model = original_model_arg
        except Exception as e:
            print(f"[COMPARE][unrolling][{model_name}] Erreur lors du rendu qualitatif : {e}")

    # --- Strategie 'random_search' ---
    for model_name in RANDOM_SEARCH_MODELS:
        metrics_per_sample, params_path = _evaluate_random_search_on_testset(model_name, args, dataset, device)
        if metrics_per_sample is None:
            print(f"[COMPARE][random_search][{model_name}] Aucun best_params.json trouve, ignore.")
            continue
        _record(model_name, 'random_search', metrics_per_sample, params_path)
        print(f"[COMPARE][random_search][{model_name}] MSE={summary_rows[-1]['mse_mean']:.4e} "
              f"SNR={summary_rows[-1]['snr_mean']:.4e} (params: {params_path})")

    for model_name in RANDOM_SEARCH_MODELS:
        if not any(r['model'] == model_name and r['strategy'] == 'random_search' for r in summary_rows):
            continue
        try:
            params_path = find_latest_best_params(model_name, 'random_search', data_folder)
            original_model_arg = args.model
            args.model = model_name
            with open(params_path, 'r') as f:
                best_params = json.load(f)

            sample = dataset[plot_idx]
            xt, y = sample[0], sample[1]
            xt = xt.to(device).double().unsqueeze(0)
            y = y.to(device).double().unsqueeze(0)
            N_dim, M_dim = xt.shape[1], y.shape[1]
            x0 = y.sum(1, keepdim=True).repeat(1, N_dim) / (M_dim * N_dim)

            algo, static = get_algo_and_static(args, N_dim, M_dim, device)
            with torch.no_grad():
                xh = run_iterative_algo(
                    model_name, algo, y, x0, static,
                    hp=best_params, max_iter=args.algo_iters
                )
            loss = _compute_all_metrics(xh, xt)[plot_criterion]
            signal_results[f"{model_name}_random_search"] = (xh.squeeze(0).cpu().numpy(), loss)
            args.model = original_model_arg
        except Exception as e:
            print(f"[COMPARE][random_search][{model_name}] Erreur lors du rendu qualitatif : {e}")

    if not summary_rows:
        print("[COMPARE] Aucun modele n'a pu etre evalue. Verifiez que des runs existent dans 'runs/'.")
        return

    xt_plot = dataset[plot_idx][0].double().cpu().numpy()
    if signal_results:
        _plot_comparison(xt_plot, signal_results, plot_idx, plot_criterion, path_plots, data_folder)
    _save_report(summary_rows, path_logs, data_folder)


def _plot_comparison(xt_np, results, idx, criterion_name, path_plots, data_folder):
    """Trace le signal vrai et toutes les restaurations, plus un histogramme des pertes."""
    fig, (ax_sig, ax_loss) = plt.subplots(2, 1, figsize=(12, 9))

    ax_sig.plot(xt_np, label='Signal vrai', color='black', linewidth=2)
    for key, (xh_np, loss) in results.items():
        ax_sig.plot(xh_np, '--', label=f"{key} ({criterion_name}={loss:.3e})")
    ax_sig.set_title(f"Comparaison des restaurations - Signal test #{idx} ({data_folder})")

    ax_sig.legend(fontsize=8)
    ax_sig.grid(True)

    names = list(results.keys())
    losses = [results[k][1] for k in names]
    ax_loss.bar(range(len(names)), losses, color='steelblue')
    ax_loss.set_xticks(range(len(names)))
    ax_loss.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax_loss.set_ylabel(criterion_name)
    ax_loss.set_title(f"{criterion_name} par modele/strategie")
    ax_loss.grid(True, axis='y')

    plt.tight_layout()
    out_path = os.path.join(path_plots, f'compare_signal_{idx}.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[COMPARE] Graphique sauvegarde : {out_path}")


def _save_report(summary_rows, path_logs, data_folder):
    """Sauvegarde le rapport comparatif complet en CSV et JSON.

    Le CSV contient, pour chaque (modele, strategie) : le nombre
    d'echantillons evalues, la moyenne et l'ecart-type de la MSE et du SNR,
    ainsi que le chemin des poids/parametres utilises.
    """
    fieldnames = ['model', 'strategy', 'data_folder', 'n_samples',
                  'mse_mean', 'mse_std', 'snr_mean', 'snr_std', 'source']

    sorted_rows = sorted(summary_rows, key=lambda r: r['mse_mean'])

    csv_path = os.path.join(path_logs, 'compare_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)

    json_path = os.path.join(path_logs, 'compare_report.json')
    with open(json_path, 'w') as f:
        json.dump(sorted_rows, f, indent=4)

    txt_path = os.path.join(path_logs, 'compare_report.txt')
    with open(txt_path, 'w') as f:
        f.write(f"=== Comparaison complete du jeu de test | Data: {data_folder} ===\n\n")
        header = f"{'model':<10s} {'strategy':<15s} {'n':>6s} {'mse_mean':>12s} {'mse_std':>12s} {'snr_mean':>12s} {'snr_std':>12s}\n"
        f.write(header)
        f.write('-' * len(header) + '\n')
        for row in sorted_rows:
            f.write(f"{row['model']:<10s} {row['strategy']:<15s} {row['n_samples']:>6d} "
                     f"{row['mse_mean']:>12.4e} {row['mse_std']:>12.4e} "
                     f"{row['snr_mean']:>12.4e} {row['snr_std']:>12.4e}\n")

    print(f"[COMPARE] Rapport CSV sauvegarde : {csv_path}")
    print(f"[COMPARE] Rapport JSON sauvegarde : {json_path}")
    print(f"[COMPARE] Rapport texte sauvegarde : {txt_path}")
