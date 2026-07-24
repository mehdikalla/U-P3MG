"""
Mode de comparaison : applique tous les modeles algorithmiques (unrolling et
random_search), a l'exclusion des modeles purement deep learning (fcae, fcun,
fctn), sur un signal tire aleatoirement du jeu de test, puis restitue les
signaux restaures ainsi que les pertes associees.

Les poids/parametres de chaque modele sont charges depuis le dernier run
disponible pour la combinaison (modele, strategie) dans le dossier 'runs/'.
"""

import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.models import NET_ARCHITECTURES
from src.strategies.network import init_static_params, get_criterion
from src.strategies.random_search import get_algo_and_static, run_iterative_algo

# Modeles algorithmiques disponibles pour chaque strategie (DL exclus).
UNROLLING_MODELS = ['p3mg', 'ista', 'hq', 'pd', 'pmms']
RANDOM_SEARCH_MODELS = ['p3mg', 'ista', 'hq', 'pd', 'pmms']


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



def _build_unrolled_model(model_name, args):
    """Instancie l'architecture unrolled correspondant a model_name."""
    ModelClass = NET_ARCHITECTURES[model_name]
    if model_name in ['p3mg', 'hq']:
        return ModelClass(num_layers=args.num_layers, num_pd_layers=args.num_pd_layers)
    return ModelClass(num_layers=args.num_layers)


def _compute_metric(xh, xt, criterion_name):
    """Calcule la metrique scalaire (MSE, SNR, TSNR) entre xh et xt (batch=1)."""
    if criterion_name == 'MSE':
        return torch.mean((xh - xt) ** 2).item()
    if criterion_name in ['SNR', 'TSNR']:
        noise = torch.mean((xt - xh) ** 2)
        sig = torch.mean(xt ** 2)
        return (-10 * torch.log10(sig / (noise + 1e-12))).item()
    return torch.mean((xh - xt) ** 2).item()


def run(dataset, args, paths):
    """
    Point d'entree du mode 'compare'.

    Args:
        dataset : instance de MyDataset (typiquement le split de test) dont
                   un signal sera tire aleatoirement.
        args    : namespace argparse contenant la configuration.
        paths   : tuple (base_dir, path_checkpoints, path_plots, path_logs)
                   du run 'compare' en cours.
    """
    device = args.device
    criterion_name = getattr(args, 'criterion', 'MSE')
    data_folder = getattr(args, 'data_folder', 'data_1').strip().lower()
    _, _, path_plots, path_logs = paths

    if len(dataset) == 0:

        print("[COMPARE] Jeu de donnees vide, impossible de tirer un signal.")
        return

    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    xt, y = sample[0], sample[1]
    xt = xt.to(device).double().unsqueeze(0)
    y = y.to(device).double().unsqueeze(0)

    N_dim, M_dim = xt.shape[1], y.shape[1]
    x0 = y.sum(1, keepdim=True).repeat(1, N_dim) / (M_dim * N_dim)

    print(f"--- [COMPARE] Signal test index={idx} | N={N_dim} M={M_dim} | Data: {data_folder} ---")


    results = {}  # cle -> (xh_numpy, loss)
    original_model_arg = getattr(args, 'model', None)

    # --- Strategie 'unrolling' ---
    for model_name in UNROLLING_MODELS:
        ckpt_path = find_latest_checkpoint(model_name, 'unrolling', data_folder)
        if ckpt_path is None:

            print(f"[COMPARE][unrolling][{model_name}] Aucun checkpoint trouve, ignore.")
            continue
        try:
            args.model = model_name
            model = _build_unrolled_model(model_name, args).to(device).double()
            ckpt = torch.load(ckpt_path, map_location=device)
            sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            model.eval()

            static_params, _ = init_static_params(args, N_dim, M_dim, device)

            current_static = static_params if model_name == 'p3mg' else None

            with torch.no_grad():
                xp, _, _ = model(current_static, None, x0, y)

            loss = _compute_metric(xp, xt, criterion_name)
            results[f"{model_name}_unrolling"] = (xp.squeeze(0).cpu().numpy(), loss)
            print(f"[COMPARE][unrolling][{model_name}] Loss={loss:.4e} (poids: {ckpt_path})")
        except Exception as e:
            print(f"[COMPARE][unrolling][{model_name}] Erreur : {e}")

    # --- Strategie 'random_search' ---
    for model_name in RANDOM_SEARCH_MODELS:
        params_path = find_latest_best_params(model_name, 'random_search', data_folder)
        if params_path is None:

            print(f"[COMPARE][random_search][{model_name}] Aucun best_params.json trouve, ignore.")
            continue
        try:
            args.model = model_name
            with open(params_path, 'r') as f:
                best_params = json.load(f)

            algo, static = get_algo_and_static(args, N_dim, M_dim, device)

            with torch.no_grad():
                xh = run_iterative_algo(
                    model_name, algo, y, x0, static,
                    hp=best_params, max_iter=args.algo_iters
                )

            loss = _compute_metric(xh, xt, criterion_name)
            results[f"{model_name}_random_search"] = (xh.squeeze(0).cpu().numpy(), loss)
            print(f"[COMPARE][random_search][{model_name}] Loss={loss:.4e} (params: {params_path})")
        except Exception as e:
            print(f"[COMPARE][random_search][{model_name}] Erreur : {e}")

    args.model = original_model_arg

    if not results:
        print("[COMPARE] Aucun modele n'a pu etre evalue. Verifiez que des runs existent dans 'runs/'.")
        return


    _plot_comparison(xt.squeeze(0).cpu().numpy(), results, idx, criterion_name, path_plots, data_folder)
    _save_report(results, idx, criterion_name, path_logs, data_folder)



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


def _save_report(results, idx, criterion_name, path_logs, data_folder):
    """Sauvegarde un rapport texte et JSON des pertes obtenues."""
    lines = [f"=== Comparaison - Signal test index {idx} ({criterion_name}) | Data: {data_folder} ===\n"]

    for key, (_, loss) in sorted(results.items(), key=lambda kv: kv[1][1]):
        lines.append(f"{key:<30s} : {loss:.6e}\n")

    report_path = os.path.join(path_logs, 'compare_report.txt')
    with open(report_path, 'w') as f:
        f.writelines(lines)

    json_path = os.path.join(path_logs, 'compare_report.json')
    with open(json_path, 'w') as f:
        json.dump({k: v[1] for k, v in results.items()}, f, indent=4)

    print(f"[COMPARE] Rapport sauvegarde : {report_path}")
