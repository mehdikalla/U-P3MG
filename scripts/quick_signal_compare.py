"""
Script rapide de comparaison qualitative sur un unique signal.

Tire un signal au hasard (ou un index precis) dans le jeu de test, applique
toutes les methodes disponibles (unrolling, random_search, deep learning
pur) avec les derniers poids/parametres calibres trouves dans 'runs/', puis
sauvegarde un fichier .png par methode (signal vrai vs reconstruction) dans
un dossier dedie.

Usage:
    python scripts/quick_signal_compare.py --config config.yaml \
        --data_folder data_1 --output_dir quick_compare_results

    # Pour rejouer le meme signal :
    python scripts/quick_signal_compare.py --data_folder data_1 --index 42
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import NET_ARCHITECTURES, FULLY_LEARNED_MODELS
from src.strategies.network import init_static_params
from src.strategies.random_search import get_algo_and_static, run_iterative_algo
from src.strategies.compare import (
    _build_unrolled_model,
    _build_dl_model,
    _compute_all_metrics,
    find_latest_checkpoint,
    find_latest_best_params,
    UNROLLING_MODELS,
    RANDOM_SEARCH_MODELS,
    DL_MODELS,
)
from Dataset.module import MyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Comparaison qualitative rapide sur un signal unique")
    parser.add_argument('--config', type=str, default=None, help="Fichier YAML de configuration (optionnel)")
    parser.add_argument('--dataset_dir', type=str, default='./Dataset')
    parser.add_argument('--data_folder', type=str, default='data_1')
    parser.add_argument('--output_dir', type=str, default='quick_compare_results',
                         help="Dossier racine ou sera cree un sous-dossier horodate")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None, help="Graine pour le tirage aleatoire du signal")
    parser.add_argument('--index', type=int, default=None, help="Index precis du signal a utiliser (sinon aleatoire)")
    parser.add_argument('--criterion', type=str, default='SNR', choices=['MSE', 'SNR', 'TSNR'])
    parser.add_argument('--algo_iters', type=int, default=200)
    parser.add_argument('--num_layers', type=int, default=25)
    parser.add_argument('--num_pd_layers', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=1e-5)
    parser.add_argument('--eta', type=float, default=1e-2)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--delta_cvx', type=float, default=0.01)
    parser.add_argument('--delta_ncvx', type=float, default=0.01)

    args = parser.parse_args()

    if args.config:
        if not os.path.isfile(args.config):
            print(f"[ERREUR] Fichier de configuration introuvable : {args.config}")
            sys.exit(1)
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        passed_args = [a.strip('-').split('=')[0] for a in sys.argv if a.startswith('-')]
        for key, value in yaml_config.items():
            if hasattr(args, key) and key not in passed_args:
                setattr(args, key, value)

    # 'model' est requis par les fonctions de src.strategies.compare / random_search
    # mais est mis a jour dynamiquement pour chaque methode evaluee.
    args.model = 'p3mg'
    return args


def _get_test_dataset(args):
    base_path = os.path.join(args.dataset_dir, args.data_folder)
    if not os.path.isdir(base_path):
        print(f"[AVERTISSEMENT] Dossier '{base_path}' introuvable, utilisation de '{args.dataset_dir}' directement.")
        base_path = args.dataset_dir
    test_path = os.path.join(base_path, "test.pt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Impossible de trouver {test_path}.")
    return MyDataset(test_path, initial_x0=None, return_name=False)


def _save_signal_plot(xt_np, xh_np, title, metrics, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xt_np, color='black', linewidth=2)
    ax.plot(xh_np, '--', color='tab:orange')
    metric_str = " | ".join(f"{k}={v:.4e}" for k, v in metrics.items())
    ax.set_title(f"{title}\n{metric_str}")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[QUICK_COMPARE] Sauvegarde : {out_path}")


def run_unrolling(model_name, args, xt, y, x0, N_dim, M_dim, device, output_dir):
    data_folder = args.data_folder.strip().lower()
    ckpt_path = find_latest_checkpoint(model_name, 'unrolling', data_folder)
    if ckpt_path is None:
        print(f"[QUICK_COMPARE][unrolling][{model_name}] Aucun checkpoint trouve, ignore.")
        return

    original_model = args.model
    args.model = model_name
    try:
        model = _build_unrolled_model(model_name, args, ckpt_path=ckpt_path).to(device).double()
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()

        static_params, _ = init_static_params(args, N_dim, M_dim, device)
        current_static = static_params if model_name in ('p3mg', 'pmms') else None

        with torch.no_grad():
            xp, _, _ = model(current_static, None, x0, y)
        metrics = _compute_all_metrics(xp, xt)
        title = f"{model_name.upper()} (unrolling) - poids: {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}"
        out_path = os.path.join(output_dir, f"{model_name}_unrolling.png")
        _save_signal_plot(xt.squeeze(0).cpu().numpy(), xp.squeeze(0).cpu().numpy(), title, metrics, out_path)
    except Exception as e:
        print(f"[QUICK_COMPARE][unrolling][{model_name}] Erreur : {e}")
    finally:
        args.model = original_model


def run_random_search(model_name, args, xt, y, x0, N_dim, M_dim, device, output_dir):
    data_folder = args.data_folder.strip().lower()
    params_path = find_latest_best_params(model_name, 'random_search', data_folder)
    if params_path is None:
        print(f"[QUICK_COMPARE][random_search][{model_name}] Aucun best_params.json trouve, ignore.")
        return

    original_model = args.model
    args.model = model_name
    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)

        algo, static = get_algo_and_static(args, N_dim, M_dim, device)
        with torch.no_grad():
            xh = run_iterative_algo(model_name, algo, y, x0, static, hp=best_params, max_iter=args.algo_iters)
        metrics = _compute_all_metrics(xh, xt)
        title = f"{model_name.upper()} (random_search) - params: {os.path.basename(os.path.dirname(os.path.dirname(params_path)))}"
        out_path = os.path.join(output_dir, f"{model_name}_random_search.png")
        _save_signal_plot(xt.squeeze(0).cpu().numpy(), xh.squeeze(0).cpu().numpy(), title, metrics, out_path)
    except Exception as e:
        print(f"[QUICK_COMPARE][random_search][{model_name}] Erreur : {e}")
    finally:
        args.model = original_model


def run_deep_learning(model_name, args, xt, y, x0, N_dim, M_dim, device, output_dir):
    data_folder = args.data_folder.strip().lower()
    ckpt_path = find_latest_checkpoint(model_name, 'unrolling', data_folder)
    if ckpt_path is None:
        print(f"[QUICK_COMPARE][deep_learning][{model_name}] Aucun checkpoint trouve, ignore.")
        return

    try:
        model = _build_dl_model(model_name, N_dim, M_dim).to(device).double()
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()

        with torch.no_grad():
            xp, _, _ = model(None, None, x0, y)
        metrics = _compute_all_metrics(xp, xt)
        title = f"{model_name.upper()} (deep_learning) - poids: {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}"
        out_path = os.path.join(output_dir, f"{model_name}_deep_learning.png")
        _save_signal_plot(xt.squeeze(0).cpu().numpy(), xp.squeeze(0).cpu().numpy(), title, metrics, out_path)
    except Exception as e:
        print(f"[QUICK_COMPARE][deep_learning][{model_name}] Erreur : {e}")


def main():
    args = parse_args()
    device = args.device

    dataset = _get_test_dataset(args)
    n_total = len(dataset)
    if n_total == 0:
        print("[QUICK_COMPARE] Jeu de test vide, impossible de continuer.")
        sys.exit(1)

    if args.index is not None:
        idx = args.index
        if not (0 <= idx < n_total):
            print(f"[ERREUR] Index {idx} hors bornes (jeu de test : {n_total} signaux).")
            sys.exit(1)
    else:
        if args.seed is not None:
            import random
            random.seed(args.seed)
            idx = random.randint(0, n_total - 1)
        else:
            import random
            idx = random.randint(0, n_total - 1)

    print(f"[QUICK_COMPARE] Signal selectionne : index {idx} / {n_total} (data_folder={args.data_folder})")

    sample = dataset[idx]
    xt, y = sample[0], sample[1]
    xt = xt.to(device).double().unsqueeze(0)
    y = y.to(device).double().unsqueeze(0)
    N_dim, M_dim = xt.shape[1], y.shape[1]
    x0 = y.sum(1, keepdim=True).repeat(1, N_dim) / (M_dim * N_dim)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{timestamp}_signal{idx}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[QUICK_COMPARE] Resultats sauvegardes dans : {output_dir}")

    for model_name in UNROLLING_MODELS:
        run_unrolling(model_name, args, xt, y, x0, N_dim, M_dim, device, output_dir)

    for model_name in RANDOM_SEARCH_MODELS:
        run_random_search(model_name, args, xt, y, x0, N_dim, M_dim, device, output_dir)

    for model_name in DL_MODELS:
        run_deep_learning(model_name, args, xt, y, x0, N_dim, M_dim, device, output_dir)

    # Sauvegarde du signal vrai seul, pour reference.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xt.squeeze(0).cpu().numpy(), color='black', linewidth=2)
    ax.set_title(f"True signal - index {idx} ({args.data_folder})")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_reference.png"))
    plt.close(fig)

    print("[QUICK_COMPARE] Termine.")


if __name__ == "__main__":
    main()
