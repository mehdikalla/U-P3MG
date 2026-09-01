"""
Mode de comparaison : applique tous les modeles disponibles (unrolling,
random_search et deep learning pur), sur l'integralite du jeu de test, en
calculant a la fois la MSE et le SNR (uniquement en inference, avec les
poids/parametres deja calibres), puis produit un rapport CSV comparatif
(moyenne et ecart-type par modele/strategie) ainsi qu'une visualisation
individuelle (un graphe par methode) sur un signal tire aleatoirement.

Les poids/parametres de chaque modele sont charges depuis le dernier run
disponible pour la combinaison (modele, strategie) dans le dossier 'runs/'.
"""

import os
import csv
import json
import random
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from src.models import NET_ARCHITECTURES, FULLY_LEARNED_MODELS
from src.strategies.network import init_static_params
from src.strategies.random_search import get_algo_and_static, run_iterative_algo

# Modeles algorithmiques disponibles pour chaque strategie.
UNROLLING_MODELS = ['p3mg', 'ista', 'hq', 'pd', 'pmms']
RANDOM_SEARCH_MODELS = ['p3mg', 'ista', 'hq', 'pd', 'pmms']

# Modeles purement deep learning (pas d'algorithme itteratif statique),
# evalues uniquement en strategie 'unrolling' (entrainement par gradient).
DL_MODELS = sorted(FULLY_LEARNED_MODELS)


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


def _load_ckpt_arch_config(ckpt_path):
    """Recupere (num_layers, num_pd_layers) reellement utilises a l'entrainement.

    Le fichier 'logs/run_config.json', ecrit par src.strategies.network.save_config
    au lancement de l'entrainement, est la source de verite pour l'architecture du
    modele associe a un checkpoint (cf. scripts/build_ablation_summary.py, qui
    applique deja ce meme principe). A defaut, retombe sur (None, None), auquel
    cas l'appelant doit utiliser les valeurs courantes de 'args' (au risque d'un
    'size mismatch' si l'architecture a change depuis l'entrainement).
    """
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))  # .../checkpoints/best_model.pt -> run_dir
    config_path = os.path.join(run_dir, 'logs', 'run_config.json')
    if not os.path.isfile(config_path):
        return None, None
    try:
        with open(config_path, 'r') as f:
            ckpt_config = json.load(f)
        num_layers = int(ckpt_config['num_layers']) if 'num_layers' in ckpt_config else None
        num_pd_layers = int(ckpt_config['num_pd_layers']) if 'num_pd_layers' in ckpt_config else None
        return num_layers, num_pd_layers
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"[COMPARE] [WARN] Lecture de {config_path} impossible ({e}), "
              f"utilisation des parametres d'architecture courants.")
        return None, None


def _build_unrolled_model(model_name, args, M_dim=None, ckpt_path=None):
    """Instancie l'architecture unrolled correspondant a model_name.

    Le parametre M_dim est requis pour HQ afin de dimensionner correctement
    les couches lineaires internes (fc_cvx/fc_ncvx), sous peine de charger un
    state_dict incompatible en silence (strict=False).

    Si ckpt_path est fourni, num_layers/num_pd_layers sont lus depuis le
    'run_config.json' du run d'origine plutot que depuis 'args', afin d'eviter
    tout 'size mismatch' lorsque l'architecture courante (config.yaml/CLI) a
    change depuis l'entrainement du checkpoint charge.
    """
    ModelClass = NET_ARCHITECTURES[model_name]
    num_layers, num_pd_layers = args.num_layers, args.num_pd_layers
    if ckpt_path is not None:
        ckpt_num_layers, ckpt_num_pd_layers = _load_ckpt_arch_config(ckpt_path)
        if ckpt_num_layers is not None:
            num_layers = ckpt_num_layers
        if ckpt_num_pd_layers is not None:
            num_pd_layers = ckpt_num_pd_layers
    if model_name == 'hq':
        kwargs = {'num_layers': num_layers, 'num_pd_layers': num_pd_layers}
        if M_dim is not None:
            kwargs['M_dim'] = M_dim
        return ModelClass(**kwargs)
    if model_name == 'p3mg':
        return ModelClass(num_layers=num_layers, num_pd_layers=num_pd_layers)
    return ModelClass(num_layers=num_layers)



def _build_dl_model(model_name, N_dim, M_dim):
    """Instancie l'architecture deep learning pure correspondant a model_name."""
    ModelClass = NET_ARCHITECTURES[model_name]
    return ModelClass(N_dim=N_dim, M_dim=M_dim)


def _sync_device(device):
    """Synchronise le device CUDA courant, si applicable, afin de garantir
    des mesures de temps fiables (les appels CUDA sont asynchrones par defaut).
    Accepte aussi bien une str ('cuda'/'cpu') qu'un torch.device.
    """
    device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
    if device_type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()


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
        model = _build_unrolled_model(model_name, args, ckpt_path=ckpt_path).to(device).double()
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()

        metrics_per_sample = {m: [] for m in REPORT_METRICS}
        static_params = None
        n_total = len(dataset)
        progress_step = max(1, n_total // 10)
        elapsed_time = 0.0

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

                start_t = time.perf_counter()
                xp, _, _ = model(current_static, None, x0, y)
                _sync_device(device)
                elapsed_time += time.perf_counter() - start_t

                sample_metrics = _compute_all_metrics(xp, xt)
                for m in REPORT_METRICS:
                    metrics_per_sample[m].append(sample_metrics[m])

                if (idx + 1) % progress_step == 0 or (idx + 1) == n_total:
                    print(f"[COMPARE][unrolling][{model_name}] Progression : {idx + 1}/{n_total} signaux evalues.")

        return metrics_per_sample, ckpt_path, elapsed_time

    finally:
        args.model = original_model_arg


def _evaluate_random_search_on_testset(model_name, args, dataset, device):
    """Evalue un modele 'random_search' sur l'integralite du jeu de test.

    Retourne (metrics_per_sample, params_path, elapsed_time) ou
    (None, None, None) si aucun 'best_params.json' n'est disponible.
    elapsed_time est le temps total (en secondes) passe dans
    run_iterative_algo sur l'ensemble du jeu de test.
    """
    data_folder = getattr(args, 'data_folder', 'data_1').strip().lower()
    params_path = find_latest_best_params(model_name, 'random_search', data_folder)
    if params_path is None:
        return None, None, None

    original_model_arg = args.model
    args.model = model_name
    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)

        metrics_per_sample = {m: [] for m in REPORT_METRICS}
        algo, static = None, None
        n_total = len(dataset)
        progress_step = max(1, n_total // 10)
        elapsed_time = 0.0

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

                start_t = time.perf_counter()
                xh = run_iterative_algo(
                    model_name, algo, y, x0, static,
                    hp=best_params, max_iter=args.algo_iters
                )
                _sync_device(device)
                elapsed_time += time.perf_counter() - start_t

                sample_metrics = _compute_all_metrics(xh, xt)
                for m in REPORT_METRICS:
                    metrics_per_sample[m].append(sample_metrics[m])

                if (idx + 1) % progress_step == 0 or (idx + 1) == n_total:
                    print(f"[COMPARE][random_search][{model_name}] Progression : {idx + 1}/{n_total} signaux evalues.")

        return metrics_per_sample, params_path, elapsed_time

    finally:
        args.model = original_model_arg


def _evaluate_dl_on_testset(model_name, args, dataset, device):
    """Evalue un modele deep learning pur sur l'integralite du jeu de test.

    Retourne (metrics_per_sample, checkpoint_path, elapsed_time) ou
    (None, None, None) si aucun checkpoint n'est disponible. Les modeles DL
    sont toujours entraines/charges sous la strategie 'unrolling'
    (entrainement par retropropagation).
    """
    data_folder = getattr(args, 'data_folder', 'data_1').strip().lower()
    ckpt_path = find_latest_checkpoint(model_name, 'unrolling', data_folder)
    if ckpt_path is None:
        return None, None, None

    try:
        metrics_per_sample = {m: [] for m in REPORT_METRICS}
        model = None
        n_total = len(dataset)
        progress_step = max(1, n_total // 10)
        elapsed_time = 0.0

        with torch.no_grad():
            for idx in range(n_total):
                sample = dataset[idx]
                xt, y = sample[0], sample[1]
                xt = xt.to(device).double().unsqueeze(0)
                y = y.to(device).double().unsqueeze(0)
                N_dim, M_dim = xt.shape[1], y.shape[1]

                if model is None:
                    model = _build_dl_model(model_name, N_dim, M_dim).to(device).double()
                    ckpt = torch.load(ckpt_path, map_location=device)
                    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
                    model.load_state_dict(sd, strict=False)
                    model.eval()

                x0 = y.sum(1, keepdim=True).repeat(1, N_dim) / (M_dim * N_dim)

                start_t = time.perf_counter()
                xp, _, _ = model(None, None, x0, y)
                _sync_device(device)
                elapsed_time += time.perf_counter() - start_t

                sample_metrics = _compute_all_metrics(xp, xt)
                for m in REPORT_METRICS:
                    metrics_per_sample[m].append(sample_metrics[m])

                if (idx + 1) % progress_step == 0 or (idx + 1) == n_total:
                    print(f"[COMPARE][deep_learning][{model_name}] Progression : {idx + 1}/{n_total} signaux evalues.")

        return metrics_per_sample, ckpt_path, elapsed_time

    except Exception as e:
        print(f"[COMPARE][deep_learning][{model_name}] Erreur lors de l'evaluation : {e}")
        return None, None, None


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

    def _record(model_name, strategy, metrics_per_sample, source_path, elapsed_time=None):
        n_samples = len(metrics_per_sample['MSE'])
        row = {'model': model_name, 'strategy': strategy, 'data_folder': data_folder,
               'n_samples': n_samples, 'source': source_path}
        for m in REPORT_METRICS:
            values = np.array(metrics_per_sample[m])
            row[f'{m.lower()}_mean'] = float(np.mean(values))
            row[f'{m.lower()}_std'] = float(np.std(values))
        row['total_time_sec'] = float(elapsed_time) if elapsed_time is not None else None
        row['avg_time_per_signal_sec'] = (
            float(elapsed_time) / n_samples if elapsed_time is not None and n_samples > 0 else None
        )
        summary_rows.append(row)

    # --- Strategie 'unrolling' ---
    for model_name in UNROLLING_MODELS:
        metrics_per_sample, ckpt_path, elapsed_time = _evaluate_unrolling_on_testset(model_name, args, dataset, device)
        if metrics_per_sample is None:
            print(f"[COMPARE][unrolling][{model_name}] Aucun checkpoint trouve, ignore.")
            continue
        _record(model_name, 'unrolling', metrics_per_sample, ckpt_path, elapsed_time)
        print(f"[COMPARE][unrolling][{model_name}] MSE={summary_rows[-1]['mse_mean']:.4e} "
              f"SNR={summary_rows[-1]['snr_mean']:.4e} "
              f"Temps={summary_rows[-1]['total_time_sec']:.4f}s "
              f"({summary_rows[-1]['avg_time_per_signal_sec']:.4e}s/signal) (poids: {ckpt_path})")

        idx_pos = metrics_per_sample[plot_criterion]  # reuse loop below for plot signal
    # Recalcule le signal unique pour la visualisation qualitative (unrolling)
    for model_name in UNROLLING_MODELS:
        if not any(r['model'] == model_name and r['strategy'] == 'unrolling' for r in summary_rows):
            continue
        try:
            ckpt_path = find_latest_checkpoint(model_name, 'unrolling', data_folder)
            original_model_arg = args.model
            args.model = model_name
            model = _build_unrolled_model(model_name, args, ckpt_path=ckpt_path).to(device).double()
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
            metrics = _compute_all_metrics(xp, xt)
            signal_results[f"{model_name}_unrolling"] = (xp.squeeze(0).cpu().numpy(), metrics)
            args.model = original_model_arg
        except Exception as e:
            print(f"[COMPARE][unrolling][{model_name}] Erreur lors du rendu qualitatif : {e}")

    # --- Strategie 'random_search' ---
    for model_name in RANDOM_SEARCH_MODELS:
        metrics_per_sample, params_path, elapsed_time = _evaluate_random_search_on_testset(model_name, args, dataset, device)
        if metrics_per_sample is None:
            print(f"[COMPARE][random_search][{model_name}] Aucun best_params.json trouve, ignore.")
            continue
        _record(model_name, 'random_search', metrics_per_sample, params_path, elapsed_time)
        print(f"[COMPARE][random_search][{model_name}] MSE={summary_rows[-1]['mse_mean']:.4e} "
              f"SNR={summary_rows[-1]['snr_mean']:.4e} "
              f"Temps={summary_rows[-1]['total_time_sec']:.4f}s "
              f"({summary_rows[-1]['avg_time_per_signal_sec']:.4e}s/signal) (params: {params_path})")

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
            metrics = _compute_all_metrics(xh, xt)
            signal_results[f"{model_name}_random_search"] = (xh.squeeze(0).cpu().numpy(), metrics)
            args.model = original_model_arg
        except Exception as e:
            print(f"[COMPARE][random_search][{model_name}] Erreur lors du rendu qualitatif : {e}")

    # --- Modeles deep learning purs ---
    for model_name in DL_MODELS:
        metrics_per_sample, ckpt_path, elapsed_time = _evaluate_dl_on_testset(model_name, args, dataset, device)
        if metrics_per_sample is None:
            print(f"[COMPARE][deep_learning][{model_name}] Aucun checkpoint trouve, ignore.")
            continue
        _record(model_name, 'deep_learning', metrics_per_sample, ckpt_path, elapsed_time)
        print(f"[COMPARE][deep_learning][{model_name}] MSE={summary_rows[-1]['mse_mean']:.4e} "
              f"SNR={summary_rows[-1]['snr_mean']:.4e} "
              f"Temps={summary_rows[-1]['total_time_sec']:.4f}s "
              f"({summary_rows[-1]['avg_time_per_signal_sec']:.4e}s/signal) (poids: {ckpt_path})")

    for model_name in DL_MODELS:
        if not any(r['model'] == model_name and r['strategy'] == 'deep_learning' for r in summary_rows):
            continue
        try:
            ckpt_path = find_latest_checkpoint(model_name, 'unrolling', data_folder)
            sample = dataset[plot_idx]
            xt, y = sample[0], sample[1]
            xt = xt.to(device).double().unsqueeze(0)
            y = y.to(device).double().unsqueeze(0)
            N_dim, M_dim = xt.shape[1], y.shape[1]
            x0 = y.sum(1, keepdim=True).repeat(1, N_dim) / (M_dim * N_dim)

            model = _build_dl_model(model_name, N_dim, M_dim).to(device).double()
            ckpt = torch.load(ckpt_path, map_location=device)
            sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            model.eval()

            with torch.no_grad():
                xp, _, _ = model(None, None, x0, y)
            metrics = _compute_all_metrics(xp, xt)
            signal_results[f"{model_name}_deep_learning"] = (xp.squeeze(0).cpu().numpy(), metrics)
        except Exception as e:
            print(f"[COMPARE][deep_learning][{model_name}] Erreur lors du rendu qualitatif : {e}")

    if not summary_rows:
        print("[COMPARE] Aucun modele n'a pu etre evalue. Verifiez que des runs existent dans 'runs/'.")
        return

    xt_plot = dataset[plot_idx][0].double().cpu().numpy()
    if signal_results:
        _plot_comparison(xt_plot, signal_results, plot_idx, plot_criterion, path_plots, data_folder)

    _plot_timing_comparison(summary_rows, path_plots, data_folder)

    _save_report(summary_rows, path_logs, data_folder)


def _plot_timing_comparison(summary_rows, path_plots, data_folder):
    """Trace un graphe recapitulatif du temps moyen de parcours du jeu de
    test (par signal) pour chaque combinaison (modele, strategie).

    Les lignes sans temps mesure (elapsed_time indisponible) sont ignorees.
    """
    rows_with_time = [r for r in summary_rows if r.get('avg_time_per_signal_sec') is not None]
    if not rows_with_time:
        print("[COMPARE] Aucun temps mesure disponible, graphe de timing ignore.")
        return

    rows_with_time = sorted(rows_with_time, key=lambda r: r['avg_time_per_signal_sec'])
    names = [f"{r['model']}_{r['strategy']}" for r in rows_with_time]
    avg_times = [r['avg_time_per_signal_sec'] for r in rows_with_time]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(names)), avg_times, color='indianred')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Model / strategy')
    ax.set_ylabel('Average time per signal (s)')
    ax.set_yscale('log')
    ax.set_title(f"Average inference time per signal by model/strategy ({data_folder})")
    ax.grid(True, axis='y', which='both')
    plt.tight_layout()
    timing_path = os.path.join(path_plots, 'compare_timing.png')
    plt.savefig(timing_path)
    plt.close(fig)
    print(f"[COMPARE] Graphique de temps sauvegarde : {timing_path}")


def _plot_comparison(xt_np, results, idx, criterion_name, path_plots, data_folder):
    """Trace un graphe individuel par methode (signal vrai vs restauration),
    puis un graphe recapitulatif comparant toutes les methodes sur la
    metrique choisie.

    Un sous-dossier 'signals/' regroupe les graphes individuels, afin de ne
    pas encombrer 'path_plots' avec un fichier par methode.
    """
    signals_dir = os.path.join(path_plots, 'signals')
    os.makedirs(signals_dir, exist_ok=True)

    for key, (xh_np, metrics) in results.items():
        mse = metrics.get('MSE')
        snr = metrics.get('SNR')
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(xt_np, label='True signal', color='black', linewidth=2)
        ax.plot(xh_np, '--', label=f"Reconstruction ({key})", color='tab:orange')
        ax.set_title(f"{key} - Test signal #{idx} ({data_folder})\nMSE={mse:.3e} | SNR={snr:.3e}")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.grid(True)
        plt.tight_layout()
        out_path = os.path.join(signals_dir, f'compare_signal_{idx}_{key}.png')
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[COMPARE] Graphique sauvegarde : {out_path}")

    # Graphe recapitulatif (histogramme des pertes par methode, sur la metrique choisie).
    fig, ax_loss = plt.subplots(figsize=(12, 5))
    names = list(results.keys())
    losses = [results[k][1][criterion_name] for k in names]
    ax_loss.bar(range(len(names)), losses, color='steelblue')
    ax_loss.set_xticks(range(len(names)))
    ax_loss.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax_loss.set_xlabel('Model / strategy')
    ax_loss.set_ylabel(criterion_name)
    ax_loss.set_title(f"{criterion_name} by model/strategy - Test signal #{idx} ({data_folder})")
    ax_loss.grid(True, axis='y')
    plt.tight_layout()
    summary_path = os.path.join(path_plots, f'compare_summary_{idx}.png')
    plt.savefig(summary_path)
    plt.close(fig)
    print(f"[COMPARE] Graphique recapitulatif sauvegarde : {summary_path}")



def _save_report(summary_rows, path_logs, data_folder):
    """Sauvegarde le rapport comparatif complet en CSV et JSON.

    Le CSV contient, pour chaque (modele, strategie) : le nombre
    d'echantillons evalues, la moyenne et l'ecart-type de la MSE et du SNR,
    le temps total (secondes) passe a parcourir le jeu de test ainsi que le
    temps moyen par signal, et le chemin des poids/parametres utilises.
    """
    fieldnames = ['model', 'strategy', 'data_folder', 'n_samples',
                  'mse_mean', 'mse_std', 'snr_mean', 'snr_std',
                  'total_time_sec', 'avg_time_per_signal_sec', 'source']

    sorted_rows = sorted(summary_rows, key=lambda r: r['mse_mean'])

    csv_path = os.path.join(path_logs, 'compare_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    json_path = os.path.join(path_logs, 'compare_report.json')
    with open(json_path, 'w') as f:
        json.dump(sorted_rows, f, indent=4)

    txt_path = os.path.join(path_logs, 'compare_report.txt')
    with open(txt_path, 'w') as f:
        f.write(f"=== Comparaison complete du jeu de test | Data: {data_folder} ===\n\n")
        header = (f"{'model':<10s} {'strategy':<15s} {'n':>6s} {'mse_mean':>12s} {'mse_std':>12s} "
                   f"{'snr_mean':>12s} {'snr_std':>12s} {'total_time_s':>14s} {'avg_time_s/sig':>16s}\n")
        f.write(header)
        f.write('-' * len(header) + '\n')
        for row in sorted_rows:
            total_time = row.get('total_time_sec')
            avg_time = row.get('avg_time_per_signal_sec')
            total_time_str = f"{total_time:>14.4f}" if total_time is not None else f"{'N/A':>14s}"
            avg_time_str = f"{avg_time:>16.4e}" if avg_time is not None else f"{'N/A':>16s}"
            f.write(f"{row['model']:<10s} {row['strategy']:<15s} {row['n_samples']:>6d} "
                     f"{row['mse_mean']:>12.4e} {row['mse_std']:>12.4e} "
                     f"{row['snr_mean']:>12.4e} {row['snr_std']:>12.4e} "
                     f"{total_time_str} {avg_time_str}\n")

    print(f"[COMPARE] Rapport CSV sauvegarde : {csv_path}")
    print(f"[COMPARE] Rapport JSON sauvegarde : {json_path}")
    print(f"[COMPARE] Rapport texte sauvegarde : {txt_path}")
