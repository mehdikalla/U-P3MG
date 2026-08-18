import os
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np

from src.utils.functions import snr_loss, tsnr_loss
from src.utils.plotting_manager import PlottingManager

def get_criterion(name):
    if name == 'MSE': return nn.MSELoss(reduction='mean')
    elif name == 'SNR': return snr_loss()
    elif name == 'TSNR': return tsnr_loss()
    else: raise ValueError(f"Loss '{name}' non reconnue.")

def _unpack_batch(batch, device):
    if len(batch) == 3: xt, y, x0 = batch
    else: xt, y = batch[0], batch[1]; x0 = None
    xt, y = xt.to(device).double(), y.to(device).double()
    if x0 is not None: x0 = x0.to(device).double()
    return xt, y, x0

def get_algo_and_static(args, N_dim, M_dim, device):
    """Charge le bon algorithme mathématique en fonction de args.model"""
    model_name = args.model.strip().lower()
    dx = torch.zeros(1, N_dim).double().to(device)
    dy = torch.zeros(1, M_dim).double().to(device)
    fixed_params = [args.alpha, args.beta, args.eta]

    if model_name == 'p3mg':
        from src.models.p3mg.algo import P3MG_algo
        num_pd = getattr(args, 'num_pd_layers', 5)
        algo = P3MG_algo(num_pd_layers=num_pd).to(device).double()
        static = algo.init_P3MG(fixed_params, dx, dy)
        return algo, static

    elif model_name == 'pd':
        # Aligne sur PrimalDual_algo (src/models/p3mg/primal_dual/algo.py) :
        # init_PD retourne (w0, sub_static). sub_static = [Hmat, L2] est
        # conserve tel quel comme "static" du random search ; w0 = [p0, d0]
        # n'est pas reutilise ici (chaque appel de run_iterative_algo
        # reinitialise son propre etat via algo.init_PD(x0, y)).
        from src.models.pd.algo import PD_Standalone_algo
        algo = PD_Standalone_algo().to(device).double()
        _, sub_static = algo.init_PD(dx, dy)
        return algo, sub_static

    elif model_name == 'pmms':
        from src.models.pmms.algo import PMMS_algo
        algo = PMMS_algo().to(device).double()
        eta = getattr(args, 'eta', 1e-2)
        sigma = getattr(args, 'sigma', 1e-5)
        beta = getattr(args, 'beta', 1e-5)
        static = (eta, sigma, beta)
        return algo, static
        
    elif model_name == 'ista':
        from src.models.ista.algo import ISTA_algo
        algo = ISTA_algo().to(device).double()
        static = algo.init_ISTA(dx, dy)
        return algo, static

    elif model_name == 'hq':
        from src.models.hq.algo import HQ_algo
        algo = HQ_algo().to(device).double()
        static = algo.init_HQ(dx, dy)
        return algo, static
        
    else:
        raise NotImplementedError(f"Le Random Search n'est pas configuré pour le modèle : {model_name.upper()}.")

def run_iterative_algo(model_name, algo, y, x0, static, hp, max_iter=100):
    """Exécute la boucle itérative adaptée au modèle avec ses hyperparamètres explicites."""
    device = y.device

    if model_name == 'p3mg':
        lmbd = torch.tensor(hp['lmbd'], device=device, dtype=torch.float64)
        tau = torch.full((algo.num_pd_layers,), hp['tau'], device=device, dtype=torch.float64)
        x, dyn = algo.iter_P3MG_base(static, x0, y, lmbd, tau)
        for _ in range(1, max_iter):
            x, dyn = algo.iter_P3MG(static, dyn, x, y, lmbd, tau)
        return x

    elif model_name == 'pd':
        # Seul tau est explore (unique hyperparametre du modele PD
        # Standalone, aligne sur P3MG/primal_dual). sigma est reparametre
        # de maniere deterministe dans PD_Standalone_algo.iter_PD a partir
        # de tau, garantissant par construction la condition de stabilite
        # du schema de Chambolle-Pock (tau * sigma * ||H||^2 <= 1).
        tau = torch.tensor(hp['tau'], device=device, dtype=torch.float64)

        sub_static = static
        w0, sub_static = algo.init_PD(x0, y)
        w = w0
        for _ in range(max_iter):
            w = algo.iter_PD(sub_static, w, y, tau)
        p, d = w
        return p

    elif model_name == 'pmms':
        nu = float(hp['nu'])
        eta, sigma, beta = static
        static_pmms, dyn = algo.init_PMMS(x0, y, sigma=sigma, beta=beta, eta=eta)
        x = x0
        for _ in range(max_iter):
            x, dyn = algo.iter_PMMS(static_pmms, dyn, x, y, nu)
        return x
        
    elif model_name == 'ista':
        Hmat, L = static
        gamma = 1.0 / L
        lmbd = float(hp['lmbd'])
        x = x0
        for _ in range(max_iter):
            x = algo.iter_ISTA(x, y, Hmat, gamma, lmbd)
        return x

    elif model_name == 'hq':
        # Maintenant static contient uniquement 2 éléments : Hmat et Ht_H
        Hmat, Ht_H = static
        
        # Calcul dynamique de Ht_y nécessaire pour le HQ
        Ht_y = torch.matmul(y, Hmat).contiguous()
        
        gamma = float(hp['gamma'])
        lmbd_cvx = float(hp['lmbd_cvx'])
        lmbd_ncvx = float(hp['lmbd_ncvx'])
        
        x = x0
        for _ in range(max_iter):
            # Passage des arguments corrigés
            x = algo.iter_HQ(x, y, Hmat, Ht_H, gamma, lmbd_cvx, lmbd_ncvx)
        return x

# =============================================================================
# TRAIN (Calibration)
# =============================================================================
def train(loader, args, paths):
    device = args.device
    criterion = get_criterion(args.criterion)
    path_checkpoints, _, path_logs = paths[1], paths[2], paths[3]
    model_name = args.model.strip().lower()
    
    print(f"--- [TRAIN] {model_name.upper()} (RANDOM_SEARCH) | Trials: {args.n_trials} | Loss: {args.criterion} ---")

    full_data = list(loader)
    subset_size = max(1, int(len(full_data) * 0.10))
    calibration_set = full_data[:subset_size]
    
    sample = calibration_set[0]
    xt_s, y_s, _ = _unpack_batch(sample, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    algo, static = get_algo_and_static(args, N_dim, M_dim, device)
    
    lmbd_min, lmbd_max = args.lmbd_bounds
    lmbd_ist_min, lmbd_ist_max = getattr(args, 'lmbd_ist_bounds', (lmbd_min, lmbd_max))
    nu_min, nu_max = getattr(args, 'nu_bounds', (1e-6, 1e-3))
    tau_min, tau_max = args.tau_bounds
    algo_iters = args.algo_iters
    
    l_mid_log = (np.log10(float(lmbd_min)) + np.log10(float(lmbd_max))) / 2
    best_loss = float('inf')
    
    best_params = {}
    
    start = time.time()
    for i in range(args.n_trials):
        log_l_min, log_l_max = np.log10(float(lmbd_min)), np.log10(float(lmbd_max))
        log_list_min, log_list_max = np.log10(float(lmbd_ist_min)), np.log10(float(lmbd_ist_max))
        log_nu_min, log_nu_max = np.log10(float(nu_min)), np.log10(float(nu_max))
        hp = {}
        
        # Attribution explicite des hyperparamètres selon le modèle
        if model_name == 'hq':
            hp['lmbd_cvx'] = 10 ** random.uniform(log_l_min, log_l_max)
            hp['lmbd_ncvx'] = 10 ** random.uniform(log_l_min, log_l_max)
            hp['gamma'] = random.uniform(float(tau_min), float(tau_max))
        elif model_name == 'pmms':
            if i == 0:
                hp['nu'] = 8.0e-5
            else:
                hp['nu'] = 10 ** random.uniform(log_nu_min, log_nu_max)

        elif model_name == 'ista':
            hp['lmbd'] = 10 ** random.uniform(log_list_min, log_list_max)
        elif model_name == 'p3mg':
            hp['lmbd'] = 10 ** random.uniform(log_l_min, log_l_max)
            hp['tau'] = random.uniform(float(tau_min), float(tau_max))
        elif model_name == 'pd':
            # Modele Primal-Dual standalone : un unique hyperparametre
            # recherche, tau, dans les memes bornes (args.tau_bounds) que
            # celles utilisees pour l'apprentissage en unrolling
            # (PD_Standalone_model, src/models/pd/net.py) et pour P3MG.
            hp['tau'] = random.uniform(float(tau_min), float(tau_max))

        
        val_loss = 0.0
        with torch.no_grad():
            for batch in calibration_set:
                xt, y, x0 = _unpack_batch(batch, device)
                if x0 is None: 
                    x0 = y.sum(1, keepdim=True).repeat(1, N_dim)/(M_dim*N_dim)
                
                xh = run_iterative_algo(model_name, algo, y, x0, static, hp, max_iter=algo_iters)
                val_loss += criterion(xh, xt).item()
        
        avg_loss = val_loss / len(calibration_set)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = hp.copy()
            
            hp_str = " ".join([f"{k}={v:.4e}" if 'lmbd' in k or 'nu' in k else f"{k}={v:.4f}" for k, v in hp.items()])
            print(f"   [{i+1}/{args.n_trials}] New Best! {hp_str} | Loss={best_loss:.4e}")

    print(f"[RESULT] Best Params: {best_params}")
    with open(os.path.join(path_checkpoints, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"--- [TRAIN] Terminé en {(time.time()-start)/60:.2f} min ---")

# =============================================================================
# TEST (Application & Reporting Complet)
# =============================================================================
def find_latest_best_params(model_name, strategy, data_folder=None, current_base_dir=None, run_group=None):
    """
    Recherche le fichier 'best_params.json' le plus récent pour un modèle,
    une stratégie et un dossier de donnees donnés dans
    'runs/<model>/<strategy>/<data_folder>/*' (ou
    'runs/<run_group>/<model>/<strategy>/<data_folder>/*' si run_group est
    fourni), en excluant le dossier du run en cours.
    """
    root = os.path.join("runs", run_group.strip().lower()) if run_group else "runs"
    if data_folder:
        strategy_dir = os.path.join(root, model_name, strategy, data_folder.strip().lower())
    else:
        strategy_dir = os.path.join(root, model_name, strategy)

    if not os.path.isdir(strategy_dir):
        return None


    run_dirs = sorted(
        (d for d in os.listdir(strategy_dir) if os.path.isdir(os.path.join(strategy_dir, d))),
        reverse=True
    )

    for run_name in run_dirs:
        run_path = os.path.join(strategy_dir, run_name)
        if current_base_dir and os.path.abspath(run_path) == os.path.abspath(current_base_dir):
            continue
        candidate = os.path.join(run_path, 'checkpoints', 'best_params.json')
        if os.path.exists(candidate):
            return candidate

    return None

def test(loader, args, paths):
    device = args.device
    criterion_name = args.criterion
    path_checkpoints, path_plots, path_logs = paths[1], paths[2], paths[3]
    model_name = args.model.strip().lower()
    
    print(f"--- [TEST] {model_name.upper()} (RANDOM_SEARCH) | Metric: {criterion_name} | Samples: {len(loader.dataset)} ---")

    checkpoint_override = getattr(args, 'checkpoint', None)
    if checkpoint_override:
        params_path = checkpoint_override
    elif args.mode == 'test':
        params_path = find_latest_best_params(model_name, args.strategy, data_folder=(getattr(args, 'run_tag', None) or getattr(args, 'data_folder', None)), current_base_dir=paths[0], run_group=getattr(args, 'run_group', None))



        if params_path:
            print(f"[INFO] Aucun --checkpoint fourni. Utilisation des paramètres les plus récents : {params_path}")
        else:
            params_path = os.path.join(path_checkpoints, 'best_params.json')
    else:
        params_path = os.path.join(path_checkpoints, 'best_params.json')

    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print(f"[INFO] Paramètres chargés depuis {params_path} : {best_params}")
    except:

        # Fallback explicite
        if model_name == 'hq':
            best_params = {'lmbd_cvx': 1.0, 'lmbd_ncvx': 1.0, 'gamma': 1.0}
        elif model_name == 'pmms':
            best_params = {'nu': 8.0e-5}
        elif model_name == 'ista':
            best_params = {'lmbd': 1.0}
        elif model_name == 'pd':
            best_params = {'tau': 0.5}
        else:
            best_params = {'lmbd': 1.0, 'tau': 0.5}

        print("[WARN] Paramètres par défaut chargés.")

    sample = next(iter(loader))
    xt_s, y_s, _ = _unpack_batch(sample, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    algo, static = get_algo_and_static(args, N_dim, M_dim, device)
    fixed_static_params = [args.alpha, args.beta, args.eta]
    
    saved_samples = [] 
    
    def compute_metric(xh, xt):
        if criterion_name == 'MSE': return torch.mean((xh - xt)**2, dim=1)
        if criterion_name in ['SNR', 'TSNR']:
             n = torch.mean((xt - xh)**2, dim=1); s = torch.mean(xt**2, dim=1)
             return -10 * torch.log10(s / (n + 1e-12))
        return torch.mean((xh - xt)**2, dim=1)

    hp_str = " ".join([f"{k}={v:.4e}" if 'lmbd' in k or 'nu' in k else f"{k}={v:.4f}" for k, v in best_params.items()])
    print(f"[INFO] Application sur {len(loader)} batchs avec {hp_str}...")

    with torch.no_grad():
        for batch in loader:
            xt, y, x0 = _unpack_batch(batch, device)
            if x0 is None: x0 = y.sum(1, keepdim=True).repeat(1, N_dim)/(M_dim*N_dim)
            
            xh = run_iterative_algo(
                model_name, algo, y, x0, static, 
                hp=best_params, 
                max_iter=args.algo_iters
            )
            
            metrics = compute_metric(xh, xt).cpu()
            xt_c, xh_c = xt.cpu(), xh.cpu()
            for k in range(xt.size(0)):
                saved_samples.append((metrics[k].item(), xt_c[k].numpy(), xh_c[k].numpy()))

    if not saved_samples:
        print("[WARN] Aucun résultat généré.")
        return 0

    saved_samples.sort(key=lambda x: x[0])
    arr = np.array([x[0] for x in saved_samples])
    
    mean_v = np.mean(arr)
    std_v = np.std(arr)
    med_v = np.median(arr)
    min_v = np.min(arr)
    max_v = np.max(arr)

    print(f"[RESULT] Mean {criterion_name}: {mean_v:.4e} | Median: {med_v:.4e} | Std: {std_v:.4e} | Best: {min_v:.4e} | Worst: {max_v:.4e}")

    table_str = (
        f"\n+-----------------------------------------+\n"
        f"|        RESULTATS TEST ({criterion_name:<5})        |\n"
        f"+-----------------------+-----------------+\n"
        f"| Mean                  | {mean_v:<15.4e} |\n"
        f"| Median                | {med_v:<15.4e} |\n"
        f"| Std                   | {std_v:<15.4e} |\n"
        f"| Min (Best)            | {min_v:<15.4e} |\n"
        f"| Max (Worst)           | {max_v:<15.4e} |\n"
        f"+-----------------------+-----------------+\n"
    )
    with open(os.path.join(path_logs, 'test_results_table.txt'), 'w') as f:
        f.write(table_str)

    pm = PlottingManager(
        model=None, 
        p3mg_tmp=algo, 
        val_loader=None, 
        N_dim=N_dim, 
        M_dim=M_dim, 
        static_params=fixed_static_params,
        device=device, 
        path_plots=path_plots, 
        criterion=None, 
        metric_name=criterion_name
    )
    
    if hasattr(pm, 'plot_statistical_samples'):
        pm.plot_statistical_samples(saved_samples, criterion_name, prefix="test_")
    if hasattr(pm, 'plot_test_error_distribution'):
        pm.plot_test_error_distribution(arr, metric_name=criterion_name)
    
    return mean_v
