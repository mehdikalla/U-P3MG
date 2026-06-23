import os
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np

# Imports des utilitaires
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

# =============================================================================
# ROUTEUR DYNAMIQUE DES ALGORITHMES
# =============================================================================
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
        from src.models.pd.algo import PD_Standalone_algo
        algo = PD_Standalone_algo().to(device).double()
        Hmat, p0, d0 = algo.init_PD(dx, dy)
        static = Hmat
        return algo, static

    elif model_name == 'pmms':
        from src.models.pmms.algo import PMMS_algo
        algo = PMMS_algo().to(device).double()
        # Pour PMMS, les vraies matrices statiques dépendent des paramètres qu'on 
        # va faire varier. On transmet donc juste eta et nu via 'static'.
        eta = getattr(args, 'eta', 0.01)
        nu = getattr(args, 'nu', 0.1)
        static = (eta, nu)
        return algo, static
        
    else:
        raise NotImplementedError(f"Le Random Search n'est pas configuré pour le modèle : {model_name.upper()}.")

def run_iterative_algo(model_name, algo, y, x0, static, lmbd_val, tau_val, max_iter=100):
    """Exécute la boucle itérative adaptée au modèle choisi."""
    device = y.device

    if model_name == 'p3mg':
        lmbd = torch.tensor(lmbd_val, device=device, dtype=torch.float64)
        tau = torch.full((algo.num_pd_layers,), tau_val, device=device, dtype=torch.float64)
        x, dyn = algo.iter_P3MG_base(static, x0, y, lmbd, tau)
        for _ in range(1, max_iter):
            x, dyn = algo.iter_P3MG(static, dyn, x, y, lmbd, tau)
        return x

    elif model_name == 'pd':
        tau = torch.tensor(tau_val, device=device, dtype=torch.float64)
        sigma = torch.tensor(lmbd_val, device=device, dtype=torch.float64)
        rho = torch.tensor(0.1, device=device, dtype=torch.float64)
        Hmat = static
        _, p0, d0 = algo.init_PD(x0, y)
        p, p_old, d, d_old = p0, p0, d0, d0
        for _ in range(max_iter):
            p_new, d_new = algo.iter_PD(p, p_old, d, d_old, y, Hmat, tau, sigma, rho)
            p_old = p
            d_old = d
            p = p_new
            d = d_new
        return p

    elif model_name == 'pmms':
        # Mapping des variables du Random Search : lmbd_val -> sigma | tau_val -> beta
        eta, nu = static
        sigma = float(lmbd_val)
        beta = float(tau_val)
        
        # Le PMMS recalcule ses constantes (ex: Cg2) en fonction des nouveaux hyperparamètres
        static_pmms, dynamic = algo.init_PMMS(x0, y, sigma=sigma, beta=beta, eta=eta, nu=nu)
        
        x = x0
        for _ in range(max_iter):
            x, dynamic = algo.iter_PMMS(static_pmms, dynamic, x, y)
        return x
# =============================================================================
# TRAIN (Calibration)
# =============================================================================
def train(loader, args, paths):
    device = args.device
    criterion = get_criterion(args.criterion)
    path_checkpoints, _, path_logs = paths[1], paths[2], paths[3]
    model_name = args.model.strip().lower()
    
    print(f"--- [RANDOM SEARCH] Calibration {model_name.upper()} | Alpha={args.alpha}, Beta={args.beta} ---")

    # Sous-ensemble de calibration (10%)
    full_data = list(loader)
    subset_size = max(1, int(len(full_data) * 0.10))
    calibration_set = full_data[:subset_size]
    
    # Dimensions
    sample = calibration_set[0]
    xt_s, y_s, _ = _unpack_batch(sample, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    # Chargement dynamique Algo + Static
    algo, static = get_algo_and_static(args, N_dim, M_dim, device)
    
    lmbd_min, lmbd_max = args.lmbd_bounds
    tau_min, tau_max = args.tau_bounds
    algo_iters = args.algo_iters
    
    # Sécurisation des conversions en float pour np.log10 (YAML bug)
    l_mid_log = (np.log10(float(lmbd_min)) + np.log10(float(lmbd_max))) / 2
    best_loss = float('inf')
    best_params = {'lmbd': 10**l_mid_log, 'tau': (float(tau_min)+float(tau_max))/2}
    
    start = time.time()
    for i in range(args.n_samples):
        # Tirage
        log_l_min, log_l_max = np.log10(float(lmbd_min)), np.log10(float(lmbd_max))
        curr_lmbd = 10 ** random.uniform(log_l_min, log_l_max)
        curr_tau = random.uniform(float(tau_min), float(tau_max))
        
        val_loss = 0.0
        with torch.no_grad():
            for batch in calibration_set:
                xt, y, x0 = _unpack_batch(batch, device)
                if x0 is None: 
                    x0 = y.sum(1, keepdim=True).repeat(1, N_dim)/(M_dim*N_dim)
                
                xh = run_iterative_algo(model_name, algo, y, x0, static, curr_lmbd, curr_tau, max_iter=algo_iters)
                val_loss += criterion(xh, xt).item()
        
        avg_loss = val_loss / len(calibration_set)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {'lmbd': curr_lmbd, 'tau': curr_tau}
            print(f"   [{i+1}/{args.n_samples}] New Best! L={curr_lmbd:.4e} T={curr_tau:.4f} | Loss={best_loss:.4e}")

    print(f"[RESULT] Best Params: {best_params} (Time: {time.time()-start:.1f}s)")
    with open(os.path.join(path_checkpoints, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)

# =============================================================================
# TEST (Application & Reporting Complet)
# =============================================================================
def test(loader, args, paths):
    device = args.device
    criterion_name = args.criterion
    path_checkpoints, path_plots, path_logs = paths[1], paths[2], paths[3]
    model_name = args.model.strip().lower()
    
    print(f"--- [RANDOM SEARCH] Test Final {model_name.upper()} ---")
    
    try:
        with open(os.path.join(path_checkpoints, 'best_params.json'), 'r') as f:
            best_params = json.load(f)
        print(f"[INFO] Paramètres chargés : {best_params}")
    except:
        best_params = {'lmbd': 1.0, 'tau': 0.5} 
        print("[WARN] Paramètres par défaut chargés.")

    sample = next(iter(loader))
    xt_s, y_s, _ = _unpack_batch(sample, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    # Chargement dynamique Algo + Static
    algo, static = get_algo_and_static(args, N_dim, M_dim, device)
    fixed_static_params = [args.alpha, args.beta, args.eta]
    
    saved_samples = [] 
    
    def compute_metric(xh, xt):
        if criterion_name == 'MSE': return torch.mean((xh - xt)**2, dim=1)
        if criterion_name in ['SNR', 'TSNR']:
             n = torch.mean((xt - xh)**2, dim=1); s = torch.mean(xt**2, dim=1)
             return -10 * torch.log10(s / (n + 1e-12))
        return torch.mean((xh - xt)**2, dim=1)

    print(f"[INFO] Application sur {len(loader)} batchs avec L={best_params['lmbd']:.4e}, T={best_params['tau']:.4f}...")

    with torch.no_grad():
        for batch in loader:
            xt, y, x0 = _unpack_batch(batch, device)
            if x0 is None: x0 = y.sum(1, keepdim=True).repeat(1, N_dim)/(M_dim*N_dim)
            
            xh = run_iterative_algo(model_name, algo, y, x0, static, best_params['lmbd'], best_params['tau'], max_iter=args.algo_iters)
            
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

    print(f"[RESULT] Mean {criterion_name}: {mean_v:.4e}")
    
    table_str = (
        f"\n+-----------------------------------------+\n"
        f"|  RESULTATS RANDOM SEARCH ({criterion_name:<5})  |\n"
        f"+-----------------------+-----------------+\n"
        f"| Moyenne               | {mean_v:<15.4e} |\n"
        f"| Mediane               | {np.median(arr):<15.4e} |\n"
        f"| Min (Best)            | {np.min(arr):<15.4e} |\n"
        f"| Max (Worst)           | {np.max(arr):<15.4e} |\n"
        f"+-----------------------+-----------------+\n"
    )
    with open(os.path.join(path_logs, 'rs_results_table.txt'), 'w') as f:
        f.write(table_str)

    # NOUVELLE MÉTHODE DE PLOT UNIVERSELLE
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
        pm.plot_statistical_samples(saved_samples, criterion_name, prefix="RS_test_")
    if hasattr(pm, 'plot_test_error_distribution'):
        pm.plot_test_error_distribution(arr, metric_name=criterion_name)
    
    return mean_v