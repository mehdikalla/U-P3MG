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
from src.models.p3mg.algo import P3MG_algo

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

def run_iterative_algo(p3mg_algo, y, x0, static, lmbd_val, tau_val, max_iter=100):
    """
    Exécute l'algo P3MG avec:
      - static : contient (Hmat, alpha, beta, eta...) FIXES
      - lmbd_val, tau_val : paramètres DYNAMIQUES (optimisés)
    """
    device = y.device
    # 1. Lambda reste un scalaire
    lmbd = torch.tensor(lmbd_val, device=device, dtype=torch.float64)
    # 2. Tau doit être un VECTEUR 1D pour être indexable dans le modèle PD
    tau = torch.full((p3mg_algo.num_pd_layers,), tau_val, device=device, dtype=torch.float64)
    
    # Init (Layer 0)
    x, dyn = p3mg_algo.iter_P3MG_base(static, x0, y, lmbd, tau)
    
    # Iterations (Layer 1..K)
    for k in range(1, max_iter):
        x, dyn = p3mg_algo.iter_P3MG(static, dyn, x, y, lmbd, tau)
        
    return x

# =============================================================================
# TRAIN (Calibration)
# =============================================================================
def train(loader, args, paths):
    device = args.device
    criterion = get_criterion(args.criterion)
    path_checkpoints, _, path_logs = paths[1], paths[2], paths[3]
    
    print(f"--- [RANDOM SEARCH] Calibration (Lambda/Tau) | Fixed: Alpha={args.alpha}, Beta={args.beta}, Eta={args.eta} ---")

    # 1. Sous-ensemble de calibration (10%)
    full_data = list(loader)
    subset_size = max(1, int(len(full_data) * 0.10))
    calibration_set = full_data[:subset_size]
    
    # 2. Instanciation (Algo pur)
    num_pd = getattr(args, 'num_pd_layers', 5)
    p3mg_algo = P3MG_algo(num_pd_layers=num_pd).to(device).double()
    
    # Dimensions & Static (Calculés UNE SEULE FOIS avec les params FIXES)
    sample = calibration_set[0]
    xt_s, y_s, _ = _unpack_batch(sample, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    dx = torch.zeros(1, N_dim).double().to(device)
    dy = torch.zeros(1, M_dim).double().to(device)
    
    fixed_static_params = [args.alpha, args.beta, args.eta]
    static = p3mg_algo.init_P3MG(fixed_static_params, dx, dy)
    
    # 3. Random Search sur Lambda et Tau
    lmbd_min, lmbd_max = args.lmbd_bounds
    tau_min, tau_max = args.tau_bounds
    algo_iters = args.algo_iters
    
    # Init Best Params (Moyenne géométrique pour Lambda, Arithmétique pour Tau)
    l_mid_log = (np.log10(lmbd_min) + np.log10(lmbd_max)) / 2
    best_loss = float('inf')
    best_params = {'lmbd': 10**l_mid_log, 'tau': (tau_min+tau_max)/2}
    
    start = time.time()
    
    for i in range(args.n_samples):
        # A. Tirage
        # Lambda : Échelle Logarithmique (10^x)
        log_l_min = np.log10(lmbd_min)
        log_l_max = np.log10(lmbd_max)
        curr_lmbd = 10 ** random.uniform(log_l_min, log_l_max)
        
        # Tau : Échelle Linéaire (Classique pour un step size)
        curr_tau = random.uniform(tau_min, tau_max)
        
        # B. Eval
        val_loss = 0.0
        with torch.no_grad():
            for batch in calibration_set:
                xt, y, x0 = _unpack_batch(batch, device)
                if x0 is None: 
                    x0 = y.sum(1, keepdim=True).repeat(1, N_dim)/(M_dim*N_dim)
                
                xh = run_iterative_algo(p3mg_algo, y, x0, static, curr_lmbd, curr_tau, max_iter=algo_iters)
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
    
    print("--- [RANDOM SEARCH] Test Final ---")
    
    # 1. Chargement Params
    try:
        with open(os.path.join(path_checkpoints, 'best_params.json'), 'r') as f:
            best_params = json.load(f)
        print(f"[INFO] Paramètres chargés : {best_params}")
    except:
        best_params = {'lmbd': 1.0, 'tau': 0.5} # Fallback
        print("[WARN] Paramètres par défaut chargés.")

    # 2. Instanciation
    num_pd = getattr(args, 'num_pd_layers', 5)
    p3mg_algo = P3MG_algo(num_pd_layers=num_pd).to(device).double()
    
    # Init Static
    sample = next(iter(loader))
    xt_s, y_s, _ = _unpack_batch(sample, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    dx, dy = torch.zeros(1, N_dim).double().to(device), torch.zeros(1, M_dim).double().to(device)
    
    # On prépare les static_params correctement
    fixed_static_params = [args.alpha, args.beta, args.eta]
    static = p3mg_algo.init_P3MG(fixed_static_params, dx, dy)
    
    # 3. Benchmark
    saved_samples = [] # Stockera (loss_val, xt, xh)
    
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
            
            xh = run_iterative_algo(
                p3mg_algo, y, x0, static, 
                best_params['lmbd'], best_params['tau'], 
                max_iter=args.algo_iters
            )
            
            metrics = compute_metric(xh, xt).cpu()
            xt_c, xh_c = xt.cpu(), xh.cpu()
            for k in range(xt.size(0)):
                saved_samples.append((metrics[k].item(), xt_c[k], xh_c[k]))

    # 4. Statistiques et Plots (Comme network.py)
    if not saved_samples:
        print("[WARN] Aucun résultat généré.")
        return 0

    saved_samples.sort(key=lambda x: x[0])
    vals = [x[0] for x in saved_samples]
    arr = np.array(vals)

    mean_v = np.mean(arr)
    med_v  = np.median(arr)
    std_v  = np.std(arr)
    min_v  = np.min(arr) # Best
    max_v  = np.max(arr) # Worst

    print(f"[RESULT] Mean {criterion_name}: {mean_v:.4e}")
    
    # Sauvegarde Tableau
    table_str = (
        f"\n+-----------------------------------------+\n"
        f"|  RESULTATS RANDOM SEARCH ({criterion_name:<5})  |\n"
        f"+-----------------------+-----------------+\n"
        f"| Nombre d'echantillons | {len(arr):<15} |\n"
        f"| Moyenne               | {mean_v:<15.4e} |\n"
        f"| Mediane               | {med_v:<15.4e} |\n"
        f"| Ecart-Type (Std)      | {std_v:<15.4e} |\n"
        f"| Min (Best)            | {min_v:<15.4e} |\n"
        f"| Max (Worst)           | {max_v:<15.4e} |\n"
        f"+-----------------------+-----------------+\n"
    )
    print(table_str)
    with open(os.path.join(path_logs, 'rs_results_table.txt'), 'w') as f:
        f.write(table_str)

    # Initialisation PlottingManager
    pm = PlottingManager(
        model=None, 
        p3mg_tmp=p3mg_algo, 
        val_loader=None, 
        N_dim=N_dim, 
        M_dim=M_dim, 
        static_params=fixed_static_params,
        device=device, 
        path_plots=path_plots, 
        criterion=None, 
        metric_name=criterion_name
    )

    # Sélection des échantillons représentatifs
    best_sample = saved_samples[0]
    worst_sample = saved_samples[-1]
    med_sample = saved_samples[len(saved_samples)//2]
    idx_mean = (np.abs(arr - mean_v)).argmin()
    mean_sample = saved_samples[idx_mean]

    # Fonction helper pour tracer
    def safe_plot(sample, tag):
        try:
            pm.plot_signals(
                sample[1].unsqueeze(0), 
                sample[2].unsqueeze(0), 
                f"RS_{tag}", 
                criterion_name, 
                sample[0]
            )
        except Exception as e:
            # On affiche l'erreur complète pour débugger si ça replante
            import traceback
            traceback.print_exc()
            print(f"[WARN] Erreur Plot {tag}: {e}")

    safe_plot(best_sample, "BEST")
    safe_plot(worst_sample, "WORST")
    safe_plot(med_sample, "MEDIAN")
    safe_plot(mean_sample, "MEAN")

    # Histogramme de distribution
    if hasattr(pm, 'plot_test_error_distribution'):
        try:
            pm.plot_test_error_distribution(arr, metric_name=criterion_name)
        except Exception as e:
            print(f"[WARN] Erreur Histogramme: {e}")
    
    return mean_v