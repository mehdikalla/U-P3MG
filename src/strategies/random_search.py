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
    """Factory pour la fonction de perte."""
    if name == 'MSE':
        return nn.MSELoss(reduction='mean')
    elif name == 'SNR':
        return snr_loss()
    elif name == 'TSNR':
        return tsnr_loss()
    else:
        raise ValueError(f"Loss '{name}' non reconnue.")

def _unpack_batch(batch, device):
    """Helper pour récupérer xt, y, x0 depuis un batch."""
    if len(batch) == 3:
        xt, y, x0 = batch
    else:
        xt, y = batch[0], batch[1]
        x0 = None
    
    xt = xt.to(device).double()
    y  = y.to(device).double()
    if x0 is not None:
        x0 = x0.to(device).double()
    return xt, y, x0

def run_iterative_algo(p3mg_net, y, x0, static, lmbd_val, tau_val, max_iter=100):
    """
    Exécute l'algorithme P3MG itératif complet (boucle for) avec les params fixés.
    """
    device = y.device
    
    # 1. Conversion des scalaires en tenseurs (double précision)
    # P3MGNet attend généralement des tenseurs
    lmbd = torch.tensor(lmbd_val, device=device).double()
    tau = torch.tensor(tau_val, device=device).double()
    
    # 2. Initialisation (Layer 0)
    x, dyn = p3mg_net.iter_P3MG_base(static, x0, y, lmbd, tau)
    
    # 3. Boucle itérative (Layers 1 to K)
    for k in range(1, max_iter):
        x, dyn = p3mg_net.iter_P3MG(static, dyn, x, y, lmbd, tau)
        
    return x

# =============================================================================
# 1. FONCTION DE CALIBRATION (TRAIN)
# =============================================================================
def train(loader, args, paths):
    """
    Calibration : Cherche les meilleurs hyperparamètres (Lambda, Tau) 
    sur un sous-ensemble (10%) du loader.
    """
    device = args.device
    criterion_name = args.criterion if hasattr(args, 'criterion') else 'MSE'
    criterion = get_criterion(criterion_name)
    path_save, path_checkpoints, path_plots, path_logs = paths
    
    print(f"--- [RANDOM SEARCH - TRAIN] Calibration (Lambda/Tau) | Loss: {criterion_name} ---")
    
    # 1. Création du sous-ensemble de calibration (10%)
    full_data = list(loader)
    subset_size = max(1, int(len(full_data) * 0.10))
    calibration_set = full_data[:subset_size]
    print(f"[INFO] Dataset total: {len(full_data)} batchs -> Calibration sur: {subset_size} batchs")

    # 2. Instanciation de l'Algorithme
    p3mg_algo = P3MG_algo(num_layers=1).to(device).double()
    
    # Récupération dimensions depuis le premier batch
    sample_batch = calibration_set[0]
    xt_s, y_s, _ = _unpack_batch(sample_batch, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    dx = torch.zeros(1, N_dim).double().to(device)
    dy = torch.zeros(1, M_dim).double().to(device)
    
    # 3. Récupération des plages de recherche depuis args
    # On suppose que args contient lmbd_bounds=[min, max] et tau_bounds=[min, max]
    lmbd_min, lmbd_max = args.lmbd_bounds
    tau_min, tau_max = args.tau_bounds
    
    print(f"[INFO] Plages de recherche : Lambda=[{lmbd_min}, {lmbd_max}], Tau=[{tau_min}, {tau_max}]")
    print(f"[INFO] Lancement de {args.n_samples} configurations aléatoires...")
    
    best_loss = float('inf')
    # Valeurs par défaut si jamais on ne trouve rien (ou si n_samples=0)
    best_params = {'lmbd': (lmbd_min+lmbd_max)/2, 'tau': (tau_min+tau_max)/2}
    
    start_time = time.time()
    algo_iters = getattr(args, 'algo_iters', 100) # Nombre d'itérations de l'algo P3MG

    # 4. Boucle de Random Search
    for i in range(args.n_samples):
        # A. Tirage aléatoire
        curr_lmbd = random.uniform(lmbd_min, lmbd_max)
        curr_tau = random.uniform(tau_min, tau_max)
        
        # B. Évaluation sur le calibration_set
        total_val_loss = 0.0
        
        # Init static params pour cette configuration
        # Note: init_P3MG prend une liste de params [lmbd, tau]
        static = p3mg_algo.init_P3MG([curr_lmbd, curr_tau], dx, dy)
        
        with torch.no_grad():
            for batch in calibration_set:
                xt, y, x0 = _unpack_batch(batch, device)
                if x0 is None:
                    mean_v = y.sum(1, keepdim=True)/(M_dim*N_dim)
                    x0 = mean_v.repeat(1, N_dim)
                
                # Exécution de l'algo itératif
                xh = run_iterative_algo(p3mg_algo, y, x0, static, curr_lmbd, curr_tau, max_iter=algo_iters)
                
                # Le criterion renvoie -SNR pour SNR/TSNR, donc on minimise toujours 'loss'
                loss = criterion(xh, xt)
                total_val_loss += loss.item()
        
        avg_loss = total_val_loss / len(calibration_set)
        
        # C. Mise à jour du meilleur
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {'lmbd': curr_lmbd, 'tau': curr_tau}
            print(f"   [{i+1}/{args.n_samples}] New Best! L={curr_lmbd:.4f}, T={curr_tau:.4f} | Loss={best_loss:.4e}")
        
        if (i+1) % 10 == 0:
            print(f"   ... {i+1} configs testées ...")

    dt = time.time() - start_time
    print(f"--- Calibration terminée en {dt:.1f}s ---")
    print(f"[RESULT] Meilleurs paramètres : {best_params} (Loss: {best_loss:.4e})")

    # 5. Sauvegarde des meilleurs paramètres
    with open(os.path.join(path_checkpoints, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)


# =============================================================================
# 2. FONCTION DE TEST (TEST)
# =============================================================================
def test(loader, args, paths):
    """
    Test Final : Applique les paramètres calibrés (Lambda, Tau) sur TOUT le test set.
    """
    device = args.device
    criterion_name = args.criterion if hasattr(args, 'criterion') else 'MSE'
    path_checkpoints, path_plots, path_logs = paths[1], paths[2], paths[3]
    
    print(f"--- [RANDOM SEARCH - TEST] Application sur Test Set complet | Loss: {criterion_name} ---")
    
    # 1. Chargement des paramètres optimaux
    params_path = os.path.join(path_checkpoints, 'best_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print(f"[INFO] Paramètres chargés : {best_params}")
    else:
        # Fallback sur les bornes moyennes si pas de fichier (ex: test lancé sans train)
        print(f"[WARN] Pas de fichier best_params.json trouvé. Utilisation de la moyenne des bornes.")
        l_def = (args.lmbd_bounds[0] + args.lmbd_bounds[1]) / 2
        t_def = (args.tau_bounds[0] + args.tau_bounds[1]) / 2
        best_params = {'lmbd': l_def, 'tau': t_def}

    # 2. Instanciation Algo
    p3mg_algo = P3MG_algo(num_layers=1).to(device).double()
    
    sample_batch = next(iter(loader))
    xt_s, y_s, _ = _unpack_batch(sample_batch, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    dx = torch.zeros(1, N_dim).double().to(device)
    dy = torch.zeros(1, M_dim).double().to(device)
    
    # Init static avec les params optimaux
    static = p3mg_algo.init_P3MG([best_params['lmbd'], best_params['tau']], dx, dy)
    
    # 3. Boucle de Test (Benchmarking)
    algo_iters = getattr(args, 'algo_iters', 100)
    saved_samples = []

    # Helper interne métrique (identique à network.py)
    def compute_sample_metric(xh, xt, name):
        if name == 'MSE': 
            return torch.mean((xh - xt)**2, dim=1)
        elif name in ['SNR', 'TSNR']:
            noise = torch.mean((xt - xh)**2, dim=1)
            sig   = torch.mean(xt**2, dim=1)
            return -10 * torch.log10(sig / (noise + 1e-12)) 
        else: 
            return torch.mean((xh - xt)**2, dim=1)

    print(f"[INFO] Traitement de {len(loader)} batchs...")
    
    with torch.no_grad():
        for batch in loader:
            xt, y, x0 = _unpack_batch(batch, device)
            if x0 is None:
                mean_v = y.sum(1, keepdim=True)/(M_dim*N_dim)
                x0 = mean_v.repeat(1, N_dim)
            
            # Reconstruction avec paramètres fixés
            xh = run_iterative_algo(
                p3mg_algo, y, x0, static, 
                best_params['lmbd'], best_params['tau'], 
                max_iter=algo_iters
            )
            
            metrics = compute_sample_metric(xh, xt, criterion_name)
            
            metrics_cpu = metrics.cpu()
            xt_cpu = xt.cpu()
            xh_cpu = xh.cpu()
            
            for k in range(xt.size(0)):
                val = metrics_cpu[k].item()
                saved_samples.append((val, xt_cpu[k], xh_cpu[k]))

    # 4. Statistiques et Plots
    if not saved_samples:
        print("[WARN] Aucun sample.")
        return 0.0

    saved_samples.sort(key=lambda x: x[0])
    all_vals = [x[0] for x in saved_samples]
    arr = np.array(all_vals)

    mean_v = np.mean(arr)
    med_v  = np.median(arr)
    std_v  = np.std(arr)
    best_v = arr[0]
    worst_v = arr[-1]

    print(f"[RESULTATS] Mean: {mean_v:.4e} | Median: {med_v:.4e} | Best: {best_v:.4e} | Worst: {worst_v:.4e}")
    
    # Sauvegarde Table
    table_str = (
        f"\n+-----------------------------------------+\n"
        f"|    RESULTATS RANDOM SEARCH ({criterion_name:<4})   |\n"
        f"+-----------------------+-----------------+\n"
        f"| Params (Lmbd, Tau)    | {best_params['lmbd']:.4f}, {best_params['tau']:.4f}    |\n"
        f"| Mean                  | {mean_v:<15.4e} |\n"
        f"| Median                | {med_v:<15.4e} |\n"
        f"| Std                   | {std_v:<15.4e} |\n"
        f"| Min (Best)            | {best_v:<15.4e} |\n"
        f"| Max (Worst)           | {worst_v:<15.4e} |\n"
        f"+-----------------------+-----------------+\n"
    )
    with open(os.path.join(path_logs, 'rs_results_table.txt'), 'w') as f:
        f.write(table_str)

    # Utilisation PlottingManager pour les graphiques
    # On crée un manager minimal car on n'a pas de 'model' nn.Module ici
    pm = PlottingManager(
        model=None, p3mg_tmp=None, val_loader=None, 
        N_dim=N_dim, M_dim=M_dim, static_params=[], 
        device=device, path_plots=path_plots, criterion=None, metric_name=criterion_name
    )

    best_sample = saved_samples[0]
    worst_sample = saved_samples[-1]
    med_sample = saved_samples[len(saved_samples)//2]
    idx_mean = (np.abs(arr - mean_v)).argmin()
    mean_sample = saved_samples[idx_mean]

    def safe_plot(sample, tag):
        try:
            pm.plot_signals(
                sample[1].unsqueeze(0), 
                sample[2].unsqueeze(0), 
                f'rs_test_{tag}', 
                criterion_name, 
                sample[0]
            )
        except Exception as e:
            print(f"Plot fail {tag}: {e}")

    safe_plot(best_sample, "BEST")
    safe_plot(worst_sample, "WORST")
    safe_plot(med_sample, "MEDIAN")
    safe_plot(mean_sample, "MEAN")

    # Histogramme
    if hasattr(pm, 'plot_test_error_distribution'):
        pm.plot_test_error_distribution(arr, metric_name=criterion_name)

    return mean_v