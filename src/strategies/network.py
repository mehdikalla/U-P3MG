import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Imports des utilitaires existants
from src.utils.functions import snr_loss, tsnr_loss 
from src.utils.plotting_manager import PlottingManager
from src.models.p3mg.algo import P3MGNet

def get_criterion(name):
    """Factory pour la fonction de perte."""
    if name == 'MSE':
        return nn.MSELoss(reduction='mean')
    elif name == 'SNR':
        return snr_loss()
    elif name == 'TSNR':
        return tsnr_loss()
    else:
        raise ValueError(f"Loss '{name}' non reconnue. Choisir: MSE, SNR, TSNR.")

def save_config(path, args):
    """Sauvegarde la configuration du run."""
    try:
        config = {k: str(v) for k, v in vars(args).items()}
        with open(os.path.join(path, 'run_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder la config : {e}")

def _unpack_batch(batch, device):
    """Helper pour déballer le batch (gère le cas avec ou sans x0)."""
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

def init_static_params(rho, gamma, N_dim, M_dim, device):
    """Initialise les paramètres statiques via P3MGNet."""
    # Instanciation temporaire pour calculer les statiques
    p3mg_tmp = P3MGNet(num_layers=1).to(device).double()
    
    dx = torch.zeros(1, N_dim).double().to(device)
    dy = torch.zeros(1, M_dim).double().to(device)
    
    params = [rho, gamma]
    static = p3mg_tmp.init_P3MG(params, dx, dy)
    
    return static, p3mg_tmp

# =============================================================================
# 1. FONCTION D'ENTRAINEMENT (Indépendante)
# =============================================================================
def train(model, train_loader, val_loader, args, paths):
    """
    Exécute uniquement la phase d'entrainement et de validation.
    Sauvegarde les checkpoints mais NE LANCE PAS le test.
    """
    device = args.device
    criterion_name = args.criterion if hasattr(args, 'criterion') else 'MSE'
    criterion = get_criterion(criterion_name)
    
    path_save, path_checkpoints, path_plots, path_logs = paths
    save_config(path_logs, args)
    
    print(f"--- [TRAIN] Démarrage : {args.epochs} epochs | Loss: {criterion_name} ---")

    # 1. Configuration Optimiseur (spécifique P3MG avec lr*5 pour tau)
    #
    tau_params = [p for n, p in model.named_parameters() if 'tau_k' in n]
    other_params = [p for n, p in model.named_parameters() if 'tau_k' not in n]
    
    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.lr},
        {'params': tau_params, 'lr': args.lr * 5.0}
    ], lr=args.lr)

    # 2. Initialisation Statique & PlottingManager
    # Récupération dimensions depuis le loader
    sample_batch = next(iter(train_loader))
    xt_s, y_s, _ = _unpack_batch(sample_batch, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    static, p3mg_tmp = init_static_params(args.rho, args.gamma, N_dim, M_dim, device)

    plot_manager = PlottingManager(
        model=model,
        p3mg_tmp=p3mg_tmp,
        val_loader=val_loader,
        N_dim=N_dim,
        M_dim=M_dim,
        static_params=[args.rho, args.gamma],
        device=device,
        path_plots=path_plots,
        criterion=criterion,
        metric_name=criterion_name
    )

    # 3. Boucle d'époques
    tr_losses, val_losses = [], []
    best_vloss = float('inf')
    start_time = time.time()

    for ep in range(args.epochs):
        t0 = time.time()
        
        # --- TRAIN STEP ---
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            xt, y, x0 = _unpack_batch(batch, device)
            
            # Gestion x0 par défaut (Moyenne)
            if x0 is None:
                mean_v = y.sum(1, keepdim=True)/(M_dim*N_dim)
                x0 = mean_v.repeat(1, N_dim)

            optimizer.zero_grad()
            # Forward: static, dynamic=None, x0, y
            xp, _, _ = model(static, None, x0, y)
            
            loss = criterion(xp, xt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        ep_tr_loss = running_loss / len(train_loader)
        tr_losses.append(ep_tr_loss)

        # --- VAL STEP ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xt, y, x0 = _unpack_batch(batch, device)
                if x0 is None:
                    mean_v = y.sum(1, keepdim=True)/(M_dim*N_dim)
                    x0 = mean_v.repeat(1, N_dim)
                
                xp, _, _ = model(static, None, x0, y)
                val_loss += criterion(xp, xt).item()
        
        ep_val_loss = val_loss / len(val_loader)
        val_losses.append(ep_val_loss)

        # Logs
        print(f"Ep {ep+1}/{args.epochs} | Tr: {ep_tr_loss:.4e} | Val: {ep_val_loss:.4e} | T: {time.time()-t0:.1f}s")

        # --- SAUVEGARDES & PLOTS ---
        if (ep+1) % 5 == 0 or (ep+1) == args.epochs:
            ckpt_path = os.path.join(path_checkpoints, f'checkpoint_epoch{ep+1}.pt')
            torch.save({
                'epoch': ep+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': tr_losses
            }, ckpt_path)
            
            # Sauvegarde Best Model
            if ep_val_loss < best_vloss:
                best_vloss = ep_val_loss
                torch.save(model.state_dict(), os.path.join(path_checkpoints, 'best_model.pt'))

            # Plots via PlottingManager
            try:
                plot_manager.plot_losses(tr_losses, val_losses)
                plot_manager.plot_best_signals(ckpt_path)
                plot_manager.plot_learned_params_evolution(ckpt_path)
            except Exception as e:
                print(f"[WARN] Plot error: {e}")

    print(f"--- Entrainement terminé en {(time.time()-start_time)/60:.2f} min ---")


# =============================================================================
# 2. FONCTION DE TEST (Indépendante)
# =============================================================================
def test(model, test_loader, args, paths, checkpoint_path=None):
    """
    Exécute uniquement la phase de test.
    Charge un checkpoint (best_model.pt par défaut) et génère les stats/plots.
    """
    device = args.device
    criterion_name = args.criterion if hasattr(args, 'criterion') else 'MSE'
    path_checkpoints, path_plots, path_logs = paths[1], paths[2], paths[3]
    
    print(f"--- [TEST] Démarrage sur {len(test_loader.dataset)} échantillons | Metric: {criterion_name} ---")

    # 1. Chargement du checkpoint
    if checkpoint_path is None:
        # Par défaut, on cherche le best_model.pt dans le dossier checkpoints généré
        checkpoint_path = os.path.join(path_checkpoints, 'best_model.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Chargement des poids depuis : {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Gestion compatibilité (state_dict pur ou dict complet)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    else:
        print(f"[WARN] Aucun checkpoint trouvé à {checkpoint_path}. Utilisation du modèle tel quel (non entrainé ?).")

    model.eval()

    # 2. Initialisation Static & PlottingManager
    sample_batch = next(iter(test_loader))
    xt_s, y_s, _ = _unpack_batch(sample_batch, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    static, p3mg_tmp = init_static_params(args.rho, args.gamma, N_dim, M_dim, device)

    plot_manager = PlottingManager(
        model=model, p3mg_tmp=p3mg_tmp, val_loader=None,
        N_dim=N_dim, M_dim=M_dim, static_params=[args.rho, args.gamma],
        device=device, path_plots=path_plots, criterion=get_criterion(criterion_name), metric_name=criterion_name
    )

    # 3. Boucle d'évaluation
    saved_samples = [] # Stocke (valeur_loss, xt, xp)

    # Helper interne pour la métrique de tri
    def compute_sample_metric(xh, xt, name):
        if name == 'MSE': 
            return torch.mean((xh - xt)**2, dim=1)
        elif name in ['SNR', 'TSNR']:
            noise = torch.mean((xt - xh)**2, dim=1)
            sig   = torch.mean(xt**2, dim=1)
            # On minimise l'opposé du SNR pour le tri
            return -10 * torch.log10(sig / (noise + 1e-12)) 
        else: 
            return torch.mean((xh - xt)**2, dim=1)

    with torch.no_grad():
        for batch in test_loader:
            xt, y, x0 = _unpack_batch(batch, device)
            if x0 is None:
                mean_v = y.sum(1, keepdim=True)/(M_dim*N_dim)
                x0 = mean_v.repeat(1, N_dim)

            xp, _, _ = model(static, None, x0, y)
            
            # Calcul loss par échantillon
            metrics = compute_sample_metric(xp, xt, criterion_name)
            
            metrics_cpu = metrics.cpu()
            xt_cpu = xt.cpu()
            xp_cpu = xp.cpu()
            
            for k in range(xt.size(0)):
                val = metrics_cpu[k].item()
                saved_samples.append((val, xt_cpu[k], xp_cpu[k]))

    if not saved_samples:
        print("[WARN] Aucun échantillon de test.")
        return 0.0

    # 4. Analyse Statistique
    saved_samples.sort(key=lambda x: x[0]) # Tri
    all_values = [x[0] for x in saved_samples]
    arr = np.array(all_values)

    mean_v = np.mean(arr)
    med_v  = np.median(arr)
    std_v  = np.std(arr)
    best_v = arr[0]
    worst_v = arr[-1]

    print(f"[RESULTATS] Mean: {mean_v:.4e} | Median: {med_v:.4e} | Best: {best_v:.4e} | Worst: {worst_v:.4e}")

    # 5. Sauvegarde Table & Stats
    table_str = (
        f"\n+-----------------------------------------+\n"
        f"|        RESULTATS TEST ({criterion_name:<5})        |\n"
        f"+-----------------------+-----------------+\n"
        f"| Mean                  | {mean_v:<15.4e} |\n"
        f"| Median                | {med_v:<15.4e} |\n"
        f"| Std                   | {std_v:<15.4e} |\n"
        f"| Min (Best)            | {best_v:<15.4e} |\n"
        f"| Max (Worst)           | {worst_v:<15.4e} |\n"
        f"+-----------------------+-----------------+\n"
    )
    with open(os.path.join(path_logs, 'test_results_table.txt'), 'w') as f:
        f.write(table_str)

    # 6. Plots (Best, Median, Worst, Mean)
    best_sample = saved_samples[0]
    worst_sample = saved_samples[-1]
    med_sample = saved_samples[len(saved_samples)//2]
    idx_mean = (np.abs(arr - mean_v)).argmin()
    mean_sample = saved_samples[idx_mean]

    def safe_plot(sample, tag):
        try:
            plot_manager.plot_signals(
                sample[1].unsqueeze(0), 
                sample[2].unsqueeze(0), 
                f'test_{tag}', 
                criterion_name, 
                sample[0]
            )
        except Exception as e:
            print(f"Plot fail {tag}: {e}")

    safe_plot(best_sample, "BEST")
    safe_plot(worst_sample, "WORST")
    safe_plot(med_sample, "MEDIAN")
    safe_plot(mean_sample, "MEAN")

    if hasattr(plot_manager, 'plot_test_error_distribution'):
        plot_manager.plot_test_error_distribution(arr, metric_name=criterion_name)

    return mean_v