import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.utils.functions import snr_loss, tsnr_loss 
from src.utils.plotting_manager import PlottingManager

def get_criterion(name):
    if name == 'MSE':
        return nn.MSELoss(reduction='mean')
    elif name == 'SNR':
        return snr_loss()
    elif name == 'TSNR':
        return tsnr_loss()
    else:
        raise ValueError(f"Loss '{name}' non reconnue. Choisir: MSE, SNR, TSNR.")

def save_config(path, args):
    try:
        config = {k: str(v) for k, v in vars(args).items()}
        with open(os.path.join(path, 'run_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder la config : {e}")

def _unpack_batch(batch, device):
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

def init_static_params(args, N_dim, M_dim, device):
    """Initialise les paramètres statiques de manière dynamique selon le modèle."""
    model_name = args.model.strip().lower()
    dx = torch.zeros(1, N_dim).double().to(device)
    dy = torch.zeros(1, M_dim).double().to(device)
    
    if model_name == 'p3mg':
        from src.models.p3mg.algo import P3MG_algo
        algo_tmp = P3MG_algo(num_pd_layers=getattr(args, 'num_pd_layers', 5)).to(device).double()
        params = [args.alpha, args.beta, args.eta]
        static = algo_tmp.init_P3MG(params, dx, dy)
        
    elif model_name == 'hq':
        from src.models.hq.algo import HQ_algo
        algo_tmp = HQ_algo().to(device).double()
        # Initialisation une seule fois
        static = algo_tmp.init_HQ(dx, dy) 

    elif model_name == 'pmms':
        from src.models.pmms.algo import PMMS_algo
        algo_tmp = PMMS_algo().to(device).double()
        sigma = getattr(args, 'sigma', 1e-5)
        beta = getattr(args, 'beta', 1e-5)
        eta = getattr(args, 'eta', 1e-2)
        static, _ = algo_tmp.init_PMMS(dx, dy, sigma=sigma, beta=beta, eta=eta)

    else:

        algo_tmp = None
        static = None
        
    return static, algo_tmp

# =============================================================================
# 1. FONCTION D'ENTRAINEMENT (Sécurisée et Dynamique)
# =============================================================================
def train(model, train_loader, val_loader, args, paths):
    device = args.device
    criterion_name = args.criterion if hasattr(args, 'criterion') else 'MSE'
    criterion = get_criterion(criterion_name)
    model_name = args.model.strip().lower()
    
    path_save, path_checkpoints, path_plots, path_logs = paths
    save_config(path_logs, args)
    
    print(f"--- [TRAIN] {model_name.upper()} (UNROLLING) | Epochs: {args.epochs} | Loss: {criterion_name} ---")

    # Sécurisation de l'optimiseur (uniquement les variables avec requires_grad)
    tau_params = [p for n, p in model.named_parameters() if 'tau_k' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'tau_k' not in n and p.requires_grad]
    
    param_groups = []
    if len(other_params) > 0:
        param_groups.append({'params': other_params, 'lr': args.lr})
    if len(tau_params) > 0:
        param_groups.append({'params': tau_params, 'lr': args.lr * 5.0})
        
    has_parameters = len(param_groups) > 0
    
    if has_parameters:
        optimizer = optim.Adam(param_groups, lr=args.lr)
    else:
        optimizer = None
        print("[INFO] Aucun paramètre apprenable détecté. Évaluation sans rétropropagation.")

    sample_batch = next(iter(train_loader))
    xt_s, y_s, _ = _unpack_batch(sample_batch, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    static_params, algo_tmp = init_static_params(args, N_dim, M_dim, device)

    plot_manager = PlottingManager(
        model=model,
        p3mg_tmp=algo_tmp,
        val_loader=val_loader,
        N_dim=N_dim,
        M_dim=M_dim,
        static_params=[args.alpha, args.beta, args.eta],
        device=device,
        path_plots=path_plots,
        criterion=criterion,
        metric_name=criterion_name
    )

    tr_losses, val_losses = [], []
    best_vloss = float('inf')
    start_time = time.time()

    for ep in range(args.epochs):
        t0 = time.time()
        
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            xt, y, x0 = _unpack_batch(batch, device)
            
            if x0 is None:
                mean_v = y.sum(1, keepdim=True)/(M_dim*N_dim)
                x0 = mean_v.repeat(1, N_dim)

            if has_parameters:
                optimizer.zero_grad()
            
            # CORRECTION MAJEURE: Ne passer les statiques que pour P3MG.
            # ISTA/PMMS doivent recevoir `None` sinon ils by-passent leurs poids appris !
            current_static = static_params
            xp, _, _ = model(current_static, None, x0, y)
            
            loss = criterion(xp, xt)
            
            if has_parameters and loss.requires_grad:
                if torch.isnan(loss).any():
                    print("ALERT: NaN détecté avant backward ! Vérifiez vos lambdas.")
                    return # Arrêtez l'entraînement
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
        
        ep_tr_loss = running_loss / len(train_loader)
        tr_losses.append(ep_tr_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                xt, y, x0 = _unpack_batch(batch, device)
                if x0 is None:
                    mean_v = y.sum(1, keepdim=True)/(M_dim*N_dim)
                    x0 = mean_v.repeat(1, N_dim)
                
                current_static = static_params 
                xp, _, _ = model(current_static, None, x0, y)
                val_loss += criterion(xp, xt).item()
        
        ep_val_loss = val_loss / len(val_loader)
        val_losses.append(ep_val_loss)

        print(f"Ep {ep+1}/{args.epochs} | Tr: {ep_tr_loss:.4e} | Val: {ep_val_loss:.4e} | T: {time.time()-t0:.1f}s")

        if (ep+1) % 5 == 0 or (ep+1) == args.epochs:
            ckpt_path = os.path.join(path_checkpoints, f'checkpoint_epoch{ep+1}.pt')
            
            save_dict = {
                'epoch': ep+1,
                'model_state_dict': model.state_dict(),
                'train_losses': tr_losses
            }
            if has_parameters:
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
                
            torch.save(save_dict, ckpt_path)
            
            if ep_val_loss < best_vloss:
                best_vloss = ep_val_loss
                torch.save(model.state_dict(), os.path.join(path_checkpoints, 'best_model.pt'))

            try:
                plot_manager.plot_losses(tr_losses, val_losses)
                plot_manager.plot_best_signals(ckpt_path)
                plot_manager.plot_learned_params_evolution(ckpt_path)
            except Exception:
                pass 

    print(f"--- [TRAIN] Terminé en {(time.time()-start_time)/60:.2f} min ---")

# =============================================================================
# 2. FONCTION DE TEST (Dynamique)
# =============================================================================
def test(model, test_loader, args, paths, checkpoint_path=None):
    device = args.device
    criterion_name = args.criterion if hasattr(args, 'criterion') else 'MSE'
    path_checkpoints, path_plots, path_logs = paths[1], paths[2], paths[3]
    model_name = args.model.strip().lower()
    
    print(f"--- [TEST] {model_name.upper()} (UNROLLING) | Metric: {criterion_name} | Samples: {len(test_loader.dataset)} ---")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(path_checkpoints, 'best_model.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Chargement des poids depuis : {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    else:
        print(f"[WARN] Aucun checkpoint trouvé. Utilisation du modèle tel quel.")

    model.eval()

    sample_batch = next(iter(test_loader))
    xt_s, y_s, _ = _unpack_batch(sample_batch, device)
    N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
    
    static_params, algo_tmp = init_static_params(args, N_dim, M_dim, device)

    plot_manager = PlottingManager(
        model=model, p3mg_tmp=algo_tmp, val_loader=None,
        N_dim=N_dim, M_dim=M_dim, static_params=[args.alpha, args.beta, args.eta],
        device=device, path_plots=path_plots, criterion=get_criterion(criterion_name), metric_name=criterion_name
    )

    saved_samples = [] 

    def compute_sample_metric(xh, xt, name):
        if name == 'MSE': 
            return torch.mean((xh - xt)**2, dim=1)
        elif name in ['SNR', 'TSNR']:
            noise = torch.mean((xt - xh)**2, dim=1)
            sig   = torch.mean(xt**2, dim=1)
            return -10 * torch.log10(sig / (noise + 1e-12)) 
        else: 
            return torch.mean((xh - xt)**2, dim=1)

    with torch.no_grad():
        for batch in test_loader:
            xt, y, x0 = _unpack_batch(batch, device)
            if x0 is None:
                mean_v = y.sum(1, keepdim=True)/(M_dim*N_dim)
                x0 = mean_v.repeat(1, N_dim)

            current_static = static_params if model_name == 'p3mg' else None
            xp, _, _ = model(current_static, None, x0, y)
            
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

        saved_samples.sort(key=lambda x: x[0]) 
        all_values = [x[0] for x in saved_samples]
        arr = np.array(all_values)

        mean_v = np.mean(arr)
        med_v  = np.median(arr)
        std_v  = np.std(arr)
        best_v = arr[0]
        worst_v = arr[-1]

        print(f"[RESULT] Mean {criterion_name}: {mean_v:.4e} | Median: {med_v:.4e} | Std: {std_v:.4e} | Best: {best_v:.4e} | Worst: {worst_v:.4e}")

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

        def safe_plot(sample, tag):
            try:
                plot_manager.plot_signals(
                    sample[1].unsqueeze(0), 
                    sample[2].unsqueeze(0), 
                    f'test_{tag}', 
                    criterion_name, 
                    sample[0]
                )
            except Exception:
                pass

        safe_plot(saved_samples[0], "BEST")
        safe_plot(saved_samples[-1], "WORST")
        safe_plot(saved_samples[len(saved_samples)//2], "MEDIAN")
        safe_plot(saved_samples[(np.abs(arr - mean_v)).argmin()], "MEAN")

        if hasattr(plot_manager, 'plot_test_error_distribution'):
            plot_manager.plot_test_error_distribution(arr, metric_name=criterion_name)

        return mean_v