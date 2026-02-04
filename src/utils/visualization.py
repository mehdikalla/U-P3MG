# Fichier: src/utils/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import shutil

def plot_gs_loss_params(path_save: str, path_logs: str, path_plots: str):
    """(Fonction existante pour le Grid Search classique)"""
    print(f"[INFO] Consolidation des résultats GS depuis : {path_logs}")
    data = []
    jsons = glob.glob(os.path.join(path_logs, 'gs_result_*.json'))
    if not jsons: return

    for j in jsons:
        try:
            with open(j, 'r') as f: data.append(json.load(f))
        except: pass
    if not data: return

    raw_l = np.array([d['lambda'] for d in data])
    raw_t = np.array([d['tau'] for d in data])
    raw_loss = np.array([d['loss'] for d in data])
    best_idx = np.argmin(raw_loss)
    best_l, best_t, best_loss = raw_l[best_idx], raw_t[best_idx], raw_loss[best_idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(raw_l, raw_loss, c=raw_t, cmap='viridis', label='Points', alpha=0.7)
    plt.colorbar(label=r'$\tau$')
    plt.plot(best_l, best_loss, '*', markersize=15, label='Optimum')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Loss')
    plt.title('Projection : Loss vs Lambda')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(path_save, 'GLOBAL_GS_SCATTER_LAMBDA.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(raw_t, raw_loss, c=np.log10(raw_l), cmap='plasma', label='Points', alpha=0.7)
    plt.colorbar(label=r'$\log_{10}(\lambda)$')
    plt.plot(best_t, best_loss, '*', markersize=15, label='Optimum')
    plt.xlabel(r'$\tau$')
    plt.ylabel('Loss')
    plt.title('Projection : Loss vs Tau')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(path_save, 'GLOBAL_GS_SCATTER_TAU.png'))
    plt.close()

def plot_signals_gs(true, pred, l_val, t_val, path):
    """(Fonction existante)"""
    pat_l = f"{l_val:.2e}".replace('+','')
    pat_t = f"{t_val:.2f}"
    fname = f"signal_L{pat_l}_T{pat_t}.png"
    plt.figure(figsize=(10,3)); plt.plot(true[0].cpu().numpy(), label='True')
    plt.plot(pred[0].detach().cpu().numpy(), '--', label='Pred')
    plt.grid()
    plt.title(f'L={l_val:.2e} T={t_val:.2f}')
    plt.legend()
    plt.savefig(os.path.join(path, fname))
    plt.close()

# --- NOUVELLES FONCTIONS ORACLE ---

def plot_oracle_analysis(best_params, best_losses, oracle_dir, loss_name="Loss"):
    """
    Génère les graphiques pour l'analyse Oracle avec le nom de la Loss.
    """
    os.makedirs(oracle_dir, exist_ok=True)
    
    lambdas = best_params[:, 0]
    taus = best_params[:, 1]
    
    # 1. Lambda Hist
    plt.figure(figsize=(8, 5))
    plt.hist(np.log10(lambdas), bins=20, color='teal', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Optimal Lambdas ({loss_name})")
    plt.xlabel("Log10(Lambda)")
    plt.ylabel("Number of signals")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(oracle_dir, "distrib_best_lambdas.png"))
    plt.close()

    # 2. Tau Hist
    plt.figure(figsize=(8, 5))
    plt.hist(taus, bins=20, color='orange', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Optimal Taus ({loss_name})")
    plt.xlabel("Tau")
    plt.ylabel("Number of signals")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(oracle_dir, "distrib_best_taus.png"))
    plt.close()
    
    # 3. Scatter
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(taus, np.log10(lambdas), c=best_losses, cmap='viridis', alpha=0.6, s=15)
    plt.colorbar(sc, label=f'Best {loss_name}')
    plt.title(f"Map of Best Parameters (Lambda vs Tau)")
    plt.xlabel("Tau")
    plt.ylabel("Log10(Lambda)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(oracle_dir, "scatter_best_params.png"))
    plt.close()

    # 4. LOSS Histogram
    mean_val = np.mean(best_losses)
    median_val = np.median(best_losses)
    
    plt.figure(figsize=(10, 6))
    plt.hist(best_losses, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2e}')
    plt.axvline(median_val, color='green', linestyle='dotted', linewidth=2, label=f'Median: {median_val:.2e}')
    
    plt.title(f"Oracle {loss_name} Distribution (Test Set)")
    plt.xlabel(f"{loss_name} Value")
    plt.ylabel("Number of signals")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(oracle_dir, "distrib_best_losses.png"))
    plt.close()

def plot_oracle_reconstruction(x_true, y_input, x_oracle, best_l, best_t, oracle_dir, index, loss_val=None, loss_name="Loss"):
    """
    Trace une comparaison : Signal Bruité vs Vrai vs Oracle avec Titre dynamique.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(x_true, 'k', label='Ground Truth', linewidth=1.5, alpha=0.8)
    plt.plot(x_oracle, 'r--', label=f'Oracle (L={best_l:.1e}, T={best_t:.2f})', linewidth=1.5)
    
    title_str = f"Oracle Reconstruction - {index}"
    if loss_val is not None:
        title_str += f"\n{loss_name}: {loss_val:.4e}"
        
    plt.title(title_str)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(oracle_dir, f"oracle_signal_{index}.png"))
    plt.close()