# Fichier: src/utils/plotting_manager.py
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class PlottingManager:
    def __init__(self, model, p3mg_tmp, val_loader, N_dim, M_dim, static_params, device, path_plots, criterion=None, metric_name="MSE"):
        self.model = model
        self.p3mg_tmp = p3mg_tmp
        self.val_loader = val_loader
        self.N_dim = N_dim
        self.M_dim = M_dim
        self.static_params = static_params
        self.device = device
        self.path_plots = path_plots
        
        # On utilise le critère passé en argument (ex: snr_loss), sinon MSE par défaut
        self.criterion = criterion if criterion is not None else nn.MSELoss(reduction="mean")
        self.metric_name = metric_name

    def _unpack(self, batch):
        if len(batch) == 3: return batch[0], batch[1], batch[2]
        return batch[0], batch[1], None

    def plot_losses(self, tr, val):
        plt.figure()
        plt.plot(tr, label='Train Loss')
        plt.plot(val, label='Val Loss')
        plt.legend()
        plt.grid()
        plt.title(f'Training and Validation {self.metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(self.metric_name)
        plt.savefig(os.path.join(self.path_plots, 'loss.png'))
        plt.close()
    
    def plot_signals(self, true, pred, identifier, metric_name=None, metric_value=None):
        """
        Trace les signaux. 
        Si metric_name n'est pas fourni, utilise self.metric_name.
        """
        if metric_name is None:
            metric_name = self.metric_name

        for i in range(min(3, true.size(0))):
            plt.figure(figsize=(10,3))
            plt.plot(true[i].cpu().numpy(), label='True Signal')
            plt.plot(pred[i].detach().cpu().numpy(), '--', label='Predicted Signal')
            plt.legend()
            plt.grid()
            
            # Titre dynamique
            title_str = f"Sample {i}"
            if metric_value is not None:
                title_str += f" | {metric_name}: {metric_value:.4e}"
            elif identifier is not None:
                title_str += f" | {identifier}"
            
            plt.title(title_str)
            
            # Nom de fichier safe
            safe_id = str(identifier).replace(" ", "_").replace(":", "").replace(".", "p")
            plt.savefig(os.path.join(self.path_plots, f'sig_{safe_id}_s{i}.png'))
            plt.close()
    
    def plot_best_signals(self, path):
        if not os.path.isfile(path): return
        try:
            ckpt = torch.load(path, map_location=self.device)
            # Gestion state_dict
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(sd, strict=False)
        except:
            return

        self.model.eval()
        if self.val_loader is None: return

        best_sample_loss = float('inf')
        best_xt, best_xp = None, None
        
        dx = torch.zeros(1, self.N_dim).double().to(self.device)
        dy = torch.zeros(1, self.M_dim).double().to(self.device)
        st = self.p3mg_tmp.init_P3MG(list(self.static_params), dx, dy)

        with torch.no_grad():
            for batch in self.val_loader:
                xt, y, x0 = self._unpack(batch)
                xt, y = xt.to(self.device).double(), y.to(self.device).double()
                if x0 is None: x0 = y.sum(1,keepdim=True).repeat(1,self.N_dim)/(self.M_dim*self.N_dim)
                xp, _, _ = self.model(st, None, x0, y)

                # Utilisation du critère configuré (MSE, SNR...)
                current_loss = self.criterion(xp, xt).item()
                
                # On cherche à minimiser la Loss (NB: SNR loss renvoie -SNR, donc minimiser est correct)
                if current_loss < best_sample_loss:
                    best_sample_loss = current_loss
                    best_xt = xt[0].cpu().numpy()
                    best_xp = xp[0].detach().cpu().numpy()
        
        if best_xt is not None:
            plt.figure(figsize=(10,3))
            plt.plot(best_xt, label='True')
            plt.plot(best_xp, '--', label='Pred')
            # Titre adapté avec le nom de la loss
            plt.title(f'Best Model (Lowest Sample {self.metric_name}: {best_sample_loss:.4e})')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(self.path_plots, 'best_sig.png'))
            plt.close()
   
    def plot_learned_params_evolution(self, path):
        if not os.path.isfile(path): return
        try:
            ckpt = torch.load(path, map_location=self.device)
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(sd, strict=False)
        except:
            return
            
        S = nn.Softplus()
        N, M = self.model.num_layers, self.model.num_pd_layers
        lmbd_vals, tau_rows = [], []
        dummy_y = torch.ones(1, self.M_dim).double().to(self.device)
        dummy_x = torch.zeros(1, self.N_dim).double().to(self.device)
        dummy_static = self.p3mg_tmp.init_P3MG(list(self.static_params), dummy_x, dummy_y) 
        
        with torch.no_grad(): 
            _, _, dl = self.model(dummy_static, None, dummy_x, dummy_y)
            
        for k, layer in enumerate(self.model.Layers):
            if k < len(dl): lmbd_vals.append(dl[k].squeeze().item())
            if hasattr(layer, 'tau_k'): tau_rows.append(S(layer.tau_k).detach().cpu().numpy().flatten())
            
        if lmbd_vals:
            plt.figure(figsize=(10,6))
            plt.plot(range(1, N+1), lmbd_vals, 'o-')
            plt.title('Learned Lambda Evolution')
            plt.grid(True)
            plt.savefig(os.path.join(self.path_plots, 'learnt_lambda_curve.png'))
            plt.close()
        if tau_rows:
            tau_mat = np.array(tau_rows)
            plt.figure(figsize=(12, 6))
            plt.imshow(tau_mat, aspect='auto', cmap='viridis', origin='lower', extent=[0.5, M+0.5, 0.5, N+0.5])
            plt.colorbar()
            plt.title('Learned Tau Heatmap')
            plt.savefig(os.path.join(self.path_plots, 'learnt_tau_heatmap.png'))
            plt.close()

    def plot_test_error_distribution(self, losses, metric_name=None):
        """Trace et sauvegarde l'histogramme des erreurs avec le nom de la Loss."""
        if metric_name is None:
            metric_name = self.metric_name

        losses = np.array(losses)
        mean_val = np.mean(losses)
        median_val = np.median(losses)
        
        plt.figure(figsize=(10, 6))
        plt.hist(losses, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2e}')
        plt.axvline(median_val, color='green', linestyle='dotted', linewidth=2, label=f'Median: {median_val:.2e}')
        
        plt.title(f'Test Set {metric_name} Distribution')
        plt.xlabel(f'{metric_name} Value')
        plt.ylabel('Number of Samples')
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_plots, 'test_error_distribution.png'))
        plt.close()