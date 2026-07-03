import torch
import torch.nn as nn

from src.models.FC_block import FC_block

class FCAE_model(nn.Module):
    """
    Autoencodeur Fully Connected pour la reconstruction de signaux.
    """
    def __init__(self, M_dim, N_dim):
        super().__init__()
        
        self.layers = FC_block([M_dim, 50, 25, 12, 12, 25, 50, N_dim])
    
    def forward(self, static, dynamic, x0, y):
        """
        Passe avant du modèle FCAE.
        """
        x_pred = self.layers(y)
        
        return x_pred, None, None