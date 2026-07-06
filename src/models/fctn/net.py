import torch
import torch.nn as nn

from src.models.Transformer_block import Transformer_Block

class FCTN_model(nn.Module):
    """
    Fully Connected Transformer Network (FCTN).
    Combine des couches de projection denses (Fully Connected) avec un encodeur 
    Transformer pour la reconstruction de signaux.
    """
    def __init__(
        self, 
        N_dim: int, 
        M_dim: int, 
        d_model: int = 64, 
        nhead: int = 8, 
        num_layers: int = 4, 
        dim_feedforward: int = 256, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.N_dim = N_dim
        self.M_dim = M_dim
        
        # 1. Couche Fully Connected pour aligner la dimension de l'observation (y) sur celle du signal (x0)
        self.fc_y = nn.Linear(M_dim, N_dim)
        
        # 2. Embedding Fully Connected
        # Transforme chaque paire (x0_i, y_proj_i) en un vecteur de dimension d_model
        self.embedding = nn.Linear(2, d_model)
        
        # 3. Réseau Transformer
        self.transformer = Transformer_Block(
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        # 4. Couche Fully Connected de sortie
        # Projette la dimension latente d_model vers la dimension scalaire finale du signal
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, static, dynamic, x0: torch.Tensor, y: torch.Tensor):
        """
        Passe avant du modèle FCTN.
        
        Args:
            static: Variables statiques (non utilisées ici, présentes pour compatibilité).
            dynamic: Variables dynamiques (non utilisées ici, présentes pour compatibilité).
            x0 (torch.Tensor): Signal initial. Shape: (batch_size, N_dim)
            y (torch.Tensor): Observation. Shape: (batch_size, M_dim)
            
        Returns:
            Tuple contenant le signal reconstruit et deux variables nulles pour compatibilité.
        """
        y_proj = self.fc_y(y)
        seq_input = torch.stack([x0, y_proj], dim=-1)

        emb = self.embedding(seq_input)

        encoded = self.transformer(emb)
        
        x_pred = self.fc_out(encoded).squeeze(-1)
        
        return x_pred, None, None