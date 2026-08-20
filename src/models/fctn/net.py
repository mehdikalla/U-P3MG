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
        # Activation Softplus finale afin de garantir un signal reconstruit
        # strictement positif.
        self.output_activation = nn.Softplus()

        # Initialisation dediee de la couche de sortie.
        # Avec l'initialisation par defaut de nn.Linear, la sortie de fc_out
        # est proche de 0, ce qui donne Softplus(0) = ln(2) ~= 0.69. Cette
        # valeur est plusieurs ordres de grandeur au-dessus de l'echelle
        # typique des signaux cibles de ce projet (souvent << 1, avec une
        # forte proportion de zeros). L'erreur initiale enorme qui en
        # resulte provoque une mise a jour Adam tres agressive des la
        # premiere iteration, qui pousse les pre-activations vers de fortes
        # valeurs negatives ou Softplus sature (gradient quasi nul). Le
        # reseau reste alors bloque a predire une sortie quasi nulle pour
        # tous les echantillons, et la loss stagne au niveau de
        # mean(x_true**2) (predire 0 partout), ce qui se manifeste comme
        # une loss constante des la deuxieme epoque.
        # On initialise donc les poids/biais de la couche de sortie a des
        # valeurs proches de zero afin que la sortie initiale du reseau
        # (avant Softplus) soit elle-meme proche de zero, evitant ainsi le
        # saut d'erreur initial et la saturation de l'activation.
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.fc_out.bias, -5.0)


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
        
        x_pred = self.output_activation(self.fc_out(encoded)).squeeze(-1)
        
        return x_pred, None, None
