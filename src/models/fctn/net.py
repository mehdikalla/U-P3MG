import math

import torch
import torch.nn as nn

from src.models.Transformer_block import Transformer_Block


class FCTN_model(nn.Module):
    """
    Fully Connected Transformer Network (FCTN).

    Architecture par patchs combinee a un contexte global non lineaire :
    contrainte de positivite via mise au carre + LayerNorm (evite le
    blocage du gradient d'une Softplus saturante), et vecteur de contexte
    global (MLP sur `y`) diffuse a chaque jeton pour apprendre le melange
    global que l'auto-attention seule ne peut pas capturer.
    """

    def __init__(
        self,
        N_dim: int,
        M_dim: int,
        patch_size: int = 20,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        context_hidden: int = 256,
    ) -> None:
        super().__init__()

        self.N_dim = N_dim
        self.M_dim = M_dim
        self.patch_size = patch_size

        # Nombre de patchs pour couvrir N_dim, avec padding a droite si besoin.
        self.num_patches = math.ceil(N_dim / patch_size)
        self.padded_dim = self.num_patches * patch_size
        self.pad_amount = self.padded_dim - N_dim

        # 1. Contexte global (role similaire a l'encodeur d'un FCAE) : apprend
        # le melange global necessaire pour inverser `Hmat`.
        self.global_context = nn.Sequential(
            nn.Linear(M_dim, context_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(context_hidden, d_model),
        )
        # Initialisation Kaiming (couche cachee, suivie de ReLU) et Xavier
        # (projection finale sans activation), pour accelerer la convergence.
        nn.init.kaiming_normal_(self.global_context[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.global_context[0].bias)
        nn.init.xavier_normal_(self.global_context[-1].weight)
        nn.init.zeros_(self.global_context[-1].bias)

        # 1bis. Decodeur dense direct (chemin residuel, type FCAE), raffine
        # ensuite par le Transformer (a priori local + contexte + attention).
        self.dense_decoder = nn.Sequential(
            nn.Linear(M_dim, context_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(context_hidden, context_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(context_hidden, N_dim),
        )
        # Kaiming pour les couches cachees (ReLU), Xavier pour la sortie.
        nn.init.kaiming_normal_(self.dense_decoder[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.dense_decoder[0].bias)
        nn.init.kaiming_normal_(self.dense_decoder[3].weight, nonlinearity='relu')
        nn.init.zeros_(self.dense_decoder[3].bias)
        nn.init.xavier_normal_(self.dense_decoder[-1].weight)
        nn.init.zeros_(self.dense_decoder[-1].bias)

        # 2. Embedding des patchs de x0 (a priori local par position).
        self.patch_embedding = nn.Linear(patch_size, d_model)
        nn.init.xavier_normal_(self.patch_embedding.weight)
        nn.init.zeros_(self.patch_embedding.bias)

        # 3. Encodage positionnel appris (un vecteur par position de patch).
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        # 4. Reseau Transformer (encodeur pur, auto-attention entre patchs).
        self.transformer = Transformer_Block(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # 5. Normalisation avant la sortie : stabilise l'echelle des
        # pre-activations et evite le gradient quasi nul d'une Softplus saturante.
        self.pre_out_norm = nn.LayerNorm(d_model)

        # 6. Sortie : projette chaque jeton vers un patch residuel, ajoute a l'estimation dense.
        self.fc_out = nn.Linear(d_model, patch_size)
        # Initialisation proche de zero : le modele demarre comme le
        # decodeur dense puis apprend progressivement le raffinement residuel.
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc_out.bias)

        # Contrainte de positivite via mise au carre plutot qu'une Softplus/ELU/ReLU saturante.


    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Decoupe un tenseur (batch_size, N_dim) en patchs contigus.

        Applique un padding a droite si necessaire, puis reorganise en
        (batch_size, num_patches, patch_size).
        """
        if self.pad_amount > 0:
            x = nn.functional.pad(x, (0, self.pad_amount))
        return x.view(x.shape[0], self.num_patches, self.patch_size)

    def forward(self, static, dynamic, x0: torch.Tensor, y: torch.Tensor):
        """
        Passe avant du modele FCTN.

        Args:
            static: Variables statiques (non utilisees ici, presentes pour
                compatibilite).
            dynamic: Variables dynamiques (non utilisees ici, presentes
                pour compatibilite).
            x0 (torch.Tensor): Signal initial. Shape: (batch_size, N_dim)
            y (torch.Tensor): Observation. Shape: (batch_size, M_dim)

        Returns:
            Tuple contenant le signal reconstruit et deux variables nulles
            pour compatibilite.
        """
        # Estimation dense directe (chemin fiable, bien conditionne, cf. FCAE).
        dense_estimate = self.dense_decoder(y)

        # Contexte global diffuse a tous les patchs : (batch, d_model) -> (batch, 1, d_model)
        global_ctx = self.global_context(y).unsqueeze(1)

        x0_patches = self._to_patches(x0)

        emb = self.patch_embedding(x0_patches) + global_ctx + self.positional_embedding

        encoded = self.transformer(emb)
        encoded = self.pre_out_norm(encoded)

        raw_patches = self.fc_out(encoded)
        residual = raw_patches.reshape(x0.shape[0], self.padded_dim)
        if self.pad_amount > 0:
            residual = residual[:, : self.N_dim]

        # Raffinement additif de l'estimation dense via l'auto-attention ;
        # contrainte de positivite appliquee une seule fois, sur la somme.
        x_pred = (dense_estimate + residual) ** 2

        return x_pred, None, None
