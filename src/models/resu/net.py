import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from src.models.ResU_block import BasicConv, ResUNetBlock, ResUNetUpBlock
from src.models.Transformer_block import Transformer_Block


class ResU_model(nn.Module):
    """
    Residual U-Net Fully Convolutional (ResU) pour la reconstruction de
    signaux 1D.

    Architecture encodeur-decodeur convolutive avec connexions residuelles
    internes (ResUNetConv) et connexions de saut (skip connections) entre
    l'encodeur et le decodeur, conformement a l'architecture ResU-Net
    classique adaptee au signal 1D (Conv1d).

    Args:
        N_dim: Dimension du signal reconstruit (sortie).
        M_dim: Dimension de l'observation (entree).
        num_layers: Profondeur du reseau, c'est-a-dire le nombre de niveaux
            de l'encodeur (et symetriquement du decodeur).
        base_channels: Nombre de canaux de la premiere couche de
            l'encodeur. Double a chaque niveau suivant.
        num_convs: Nombre de convolutions residuelles par bloc.
        use_batchnorm: Active la normalisation par batch dans les blocs
            convolutifs.
        use_transformer: Active un module d'attention (Transformer encoder)
            applique sur la representation du goulot d'etranglement, afin
            de capturer des dependances a longue portee complementaires
            aux convolutions locales.
        transformer_nhead: Nombre de tetes d'attention du Transformer.
        transformer_num_layers: Nombre de couches d'encodeur Transformer.
        transformer_dim_feedforward: Dimension du reseau feedforward interne
            du Transformer.
        transformer_dropout: Taux de dropout applique dans le Transformer.
        dropout: Taux de dropout applique dans les blocs convolutifs
            (encodeur, bottleneck et decodeur). Complementaire au dropout
            propre au Transformer (transformer_dropout), il reduit le
            surapprentissage du chemin purement convolutif.
    """

    def __init__(
        self,
        N_dim: int = 800,
        M_dim: int = 100,
        num_layers: int = 3,
        base_channels: int = 16,
        num_convs: int = 2,
        use_batchnorm: bool = True,
        use_transformer: bool = True,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        transformer_dropout: float = 0.1,
        dropout: float = 0.05,
    ) -> None:

        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers doit etre superieur ou egal a 1.")

        self.N_dim = N_dim
        self.M_dim = M_dim
        self.num_layers = num_layers

        # Projection de l'observation vers (batch, canal=1, longueur=M_dim) pour les convolutions 1D.
        self.input_proj = nn.Identity()

        channel_widths = [base_channels * (2 ** i) for i in range(num_layers)]

        # --- Encodeur ---
        self.encoder_blocks = nn.ModuleList()
        in_channels = 1
        for width in channel_widths:
            self.encoder_blocks.append(
                ResUNetBlock(in_channels, width, num_convs, use_batchnorm, dropout)
            )
            in_channels = width

        # --- Goulot d'etranglement (bottleneck) ---
        bottleneck_channels = channel_widths[-1] * 2
        self.bottleneck = BasicConv(channel_widths[-1], bottleneck_channels, use_batchnorm, dropout)

        # --- Module d'attention (Transformer) sur le goulot d'etranglement ---
        # Le nombre de tetes doit diviser le nombre de canaux du bottleneck.
        self.use_transformer = use_transformer
        if use_transformer:
            adjusted_nhead = transformer_nhead
            while adjusted_nhead > 1 and bottleneck_channels % adjusted_nhead != 0:
                adjusted_nhead -= 1
            self.transformer = Transformer_Block(
                d_model=bottleneck_channels,
                nhead=adjusted_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                batch_first=True,
                norm_first=True,
            )
        else:
            self.transformer = None

        # --- Decodeur ---

        self.decoder_blocks = nn.ModuleList()
        in_channels = bottleneck_channels
        for width in reversed(channel_widths):
            self.decoder_blocks.append(
                ResUNetUpBlock(in_channels, width, width, num_convs, use_batchnorm, dropout)
            )
            in_channels = width

        # --- Projection finale vers la dimension du signal reconstruit ---
        self.output_proj = nn.Conv1d(channel_widths[0], 1, kernel_size=1)
        nn.init.kaiming_normal_(self.output_proj.weight, nonlinearity='linear')
        nn.init.zeros_(self.output_proj.bias)
        self.fc_out = nn.Linear(M_dim, N_dim)
        nn.init.xavier_normal_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        self.output_activation = nn.Softplus()

    def forward(self, static, dynamic, x0: tc.Tensor, y: tc.Tensor):
        """
        Passe avant du modele ResU.

        Args:
            static: Variables statiques (non utilisees, presentes pour
                compatibilite avec l'interface commune des modeles du
                projet).
            dynamic: Variables dynamiques (non utilisees, idem).
            x0: Signal initial. Non utilise directement, conserve pour
                compatibilite.
            y: Observation bruitee. Shape: (batch_size, M_dim).

        Returns:
            Tuple (x_pred, None, None) ou x_pred est le signal reconstruit
            de shape (batch_size, N_dim).
        """
        h = y.unsqueeze(1)  # (batch, 1, M_dim)

        skips = []
        for block in self.encoder_blocks:
            h, skip = block(h)
            skips.append(skip)

        h = self.bottleneck(h)

        if self.use_transformer:
            # (batch, channels, length) -> (batch, length, channels)
            h_seq = h.permute(0, 2, 1)
            h_seq = self.transformer(h_seq)
            h = h_seq.permute(0, 2, 1)

        for block, skip in zip(self.decoder_blocks, reversed(skips)):

            h = block(h, skip)

        h = self.output_proj(h).squeeze(1)  # (batch, M_dim)
        x_pred = self.fc_out(h)
        x_pred = self.output_activation(x_pred)

        return x_pred, None, None
