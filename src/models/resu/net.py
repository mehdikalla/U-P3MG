import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from src.models.ResU_block import BasicConv, ResUNetBlock, ResUNetUpBlock


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
    """

    def __init__(
        self,
        N_dim: int = 800,
        M_dim: int = 100,
        num_layers: int = 3,
        base_channels: int = 16,
        num_convs: int = 2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers doit etre superieur ou egal a 1.")

        self.N_dim = N_dim
        self.M_dim = M_dim
        self.num_layers = num_layers

        # Projection initiale de l'observation (scalaire par position) vers
        # un tenseur (batch, canal=1, longueur=M_dim) exploitable par les
        # convolutions 1D.
        self.input_proj = nn.Identity()

        channel_widths = [base_channels * (2 ** i) for i in range(num_layers)]

        # --- Encodeur ---
        self.encoder_blocks = nn.ModuleList()
        in_channels = 1
        for width in channel_widths:
            self.encoder_blocks.append(
                ResUNetBlock(in_channels, width, num_convs, use_batchnorm)
            )
            in_channels = width

        # --- Goulot d'etranglement (bottleneck) ---
        bottleneck_channels = channel_widths[-1] * 2
        self.bottleneck = BasicConv(channel_widths[-1], bottleneck_channels, use_batchnorm)

        # --- Decodeur ---
        self.decoder_blocks = nn.ModuleList()
        in_channels = bottleneck_channels
        for width in reversed(channel_widths):
            self.decoder_blocks.append(
                ResUNetUpBlock(in_channels, width, width, num_convs, use_batchnorm)
            )
            in_channels = width

        # --- Projection finale vers la dimension du signal reconstruit ---
        self.output_proj = nn.Conv1d(channel_widths[0], 1, kernel_size=1)
        self.fc_out = nn.Linear(M_dim, N_dim)

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

        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            h = block(h, skip)

        h = self.output_proj(h).squeeze(1)  # (batch, M_dim)
        x_pred = self.fc_out(h)

        return x_pred, None, None
