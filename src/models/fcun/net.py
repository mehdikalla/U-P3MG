import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """
    Bloc dense (Fully Connected) avec activation ReLU et normalisation optionnelle.
    Sert d'unite de base pour l'encodeur et le decodeur du FCUN.
    """

    def __init__(self, in_dim: int, out_dim: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FCUN_model(nn.Module):
    """
    Fully Connected U-Net (FCUN).

    Architecture encodeur-decodeur entierement connectee (sans convolution),
    reproduisant la logique d'un U-Net classique : reduction progressive de la
    dimension latente (encodeur), remontee progressive (decodeur) et connexions
    de saut (skip connections) entre couches de meme niveau afin de preserver
    l'information haute resolution perdue lors de la compression.

    Args:
        N_dim: Dimension du signal reconstruit (sortie).
        M_dim: Dimension de l'observation (entree).
        num_layers: Profondeur du reseau, c'est-a-dire le nombre de niveaux de
            l'encodeur (et symetriquement du decodeur).
        base_width: Largeur de la premiere couche cachee. Chaque niveau suivant
            de l'encodeur divise cette largeur par deux (largeur minimale
            forcee a 8) ; le decodeur remonte symetriquement.
        use_batchnorm: Active la normalisation par batch dans les blocs denses.
    """

    def __init__(
        self,
        N_dim: int = 800,
        M_dim: int = 100,
        num_layers: int = 4,
        base_width: int = 256,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers doit etre superieur ou egal a 1.")

        self.N_dim = N_dim
        self.M_dim = M_dim
        self.num_layers = num_layers

        # Calcul des largeurs successives de l'encodeur (division par 2, largeur
        # minimale de 8 afin d'eviter un goulot d'etranglement degenere).
        encoder_widths = [max(base_width // (2 ** i), 8) for i in range(num_layers)]

        # --- Encodeur ---
        self.encoder_blocks = nn.ModuleList()
        in_dim = M_dim
        for width in encoder_widths:
            self.encoder_blocks.append(DenseBlock(in_dim, width, use_batchnorm))
            in_dim = width

        # --- Goulot d'etranglement (bottleneck) ---
        bottleneck_dim = max(encoder_widths[-1] // 2, 4)
        self.bottleneck = DenseBlock(encoder_widths[-1], bottleneck_dim, use_batchnorm)

        # --- Decodeur ---
        # Chaque bloc decodeur recoit la concatenation de la sortie du niveau
        # precedent et de la sortie de l'encodeur au meme niveau (skip connection).
        decoder_widths = list(reversed(encoder_widths))
        self.decoder_blocks = nn.ModuleList()
        in_dim = bottleneck_dim
        for width in decoder_widths:
            self.decoder_blocks.append(DenseBlock(in_dim + width, width, use_batchnorm))
            in_dim = width

        # --- Projection finale vers la dimension du signal reconstruit ---
        # Suivie d'une activation Softplus afin de garantir un signal
        # reconstruit strictement positif.
        self.fc_out = nn.Linear(decoder_widths[-1], N_dim)
        self.output_activation = nn.Softplus()

    def forward(self, static, dynamic, x0: torch.Tensor, y: torch.Tensor):
        """
        Passe avant du modele FCUN.

        Args:
            static: Variables statiques (non utilisees, presentes pour compatibilite
                avec l'interface commune des modeles du projet).
            dynamic: Variables dynamiques (non utilisees, idem).
            x0: Signal initial. Non utilise directement, conserve pour compatibilite.
            y: Observation bruitee. Shape: (batch_size, M_dim).

        Returns:
            Tuple (x_pred, None, None) ou x_pred est le signal reconstruit de
            shape (batch_size, N_dim).
        """
        skip_connections = []
        h = y

        for block in self.encoder_blocks:
            h = block(h)
            skip_connections.append(h)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            h = torch.cat([h, skip], dim=-1)
            h = block(h)

        x_pred = self.output_activation(self.fc_out(h))

        return x_pred, None, None
