import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """
    Bloc dense (Fully Connected) avec activation ReLU, normalisation par batch
    et dropout optionnels. Sert d'unite de base pour l'encodeur et le
    decodeur du FCAE.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        # Initialisation Kaiming (He), adaptee a l'activation ReLU qui suit,
        # pour accelerer la convergence par rapport a l'init par defaut.
        nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        nn.init.zeros_(linear.bias)

        layers = [linear]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Encoder(nn.Module):
    """
    Encodeur Fully Connected : compresse l'observation bruitee vers un
    espace latent de dimension reduite (goulot d'etranglement).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list,
        latent_dim: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        blocks = [
            DenseBlock(dims[i], dims[i + 1], use_batchnorm, dropout)
            for i in range(len(dims) - 1)
        ]
        self.hidden = nn.Sequential(*blocks)
        # Projection finale vers l'espace latent sans activation, afin de ne
        # pas contraindre artificiellement le signe/l'amplitude du code latent.
        self.to_latent = nn.Linear(dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return self.to_latent(h)


class Decoder(nn.Module):
    """
    Decodeur Fully Connected : reconstruit le signal a partir du code
    latent produit par l'encodeur.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list,
        out_dim: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [latent_dim] + list(hidden_dims)
        blocks = [
            DenseBlock(dims[i], dims[i + 1], use_batchnorm, dropout)
            for i in range(len(dims) - 1)
        ]
        self.hidden = nn.Sequential(*blocks)
        # Projection finale suivie d'une Softplus garantissant un signal positif.
        self.to_output = nn.Linear(dims[-1], out_dim)
        nn.init.xavier_normal_(self.to_output.weight)
        nn.init.zeros_(self.to_output.bias)
        self.output_activation = nn.Softplus()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.hidden(z)
        return self.output_activation(self.to_output(h))


class FCAE_model(nn.Module):
    """
    Autoencodeur Fully Connected (FCAE) pour la reconstruction de signaux.

    Architecture encodeur-decodeur symetrique avec un goulot d'etranglement
    (espace latent) explicite : l'encodeur compresse l'observation bruitee
    `y` vers un code latent de dimension reduite, et le decodeur reconstruit
    le signal `x` de dimension `N_dim` a partir de ce code. La normalisation
    par batch et le dropout sont utilises pour stabiliser l'entrainement et
    limiter le surapprentissage.

    Args:
        M_dim: Dimension de l'observation (entree).
        N_dim: Dimension du signal reconstruit (sortie).
        latent_dim: Dimension de l'espace latent (goulot d'etranglement).
        hidden_dims: Largeurs des couches cachees de l'encodeur, dans l'ordre
            entree -> latent. Le decodeur utilise les largeurs symetriques.
        use_batchnorm: Active la normalisation par batch dans les blocs denses.
        dropout: Taux de dropout applique apres chaque bloc dense (0.0 pour
            desactiver).
    """

    def __init__(
        self,
        M_dim: int,
        N_dim: int,
        latent_dim: int = 12,
        hidden_dims: list = None,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [50, 25]

        self.M_dim = M_dim
        self.N_dim = N_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            in_dim=M_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            out_dim=N_dim,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """Projette l'observation `y` dans l'espace latent."""
        return self.encoder(y)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruit le signal a partir du code latent `z`."""
        return self.decoder(z)

    def forward(self, static, dynamic, x0: torch.Tensor, y: torch.Tensor):
        """
        Passe avant du modele FCAE.

        Args:
            static: Variables statiques (non utilisees, presentes pour
                compatibilite avec l'interface commune des modeles du projet).
            dynamic: Variables dynamiques (non utilisees, idem).
            x0: Signal initial. Non utilise directement, conserve pour
                compatibilite.
            y: Observation bruitee. Shape: (batch_size, M_dim).

        Returns:
            Tuple (x_pred, None, None) ou x_pred est le signal reconstruit
            de shape (batch_size, N_dim).
        """
        z = self.encode(y)
        x_pred = self.decode(z)

        return x_pred, None, None
