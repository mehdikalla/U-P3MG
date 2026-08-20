import math

import torch
import torch.nn as nn

from src.models.Transformer_block import Transformer_Block


class FCTN_model(nn.Module):
    """
    Fully Connected Transformer Network (FCTN).

    Architecture par patchs combinee a un contexte global non lineaire.

    Deux problemes ont ete identifies et corriges par rapport aux
    versions precedentes de ce modele :

      1. Blocage du gradient (loss constante des la 2e epoque), du a une
         activation de sortie saturante (Softplus) mal initialisee par
         rapport a l'echelle tres faible des signaux cibles. Corrige par
         une contrainte de positivite via mise au carre (gradient jamais
         nul) et une LayerNorm avant la sortie.
      2. Absence de melange global de l'observation (la loss decroit mais
         reste totalement decorrelee du signal vrai). La matrice
         d'observation `Hmat` est dense : chaque composante du signal
         reconstruit depend en principe de la totalite de `y`. Fournir a
         chaque jeton (patch) uniquement une projection lineaire par
         position de `y` ne permet pas d'apprendre ce melange global via
         la seule auto-attention entre patchs. On introduit donc un
         vecteur de contexte global non lineaire (MLP applique a `y` dans
         son integralite, a la maniere de l'encodeur d'un autoencodeur
         dense) qui est diffuse (broadcast) a chaque jeton avant
         l'encodeur Transformer, qui affine ensuite cette information en
         tenant compte de la position et du voisinage de chaque portion
         du signal.

    Le signal initial `x0` est decoupe en patchs (segments contigus) qui
    fournissent un a priori local par position ; le contexte global
    (derive de `y`) et l'encodage positionnel appris sont ajoutes a chaque
    jeton avant le passage dans l'encodeur Transformer.
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

        # Nombre de patchs necessaires pour couvrir N_dim, avec padding a
        # droite si N_dim n'est pas un multiple exact de patch_size.
        self.num_patches = math.ceil(N_dim / patch_size)
        self.padded_dim = self.num_patches * patch_size
        self.pad_amount = self.padded_dim - N_dim

        # 1. Contexte global non lineaire derive de l'observation complete
        # `y`. Cette branche joue le meme role que l'encodeur d'un
        # autoencodeur dense (cf. FCAE) : elle apprend le melange global
        # necessaire pour inverser (approximativement) l'operateur
        # d'observation dense `Hmat`, ce qu'une simple projection lineaire
        # par position de patch ne peut pas apprendre efficacement seule.
        self.global_context = nn.Sequential(
            nn.Linear(M_dim, context_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden, d_model),
        )

        # 1bis. Decodeur dense direct (chemin residuel).
        # `Hmat` est generalement tres mal conditionnee, voire de rang tres
        # inferieur a N_dim (probleme inverse mal pose). Dans ce contexte,
        # un decodeur entierement connecte (a la maniere du FCAE) reste la
        # facon la plus directe et la plus fiable d'apprendre le melange
        # global observation -> signal. Ce chemin dense produit une
        # premiere estimation du signal complet, que l'encodeur Transformer
        # (via patch_embedding/positional_embedding/global_ctx) vient
        # ensuite raffiner de maniere residuelle (a priori local + contexte
        # + relations inter-patchs). Cette combinaison permet de conserver
        # un veritable encodeur Transformer au coeur de l'architecture tout
        # en beneficiant de la robustesse d'un decodeur dense pour le
        # melange global difficile a apprendre par la seule auto-attention.
        self.dense_decoder = nn.Sequential(
            nn.Linear(M_dim, context_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden, context_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(context_hidden, N_dim),
        )

        # 2. Embedding des patchs de x0 (a priori local par position).
        self.patch_embedding = nn.Linear(patch_size, d_model)

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

        # 5. Normalisation appliquee juste avant la projection de sortie.
        # Contraint l'echelle des pre-activations independamment de la
        # derive des poids/biais au cours de l'entrainement, ce qui evite
        # qu'une activation de positivite saturante (Softplus, ELU, ...)
        # ne se retrouve durablement dans sa zone a gradient quasi nul.
        self.pre_out_norm = nn.LayerNorm(d_model)

        # 6. Couche Fully Connected de sortie : projette chaque jeton de
        # dimension d_model vers un patch residuel de longueur patch_size,
        # ajoute (dans l'espace positif, via mise au carre) a l'estimation
        # dense directe pour produire le signal reconstruit final.
        self.fc_out = nn.Linear(d_model, patch_size)
        # Initialisation proche de zero afin que la contribution residuelle
        # du Transformer soit negligeable en debut d'entrainement : le
        # modele demarre donc essentiellement comme le decodeur dense
        # (chemin fiable et bien conditionne), puis apprend progressivement
        # a exploiter le raffinement local/contextuel apporte par
        # l'auto-attention.
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc_out.bias)

        # Contrainte de positivite du signal reconstruit via une mise au
        # carre plutot qu'une activation de type Softplus/ELU/ReLU.
        # Justification : les signaux cibles de ce projet sont a tres
        # faible echelle (souvent << 1) avec une forte proportion de
        # valeurs nulles. Une activation saturante cote negatif
        # (Softplus, ELU, ReLU) expose le reseau a un phenomene de
        # "neurone mort" : des que l'optimiseur pousse les pre-activations
        # vers de fortes valeurs negatives (ce qui se produit tres vite ici
        # car la sortie initiale est bien plus grande que la cible), le
        # gradient de l'activation devient quasi nul et l'apprentissage se
        # bloque durablement, la loss se stabilisant a la valeur triviale
        # "prediction nulle partout" (mean(x_true**2)). La mise au carre
        # n'a pas ce probleme : son gradient (2x) reste proportionnel a
        # l'ecart a zero, quel que soit le signe de la pre-activation, ce
        # qui permet a l'optimiseur de corriger l'echelle de sortie sans
        # jamais se retrouver bloque dans une zone a gradient nul.
        # L'initialisation par defaut de nn.Linear (poids/biais proches de
        # zero) est ici directement adaptee, car elle produit une sortie
        # initiale elle-meme proche de zero.


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
        # Estimation dense directe du signal complet a partir de
        # l'observation (chemin fiable, bien conditionne, cf. FCAE).
        dense_estimate = self.dense_decoder(y)

        # Contexte global non lineaire issu de l'observation complete,
        # diffuse identiquement a tous les patchs d'un meme echantillon :
        # (batch, d_model) -> (batch, 1, d_model)
        global_ctx = self.global_context(y).unsqueeze(1)

        x0_patches = self._to_patches(x0)

        emb = self.patch_embedding(x0_patches) + global_ctx + self.positional_embedding

        encoded = self.transformer(emb)
        encoded = self.pre_out_norm(encoded)

        raw_patches = self.fc_out(encoded)
        residual = raw_patches.reshape(x0.shape[0], self.padded_dim)
        if self.pad_amount > 0:
            residual = residual[:, : self.N_dim]

        # Le Transformer apprend un raffinement additif (positif ou
        # negatif) de l'estimation dense directe, exploitant le contexte
        # local par patch et les relations inter-patchs via
        # l'auto-attention. La contrainte de positivite du signal final
        # est appliquee une seule fois, sur la somme, via une mise au
        # carre (gradient jamais nul, cf. commentaire plus haut).
        x_pred = (dense_estimate + residual) ** 2

        return x_pred, None, None
