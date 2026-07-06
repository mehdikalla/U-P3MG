# -----------------------------------------------------------------------------
# Import des composants des différents modèles
# -----------------------------------------------------------------------------

# --- Modèle 1 : P3MG ---
from src.models.p3mg.net import P3MG_model
from src.models.p3mg.algo import P3MG_algo

# --- Modèle 2 : ISTA ---
from src.models.ista.net import ISTA_model
from src.models.ista.algo import ISTA_algo

# --- Modèle 3 : HQ ---
from src.models.hq.net import HQ_model
from src.models.hq.algo import HQ_algo

# --- Modèle 4 : Primal-Dual (Autonome) ---
from src.models.pd.net import PD_Standalone_model
from src.models.pd.algo import PD_Standalone_algo

# --- Modèle 5 : PMMS ---
from src.models.pmms.net import PMMS_model
from src.models.pmms.algo import PMMS_algo

# --- Modèle 6 : FCAE (Fully Connected Autoencoder) ---
from src.models.fcae.net import FCAE_model

# --- Modèle 7 : FCTN (Fully Connected Transformer Network) ---
from src.models.fctn.net import FCTN_model

# --- Modèle 8 : FCUN (Fully Connected U-Net) ---
from src.models.fcun.net import FCUN_model

# -----------------------------------------------------------------------------
# Registres (Dictionnaires)
# C'est ce que le main.py va utiliser pour sélectionner le modèle
# -----------------------------------------------------------------------------


NET_ARCHITECTURES = {
    'p3mg': P3MG_model,
    'ista': ISTA_model,
    'hq': HQ_model,
    'pd': PD_Standalone_model,
    'pmms': PMMS_model,
    'fcae': FCAE_model,
    'fctn': FCTN_model,
    'fcun': FCUN_model,
}

ALGORITHMS = {
    'p3mg': P3MG_algo,
    'ista': ISTA_algo,
    'hq': HQ_algo,
    'pd': PD_Standalone_algo,
    'pmms': PMMS_algo,
}

# Modèles purement "deep learning" ne reposant sur aucun algorithme itératif
# statique (pas de dictionnaire `static`/`dynamic` physique à initialiser).
FULLY_LEARNED_MODELS = {'fcae', 'fctn', 'fcun'}

