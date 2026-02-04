# -----------------------------------------------------------------------------
# Import des composants des différents modèles
# -----------------------------------------------------------------------------

# --- Modèle 1 : P3MG ---
# On importe la classe NN (pour l'unrolling) et la fonction algo (pour le grid search)
# NOTE : Vérifie bien le nom exact de la classe dans net.py (ex: P3MG)
# et le nom exact de la fonction dans algo.py (ex: p3mg_algorithm)
from .p3mg.net import P3MG
from .p3mg.algo import p3mg_algorithm

# --- Modèle 2 (Futur) ---
# from .model2.net import Model2
# from .model2.algo import model2_algorithm


# -----------------------------------------------------------------------------
# Registres (Dictionnaires)
# C'est ce que le main.py va utiliser pour sélectionner le modèle
# -----------------------------------------------------------------------------

# 1. Registre des Architectures (Pour la stratégie 'unrolling')
NET_ARCHITECTURES = {
    'p3mg': P3MG,
    # 'model2': Model2,
    # 'model3': Model3,
}

# 2. Registre des Algorithmes Mathématiques (Pour la stratégie 'random_search')
ALGORITHMS = {
    'p3mg': p3mg_algorithm,
    # 'model2': model2_algorithm,
    # 'model3': model3_algorithm,
}