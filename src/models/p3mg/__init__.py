# On expose la classe du réseau de neurones (pour l'entrainement Unrolling)
# Assure-toi que la classe dans 'net.py' s'appelle bien 'P3MG' (ou 'P3MG_Net')
from src.models.p3mg.net import P3MG_model

# On expose la fonction mathématique pure (pour le Random Search)
# Assure-toi que la fonction dans 'algo.py' s'appelle bien 'p3mg_algorithm'
from src.models.p3mg.algo import P3MG_algo