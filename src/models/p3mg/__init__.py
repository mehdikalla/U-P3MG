# On expose la classe du réseau de neurones (pour l'entrainement Unrolling)
# Assure-toi que la classe dans 'net.py' s'appelle bien 'P3MG' (ou 'P3MG_Net')
from .net import P3MG

# On expose la fonction mathématique pure (pour le Random Search)
# Assure-toi que la fonction dans 'algo.py' s'appelle bien 'p3mg_algorithm'
from .algo import p3mg_algorithm