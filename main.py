import argparse
import torch
import sys
import os
import time

# --- Imports du Framework ---
# Charge les dictionnaires de modèles définis dans src/models/__init__.py
from src.models import NET_ARCHITECTURES

# Charge les stratégies (moteurs de résolution)
from src.strategies import network
from src.strategies import random_search

# Chargeur de données
from Dataset.module import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Framework de Reconstruction (Unrolling & Grid Search)")

    # --- 1. Choix du Modèle et de la Stratégie ---
    parser.add_argument('--model', type=str, default='p3mg', choices=['p3mg', 'model2'],
                        help="Architecture du modèle à utiliser.")
    parser.add_argument('--strategy', type=str, default='unrolling', choices=['unrolling', 'random_search'],
                        help="Méthode de résolution : 'unrolling' (NN) ou 'random_search' (Algo itératif).")
    parser.add_argument('--mode', type=str, default='full', choices=['train', 'test', 'full'],
                        help="Action à effectuer : 'train' (apprendre/calibrer), 'test' (évaluer), ou 'full' (les deux).")
    
    # --- 2. Paramètres Généraux & Hardware ---
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device de calcul (cuda/cpu).")
    parser.add_argument('--seed', type=int, default=42, help="Graine aléatoire.")
    parser.add_argument('--batch_size', type=int, default=4, help="Taille des batchs.")
    parser.add_argument('--criterion', type=str, default='MSE', choices=['MSE', 'SNR', 'TSNR'],
                        help="Fonction de coût.")

    # --- 3. Paramètres Unrolling (NN) ---
    parser.add_argument('--epochs', type=int, default=100, help="Nombre d'époques (Unrolling).")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (Unrolling).")
    parser.add_argument('--num_layers', type=int, default=8, help="Nombre de couches déroulées (K).")
    parser.add_argument('--num_pd_layers', type=int, default=5, help="Nombre de sous-couches Primal-Dual (T).")
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help="Chemin vers un checkpoint (.pt) pour le mode 'test'. Si vide, cherche dans le dossier courant.")

    # --- 4. Paramètres Random Search & Algo ---
    parser.add_argument('--n_samples', type=int, default=50, help="Nombre de configurations à tester (Calibration).")
    parser.add_argument('--algo_iters', type=int, default=200, help="Nombre d'itérations pour l'algo itératif (P3MG pur).")
    
    # Paramètres Statiques initiaux (utilisés par les deux stratégies comme point de départ ou constante)
    parser.add_argument('--rho', type=float, default=1.0, help="Paramètre statique Rho initial.")
    parser.add_argument('--gamma', type=float, default=0.1, help="Paramètre statique Gamma initial.")

    # Bornes de recherche (Random Search) - Lambda (~Rho) et Tau (~Gamma)
    parser.add_argument('--lmbd_min', type=float, default=0.1, help="Borne Min pour Lambda.")
    parser.add_argument('--lmbd_max', type=float, default=5.0, help="Borne Max pour Lambda.")
    parser.add_argument('--tau_min', type=float, default=0.01, help="Borne Min pour Tau.")
    parser.add_argument('--tau_max', type=float, default=2.0, help="Borne Max pour Tau.")

    return parser.parse_args()

def setup_paths(args):
    """Prépare l'arborescence de sauvegarde (runs/MODEL_STRATEGY_DATE/)."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.model}_{args.strategy}_{timestamp}"
    
    # Si on fait juste un test avec un checkpoint spécifique, on peut vouloir écrire ailleurs
    # Mais pour simplifier, on crée toujours un nouveau dossier de logs
    base_dir = os.path.join("runs", run_name)
    
    paths = (
        base_dir,                             # path_save
        os.path.join(base_dir, 'checkpoints'), # path_checkpoints
        os.path.join(base_dir, 'plots'),       # path_plots
        os.path.join(base_dir, 'logs')         # path_logs
    )
    
    # Création des dossiers seulement si on lance un train ou un full
    # Si c'est juste un test, on crée quand même pour sauver les résultats du test
    for p in paths:
        os.makedirs(p, exist_ok=True)
        
    return paths

def main():
    args = parse_args()
    
    # Consolidation des bornes pour le Random Search
    args.lmbd_bounds = (args.lmbd_min, args.lmbd_max)
    args.tau_bounds = (args.tau_min, args.tau_max)

    print(f"=== Lancement : {args.model.upper()} | Stratégie : {args.strategy.upper()} | Mode : {args.mode.upper()} ===")
    
    # 1. Chargement des Données
    print("--- Chargement du Dataset ---")
    train_loader, val_loader, test_loader = load_dataset(batch_size=args.batch_size)
    
    # 2. Préparation des Dossiers
    paths = setup_paths(args)
    print(f"[INFO] Résultats sauvegardés dans : {paths[0]}")

    # =========================================================================
    # STRATÉGIE 1 : UNROLLING (Réseau de Neurones)
    # =========================================================================
    if args.strategy == 'unrolling':
        # A. Instanciation du Modèle
        if args.model not in NET_ARCHITECTURES:
            raise ValueError(f"Architecture '{args.model}' introuvable dans src.models.")
        
        ModelClass = NET_ARCHITECTURES[args.model]
        
        # On passe les dimensions explicites au constructeur
        model = ModelClass(
            num_layers=args.num_layers,
            num_pd_layers=args.num_pd_layers
        ).to(args.device).double() # Important : Double précision pour P3MG
        
        print(f"[INFO] Modèle {args.model} instancié (Layers={args.num_layers}, PD-Layers={args.num_pd_layers}).")

        # B. Mode TRAIN
        if args.mode in ['train', 'full']:
            network.train(model, train_loader, val_loader, args, paths)
        
        # C. Mode TEST
        if args.mode in ['test', 'full']:
            # Si on a fait 'full', le checkpoint est déjà dans paths[1] (créé par train)
            # Si on fait juste 'test', il faut peut-être utiliser args.checkpoint
            ckpt_to_load = None
            
            if args.mode == 'test' and args.checkpoint:
                ckpt_to_load = args.checkpoint
            elif args.mode == 'full':
                # On prend celui qu'on vient de générer
                ckpt_to_load = os.path.join(paths[1], 'best_model.pt')
            
            network.test(model, test_loader, args, paths, checkpoint_path=ckpt_to_load)

    # =========================================================================
    # STRATÉGIE 2 : RANDOM SEARCH (Optimisation Hyperparamètres)
    # =========================================================================
    elif args.strategy == 'random_search':
        # Note : Pour le Random Search, pas besoin d'instancier un nn.Module complexe.
        # L'algorithme mathématique est instancié à la volée dans random_search.py 
        # via P3MGNet(num_layers=1).

        # A. Mode TRAIN (Calibration sur 10% du dataset)
        if args.mode in ['train', 'full']:
            # On utilise le test_loader (ou val_loader) pour calibrer
            # random_search.train va extraire 10% de ce loader
            random_search.train(test_loader, args, paths)

        # B. Mode TEST (Application sur 100% du dataset)
        if args.mode in ['test', 'full']:
            random_search.test(test_loader, args, paths)

    print("\n=== Exécution Terminée ===")

if __name__ == "__main__":
    main()