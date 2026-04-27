import argparse
import torch
import sys
import os
import time
import yaml 
from torch.utils.data import DataLoader

from src.models import NET_ARCHITECTURES
from src.strategies import network
from src.strategies import random_search
from Dataset.module import MyDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Framework de Reconstruction")

    # Ajout de l'argument de configuration
    parser.add_argument('--config', type=str, default=None, help="Chemin vers le fichier YAML de configuration")
    
    # Conservation de tous vos arguments existants
    parser.add_argument('--model', type=str, default='p3mg', choices=['p3mg', 'ista'])
    parser.add_argument('--strategy', type=str, default='unrolling', choices=['unrolling', 'random_search'])
    parser.add_argument('--mode', type=str, default='full', choices=['train', 'test', 'full'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--criterion', type=str, default='MSE', choices=['MSE', 'SNR', 'TSNR'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_pd_layers', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--algo_iters', type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--beta",  type=float, default=1e-5)
    parser.add_argument("--eta",   type=float, default=1e-2)
    parser.add_argument('--lmbd_min', type=float, default=0.1)
    parser.add_argument('--lmbd_max', type=float, default=5.0)
    parser.add_argument('--tau_min', type=float, default=0.01)
    parser.add_argument('--tau_max', type=float, default=2.0)

    args = parser.parse_args()

    # Surcharge des arguments via le fichier YAML s'il est fourni
    if args.config:
        if not os.path.isfile(args.config):
            print(f"[ERREUR] Fichier de configuration introuvable : {args.config}")
            sys.exit(1)
            
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        for key, value in yaml_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"[AVERTISSEMENT] Paramètre '{key}' du YAML non reconnu par argparse.")

    return args

def setup_paths(args):
    """Prépare l'arborescence de sauvegarde (runs/MODEL_STRATEGY_DATE/)."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.model}_{args.strategy}_{timestamp}"
    base_dir = os.path.join("runs", run_name)
    
    paths = (
        base_dir,                             
        os.path.join(base_dir, 'checkpoints'),
        os.path.join(base_dir, 'plots'),       
        os.path.join(base_dir, 'logs')         
    )
    for p in paths:
        os.makedirs(p, exist_ok=True)
    return paths

def get_dataloaders(batch_size):
    """Charge les datasets via MyDataset et crée les DataLoaders."""
    # Chemins basés sur ton arborescence (Dataset/train.pt, etc.)
    base_path = "Dataset"
    train_path = os.path.join(base_path, "train.pt")
    val_path   = os.path.join(base_path, "val.pt")
    test_path  = os.path.join(base_path, "test.pt")
    
    # Vérification simple
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Impossible de trouver {train_path}. Vérifiez le dossier 'Dataset'.")

    print("[INFO] Instanciation des MyDataset...")
    # On instancie MyDataset
    # initial_x0 est None par défaut (géré dynamiquement dans les stratégies)
    train_ds = MyDataset(train_path, initial_x0=None, return_name=False)
    val_ds   = MyDataset(val_path,   initial_x0=None, return_name=False)
    test_ds  = MyDataset(test_path,  initial_x0=None, return_name=False)
    
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Création des loaders PyTorch
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def main():
    args = parse_args()
    
    # Consolidation des bornes pour Random Search
    args.lmbd_bounds = (args.lmbd_min, args.lmbd_max)
    args.tau_bounds = (args.tau_min, args.tau_max)

    print(f"=== Lancement : {args.model.upper()} | Stratégie : {args.strategy.upper()} | Mode : {args.mode.upper()} ===")
    
    # 1. Chargement des Données
    print("--- Préparation des DataLoaders ---")
    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size)
    
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
        model = ModelClass(
            num_layers=args.num_layers,
            num_pd_layers=args.num_pd_layers
        ).to(args.device).double()
        
        print(f"[INFO] Modèle {args.model} instancié.")

        # B. Mode TRAIN
        if args.mode in ['train', 'full']:
            network.train(model, train_loader, val_loader, args, paths)
        
        # C. Mode TEST
        if args.mode in ['test', 'full']:
            ckpt_to_load = None
            if args.mode == 'test' and args.checkpoint:
                ckpt_to_load = args.checkpoint
            elif args.mode == 'full':
                ckpt_to_load = os.path.join(paths[1], 'best_model.pt')
            
            network.test(model, test_loader, args, paths, checkpoint_path=ckpt_to_load)

    # =========================================================================
    # STRATÉGIE 2 : RANDOM SEARCH (Optimisation Hyperparamètres)
    # =========================================================================
    elif args.strategy == 'random_search':
        # Pas d'instanciation nn.Module ici, c'est géré dans random_search.py

        # A. Mode TRAIN (Calibration)
        if args.mode in ['train', 'full']:
            # Calibration sur le test_loader (ou val_loader selon préférence)
            random_search.train(test_loader, args, paths)

        # B. Mode TEST (Application)
        if args.mode in ['test', 'full']:
            random_search.test(test_loader, args, paths)

    print("\n=== Exécution Terminée ===")

if __name__ == "__main__":
    main()