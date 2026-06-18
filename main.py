import argparse
import torch
import sys
import os
import time
import yaml 
import random
import numpy as np
from torch.utils.data import DataLoader

from src.models import NET_ARCHITECTURES
from src.strategies import network
from src.strategies import random_search
from Dataset.module import MyDataset

def set_seed(seed):
    """Fixe toutes les graines aléatoires pour garantir la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Framework de Reconstruction")

    parser.add_argument('--config', type=str, default=None, help="Chemin vers le fichier YAML de configuration")
    parser.add_argument('--model', type=str, default='p3mg', choices=['p3mg', 'ista', 'hq', 'pmms', 'pd'])
    parser.add_argument('--strategy', type=str, default='unrolling', choices=['unrolling', 'random_search'])
    parser.add_argument('--mode', type=str, default='full', choices=['train', 'test', 'full'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--criterion', type=str, default='MSE', choices=['MSE', 'SNR', 'TSNR'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_layers', type=int, default=25)
    parser.add_argument('--num_pd_layers', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--algo_iters', type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--beta",  type=float, default=1e-5)
    parser.add_argument("--eta",   type=float, default=1e-2)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--delta_cvx", type=float, default=0.01)
    parser.add_argument("--delta_ncvx", type=float, default=0.01)
    parser.add_argument('--lmbd_min', type=float, default=0.1)
    parser.add_argument('--lmbd_max', type=float, default=5.0)
    parser.add_argument('--tau_min', type=float, default=0.01)
    parser.add_argument('--tau_max', type=float, default=2.0)

    args = parser.parse_args()

    if args.config:
        if not os.path.isfile(args.config):
            print(f"[ERREUR] Fichier de configuration introuvable : {args.config}")
            sys.exit(1)
            
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        passed_args = [arg.strip('-').split('=')[0] for arg in sys.argv if arg.startswith('-')]

        for key, value in yaml_config.items():
            if hasattr(args, key):
                if key not in passed_args:
                    setattr(args, key, value)
            else:
                print(f"[AVERTISSEMENT] Paramètre '{key}' du YAML non reconnu par argparse.")

    return args

def setup_paths(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{args.mode}"
    base_dir = os.path.join("runs", args.model.strip().lower(), args.strategy, run_name)
    
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
    base_path = "Dataset"
    train_path = os.path.join(base_path, "train.pt")
    val_path   = os.path.join(base_path, "val.pt")
    test_path  = os.path.join(base_path, "test.pt")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Impossible de trouver {train_path}.")

    print("[INFO] Instanciation des MyDataset...")
    train_ds = MyDataset(train_path, initial_x0=None, return_name=False)
    val_ds   = MyDataset(val_path,   initial_x0=None, return_name=False)
    test_ds  = MyDataset(test_path,  initial_x0=None, return_name=False)
    
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def main():
    args = parse_args()
    
    # Sécurisation de l'environnement aléatoire
    set_seed(args.seed)
    
    args.lmbd_bounds = (args.lmbd_min, args.lmbd_max)
    args.tau_bounds = (args.tau_min, args.tau_max)

    print(f"=== Lancement : {args.model.upper()} | Stratégie : {args.strategy.upper()} | Mode : {args.mode.upper()} ===")
    print("--- Préparation des DataLoaders ---")
    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size)
    
    paths = setup_paths(args)
    print(f"[INFO] Résultats sauvegardés dans : {paths[0]}")

    if args.strategy == 'unrolling':
        model_key = args.model.strip().lower()
        if model_key not in NET_ARCHITECTURES:
            raise ValueError(f"Architecture '{model_key}' introuvable.")
        
        ModelClass = NET_ARCHITECTURES[model_key]
        
        if model_key in ['p3mg', 'hq']:
            model = ModelClass(num_layers=args.num_layers, num_pd_layers=args.num_pd_layers)
        else:
            model = ModelClass(num_layers=args.num_layers)
            
        model = model.to(args.device).double()
        print(f"[INFO] Modèle {model_key.upper()} instancié.")

        if args.mode in ['train', 'full']:
            network.train(model, train_loader, val_loader, args, paths)
        
        if args.mode in ['test', 'full']:
            ckpt_to_load = args.checkpoint if (args.mode == 'test' and args.checkpoint) else os.path.join(paths[1], 'best_model.pt')
            network.test(model, test_loader, args, paths, checkpoint_path=ckpt_to_load)

    elif args.strategy == 'random_search':
        if args.mode in ['train', 'full']:
            random_search.train(test_loader, args, paths)
        if args.mode in ['test', 'full']:
            random_search.test(test_loader, args, paths)

    print("\n=== Exécution Terminée ===")

if __name__ == "__main__":
    main()