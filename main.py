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
from src.strategies import compare
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
    parser.add_argument('--model', type=str, default='pd', choices=['p3mg', 'ista', 'hq', 'pmms', 'ipiano', 'vmfb', 'fista', 'pd', 'fcae', 'fctn', 'fcun', 'resu'])

    parser.add_argument('--dataset_dir', type=str, default='./Dataset', help="Dossier racine contenant les générations de datasets")
    parser.add_argument('--data_folder', type=str, default='data_0', help="Sous-dossier de génération à utiliser (ex: data_0, data_1, ...)")
    parser.add_argument('--run_tag', type=str, default=None,
                         help="Espace de nommage isole pour 'runs/<model>/<strategy>/<run_tag>' et la "
                              "recherche de checkpoints. Par defaut, egal a --data_folder. Permet de "
                              "cloisonner des runs (ex: etudes d'ablation) sans qu'ils interferent avec "
                              "les runs standard/compare bases sur le meme data_folder.")
    parser.add_argument('--run_group', type=str, default=None, choices=[None, 'ablation'],
                         help="Prefixe le dossier de sortie par 'runs/<run_group>/...' au lieu de "
                              "'runs/<model>/<strategy>/...'. Permet de dedier un dossier racine complet "
                              "(ex: 'runs/ablation/') aux etudes d'ablation, totalement separe de "
                              "'runs/<model>/<strategy>/' (runs standard) et 'runs/compare/'.")




    parser.add_argument('--strategy', type=str, default='unrolling', choices=['unrolling', 'random_search'])
    parser.add_argument('--mode', type=str, default='full', choices=['train', 'test', 'full','compare'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--criterion', type=str, default='MSE', choices=['MSE', 'SNR', 'TSNR'])
    parser.add_argument('--profiler', action='store_true', default=False,
                         help="Active le profilage memoire/temps via `torch.profiler` (cf. "
                              "src.utils.torch_profiler_utils.TorchOpProfiler) durant le mode "
                              "'compare'. Desactive par defaut (le profiler est instable/bugue "
                              "sur certaines configurations) ; active-le explicitement pour les "
                              "runs 'compare_single_hp' (cf. scripts/run_compare_single_hp.slurm).")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_layers', type=int, default=25)
    parser.add_argument('--num_pd_layers', type=int, default=10)
    parser.add_argument('--mlp_hidden', type=str, default=None,
                         help="Tailles des couches cachees du MLP interne P3MG (lambda), ex: '50,25,12'")
    parser.add_argument('--use_fc_lambda', type=int, default=1, choices=[0, 1],
                         help="Si 1 (defaut), lambda du P3MG est produit par un FC_block (MLP) "
                              "dependant de y. Si 0, lambda est entraine directement comme un "
                              "nn.Parameter scalaire par couche (initialise a 8e-5), a la maniere de tau.")

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--algo_iters', type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--beta",  type=float, default=1e-5)
    parser.add_argument("--eta",   type=float, default=1e-2)
    parser.add_argument("--sigma", type=float, default=1e-5)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--delta_cvx", type=float, default=0.01)
    parser.add_argument("--delta_ncvx", type=float, default=0.01)
    parser.add_argument('--lmbd_min', type=float, default=0.1)
    parser.add_argument('--lmbd_max', type=float, default=5.0)
    parser.add_argument('--lmbd_ist_min', type=float, default=0.1,
                         help="Borne inferieure pour la recherche de Lambda ISTA (independante de P3MG)")
    parser.add_argument('--lmbd_ist_max', type=float, default=5.0,
                         help="Borne superieure pour la recherche de Lambda ISTA (independante de P3MG)")
    parser.add_argument('--tau_min', type=float, default=0.01)
    parser.add_argument('--tau_max', type=float, default=2.0)
    parser.add_argument('--tau_pd_min', type=float, default=None,
                         help="Borne inferieure specifique pour la recherche de Tau du modele PD "
                              "standalone. Si non fourni, retombe sur --tau_min.")
    parser.add_argument('--tau_pd_max', type=float, default=None,
                         help="Borne superieure specifique pour la recherche de Tau du modele PD "
                              "standalone. Si non fourni, retombe sur --tau_max.")


    parser.add_argument('--nu_min', type=float, default=1e-6,

                         help="Borne inferieure pour la recherche de nu (PMMS/iPiano/VMFB/FISTA)")
    parser.add_argument('--nu_max', type=float, default=1e-3,
                         help="Borne superieure pour la recherche de nu (PMMS/iPiano/VMFB/FISTA)")

    parser.add_argument('--lambda_tau_min', type=float, default=None,
                         help="Borne inferieure pour lambda_tau (regularisation du modele PD standalone). Si non fourni, retombe sur --lmbd_min.")
    parser.add_argument('--lambda_tau_max', type=float, default=None,
                         help="Borne superieure pour lambda_tau (regularisation du modele PD standalone). Si non fourni, retombe sur --lmbd_max.")

    parser.add_argument('--compare_unrolling_models', type=str, default=None,
                         help="Liste separee par des virgules des modeles a evaluer en strategie "
                              "'unrolling' lors du mode 'compare' (ex: 'hq,ista,p3mg'). Si non fourni, "
                              "retombe sur src.strategies.compare.UNROLLING_MODELS (par defaut : p3mg,hq).")
    parser.add_argument('--compare_random_search_models', type=str, default=None,
                         help="Liste separee par des virgules des modeles a evaluer en strategie "
                              "'random_search' lors du mode 'compare' (ex: 'pd,hq,ista,p3mg'). Si non "
                              "fourni, retombe sur src.strategies.compare.RANDOM_SEARCH_MODELS.")
    parser.add_argument('--compare_dl_models', type=str, default=None,
                         help="Liste separee par des virgules des modeles purement deep-learning a "
                              "evaluer lors du mode 'compare' (ex: 'fcun,fcae'). Si non fourni, retombe "
                              "sur src.strategies.compare.DL_MODELS.")



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
    run_tag = (args.run_tag or args.data_folder).strip().lower()
    run_group = getattr(args, 'run_group', None)
    if run_group:
        base_dir = os.path.join("runs", run_group.strip().lower(), args.model.strip().lower(), run_tag, run_name)
    elif args.mode == 'compare':
        base_dir = os.path.join("runs", "compare", run_tag, run_name)
    else:
        base_dir = os.path.join("runs", args.model.strip().lower(), args.strategy, run_tag, run_name)



    
    paths = (

        base_dir,                             
        os.path.join(base_dir, 'checkpoints'),
        os.path.join(base_dir, 'plots'),       
        os.path.join(base_dir, 'logs')         
    )
    for p in paths:
        os.makedirs(p, exist_ok=True)
    return paths

def find_latest_checkpoint(model_name, strategy, data_folder=None, current_base_dir=None, run_group=None):
    """
    Recherche le checkpoint 'best_model.pt' le plus récent pour un modèle,
    une stratégie et un dossier de données donnés, en parcourant les dossiers
    'runs/<model>/<strategy>/<data_folder>/*' (ou 'runs/<run_group>/<model>/<strategy>/<data_folder>/*'
    si run_group est fourni)

    Le dossier 'current_base_dir' (run en cours) est exclu de la recherche
    puisqu'il vient d'être créé et ne contient encore aucun poids.
    """
    root = os.path.join("runs", run_group.strip().lower()) if run_group else "runs"
    model_dir = os.path.join(root, model_name) if run_group else os.path.join(root, model_name, strategy)
    if data_folder:
        strategy_dir = os.path.join(model_dir, data_folder.strip().lower())
    else:
        strategy_dir = model_dir

    if not os.path.isdir(strategy_dir):
        return None


    run_dirs = sorted(
        (d for d in os.listdir(strategy_dir) if os.path.isdir(os.path.join(strategy_dir, d))),
        reverse=True
    )

    for run_name in run_dirs:
        run_path = os.path.join(strategy_dir, run_name)
        if current_base_dir and os.path.abspath(run_path) == os.path.abspath(current_base_dir):
            continue
        candidate = os.path.join(run_path, 'checkpoints', 'best_model.pt')
        if os.path.exists(candidate):
            return candidate

    return None

def get_dataloaders(batch_size, dataset_dir="./Dataset", data_folder="data_1"):

    """
    Charge les splits train/val/test depuis dataset_dir/data_folder.

    Retombe sur dataset_dir directement (rétrocompatibilité) si le
    sous-dossier de génération n'existe pas.
    """
    base_path = os.path.join(dataset_dir, data_folder)
    if not os.path.isdir(base_path):
        print(f"[AVERTISSEMENT] Dossier de génération '{base_path}' introuvable, "
              f"utilisation de '{dataset_dir}' directement.")
        base_path = dataset_dir

    train_path = os.path.join(base_path, "train.pt")
    val_path   = os.path.join(base_path, "val.pt")
    test_path  = os.path.join(base_path, "test.pt")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Impossible de trouver {train_path}.")

    print(f"[INFO] Chargement des données depuis : {base_path}")


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
    args.lmbd_ist_bounds = (args.lmbd_ist_min, args.lmbd_ist_max)
    args.tau_bounds = (args.tau_min, args.tau_max)
    args.tau_pd_bounds = (
        args.tau_pd_min if args.tau_pd_min is not None else args.tau_min,
        args.tau_pd_max if args.tau_pd_max is not None else args.tau_max,
    )
    args.nu_bounds = (args.nu_min, args.nu_max)
    args.lambda_tau_bounds = (
        args.lambda_tau_min if args.lambda_tau_min is not None else args.lmbd_min,
        args.lambda_tau_max if args.lambda_tau_max is not None else args.lmbd_max,
    )






    print(f"=== Lancement : {args.model.upper()} | Stratégie : {args.strategy.upper()} | Mode : {args.mode.upper()} ===")
    print("--- Préparation des DataLoaders ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.batch_size, dataset_dir=args.dataset_dir, data_folder=args.data_folder
    )

    
    paths = setup_paths(args)
    print(f"[INFO] Résultats sauvegardés dans : {paths[0]}")

    if args.mode == 'compare':
        compare.run(test_loader.dataset, args, paths)
        print("\n=== Exécution Terminée ===")
        return

    if args.strategy == 'unrolling':

        model_key = args.model.strip().lower()
        if model_key not in NET_ARCHITECTURES:
            raise ValueError(f"Architecture '{model_key}' introuvable.")
        
        ModelClass = NET_ARCHITECTURES[model_key]
        
        if model_key == 'p3mg':
            mlp_hidden = None
            if args.mlp_hidden:
                mlp_hidden = [int(v.strip()) for v in str(args.mlp_hidden).split(',') if v.strip()]
            model = ModelClass(num_layers=args.num_layers, num_pd_layers=args.num_pd_layers,
                                mlp_hidden=mlp_hidden, use_fc_lambda=bool(args.use_fc_lambda))
        elif model_key == 'hq':
            model = ModelClass(num_layers=args.num_layers, num_pd_layers=args.num_pd_layers)

        elif model_key in ['fcae', 'fctn', 'fcun', 'resu']:

            sample_batch = next(iter(train_loader))
            xt_s, y_s = sample_batch[0], sample_batch[1]
            N_dim, M_dim = xt_s.shape[1], y_s.shape[1]
            model = ModelClass(N_dim=N_dim, M_dim=M_dim)
        else:
            model = ModelClass(num_layers=args.num_layers)

            
        model = model.to(args.device).double()
        print(f"[INFO] Modèle {model_key.upper()} instancié.")

        if args.mode in ['train', 'full']:
            network.train(model, train_loader, val_loader, args, paths)
        
        if args.mode in ['test', 'full']:
            if args.checkpoint:
                ckpt_to_load = args.checkpoint
            elif args.mode == 'test':
                # Mode test seul : le dossier de run courant vient d'etre cree et
                # ne contient donc aucun poids. On recherche le run le plus recent.
                ckpt_to_load = find_latest_checkpoint(model_key, args.strategy, data_folder=(args.run_tag or args.data_folder), current_base_dir=paths[0], run_group=getattr(args, 'run_group', None))



                if ckpt_to_load:
                    print(f"[INFO] Aucun --checkpoint fourni. Utilisation du poids le plus récent : {ckpt_to_load}")
                else:
                    print("[AVERTISSEMENT] Aucun checkpoint existant trouvé dans 'runs/'.")
                    ckpt_to_load = os.path.join(paths[1], 'best_model.pt')
            else:
                ckpt_to_load = os.path.join(paths[1], 'best_model.pt')
            network.test(model, test_loader, args, paths, checkpoint_path=ckpt_to_load)


    elif args.strategy == 'random_search':
        if args.mode in ['train', 'full']:
            random_search.train(test_loader, args, paths)
        if args.mode in ['test', 'full']:
            random_search.test(test_loader, args, paths)

    print("\n=== Exécution Terminée ===")

if __name__ == "__main__":
    main()