import os
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def get_next_generation_dir(base_dir):
    """
    Détermine le prochain dossier de génération 'data_i' à créer dans base_dir.

    Parcourt les sous-dossiers existants nommés 'data_<entier>' et retourne
    le chemin du dossier suivant (index maximal + 1), créé s'il n'existe pas.
    """
    os.makedirs(base_dir, exist_ok=True)
    pattern = re.compile(r"^data_(\d+)$")
    existing_indices = []
    for entry in os.listdir(base_dir):
        match = pattern.match(entry)
        if match and os.path.isdir(os.path.join(base_dir, entry)):
            existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=-1) + 1
    generation_dir = os.path.join(base_dir, f"data_{next_index}")
    os.makedirs(generation_dir, exist_ok=True)
    return generation_dir


def _generalized_gaussian_peak(n, pos, width, beta, skew):
    """
    Calcule un pic gaussien généralisé, éventuellement asymétrique.

    Le paramètre `beta` (forme) généralise la gaussienne classique
    (beta=2 correspond à une gaussienne standard). Le paramètre `skew`
    introduit une asymétrie en utilisant une largeur différente de
    part et d'autre de la position du pic : une valeur positive
    étire le côté droit, une valeur négative étire le côté gauche.

    Args:
        n: Vecteur des indices d'échantillonnage.
        pos: Position du pic.
        width: Largeur de base du pic.
        beta: Paramètre de forme de la gaussienne généralisée.
        skew: Paramètre d'asymétrie, dans (-1, 1).

    Returns:
        Vecteur numpy représentant le pic généré.
    """
    diff = n - pos
    width_left = width * (1.0 + skew)
    width_right = width * (1.0 - skew)
    width_left = max(width_left, 1e-6)
    width_right = max(width_right, 1e-6)
    w = np.where(diff < 0, width_left, width_right)
    return np.exp(-0.5 * (np.abs(diff) / w) ** beta)


def generate_dosy_dataset(
    num_samples,
    N=800,
    M=100,
    noise_std=0.01,
    randomize_peaks=True,
    seed=42,
    skew=False,
    number=2,
    beta=2.0,
    skew_range=(-1, 1),
):
    """
    Génère un dataset de signaux RMN DOSY inspiré du code MATLAB.

    Args:
        num_samples: Nombre d'échantillons à générer.
        N: Dimension du signal xtrue.
        M: Dimension de l'observation y.
        noise_std: Écart-type du bruit gaussien additif.
        randomize_peaks: Si True, tire aléatoirement position/largeur/amplitude des pics.
        seed: Graine aléatoire pour la reproductibilité.
        skew: Si True, utilise une gaussienne généralisée asymétrique pour
            générer chaque pic (troisième paramètre de forme/asymétrie).
            Si False, un pic gaussien standard (symétrique) est utilisé.
        number: Nombre de gaussiennes fondamentales sommées pour former
            chaque signal xtrue (défaut 2).
        beta: Paramètre de forme de la gaussienne généralisée, utilisé
            uniquement si `skew` est True.
        skew_range: Intervalle (min, max) dans lequel le paramètre
            d'asymétrie de chaque pic est tiré aléatoirement lorsque
            `skew` est True et `randomize_peaks` est True.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Construction de la matrice DOSY (Hmat)
    Dmin = 1.0
    Dmax = 1000.0
    D = (np.log(Dmin) - np.log(Dmax)) / (N - 1)
    
    n = np.arange(1, N + 1)
    T = Dmin * np.exp(-D * n)
    
    tmin = 0.0
    tmax = 1.5
    t = np.linspace(tmin, tmax, M)
    
    Hmat = np.exp(-np.outer(t, T))

    # 2. Génération des signaux (xtrue) et observations (y)
    X_true_list = []
    Y_list = []

    # Zone centrale autour de laquelle les pics sont répartis. La demi-largeur
    # de la zone (spread) est élargie proportionnellement au nombre de pics
    # afin de conserver un espacement suffisant même lorsque `number` augmente,
    # évitant ainsi que l'algorithme ne soit contraint de compresser les
    # positions (et donc de faire se chevaucher les pics visuellement).
    center = 0.35
    spread = min(0.45, 0.15 + 0.1 * max(0, number - 1))
    # Facteur de séparation minimale entre pics adjacents, exprimé en
    # multiples de la somme de leurs écarts-types (règle heuristique
    # garantissant un chevauchement négligeable entre gaussiennes voisines).
    # Augmenté de 5.0 à 9.0 pour imposer une distance minimale nettement
    # plus grande entre pics consécutifs.
    separation_factor = 9.0



    for _ in range(num_samples):
        xtrue_raw = np.zeros(N)

        amplitudes = (
            np.random.uniform(0.5, 1.0, size=number) if randomize_peaks
            else np.full(number, 1.0)
        )
        shapes = (
            np.random.uniform(1.5, 2.5, size=number) if randomize_peaks
            else np.full(number, beta if skew else 2.0)
        )
        skews = (
            np.random.uniform(skew_range[0], skew_range[1], size=number)
            if (skew and randomize_peaks)
            else np.zeros(number)
        )

        if randomize_peaks:
            sigmas = np.random.uniform(3, 8, size=number)
        else:
            sigmas = np.full(number, 5.0)

        if number == 1:
            positions = np.array([center])
        else:
            # Écarts minimaux (normalisés) requis entre pics consécutifs
            # pour éviter tout chevauchement significatif.
            gaps_min = separation_factor * (sigmas[:-1] + sigmas[1:]) / N
            total_min = gaps_min.sum()
            available = 2 * spread

            # Si l'espace disponible est insuffisant pour respecter les
            # écarts minimaux, on agrandit la zone disponible (available)
            # plutôt que de compresser les écarts minimaux requis. Cela
            # garantit que la distance minimale entre pics n'est JAMAIS
            # réduite, quitte à élargir la plage de positions possibles.
            if total_min > available * 0.95:
                available = total_min / 0.95

            remaining = available - total_min

            if randomize_peaks:
                extra_gaps = np.random.dirichlet(np.ones(number + 1)) * remaining
            else:
                extra_gaps = np.full(number + 1, remaining / (number + 1))

            positions = np.empty(number)
            pos = (center - spread) + extra_gaps[0]
            positions[0] = pos
            for i in range(1, number):
                pos += gaps_min[i - 1] + extra_gaps[i]
                positions[i] = pos


        for pos, sigma, amp, shape, sk in zip(positions, sigmas, amplitudes, shapes, skews):
            if skew:
                xtrue_raw += amp * _generalized_gaussian_peak(n, pos * N, sigma, shape, sk)
            else:
                xtrue_raw += amp * np.exp(-0.5 * np.abs((n - pos * N) / sigma) ** shape)

        xtrue = xtrue_raw / np.sum(xtrue_raw)


        xblured = Hmat @ xtrue


        noise = noise_std * np.random.randn(M)
        y = xblured + noise

        X_true_list.append(xtrue)
        Y_list.append(y)
        
    X_true_tensor = torch.tensor(np.array(X_true_list), dtype=torch.float64)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float64)
    Hmat_tensor = torch.tensor(Hmat, dtype=torch.float64)
    
    return X_true_tensor, Y_tensor, Hmat_tensor


def verify_and_plot_dataset(dataset_dir):
    """
    Vérifie les dimensions des datasets sauvegardés et trace un échantillon aléatoire.
    """
    print("\n[VERIFICATION] Analyse des datasets générés...")
    
    splits = ["train.pt", "val.pt", "test.pt"]
    
    for filename in splits:
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            data = torch.load(filepath, map_location='cpu')
            X = data['xtrue']
            Y = data['yblur']
            print(f" - {filename:<10} : X_true shape = {list(X.shape)}, Y shape = {list(Y.shape)}")
        else:
            print(f" - [ERREUR] {filename} est introuvable.")

    # Chargement d'un échantillon depuis train.pt pour visualisation
    train_path = os.path.join(dataset_dir, "train.pt")
    if os.path.exists(train_path):
        data = torch.load(train_path, map_location='cpu')
        X = data['xtrue']
        Y = data['yblur']
        
        # Sélection aléatoire
        idx = np.random.randint(0, X.shape[0])
        x_sample = X[idx].numpy()
        y_sample = Y[idx].numpy()
        
        # Création de la figure
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_sample, color='blue', linewidth=1.5)
        plt.title(f"Signal Vrai (X_true) - Index {idx}")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(y_sample, color='red', linewidth=1.5)
        plt.title(f"Observation bruitée (Y) - Index {idx}")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(dataset_dir, "dataset_sample_check.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"\n[VISUALISATION] Echantillon d'illustration sauvegardé sous : {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de dataset RMN DOSY")
    parser.add_argument("--out_dir", type=str, default="./Dataset", help="Dossier de sauvegarde")
    parser.add_argument("--noise", type=float, default=0.01, help="Écart-type du bruit gaussien")
    parser.add_argument("--total_samples", type=int, default=1000, help="Nombre total d'échantillons à générer")
    parser.add_argument("--skew", action="store_true", default=False,
                         help="Utilise une gaussienne généralisée asymétrique pour générer xtrue")
    parser.add_argument("--number", type=int, default=2,
                         help="Nombre de gaussiennes fondamentales sommées pour générer chaque signal")
    parser.add_argument("--beta", type=float, default=2.0,
                         help="Paramètre de forme de la gaussienne généralisée (utilisé si --skew)")
    args = parser.parse_args()


    generation_dir = get_next_generation_dir(args.out_dir)
    print(f"[INFO] Dossier de génération : {generation_dir}")

    print(f"[INFO] Génération d'un dataset de {args.total_samples} échantillons (Bruit: {args.noise})...")


    # 1. Génération du dataset complet
    X_all, Y_all, Hmat = generate_dosy_dataset(
        num_samples=args.total_samples,
        noise_std=args.noise,
        randomize_peaks=True,
        seed=42,
        skew=args.skew,
        number=args.number,
        beta=args.beta,
    )


    # 2. Découpage 80/10/10
    num_train = int(0.8 * args.total_samples)
    num_val   = int(0.1 * args.total_samples)
    
    X_train, Y_train = X_all[:num_train], Y_all[:num_train]
    X_val, Y_val     = X_all[num_train:num_train+num_val], Y_all[num_train:num_train+num_val]
    X_test, Y_test   = X_all[num_train+num_val:], Y_all[num_train+num_val:]

    print("[INFO] Découpage effectué : 80% Train, 10% Val, 10% Test.")

    # 3. Sauvegarde
    splits = {
        "dataset.pt": (X_all, Y_all),
        "train.pt": (X_train, Y_train),
        "val.pt":   (X_val, Y_val),
        "test.pt":  (X_test, Y_test)
    }

    for filename, (X, Y) in splits.items():
        filepath = os.path.join(generation_dir, filename)
        torch.save({
            'xtrue': X,
            'yblur': Y,
            'Hmat': Hmat
        }, filepath)
        
    print(f"[SUCCÈS] Sauvegarde terminée dans {generation_dir}.")


    # 4. Vérification et Plot final
    verify_and_plot_dataset(generation_dir)

