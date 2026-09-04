import os
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def get_next_generation_dir(base_dir):
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


def generate_dosy_dataset(num_samples, N=800, M=100, noise_std=0.01, seed=42, number=2,
                           skew=False, beta=2.0, beta_min=1.5, beta_max=3.0):
    """Génère un dataset RMN DOSY simulé.

    Le paramètre de forme `beta` de chaque gaussienne généralisée est fixé à
    la valeur `beta` si `skew` est désactivé, ou tiré uniformément dans
    l'intervalle `[beta_min, beta_max]` (par pic, à chaque échantillon) si
    `skew` est activé. C'est ce paramètre `beta` qui contrôle l'asymétrie
    ("skew") de la forme des pics.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    Dmin, Dmax = 1.0, 1000.0
    D = (np.log(Dmin) - np.log(Dmax)) / (N - 1)
    
    n = np.arange(1, N + 1)
    T = Dmin * np.exp(-D * n)
    t = np.linspace(0.0, 1.5, M)
    Hmat = np.exp(-np.outer(t, T))

    X_true_list, Y_list = [], []

    for _ in range(num_samples):
        # Paramètres pour le pic 1
        A_1 = np.random.uniform(0.6, 0.8)
        mu_1 = np.random.uniform(0.45, 0.55) * N
        sigma_1 = np.random.uniform(8.0, 12.0)
        beta_1 = np.random.uniform(0.75, 1.8) if np.random.rand() < 1.3 / 3.1 else np.random.uniform(2.2, 3.5)


        # Paramètres pour le pic 2
        A_2 = np.random.uniform(0.9, 1.1)
        mu_2 = np.random.uniform(0.15, 0.25) * N
        sigma_2 = np.random.uniform(4.0, 6.0)
        beta_2 = np.random.uniform(0.75, 1.8) if np.random.rand() < 1.3 / 3.1 else np.random.uniform(2.2, 3.5)


        # Génération des signaux de base
        x1 = A_1 * np.exp(-0.5 * (np.abs((n - mu_1) / sigma_1)) ** beta_1)
        x2 = A_2 * np.exp(-0.5 * (np.abs((n - mu_2) / sigma_2)) ** beta_2)

        xtrue = x1 + x2

        # Conditionnel pour l'ajout d'un troisième pic
        if number == 3:
            A_3 = np.random.uniform(0.4, 0.6)
            mu_3 = np.random.uniform(0.75, 0.85) * N
            sigma_3 = np.random.uniform(16.0, 24.0)
            beta_3 = np.random.uniform(0.75, 1.8) if np.random.rand() < 1.3 / 3.1 else np.random.uniform(2.2, 3.5)


            x3 = A_3 * np.exp(-0.5 * (np.abs((n - mu_3) / sigma_3)) ** beta_3)
            xtrue += x3

        xtrue /= np.sum(xtrue)

        y = (Hmat @ xtrue) + (noise_std * np.random.randn(M))

        X_true_list.append(xtrue)
        Y_list.append(y)

    return (
        torch.tensor(np.array(X_true_list), dtype=torch.float64),
        torch.tensor(np.array(Y_list), dtype=torch.float64),
        torch.tensor(Hmat, dtype=torch.float64)
    )


def verify_and_plot_dataset(dataset_dir):
    print("\n[VERIFICATION] Analyse des datasets générés...")
    for filename in ["train.pt", "val.pt", "test.pt"]:
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            data = torch.load(filepath, map_location='cpu', weights_only=True)
            print(f" - {filename:<10} : X_true shape = {list(data['xtrue'].shape)}, Y shape = {list(data['yblur'].shape)}")
        else:
            print(f" - [ERREUR] {filename} est introuvable.")

    train_path = os.path.join(dataset_dir, "train.pt")
    if os.path.exists(train_path):
        data = torch.load(train_path, map_location='cpu', weights_only=True)
        idx = np.random.randint(0, data['xtrue'].shape[0])
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(data['xtrue'][idx].numpy(), color='blue', linewidth=1.5)
        plt.title(f"True signal (X_true) - Index {idx}")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(data['yblur'][idx].numpy(), color='red', linewidth=1.5)
        plt.title(f"Observation (y) - Index {idx}")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(dataset_dir, "dataset_sample_check.png")
        plt.savefig(save_path)
        plt.close()
        print(f"\n[VISUALISATION] Echantillon sauvegardé sous : {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de dataset RMN DOSY")
    parser.add_argument("--out_dir", type=str, default="./Dataset")
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--total_samples", type=int, default=1000)
    
    # Remplacement de num_peaks par number pour correspondre au script Bash
    parser.add_argument("--number", type=int, default=2, choices=[2, 3])
    
    # --skew active la variation aléatoire du paramètre de forme beta
    # (entre --beta_min et --beta_max) pour chaque pic, à la place de la
    # valeur fixe --beta.
    parser.add_argument("--skew", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--beta_min", type=float, default=1.5)
    parser.add_argument("--beta_max", type=float, default=3.0)
    
    args = parser.parse_args()

    generation_dir = get_next_generation_dir(args.out_dir)
    print(f"[INFO] Dossier de génération : {generation_dir}")
    print(f"[INFO] Génération de {args.total_samples} échantillons avec {args.number} pics (Bruit: {args.noise})...")

    X_all, Y_all, Hmat = generate_dosy_dataset(
        num_samples=args.total_samples,
        noise_std=args.noise,
        number=args.number,
        skew=args.skew,
        beta=args.beta,
        beta_min=args.beta_min,
        beta_max=args.beta_max
    )

    num_train = int(0.8 * args.total_samples)
    num_val = int(0.1 * args.total_samples)
    
    splits = {
        "dataset.pt": (X_all, Y_all),
        "train.pt": (X_all[:num_train], Y_all[:num_train]),
        "val.pt": (X_all[num_train:num_train+num_val], Y_all[num_train:num_train+num_val]),
        "test.pt": (X_all[num_train+num_val:], Y_all[num_train+num_val:])
    }

    for filename, (X, Y) in splits.items():
        filepath = os.path.join(generation_dir, filename)
        torch.save({'xtrue': X, 'yblur': Y, 'Hmat': Hmat}, filepath)

    print(f"[SUCCÈS] Sauvegarde terminée dans {generation_dir}.")
    verify_and_plot_dataset(generation_dir)