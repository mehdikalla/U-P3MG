import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_dosy_dataset(num_samples, N=800, M=100, noise_std=0.01, randomize_peaks=True, seed=42):
    """
    Génère un dataset de signaux RMN DOSY inspiré du code MATLAB.
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
    
    for _ in range(num_samples):
        if randomize_peaks:
            pos1 = np.random.uniform(0.4, 0.6) * N
            pos2 = np.random.uniform(0.1, 0.3) * N
            w1   = np.random.uniform(8, 12)
            w2   = np.random.uniform(3, 7)
            amp1 = np.random.uniform(0.5, 0.9)
        else:
            pos1 = 0.5 * N
            pos2 = 0.2 * N
            w1   = 10.0
            w2   = 5.0
            amp1 = 0.7
            
        x1 = amp1 * np.exp(-0.5 * (((n - pos1) / w1) ** 2))
        x2 = 1.0 * np.exp(-0.5 * (((n - pos2) / w2) ** 2))
        
        xtrue = (x1 + x2) / np.sum(x1 + x2)
        
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
            X = data['X_true']
            Y = data['Y']
            print(f" - {filename:<10} : X_true shape = {list(X.shape)}, Y shape = {list(Y.shape)}")
        else:
            print(f" - [ERREUR] {filename} est introuvable.")

    # Chargement d'un échantillon depuis train.pt pour visualisation
    train_path = os.path.join(dataset_dir, "train.pt")
    if os.path.exists(train_path):
        data = torch.load(train_path, map_location='cpu')
        X = data['X_true']
        Y = data['Y']
        
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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Génération d'un dataset de {args.total_samples} échantillons (Bruit: {args.noise})...")

    # 1. Génération du dataset complet
    X_all, Y_all, Hmat = generate_dosy_dataset(
        num_samples=args.total_samples,
        noise_std=args.noise,
        randomize_peaks=True,
        seed=42
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
        "train.pt": (X_train, Y_train),
        "val.pt":   (X_val, Y_val),
        "test.pt":  (X_test, Y_test)
    }

    for filename, (X, Y) in splits.items():
        filepath = os.path.join(args.out_dir, filename)
        torch.save({
            'X_true': X,
            'Y': Y,
            'Hmat': Hmat
        }, filepath)
        
    print("[SUCCÈS] Sauvegarde terminée.")

    # 4. Vérification et Plot final
    verify_and_plot_dataset(args.out_dir)
