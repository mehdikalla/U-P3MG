"""
Visualiseur interactif pour parcourir les signaux d'un dataset .pt.

Ce script charge un fichier .pt (contenant les clés 'xtrue' et 'yblur', et
optionnellement 'Hmat') et affiche une interface matplotlib permettant de
naviguer entre les échantillons du dataset via un slider et des boutons
Précédent/Suivant, ou directement au clavier (flèches gauche/droite).

Usage:
    python Dataset/visualize_dataset.py --path Dataset/train.pt
    python Dataset/visualize_dataset.py --path Dataset/test.pt --index 12
"""

import argparse
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def load_dataset(path: str):
    """
    Charge un fichier .pt et retourne les tenseurs xtrue et yblur sous forme
    de tableaux numpy, ainsi que le nombre total d'échantillons.

    Args:
        path: Chemin vers le fichier .pt à charger.

    Returns:
        Tuple (X, Y, num_samples) où X et Y sont des tableaux numpy de forme
        (K, N) et (K, M) respectivement.
    """
    data = tc.load(path, map_location="cpu")

    if "xtrue" not in data or "yblur" not in data:
        raise KeyError(
            f"Le fichier {path} doit contenir les clés 'xtrue' et 'yblur'. "
            f"Clés trouvées : {list(data.keys())}"
        )

    X = data["xtrue"]
    Y = data["yblur"]

    if X.shape[0] != Y.shape[0]:
        X = X.T
        Y = Y.T

    assert X.shape[0] == Y.shape[0], (
        f"Incohérence du nombre d'échantillons : X={X.shape[0]} vs Y={Y.shape[0]}"
    )

    return X.numpy(), Y.numpy(), X.shape[0]


class DatasetVisualizer:
    """
    Encapsule l'état et les callbacks de l'interface de visualisation
    interactive d'un dataset de signaux (xtrue / yblur).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, dataset_path: str):
        self.X = X
        self.Y = Y
        self.num_samples = X.shape[0]
        self.dataset_path = dataset_path
        self.current_index = 0

        self.fig, (self.ax_true, self.ax_obs) = plt.subplots(1, 2, figsize=(13, 5))
        plt.subplots_adjust(bottom=0.22)

        (self.line_true,) = self.ax_true.plot([], [], color="tab:blue", linewidth=1.5)
        (self.line_obs,) = self.ax_obs.plot([], [], color="tab:red", linewidth=1.5)

        self.ax_true.set_title("True signal (xtrue)")
        self.ax_obs.set_title("Observation (yblur)")
        self.ax_true.grid(True, linestyle="--", alpha=0.6)
        self.ax_obs.grid(True, linestyle="--", alpha=0.6)

        ax_slider = plt.axes([0.15, 0.08, 0.55, 0.04])
        self.slider = Slider(
            ax_slider, "Index", 0, self.num_samples - 1,
            valinit=0, valstep=1
        )
        self.slider.on_changed(self._on_slider_change)

        ax_prev = plt.axes([0.75, 0.08, 0.08, 0.05])
        ax_next = plt.axes([0.85, 0.08, 0.08, 0.05])
        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_next = Button(ax_next, "Next >")
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        self._update_plot(0)

    def _update_plot(self, idx: int):
        """
        Met à jour les deux sous-graphes pour afficher le signal correspondant
        à l'index donné, et rafraîchit les axes/limites automatiquement.
        """
        idx = int(idx)
        self.current_index = idx

        x_sample = self.X[idx]
        y_sample = self.Y[idx]

        self.line_true.set_data(np.arange(len(x_sample)), x_sample)
        self.line_obs.set_data(np.arange(len(y_sample)), y_sample)

        self.ax_true.relim()
        self.ax_true.autoscale_view()
        self.ax_obs.relim()
        self.ax_obs.autoscale_view()

        self.ax_true.set_title(f"True signal (xtrue) - Index {idx}")
        self.ax_obs.set_title(f"Observation (yblur) - Index {idx}")

        self.fig.suptitle(
            f"Dataset : {self.dataset_path} "
            f"({idx + 1}/{self.num_samples})"
        )
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, val):
        self._update_plot(val)

    def _on_prev(self, event):
        new_idx = max(0, self.current_index - 1)
        self.slider.set_val(new_idx)

    def _on_next(self, event):
        new_idx = min(self.num_samples - 1, self.current_index + 1)
        self.slider.set_val(new_idx)

    def _on_key_press(self, event):
        if event.key == "left":
            self._on_prev(event)
        elif event.key == "right":
            self._on_next(event)

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualiseur interactif pour un dataset .pt (xtrue / yblur)."
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Chemin vers le fichier .pt à visualiser (ex: Dataset/train.pt)"
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Index initial du signal à afficher (défaut: 0)"
    )
    args = parser.parse_args()

    X, Y, num_samples = load_dataset(args.path)
    print(f"[INFO] Dataset chargé : {args.path} ({num_samples} échantillons)")

    visualizer = DatasetVisualizer(X, Y, args.path)
    start_idx = min(max(args.index, 0), num_samples - 1)
    visualizer.slider.set_val(start_idx)
    visualizer.show()


if __name__ == "__main__":
    main()