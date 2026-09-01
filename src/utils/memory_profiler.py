"""
Profiler de memoire pour les methodes d'entrainement.

Ce module fournit un gestionnaire de contexte `MemoryProfiler` capable de
mesurer, pour une methode d'entrainement donnee (unrolling ou
random_search), la consommation memoire CPU (via `tracemalloc`) ainsi que
la memoire GPU (via les statistiques CUDA de PyTorch), et de sauvegarder un
rapport lisible dans le dossier de logs du run.

Exemple d'utilisation :

    from src.utils.memory_profiler import MemoryProfiler

    with MemoryProfiler(path_logs, tag='train', device=args.device) as prof:
        ... boucle d'entrainement ...

    prof.report  # dictionnaire des metriques mesurees
"""

import os
import json
import time
import tracemalloc

import torch


class MemoryProfiler:
    """Mesure l'empreinte memoire CPU/GPU d'un bloc de code (entrainement).

    Attributes:
        path_logs (str): Dossier ou ecrire le rapport JSON/texte.
        tag (str): Nom du bloc profile (ex: 'train_p3mg_unrolling').
        device (str | torch.device): Device utilise (pour activer les
            statistiques CUDA si pertinent).
        report (dict): Rempli a la sortie du bloc avec les metriques
            mesurees (memoire CPU/GPU, duree).
    """

    def __init__(self, path_logs, tag="train", device="cpu", verbose=True):
        self.path_logs = path_logs
        self.tag = tag
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.verbose = verbose
        self.report = {}
        self._start_time = None

    def __enter__(self):
        self._is_cuda = (self.device.type == "cuda") and torch.cuda.is_available()

        # Memoire CPU : suivi fin via tracemalloc.
        tracemalloc.start()

        # Memoire GPU : reinitialise les compteurs de pic pour ce device.
        if self._is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.time() - self._start_time

        cpu_current, cpu_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.report = {
            "tag": self.tag,
            "duration_sec": duration,
            "cpu_current_MB": cpu_current / (1024 ** 2),
            "cpu_peak_MB": cpu_peak / (1024 ** 2),
        }

        if self._is_cuda:
            torch.cuda.synchronize(self.device)
            self.report.update({
                "gpu_allocated_MB": torch.cuda.memory_allocated(self.device) / (1024 ** 2),
                "gpu_peak_allocated_MB": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
                "gpu_reserved_MB": torch.cuda.memory_reserved(self.device) / (1024 ** 2),
                "gpu_peak_reserved_MB": torch.cuda.max_memory_reserved(self.device) / (1024 ** 2),
            })

        self._save_report()

        if self.verbose:
            self._print_summary()

        # Ne supprime pas l'exception eventuelle : on la laisse se propager.
        return False

    def _print_summary(self):
        r = self.report
        msg = (
            f"[MEM-PROFILE] {r['tag']} | Duree: {r['duration_sec']:.1f}s | "
            f"CPU peak: {r['cpu_peak_MB']:.2f} MB"
        )
        if "gpu_peak_allocated_MB" in r:
            msg += (
                f" | GPU peak alloc: {r['gpu_peak_allocated_MB']:.2f} MB "
                f"| GPU peak reserve: {r['gpu_peak_reserved_MB']:.2f} MB"
            )
        print(msg)

    def _save_report(self):
        if not self.path_logs:
            return
        try:
            os.makedirs(self.path_logs, exist_ok=True)
            json_path = os.path.join(self.path_logs, f"memory_profile_{self.tag}.json")
            with open(json_path, "w") as f:
                json.dump(self.report, f, indent=4)

            txt_path = os.path.join(self.path_logs, "memory_profile_summary.txt")
            with open(txt_path, "a") as f:
                f.write(self._format_table())
        except Exception as e:
            print(f"[WARN] Impossible de sauvegarder le profil memoire : {e}")

    def _format_table(self):
        r = self.report
        lines = [
            "\n+-----------------------------------------------+",
            f"|   PROFIL MEMOIRE ({r['tag']:<25}) |",
            "+---------------------------+---------------------+",
            f"| Duree (s)                 | {r['duration_sec']:<19.2f} |",
            f"| CPU courant (MB)          | {r['cpu_current_MB']:<19.2f} |",
            f"| CPU pic (MB)              | {r['cpu_peak_MB']:<19.2f} |",
        ]
        if "gpu_peak_allocated_MB" in r:
            lines += [
                f"| GPU alloue (MB)           | {r['gpu_allocated_MB']:<19.2f} |",
                f"| GPU pic alloue (MB)       | {r['gpu_peak_allocated_MB']:<19.2f} |",
                f"| GPU reserve (MB)          | {r['gpu_reserved_MB']:<19.2f} |",
                f"| GPU pic reserve (MB)      | {r['gpu_peak_reserved_MB']:<19.2f} |",
            ]
        lines.append("+---------------------------+---------------------+\n")
        return "\n".join(lines)
