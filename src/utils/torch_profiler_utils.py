"""
Profilage memoire/temps base sur `torch.profiler`, tel que documente dans
https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html.

Ce module fournit un gestionnaire de contexte `TorchOpProfiler` qui encapsule
`torch.profiler.profile` (activites CPU + CUDA, `profile_memory=True`) pour
mesurer, operateur par operateur, le temps et la memoire consommes par un
bloc de code (typiquement la boucle d'inference d'un modele/algorithme lors
du mode 'compare'). A la sortie du bloc :
  - un tableau texte trie par memoire (puis par temps) est sauvegarde dans
    le dossier de logs du run (`torch_profile_<tag>.txt`) ;
  - une trace Chrome (`torch_profile_<tag>.json`) est exportee, exploitable
    via `chrome://tracing` ou https://ui.perfetto.dev pour une inspection
    visuelle detaillee ;
  - un resume agrege (temps CPU/CUDA total, pic memoire CPU/CUDA) est
    accessible via l'attribut `summary`.

Exemple d'utilisation :

    from src.utils.torch_profiler_utils import TorchOpProfiler

    with TorchOpProfiler(path_logs, tag=f"{model_name}_{strategy}") as prof:
        for batch in loader:
            ... inference ...

    prof.summary  # dict des metriques agregees
"""

import os

import torch
from torch.profiler import profile, ProfilerActivity


class TorchOpProfiler:
    """Encapsule `torch.profiler.profile` pour mesurer temps/memoire par operateur.

    Attributes:
        path_logs (str): Dossier ou sauvegarder le rapport et la trace.
        tag (str): Nom du bloc profile (ex: 'p3mg_unrolling').
        row_limit (int): Nombre de lignes affichees/sauvegardees dans le
            tableau des operateurs les plus couteux.
        summary (dict): Rempli a la sortie du bloc avec les metriques
            agregees (temps CPU/CUDA total, pic memoire CPU/CUDA).
    """

    def __init__(self, path_logs, tag="model", device="cpu", row_limit=30, verbose=True):
        self.path_logs = path_logs
        self.tag = tag
        self.row_limit = row_limit
        self.verbose = verbose
        self.summary = {}

        device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
        self._use_cuda = (device_type == "cuda") and torch.cuda.is_available()

        activities = [ProfilerActivity.CPU]
        if self._use_cuda:
            activities.append(ProfilerActivity.CUDA)

        # record_shapes=False : torch.profiler conserve sinon un enregistrement
        # non agrege par appel d'operateur (avec la forme de chaque tenseur
        # d'entree). Sur une boucle couvrant l'integralite du jeu de test
        # (des dizaines de signaux x des dizaines de couches internes), le
        # volume d'evenements non agreges accumules en RAM hote avant
        # key_averages() sature la memoire systeme et provoque un 'Killed'
        # (OOM). Desactiver record_shapes force l'agregation par nom
        # d'operateur au fil de l'eau, sans perte sur les metriques agregees
        # (temps/memoire self CPU/CUDA) exploitees par compare.py.
        self._profiler = profile(
            activities=activities,
            profile_memory=True,
            record_shapes=False,
        )

    def __enter__(self):
        self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._profiler.__exit__(exc_type, exc_value, traceback)

        key_averages = self._profiler.key_averages()
        self.summary = self._build_summary(key_averages)
        self._save_report(key_averages)

        if self.verbose:
            self._print_summary()

        # Ne supprime pas l'exception eventuelle : on la laisse se propager.
        return False

    def _build_summary(self, key_averages):
        self_cpu_time_total = sum(e.self_cpu_time_total for e in key_averages)
        self_cuda_time_total = sum(getattr(e, "self_device_time_total", 0) for e in key_averages)
        cpu_mem_peak = max((e.cpu_memory_usage for e in key_averages), default=0)
        cuda_mem_peak = max((getattr(e, "device_memory_usage", 0) for e in key_averages), default=0)
        cpu_mem_total = sum(max(e.cpu_memory_usage, 0) for e in key_averages)
        cuda_mem_total = sum(max(getattr(e, "device_memory_usage", 0), 0) for e in key_averages)

        return {
            "tag": self.tag,
            "self_cpu_time_total_ms": self_cpu_time_total / 1e3,
            "self_cuda_time_total_ms": self_cuda_time_total / 1e3,
            "cpu_memory_peak_MB": cpu_mem_peak / (1024 ** 2),
            "cuda_memory_peak_MB": cuda_mem_peak / (1024 ** 2),
            "cpu_memory_total_alloc_MB": cpu_mem_total / (1024 ** 2),
            "cuda_memory_total_alloc_MB": cuda_mem_total / (1024 ** 2),
        }

    def _print_summary(self):
        s = self.summary
        msg = (
            f"[TORCH-PROFILE] {s['tag']} | CPU self time: {s['self_cpu_time_total_ms']:.2f} ms "
            f"| CPU mem peak: {s['cpu_memory_peak_MB']:.2f} MB"
        )
        if self._use_cuda:
            msg += (
                f" | CUDA self time: {s['self_cuda_time_total_ms']:.2f} ms "
                f"| CUDA mem peak: {s['cuda_memory_peak_MB']:.2f} MB"
            )
        print(msg)

    def _save_report(self, key_averages):
        if not self.path_logs:
            return
        try:
            os.makedirs(self.path_logs, exist_ok=True)

            sort_key = "self_cuda_memory_usage" if self._use_cuda else "self_cpu_memory_usage"
            table_str = key_averages.table(sort_by=sort_key, row_limit=self.row_limit)

            txt_path = os.path.join(self.path_logs, f"torch_profile_{self.tag}.txt")
            with open(txt_path, "w") as f:
                f.write(f"=== Profil torch.profiler : {self.tag} ===\n\n")
                f.write(table_str)
                f.write("\n\n=== Resume agrege ===\n")
                for k, v in self.summary.items():
                    f.write(f"{k}: {v}\n")

            trace_path = os.path.join(self.path_logs, f"torch_profile_{self.tag}.json")
            self._profiler.export_chrome_trace(trace_path)
        except Exception as e:
            print(f"[WARN] Impossible de sauvegarder le profil torch.profiler ({self.tag}) : {e}")
