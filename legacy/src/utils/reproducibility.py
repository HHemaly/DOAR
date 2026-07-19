"""
reproducibility.py — deterministic seeds and an environment manifest.

Every experiment writes a manifest so results can be reproduced:
Python version, key library versions, device, GPU name, random seed,
git commit hash, and timestamp (passed in — never generated internally
so that a re-run with the same inputs is byte-identical).
"""

from __future__ import annotations
import os
import sys
import json
import platform
import subprocess
from pathlib import Path

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and (if present) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cudnn — slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _lib_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not_installed"


def device_info() -> dict:
    info = {"device": "cpu", "gpu_name": None, "cuda_available": False}
    try:
        import torch
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def build_manifest(seed: int, timestamp: str, extra: dict | None = None) -> dict:
    """
    Build a reproducibility manifest.

    timestamp must be passed in by the caller (do not call datetime here so
    the function stays pure and re-runs are identical given the same inputs).
    """
    manifest = {
        "timestamp":       timestamp,
        "python_version":  sys.version.split()[0],
        "platform":        platform.platform(),
        "seed":            seed,
        "git_commit":      _git_commit(),
        "library_versions": {
            "numpy":       _lib_version("numpy"),
            "opencv":      _lib_version("cv2"),
            "torch":       _lib_version("torch"),
            "torchvision": _lib_version("torchvision"),
            "sklearn":     _lib_version("sklearn"),
            "matplotlib":  _lib_version("matplotlib"),
            "PIL":         _lib_version("PIL"),
        },
        **device_info(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(manifest: dict, out_dir: str, filename: str = "reproducibility.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path
