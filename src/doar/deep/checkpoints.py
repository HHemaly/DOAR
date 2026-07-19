from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    import torch
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
