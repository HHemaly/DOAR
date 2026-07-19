from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from ..dataset import CLASSES
from .augmentations import build_transforms
from .registry import build_model


def predict_image(image: str, checkpoint: str, device: str = "auto") -> dict:
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    payload = torch.load(checkpoint, map_location=selected, weights_only=False)
    if tuple(payload["classes"]) != CLASSES:
        raise ValueError("Checkpoint class mapping mismatch")
    model = build_model(payload["model_name"], len(CLASSES), pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.to(selected).eval()
    tensor = build_transforms(payload["image_size"], False)(Image.open(image).convert("RGB"))
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor[None].to(selected)), 1)[0].cpu().numpy()
    order = np.argsort(probabilities)[::-1]
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    return {
        "status": "available",
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
        "top_class": CLASSES[int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "top_two_margin": float(probabilities[order[0]] - probabilities[order[1]]),
        "entropy": entropy,
        "uncertainty": "high" if probabilities[order[0]] < .5 else "moderate",
        "calibration_status": payload.get("calibration_status", "uncalibrated"),
        "model_name": payload["model_name"],
        "model_family": "deep_image",
        "checkpoint": Path(checkpoint).name,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "model_version": payload.get("model_version", "doar_deep_v1"),
        "preprocessing_version": payload["preprocessing_version"],
        "evidence_id": "ev_emotion_prediction",
    }
