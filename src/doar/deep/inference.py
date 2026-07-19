from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..dataset import CLASSES
from .augmentations import build_transforms
from .registry import build_model


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


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
    # Item 6: fail if the checkpoint's preprocessing is incompatible with the
    # transform we are about to apply.
    from .preprocessing import (resolve_preprocessing, assert_preprocessing_compatible,
                                build_eval_transform)
    # A5: prefer the checkpoint's stored preprocessing_spec so the exact transform
    # used at training is reproduced; validate the stored hash where present.
    applied_spec = payload.get("preprocessing_spec") or resolve_preprocessing(
        payload["model_name"], payload["image_size"])
    assert_preprocessing_compatible(payload.get("preprocessing_hash"), applied_spec)
    transform = build_eval_transform(applied_spec) or build_transforms(payload["image_size"], False)
    tensor = transform(Image.open(image).convert("RGB"))
    # Apply validation-fitted temperature scaling when available (D1). T=1.0 is
    # a no-op, so uncalibrated checkpoints behave exactly as before.
    temperature = float(payload.get("temperature", 1.0))
    with torch.no_grad():
        logits = model(tensor[None].to(selected))[0].cpu().numpy()
    raw_probabilities = _softmax(logits, 1.0)
    probabilities = _softmax(logits, temperature)
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
        "temperature": temperature,
        "raw_probabilities": {name: float(raw_probabilities[i]) for i, name in enumerate(CLASSES)},
        "calibration_status": payload.get("calibration_status", "uncalibrated"),
        "model_name": payload["model_name"],
        "model_family": "deep_image",
        "checkpoint": Path(checkpoint).name,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "model_version": payload.get("model_version", "doar_deep_v1"),
        "preprocessing_version": payload["preprocessing_version"],
        "evidence_id": "ev_emotion_prediction",
    }
