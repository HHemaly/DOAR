from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    result = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        selected = (confidence >= lower) & (confidence < lower + 1 / bins)
        if selected.any():
            result += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(result)


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def fit_temperature(validation_logits: np.ndarray, validation_labels: np.ndarray) -> dict:
    """Grid-search temperature on validation negative log likelihood only."""
    candidates = np.geomspace(.25, 5.0, 200)
    losses = []
    for temperature in candidates:
        probabilities = softmax(validation_logits, float(temperature))
        losses.append(float(-np.log(np.maximum(
            probabilities[np.arange(len(validation_labels)), validation_labels], 1e-12
        )).mean()))
    index = int(np.argmin(losses))
    before = softmax(validation_logits)
    after = softmax(validation_logits, float(candidates[index]))
    return {
        "method": "temperature_scaling",
        "fit_split": "valid",
        "temperature": float(candidates[index]),
        "validation_nll_before": losses[int(np.argmin(np.abs(candidates - 1.0)))],
        "validation_nll_after": losses[index],
        "validation_ece_before": expected_calibration_error(validation_labels, before),
        "validation_ece_after": expected_calibration_error(validation_labels, after),
        "validation_brier_before": brier_score(validation_labels, before),
        "validation_brier_after": brier_score(validation_labels, after),
    }


def save_calibration(result: dict, output: str | Path) -> None:
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")


def collect_validation_logits(checkpoint: str | Path, dataset: str | Path,
                              device: str = "auto") -> tuple:
    """Run the checkpoint's model over the VALIDATION split only and return
    (logits, labels) as numpy arrays. Test split is never touched.
    torch is imported lazily so this module stays importable without it."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only with [deep]
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    from ..dataset import CLASSES
    from .registry import build_model
    from .datasets import build_loaders

    selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    payload = torch.load(checkpoint, map_location=selected, weights_only=False)
    if tuple(payload["classes"]) != CLASSES:
        raise ValueError("Checkpoint class mapping mismatch")
    model = build_model(payload["model_name"], len(CLASSES), pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.to(selected).eval()

    # build_loaders exposes train + valid ONLY (test is never loaded). Item 1:
    # use the checkpoint's stored preprocessing spec so the calibration loader
    # matches the transform the model was trained/evaluated with.
    _, valid_loader = build_loaders(
        dataset, payload["image_size"], batch_size=32, workers=0,
        augmentation="conservative", weighted_sampler=False,
        preprocessing_spec=payload.get("preprocessing_spec"),
    )
    logits_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in valid_loader:
            logits_all.append(model(images.to(selected)).cpu().numpy())
            labels_all.extend(labels.tolist())
    return np.concatenate(logits_all, axis=0), np.asarray(labels_all)


def calibrate_checkpoint(checkpoint: str | Path, dataset: str | Path,
                         output: str | Path, device: str = "auto") -> dict:
    """Fit temperature scaling on the VALIDATION split and persist it.

    Writes calibration.json to `output`, and stamps the checkpoint in place with
    `temperature` and `calibration_status="temperature_scaled"` so that
    predict_image applies it automatically. Never uses the test split.
    """
    import torch  # lazy

    logits, labels = collect_validation_logits(checkpoint, dataset, device)
    result = fit_temperature(logits, labels)

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_calibration(result, out_dir / "calibration.json")

    # Stamp the checkpoint so inference is calibrated by default.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["temperature"] = result["temperature"]
    payload["calibration_status"] = "temperature_scaled"
    payload["calibration"] = result
    torch.save(payload, checkpoint)

    result["checkpoint"] = str(checkpoint)
    result["applied_to_checkpoint"] = True
    return result
