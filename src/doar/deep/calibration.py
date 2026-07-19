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
