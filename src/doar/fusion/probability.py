from __future__ import annotations

import numpy as np


def _validate(*sets: np.ndarray) -> list[np.ndarray]:
    values = [np.asarray(item, dtype=float) for item in sets]
    if not values or any(item.ndim != 2 for item in values):
        raise ValueError("Probability inputs must be non-empty 2D arrays")
    if len({item.shape for item in values}) != 1:
        raise ValueError("Base probability arrays must have identical shapes")
    if any(np.any(item < 0) or not np.isfinite(item).all() for item in values):
        raise ValueError("Invalid base probabilities")
    return [item / item.sum(axis=1, keepdims=True) for item in values]


def equal_late_fusion(*probability_sets: np.ndarray) -> dict:
    values = _validate(*probability_sets)
    return {
        "method": "equal_late_fusion",
        "weights": [1 / len(values)] * len(values),
        "probabilities": np.mean(values, axis=0),
        "input_type": "base_model_probabilities",
    }


def validation_weighted_late_fusion(
    objective_probabilities: np.ndarray, deep_probabilities: np.ndarray,
    validation_labels: np.ndarray,
) -> dict:
    objective, deep = _validate(objective_probabilities, deep_probabilities)
    labels = np.asarray(validation_labels, dtype=int)
    candidates = np.linspace(0, 1, 101)
    losses = []
    for objective_weight in candidates:
        fused = objective_weight * objective + (1 - objective_weight) * deep
        losses.append(float(-np.log(np.maximum(
            fused[np.arange(len(labels)), labels], 1e-12)).mean()))
    index = int(np.argmin(losses))
    weight = float(candidates[index])
    return {
        "method": "validation_weighted_late_fusion",
        "fit_split": "valid",
        "weights": [weight, 1 - weight],
        "validation_log_loss": losses[index],
        "probabilities": weight * objective + (1 - weight) * deep,
        "input_type": "base_model_probabilities",
    }


def probability_meta_features(*probability_sets: np.ndarray) -> np.ndarray:
    values = _validate(*probability_sets)
    return np.concatenate(values, axis=1)


def validate_oof_folds(sample_ids: list[str], fold_ids: list[int]) -> None:
    if len(sample_ids) != len(fold_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("OOF metadata must contain one unique fold assignment per sample")
    if len(set(fold_ids)) < 2:
        raise ValueError("Proper stacking requires at least two out-of-fold partitions")
