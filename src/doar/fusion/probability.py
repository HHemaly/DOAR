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


# ---------------------------------------------------------------------------
# D2 — runnable late-fusion / stacking orchestration
# ---------------------------------------------------------------------------

def _macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    preds = probabilities.argmax(axis=1)
    classes = sorted(set(labels.tolist()))
    f1s = []
    for c in classes:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    p = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.maximum(p, 1e-12)).mean())


def logistic_probability_meta(
    train_meta: np.ndarray, train_labels: np.ndarray, valid_meta: np.ndarray,
) -> np.ndarray:
    """Stacking meta-learner over concatenated base probabilities.

    train_meta must be OUT-OF-FOLD base probabilities to avoid stacking leakage
    (validate_oof_folds enforces the fold discipline upstream). Requires sklearn.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - needs [ml]
        raise RuntimeError('Install ML support with: pip install -e ".[ml]"') from exc
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(train_meta, train_labels)
    return clf.predict_proba(valid_meta)


def run_late_fusion(
    valid_probability_sets: list[np.ndarray], valid_labels: np.ndarray,
    method: str = "validation_weighted_late_fusion", calibrated: bool = False,
    train_probability_sets: list[np.ndarray] | None = None,
    train_labels: np.ndarray | None = None,
) -> dict:
    """Fuse base-model probabilities and select on VALIDATION only (test never
    used). Returns fused probabilities, validation metrics, and ensemble
    uncertainty. This is the runnable RQ3 late-fusion / stacking arm (D2)."""
    labels = np.asarray(valid_labels, dtype=int)

    if method == "equal_late_fusion":
        result = equal_late_fusion(*valid_probability_sets)
        fused = result["probabilities"]
    elif method == "validation_weighted_late_fusion":
        if len(valid_probability_sets) != 2:
            raise ValueError("validation_weighted_late_fusion expects exactly 2 base models")
        result = validation_weighted_late_fusion(
            valid_probability_sets[0], valid_probability_sets[1], labels)
        fused = result["probabilities"]
    elif method == "logistic_probability_meta":
        if train_probability_sets is None or train_labels is None:
            raise ValueError("logistic_probability_meta requires OOF train probabilities + labels")
        train_meta = probability_meta_features(*train_probability_sets)
        valid_meta = probability_meta_features(*valid_probability_sets)
        fused = logistic_probability_meta(train_meta, np.asarray(train_labels, dtype=int), valid_meta)
        result = {"method": "logistic_probability_meta", "fit_split": "valid",
                  "input_type": "stacked_base_probabilities", "probabilities": fused}
    else:
        raise ValueError(f"Unknown late-fusion method: {method}")

    from ..uncertainty import summarize_ensemble
    per_sample_uncertainty = summarize_ensemble(
        [s.mean(axis=0).tolist() for s in valid_probability_sets], calibrated
    ) if valid_probability_sets else {"status": "unavailable"}

    return {
        "method": method,
        "selection_split": "valid",
        "test_used": False,
        "weights": result.get("weights"),
        "validation_macro_f1": _macro_f1(labels, fused),
        "validation_log_loss": _log_loss(labels, fused),
        "ensemble_uncertainty": per_sample_uncertainty,
        "fused_probabilities_shape": list(fused.shape),
    }
