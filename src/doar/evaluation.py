"""
evaluation.py — the shared metrics + prediction-format spine (Items 3 & 8).

One common metrics function and one common prediction/probability format used by
EVERY model family (objective-feature, deep image, generic/fine-tuned embeddings,
early fusion, late fusion, stacking). Core metrics are pure-numpy so they run on
CPU without sklearn; ROC-AUC / PR-AUC use sklearn when present, else report None.

All predictions and fusion inputs align by `sample_id`, never row order.

Common metrics:
  accuracy, macro-F1, weighted-F1, balanced accuracy, per-class precision/recall/
  F1/support, confusion matrix, normalized confusion matrix, log loss, multiclass
  Brier score, ECE, macro OVR ROC-AUC (optional), macro PR-AUC (optional).
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

from .dataset import CLASSES

CLASS_ORDER = list(CLASSES)


# ---------------------------------------------------------------------------
# Metrics (pure numpy core)
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    total = len(y_true)
    value = 0.0
    for low in np.linspace(0, 1, bins, endpoint=False):
        sel = (confidence >= low) & (confidence < low + 1 / bins)
        if sel.any():
            value += sel.sum() / total * abs(correct[sel].mean() - confidence[sel].mean())
    return float(value)


def _confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def compute_metrics(y_true, y_pred, y_proba, class_names=None) -> dict:
    """Full common metric set. y_proba rows must be ordered by class_names."""
    class_names = class_names or CLASS_ORDER
    n = len(class_names)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    proba = proba / proba.sum(axis=1, keepdims=True)

    cm = _confusion(y_true, y_pred, n)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {"precision": float(prec), "recall": float(rec),
                           "f1": float(f1), "support": support}
        precisions.append(prec); recalls.append(rec); f1s.append(f1); supports.append(support)

    total = len(y_true)
    accuracy = float((y_true == y_pred).mean()) if total else 0.0
    supports_arr = np.array(supports, dtype=float)
    weight = supports_arr / supports_arr.sum() if supports_arr.sum() else np.zeros(n)

    # log loss + Brier (numpy)
    onehot = np.eye(n)[y_true]
    log_loss = float(-np.log(np.maximum(proba[np.arange(total), y_true], 1e-12)).mean()) if total else 0.0
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1))) if total else 0.0

    metrics = {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.sum(np.array(f1s) * weight)),
        "balanced_accuracy": float(np.mean(recalls)),
        "log_loss": log_loss,
        "multiclass_brier": brier,
        "ece": expected_calibration_error(y_true, proba),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm.tolist(),
        "class_order": list(class_names),
        "n": total,
    }

    # Optional AUCs via sklearn.
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        metrics["macro_ovr_roc_auc"] = float(roc_auc_score(
            y_true, proba, labels=list(range(n)), multi_class="ovr", average="macro"))
        metrics["macro_pr_auc"] = float(average_precision_score(onehot, proba, average="macro"))
    except Exception as exc:
        metrics["macro_ovr_roc_auc"] = None
        metrics["macro_pr_auc"] = None
        metrics["auc_note"] = f"AUC unavailable: {exc}"
    return metrics


# ---------------------------------------------------------------------------
# Common prediction / probability export format (aligned by sample_id)
# ---------------------------------------------------------------------------

EXPORT_FIELDS = ["sample_id", "split", "true_label", "predicted_label",
                 "confidence", "fold_id"]


def save_probability_export(path, *, sample_ids, splits, y_true, proba,
                            model_id, checkpoint_hash, calibration_status,
                            class_order=None, fold_ids=None, raw_proba=None,
                            temperature=None) -> str:
    """Write a probability export (.json) in the common format (Item 8 / B1).

    `proba` is the probability actually used downstream (calibrated when a
    temperature is applied). `raw_proba` (optional) carries the pre-calibration
    probabilities. Per-sample: sample_id, split, true_label, probabilities,
    raw_probabilities, predicted_label, confidence, fold_id. Aligned by sample_id.
    """
    class_order = class_order or CLASS_ORDER
    proba = np.asarray(proba, dtype=float)
    raw = np.asarray(raw_proba, dtype=float) if raw_proba is not None else None
    n = len(sample_ids)
    if not (len(splits) == n == len(y_true) == proba.shape[0]):
        raise ValueError("save_probability_export: length mismatch across inputs")
    if raw is not None and raw.shape != proba.shape:
        raise ValueError("save_probability_export: raw_proba shape mismatch")
    if len(set(sample_ids)) != n:
        raise ValueError("save_probability_export: duplicate sample_id values")

    rows = []
    for i, sid in enumerate(sample_ids):
        pred = int(proba[i].argmax())
        row = {
            "sample_id": str(sid),
            "split": splits[i],
            "true_label": class_order[int(y_true[i])] if y_true[i] is not None else None,
            "probabilities": {c: float(proba[i][j]) for j, c in enumerate(class_order)},
            "predicted_label": class_order[pred],
            "confidence": float(proba[i][pred]),
            "fold_id": (int(fold_ids[i]) if fold_ids is not None else None),
        }
        if raw is not None:
            row["raw_probabilities"] = {c: float(raw[i][j]) for j, c in enumerate(class_order)}
        rows.append(row)
    payload = {
        "format": "doar_probability_export_v1",
        "model_id": model_id,
        "checkpoint_hash": checkpoint_hash,
        "calibration_status": calibration_status,
        "temperature": temperature,
        "class_order": list(class_order),
        "count": n,
        "predictions": rows,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(p)


def load_probability_export(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def align_exports(exports: list[dict], split: str):
    """Align multiple probability exports by sample_id for a split (Item 8).

    Fails on: missing sample_id in any export, duplicate ids, class-order
    mismatch, or true-label disagreement. Returns
    (sample_ids, y_true, [proba_matrix per export], fold_ids_or_None).
    """
    if not exports:
        raise ValueError("No exports supplied")
    class_order = exports[0]["class_order"]
    for e in exports[1:]:
        if e["class_order"] != class_order:
            raise ValueError("Class-order mismatch across exports")

    # Build per-export {sample_id: row} for the split.
    maps = []
    for e in exports:
        m = {}
        for row in e["predictions"]:
            if row["split"] != split:
                continue
            sid = row["sample_id"]
            if sid in m:
                raise ValueError(f"Duplicate sample_id {sid!r} in export {e.get('model_id')}")
            m[sid] = row
        maps.append(m)

    common_ids = set(maps[0])
    for m in maps[1:]:
        common_ids &= set(m)
    missing = set(maps[0]) - common_ids
    if any(set(m) != common_ids for m in maps):
        # Report which ids are missing where.
        raise ValueError(
            f"sample_id sets differ across exports for split {split!r}; "
            f"cannot align (example missing: {sorted(missing)[:5]})")

    sample_ids = sorted(common_ids)
    y_true, proba_mats, fold_ids = [], [[] for _ in exports], []
    for sid in sample_ids:
        labels = {m[sid]["true_label"] for m in maps}
        if len(labels) != 1:
            raise ValueError(f"true_label disagreement for sample_id {sid!r}: {labels}")
        y_true.append(class_order.index(next(iter(labels))))
        fold_ids.append(maps[0][sid].get("fold_id"))
        for k, m in enumerate(maps):
            proba_mats[k].append([m[sid]["probabilities"][c] for c in class_order])

    proba_mats = [np.asarray(mat, dtype=float) for mat in proba_mats]
    fold_ids = fold_ids if any(f is not None for f in fold_ids) else None
    return sample_ids, np.asarray(y_true, dtype=int), proba_mats, fold_ids


def write_metrics_csv(metrics: dict, path) -> None:
    """Flatten per-class metrics to a CSV alongside metrics.json."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for name, m in metrics["per_class"].items():
            w.writerow([name, m["precision"], m["recall"], m["f1"], m["support"]])
        w.writerow([])
        for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                    "log_loss", "multiclass_brier", "ece", "macro_ovr_roc_auc",
                    "macro_pr_auc"):
            w.writerow([key, metrics.get(key)])
