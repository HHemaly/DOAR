"""
probability_export.py — repository-generated probability exports (Item 8).

Produces the common export format (see evaluation.save_probability_export) from a
saved sklearn model bundle (objective-feature or fusion). Every row carries
sample_id, split, true_label, class probabilities; the header carries model_id,
checkpoint_hash, calibration_status and class_order. fold_id is included when a
manifest supplies it (for OOF stacking).

Model inference needs sklearn (+ embeddings for fusion). The export FORMAT and
its alignment are already CPU-tested via tests/test_evaluation.py.
"""

from __future__ import annotations
import csv
import hashlib
from pathlib import Path

import numpy as np

from .dataset import CLASSES
from .evaluation import save_probability_export

META = {"image_id", "path", "split", "class"}


def _load_feature_rows(features_csv):
    with open(features_csv, newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    names = [n for n in rows[0] if n not in META]
    return names, rows


def export_probabilities(model_path, features_csv, embeddings_npz, output,
                         splits=("train", "valid")) -> dict:
    """Run a saved model over the requested splits and write a probability export.
    Test split is exportable ONLY if explicitly requested (kept out by default)."""
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - needs [ml]
        raise RuntimeError('Install ML support with: pip install -e ".[ml]"') from exc
    bundle = joblib.load(model_path)
    model = bundle["model"]
    model_id = bundle.get("model_name") or bundle.get("method") or Path(model_path).stem
    checkpoint_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    calibration_status = bundle.get("calibration", {}).get("status", "uncalibrated") \
        if isinstance(bundle.get("calibration"), dict) else "uncalibrated"

    feature_names, rows = _load_feature_rows(features_csv)
    embed_by_id = {}
    if embeddings_npz:
        cache = np.load(embeddings_npz, allow_pickle=True)
        for sid, vec in zip(cache["image_ids"].astype(str), cache["embeddings"]):
            embed_by_id[sid] = np.asarray(vec, dtype=float)

    sample_ids, split_list, y_true, X, fold_ids = [], [], [], [], []
    has_fold = rows and "fold_id" in rows[0]
    for row in rows:
        if row["split"] not in splits:
            continue
        feats = [float(row[n]) if row[n] not in ("", "nan", "NaN") else np.nan
                 for n in feature_names]
        if embeddings_npz:
            sid = row["image_id"]
            if sid not in embed_by_id:
                continue
            vec = np.concatenate((np.asarray(feats), embed_by_id[sid]))
        else:
            vec = np.asarray(feats)
        sample_ids.append(row["image_id"])
        split_list.append(row["split"])
        y_true.append(CLASSES.index(row["class"]))
        X.append(vec)
        if has_fold:
            fold_ids.append(int(row["fold_id"]))
    if not X:
        raise ValueError("No rows matched the requested splits")

    proba = model.predict_proba(np.vstack(X))
    path = save_probability_export(
        output, sample_ids=sample_ids, splits=split_list, y_true=y_true, proba=proba,
        model_id=model_id, checkpoint_hash=checkpoint_hash,
        calibration_status=calibration_status, class_order=list(CLASSES),
        fold_ids=(fold_ids if has_fold else None))
    return {"export": path, "model_id": model_id, "count": len(sample_ids),
            "splits": sorted(set(split_list)), "has_oof_folds": has_fold}
