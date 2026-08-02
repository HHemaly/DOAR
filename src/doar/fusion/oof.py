"""
oof.py — out-of-fold probability generation for genuine stacking (C1).

Generates deterministic stratified folds over the TRAIN split, trains one model
per fold on the other folds only, and predicts exactly the held-out fold — so
every training sample is predicted exactly once by a model that never saw it.
Emits the common probability-export schema with a real `fold_id`, plus fold-model
hashes and metadata. Never touches valid/test.

stratified_folds is pure-numpy (CPU-tested); training uses sklearn.
"""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..dataset import CLASSES
from ..evaluation import save_probability_export

META = {"image_id", "path", "split", "class"}


def stratified_folds(labels, n_folds: int, seed: int = 42) -> list[int]:
    """Deterministic stratified fold assignment (one fold id per sample)."""
    if n_folds < 2:
        raise ValueError("stacking requires n_folds >= 2")
    labels = np.asarray(labels)
    fold_of = np.full(len(labels), -1, dtype=int)
    rng = np.random.RandomState(seed)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            fold_of[i] = k % n_folds
    return fold_of.tolist()


def _load_train(features_csv):
    with open(features_csv, newline="", encoding="utf-8") as h:
        rows = [r for r in csv.DictReader(h) if r["split"] == "train"]
    if not rows:
        raise ValueError("No train rows for OOF generation")
    names = [n for n in rows[0] if n not in META]
    X = np.asarray([[float(r[n]) if r[n] not in ("", "nan", "NaN") else np.nan
                     for n in names] for r in rows])
    y = np.asarray([CLASSES.index(r["class"]) for r in rows])
    ids = [r["image_id"] for r in rows]
    return names, X, y, ids


def generate_oof(features_csv, output, model_name: str = "logistic_regression",
                 n_folds: int = 5, seed: int = 42) -> dict:
    """Generate OOF probabilities for objective-feature models. sklearn."""
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError as exc:  # pragma: no cover - needs [ml]
        raise RuntimeError('Install ML support with: pip install -e ".[ml]"') from exc

    names, X, y, ids = _load_train(features_csv)
    folds = np.asarray(stratified_folds(y, n_folds, seed))
    n = len(ids)
    oof = np.zeros((n, len(CLASSES)), dtype=float)
    predicted_mask = np.zeros(n, dtype=bool)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    fold_meta = []

    for k in range(n_folds):
        held = folds == k
        train_idx = ~held
        if held.sum() == 0:
            continue
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=seed))
        clf.fit(X[train_idx], y[train_idx])
        oof[held] = clf.predict_proba(X[held])
        # A sample must never be predicted by a model trained on it.
        assert not predicted_mask[held].any(), "OOF integrity: fold predicted twice"
        predicted_mask[held] = True
        mp = out / f"fold_{k}_model.joblib"
        joblib.dump({"model": clf, "fold": k, "classes": CLASSES}, mp)
        fold_meta.append({"fold": k, "n_held": int(held.sum()),
                          "model_hash": hashlib.sha256(mp.read_bytes()).hexdigest()})

    if not predicted_mask.all():
        raise ValueError("OOF integrity: some samples were never predicted")

    export_path = out / "oof_export.json"
    save_probability_export(
        export_path, sample_ids=ids, splits=["train"] * n, y_true=y.tolist(),
        proba=oof, model_id=f"oof_{model_name}", checkpoint_hash="oof_ensemble",
        calibration_status="uncalibrated", class_order=list(CLASSES),
        fold_ids=folds.tolist())
    (out / "oof_metadata.json").write_text(json.dumps({
        "model": model_name, "n_folds": n_folds, "seed": seed,
        "n_train_samples": n, "each_sample_predicted_once": bool(predicted_mask.all()),
        "folds": fold_meta, "export": str(export_path),
    }, indent=2), encoding="utf-8")
    return {"export": str(export_path), "n_folds": n_folds, "n_samples": n,
            "each_sample_predicted_once": True}
