from __future__ import annotations

import csv
import hashlib
import json
import getpass
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import CLASSES


def _dependencies():
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            'Training requires the ML extras: python -m pip install -e ".[ml]"'
        ) from exc
    return joblib, LogisticRegression, accuracy_score, balanced_accuracy_score, f1_score, make_pipeline, StandardScaler


def _image_features(path: str) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    hist = np.concatenate([np.histogram(rgb[:, :, c], bins=16, range=(0, 1))[0] for c in range(3)])
    return np.concatenate((
        hist / max(1, hist.sum()),
        rgb.mean(axis=(0, 1)),
        rgb.std(axis=(0, 1)),
        [gray.mean(), gray.std(), (gray < 0.8).mean()],
    )).astype(np.float32)


def _load(manifest: str, split: str):
    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split and row["readable"] == "True"]
    if not rows:
        raise ValueError(f"No readable rows found for split {split!r}")
    x = np.stack([_image_features(row["path"]) for row in rows])
    y = np.asarray([CLASSES.index(row["class"]) for row in rows])
    return x, y


def _metrics(y_true, y_pred, accuracy_score, balanced_accuracy_score, f1_score):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def train_model(manifest: str, output: str, seed: int = 42) -> dict:
    deps = _dependencies()
    joblib, LogisticRegression, accuracy_score, balanced_accuracy_score, f1_score, make_pipeline, StandardScaler = deps
    x_train, y_train = _load(manifest, "train")
    x_valid, y_valid = _load(manifest, "valid")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    started = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    metrics = _metrics(y_valid, model.predict(x_valid), accuracy_score, balanced_accuracy_score, f1_score)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "feature_logistic_regression.joblib"
    joblib.dump({"model": model, "classes": CLASSES, "seed": seed}, checkpoint)
    result = {
        "model": "whole-image statistical baseline",
        "selection_split": "valid",
        "test_used": False,
        "seed": seed,
        "train_samples": len(y_train),
        "valid_samples": len(y_valid),
        "valid_metrics": metrics,
        "training_seconds": elapsed,
        "checkpoint": str(checkpoint),
    }
    (out / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_model(
    manifest: str, checkpoint: str, output: str, split: str, unlock_test: bool,
    confirm_final_evaluation: bool = False, initiated_by: str | None = None,
) -> dict:
    if split == "test" and not (unlock_test and confirm_final_evaluation):
        raise PermissionError(
            "The test split is locked. Final evaluation requires both "
            "--unlock-test and --confirm-final-evaluation."
        )
    deps = _dependencies()
    joblib, _, accuracy_score, balanced_accuracy_score, f1_score, _, _ = deps
    payload = joblib.load(checkpoint)
    if tuple(payload["classes"]) != CLASSES:
        raise ValueError("Checkpoint class mapping does not match the fixed DOAR mapping")
    x, y = _load(manifest, split)
    metrics = _metrics(y, payload["model"].predict(x), accuracy_score, balanced_accuracy_score, f1_score)
    result = {"split": split, "samples": len(y), "metrics": metrics, "checkpoint": checkpoint}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if split == "test":
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "initiated_by": initiated_by or getpass.getuser(),
            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "configuration_sha256": hashlib.sha256(json.dumps({
                "split": split, "classes": CLASSES, "checkpoint": Path(checkpoint).name,
            }, sort_keys=True).encode()).hexdigest(),
            "confirmation_flags": ["unlock_test", "confirm_final_evaluation"],
            "selected_validation_results": "must be archived with the frozen experiment configuration",
        }
        with (out / "final_test_unlock_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
    return result
