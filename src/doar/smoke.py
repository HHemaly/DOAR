"""
smoke.py — end-to-end smoke experiment on a LIMITED subset (Item 3).

Builds a temporary limited dataset (ImageFolder structure, train/valid only — the
test split is never copied, read or evaluated) and runs BOTH the objective and the
optional one-epoch CNN stages on that SAME subset. Returns PASS / WARN / FAIL and
records device requested/used, CUDA usage, per-split/class sample counts, the
checkpoint, validation metrics, elapsed time and failed/skipped steps.
"""

from __future__ import annotations
import csv
import json
import shutil
import time
from pathlib import Path

from .dataset import CLASSES

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# The limited dataset only ever contains these splits (test is excluded).
LIMITED_SPLITS = ("train", "valid")


def _build_limited_dataset(dataset, dest, max_per_class):
    """Copy up to max_per_class images per (split, class) into an ImageFolder at
    `dest`, for train/valid ONLY. Returns per-split/class counts. Never reads test."""
    src = Path(dataset)
    counts = {}
    for split in LIMITED_SPLITS:
        counts[split] = {}
        for cls in CLASSES:
            src_dir = src / split / cls
            dst_dir = Path(dest) / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            if src_dir.exists():
                imgs = sorted(p for p in src_dir.iterdir()
                              if p.suffix.lower() in SUPPORTED_EXT)
                if max_per_class:
                    imgs = imgs[:max_per_class]
                for p in imgs:
                    shutil.copy2(p, dst_dir / p.name)
                    n += 1
            counts[split][cls] = n
    return counts


def run_smoke_experiment(dataset, output, max_samples_per_class: int = 5,
                         device: str = "auto", skip_deep: bool = False,
                         require_deep: bool = False) -> dict:
    from .dataset import build_manifest
    from .leakage import enforce_leakage_gate
    from .extract import extract_features
    from .experiments import run_feature_experiment
    from .probability_export import export_probabilities
    from .evaluation import load_probability_export, compute_metrics
    from .analysis import analyze_image

    started = time.perf_counter()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    steps, failed, skipped = {}, [], []

    def _fail(step, exc):
        failed.append(step)
        steps[step] = f"FAIL: {exc}"

    # Build the limited dataset (train/valid only) — the SAME subset for all stages.
    limited_dir = out / "limited_dataset"
    counts = _build_limited_dataset(dataset, limited_dir, max_samples_per_class)
    steps["samples_per_split_class"] = counts
    total_limited = sum(n for split in counts.values() for n in split.values())

    # 1 validate (best-effort).
    try:
        from .readiness import validate_dataset
        steps["validate_dataset"] = validate_dataset(str(limited_dir), str(out / "validate")).get("status")
    except Exception as exc:
        steps["validate_dataset"] = f"skipped: {exc}"
        skipped.append("validate_dataset")

    # 2 manifest of the LIMITED dataset (contains no test rows).
    manifest = out / "manifest.csv"
    build_manifest(limited_dir, manifest)

    # 3 leakage gate.
    try:
        gate = enforce_leakage_gate(manifest, out / "gate", timestamp="1970-01-01T00:00:00Z")
        steps["leakage_gate"] = gate["gate"]
    except Exception as exc:
        _fail("leakage_gate", exc)

    # 4 feature extraction on the limited manifest.
    feat_csv = out / "features" / "features.csv"
    try:
        extract_features(manifest, out / "features")
        steps["feature_extraction"] = "PASS"
    except Exception as exc:
        _fail("feature_extraction", exc)

    # 5 one small objective model (validation-selected).
    model_path = None
    try:
        lb = run_feature_experiment(str(feat_csv), out / "feature_model",
                                    models=["logistic_regression"], seeds=(42,))
        steps["objective_model_val_macro_f1"] = lb["leaderboard"][0]["mean_macro_f1"] if lb["leaderboard"] else None
        model_path = lb["runs"][0]["checkpoint"]
    except Exception as exc:
        _fail("objective_model", exc)

    # 6 optional one-epoch small_cnn on the SAME limited dataset.
    device_used, cuda_used, checkpoint = None, False, None
    if skip_deep:
        steps["small_cnn_one_epoch"] = "skipped (--skip-deep)"
        skipped.append("small_cnn")
    else:
        try:
            import torch  # noqa
            from .deep.trainers import train_image_model
            res = train_image_model(str(limited_dir), "small_cnn", str(out / "small_cnn"),
                                    seed=42, epochs=1, batch_size=4, image_size=64,
                                    device=device, workers=0, freeze_epochs=0,
                                    pretrained_weights="none")
            device_used = res.get("device")
            cuda_used = str(device_used).startswith("cuda")
            checkpoint = res.get("checkpoint")
            steps["small_cnn_one_epoch"] = "PASS"
        except Exception as exc:
            if require_deep:
                _fail("small_cnn_one_epoch", exc)      # required -> smoke fails
            else:
                steps["small_cnn_one_epoch"] = f"skipped: {exc}"
                skipped.append("small_cnn")

    # 7 validation probability export + 8 common metrics (VALID only).
    val_metrics = None
    if model_path:
        try:
            export = out / "export_valid.json"
            export_probabilities(model_path, str(feat_csv), None, str(export), splits=["valid"])
            exp = load_probability_export(export)
            rows = [r for r in exp["predictions"] if r["split"] == "valid"]
            import numpy as np
            order = exp["class_order"]
            y = np.array([order.index(r["true_label"]) for r in rows])
            proba = np.array([[r["probabilities"][c] for c in order] for r in rows])
            val_metrics = compute_metrics(y, proba.argmax(1), proba, class_names=order)
            steps["validation_metrics"] = {"macro_f1": val_metrics["macro_f1"],
                                           "accuracy": val_metrics["accuracy"]}
        except Exception as exc:
            _fail("validation_export_metrics", exc)

    # 9 one image analysis/report (train image only).
    try:
        with open(manifest, encoding="utf-8") as _mh:
            first = next((r["path"] for r in csv.DictReader(_mh)), None)
        if first:
            analyze_image(first, out / "example_case")
            steps["example_case"] = "PASS"
    except Exception as exc:
        steps["example_case"] = f"skipped: {exc}"
        skipped.append("example_case")

    # ── status ───────────────────────────────────────────────────────────────
    if failed:
        status = "FAIL"
    elif skipped:
        status = "WARN"
    else:
        status = "PASS"

    summary = {
        "status": status,
        "test_used": False,
        "device_requested": device,
        "device_used": device_used,
        "cuda_used": cuda_used,
        "samples_per_split_class": counts,
        "total_limited_samples": total_limited,
        "checkpoint": checkpoint,
        "validation_metrics": val_metrics and {"macro_f1": val_metrics["macro_f1"],
                                               "accuracy": val_metrics["accuracy"]},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "failed_steps": failed,
        "skipped_steps": skipped,
        "steps": steps,
        "output": str(out),
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
