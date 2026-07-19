"""
smoke.py — end-to-end smoke experiment (C5).

Runs the real pipeline on a tiny subset so the whole chain is exercised quickly:
  1 dataset validation, 2 manifest, 3 leakage gate, 4 limited feature extraction,
  5 one small objective model, 6 optional 1-epoch small_cnn (torch),
  7 validation probability export, 8 common validation metrics,
  9 one image analysis/report, 10 smoke summary.

NEVER unlocks or reads the test split.
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path


def _limit_manifest(manifest_csv, out_csv, max_per_class, splits=("train", "valid")):
    with open(manifest_csv, newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    fields = list(rows[0].keys()) if rows else ["image_id", "path", "split", "class"]
    kept, counts = [], defaultdict(int)
    for r in rows:
        if r["split"] not in splits:            # test never processed
            continue
        key = (r["split"], r["class"])
        if max_per_class and counts[key] >= max_per_class:
            continue
        counts[key] += 1
        kept.append(r)
    with open(out_csv, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(kept)
    return len(kept)


def run_smoke_experiment(dataset, output, max_samples_per_class: int = 5,
                         device: str = "auto") -> dict:
    from .dataset import build_manifest
    from .leakage import enforce_leakage_gate
    from .extract import extract_features
    from .experiments import run_feature_experiment
    from .probability_export import export_probabilities
    from .evaluation import load_probability_export, compute_metrics
    from .analysis import analyze_image

    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    steps = {}

    # 1-2 validate + manifest.
    try:
        from .readiness import validate_dataset
        steps["validate_dataset"] = validate_dataset(dataset, str(out / "validate")).get("status")
    except Exception as exc:
        steps["validate_dataset"] = f"skipped: {exc}"
    manifest = out / "manifest.csv"
    build_manifest(dataset, manifest)

    # 3 leakage gate (blocks on leakage).
    gate = enforce_leakage_gate(manifest, out / "gate", timestamp="1970-01-01T00:00:00Z")
    steps["leakage_gate"] = gate["gate"]

    # 4 limited feature extraction (train+valid only).
    limited = out / "manifest_limited.csv"
    n_limited = _limit_manifest(manifest, limited, max_samples_per_class)
    steps["limited_samples"] = n_limited
    extract_features(limited, out / "features")

    # 5 one small objective model (validation-selected).
    feat_csv = out / "features" / "features.csv"
    lb = run_feature_experiment(str(feat_csv), out / "feature_model",
                                models=["logistic_regression"], seeds=(42,))
    steps["objective_model_val_macro_f1"] = lb["leaderboard"][0]["mean_macro_f1"] if lb["leaderboard"] else None
    model_path = lb["runs"][0]["checkpoint"]

    # 6 optional one-epoch small_cnn (torch).
    try:
        import torch  # noqa
        from .deep.trainers import train_image_model
        train_image_model(dataset, "small_cnn", str(out / "small_cnn"), seed=42, epochs=1,
                          batch_size=4, image_size=64, device=device, workers=0,
                          freeze_epochs=0, pretrained_weights="none")
        steps["small_cnn_one_epoch"] = "ok"
    except Exception as exc:
        steps["small_cnn_one_epoch"] = f"skipped: {exc}"

    # 7 validation probability export + 8 common metrics (VALID only).
    export = out / "export_valid.json"
    export_probabilities(model_path, str(feat_csv), None, str(export), splits=["valid"])
    exp = load_probability_export(export)
    rows = [r for r in exp["predictions"] if r["split"] == "valid"]
    import numpy as np
    order = exp["class_order"]
    y = np.array([order.index(r["true_label"]) for r in rows])
    proba = np.array([[r["probabilities"][c] for c in order] for r in rows])
    metrics = compute_metrics(y, proba.argmax(1), proba, class_names=order)
    steps["validation_metrics"] = {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}

    # 9 one image analysis/report.
    first = next((r["path"] for r in csv.DictReader(open(limited, encoding="utf-8"))), None)
    if first:
        analyze_image(first, out / "example_case")
        steps["example_case"] = "ok"

    summary = {"status": "ok", "test_used": False, "steps": steps,
               "output": str(out)}
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
