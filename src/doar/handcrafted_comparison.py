"""
handcrafted_comparison.py -- Experiment B (RQ2): does adding HOG improve on
the existing colour/geometry objective features, and how does each
handcrafted-feature group perform in isolation?

Joins the real objective-features CSV (features.py::objective_feature_row,
via `extract-features`) with the real HOG-features CSV
(hog_features.py::hog_feature_row, via `extract_hog_features`) on
image_id, then runs the SAME controlled classical-classifier protocol as
Experiment A (experiments.py::_model, same 6 classifier families) across
four feature-group configurations: HOG only, colour only, geometry only,
and HOG+colour+geometry combined. "Geometry" = composition.* +
segmentation.* + shape.* (spatial/layout signal) -- deliberately excludes
quality.* (technical/scanning-artifact signal, not drawing geometry) and
stroke.* (texture, conceptually closer to what HOG already captures).

Reuses experiments.py's classifier menu and evaluation.compute_metrics --
single source of truth for classifier definitions, not duplicated here.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .dataset import CLASSES
from .experiments import DEFAULT_SEEDS, _deps, _model

META = {"image_id", "path", "split", "class"}

GROUP_PREFIXES = {
    "hog": ("hog.",),
    "colour": ("colour.",),
    "geometry": ("composition.", "segmentation.", "shape."),
}


def handcrafted_group_configs() -> list[dict]:
    return [
        {"name": "hog_only", "include_groups": ["hog"]},
        {"name": "colour_only", "include_groups": ["colour"]},
        {"name": "geometry_only", "include_groups": ["geometry"]},
        {"name": "hog_colour_geometry", "include_groups": ["hog", "colour", "geometry"]},
    ]


def select_group_columns(feature_names: list[str], include_groups: list[str]) -> list[int]:
    """Pure function: indices of columns belonging to ANY of the included
    groups (an include-list, unlike ablation.py's exclude-list semantics --
    Experiment B compares isolated groups, not "all minus one")."""
    prefixes: list[str] = []
    for group in include_groups:
        prefixes.extend(GROUP_PREFIXES[group])
    return [i for i, name in enumerate(feature_names) if any(name.startswith(p) for p in prefixes)]


def _merge_objective_and_hog(objective_features_csv: str | Path, hog_features_csv: str | Path):
    with open(objective_features_csv, newline="", encoding="utf-8") as handle:
        objective_rows = {row["image_id"]: row for row in csv.DictReader(handle)}
    with open(hog_features_csv, newline="", encoding="utf-8") as handle:
        hog_rows = {row["image_id"]: row for row in csv.DictReader(handle)}
    common_ids = sorted(set(objective_rows) & set(hog_rows))
    if not common_ids:
        raise ValueError("No image_id is common to both the objective-features and HOG CSVs")
    objective_names = [n for n in next(iter(objective_rows.values())) if n not in META]
    hog_names = [n for n in next(iter(hog_rows.values())) if n not in META]
    feature_names = objective_names + hog_names
    merged_rows = []
    for image_id in common_ids:
        obj_row, hog_row = objective_rows[image_id], hog_rows[image_id]
        if obj_row["split"] != hog_row["split"] or obj_row["class"] != hog_row["class"]:
            raise ValueError(f"split/class mismatch between feature sources for {image_id}")
        merged_rows.append({
            "image_id": image_id, "split": obj_row["split"], "class": obj_row["class"],
            **{n: obj_row[n] for n in objective_names}, **{n: hog_row[n] for n in hog_names},
        })
    return feature_names, merged_rows


def _to_matrix(rows: list[dict], feature_names: list[str], split: str):
    selected = [r for r in rows if r["split"] == split]
    if not selected:
        raise ValueError(f"No rows for split {split!r}")
    x = np.asarray([
        [float(r[n]) if r[n] not in ("", "nan", "NaN") else np.nan for n in feature_names]
        for r in selected
    ])
    y = np.asarray([CLASSES.index(r["class"]) for r in selected])
    ids = [r["image_id"] for r in selected]
    return x, y, ids


def run_handcrafted_group_comparison(
    objective_features_csv: str | Path, hog_features_csv: str | Path, output: str | Path,
    models: list[str] | None = None, seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict:
    d = _deps()
    models = models or ["logistic_regression", "linear_svm", "random_forest",
                        "extra_trees", "hist_gradient_boosting"]
    feature_names, rows = _merge_objective_and_hog(objective_features_csv, hog_features_csv)
    x_train_full, y_train, _ = _to_matrix(rows, feature_names, "train")
    x_valid_full, y_valid, valid_ids = _to_matrix(rows, feature_names, "valid")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    from .evaluation import compute_metrics
    runs = []
    for cfg in handcrafted_group_configs():
        keep = select_group_columns(feature_names, cfg["include_groups"])
        if not keep:
            raise ValueError(f"Configuration {cfg['name']!r} selected zero columns")
        x_train, x_valid = x_train_full[:, keep], x_valid_full[:, keep]
        for name in models:
            for seed in seeds:
                estimator = _model(name, seed, d)
                started = time.perf_counter()
                estimator.fit(x_train, y_train)
                elapsed = time.perf_counter() - started
                probabilities = estimator.predict_proba(x_valid)
                predictions = probabilities.argmax(axis=1)
                metrics = compute_metrics(y_valid, predictions, probabilities)
                run_dir = output / "runs" / f"{cfg['name']}_{name}_seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = run_dir / "model.joblib"
                d["joblib"].dump({
                    "model": estimator, "classes": CLASSES,
                    "feature_names": [feature_names[i] for i in keep],
                    "group_configuration": cfg["name"], "seed": seed, "model_name": name,
                }, checkpoint)
                record = {
                    "family": "handcrafted_groups", "configuration": cfg["name"],
                    "include_groups": cfg["include_groups"], "model": name, "seed": seed,
                    "selection_split": "valid", "test_used": False,
                    "n_features": len(keep), "training_seconds": elapsed,
                    "metrics": metrics, "checkpoint": str(checkpoint),
                }
                (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                runs.append(record)

    leaderboard = []
    for cfg in handcrafted_group_configs():
        for name in models:
            selected = [r for r in runs if r["configuration"] == cfg["name"] and r["model"] == name]
            leaderboard.append({
                "configuration": cfg["name"], "model": name, "seeds": list(seeds),
                "n_features": selected[0]["n_features"],
                "mean_macro_f1": float(np.mean([r["metrics"]["macro_f1"] for r in selected])),
                "std_macro_f1": float(np.std([r["metrics"]["macro_f1"] for r in selected])),
                "mean_balanced_accuracy": float(np.mean([r["metrics"]["balanced_accuracy"] for r in selected])),
                "mean_ece": float(np.mean([r["metrics"]["ece"] for r in selected])),
                "mean_training_seconds": float(np.mean([r["training_seconds"] for r in selected])),
            })
    leaderboard.sort(key=lambda row: (-row["mean_macro_f1"], row["std_macro_f1"]))

    summary = {
        "primary_selection_metric": "mean validation macro F1 across seeds",
        "test_used": False,
        "objective_features_csv": str(Path(objective_features_csv).resolve()),
        "hog_features_csv": str(Path(hog_features_csv).resolve()),
        "group_definitions": {k: list(v) for k, v in GROUP_PREFIXES.items()},
        "leaderboard": leaderboard,
        "runs": runs,
    }
    (output / "handcrafted_group_leaderboard.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "handcrafted_group_leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(leaderboard[0]) if leaderboard else ["configuration", "model"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(leaderboard)
    return summary
