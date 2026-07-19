from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from .dataset import CLASSES

META = {"image_id", "path", "split", "class"}
DEFAULT_SEEDS = (42, 123, 2026)


def _deps():
    try:
        import joblib
        import sklearn
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, classification_report,
            confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
            roc_auc_score,
        )
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError('Install experiment dependencies with: pip install -e ".[ml]"') from exc
    return locals()


def _load_features(path: str | Path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Feature table is empty")
    feature_names = [name for name in rows[0] if name not in META]
    data = {}
    for split in ("train", "valid"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            raise ValueError(f"No rows found for required split {split!r}")
        x = np.asarray([
            [float(row[name]) if row[name] not in ("", "nan", "NaN") else np.nan for name in feature_names]
            for row in selected
        ])
        y = np.asarray([CLASSES.index(row["class"]) for row in selected])
        ids = [row["image_id"] for row in selected]
        data[split] = (x, y, ids)
    return feature_names, data


def _model(name: str, seed: int, d: dict):
    common = [d["SimpleImputer"](strategy="median")]
    if name == "logistic_regression":
        return d["make_pipeline"](*common, d["StandardScaler"](), d["LogisticRegression"](
            max_iter=2000, class_weight="balanced", random_state=seed
        ))
    if name == "linear_svm":
        return d["make_pipeline"](*common, d["StandardScaler"](), d["SVC"](
            kernel="linear", probability=True, class_weight="balanced", random_state=seed
        ))
    if name == "rbf_svm":
        return d["make_pipeline"](*common, d["StandardScaler"](), d["SVC"](
            kernel="rbf", probability=True, class_weight="balanced", random_state=seed
        ))
    if name == "random_forest":
        return d["make_pipeline"](*common, d["RandomForestClassifier"](
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=seed
        ))
    if name == "extra_trees":
        return d["make_pipeline"](*common, d["ExtraTreesClassifier"](
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=seed
        ))
    if name == "hist_gradient_boosting":
        return d["make_pipeline"](*common, d["HistGradientBoostingClassifier"](
            max_iter=300, learning_rate=.05, random_state=seed
        ))
    raise ValueError(f"Unknown objective-feature model: {name}")


def run_feature_experiment(
    features_csv: str | Path,
    output: str | Path,
    models: list[str] | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict:
    d = _deps()
    models = models or ["logistic_regression", "linear_svm", "random_forest", "extra_trees",
                        "hist_gradient_boosting"]
    feature_names, data = _load_features(features_csv)
    x_train, y_train, _ = data["train"]
    x_valid, y_valid, valid_ids = data["valid"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for name in models:
        for seed in seeds:
            estimator = _model(name, seed, d)
            started = time.perf_counter()
            estimator.fit(x_train, y_train)
            elapsed = time.perf_counter() - started
            probabilities = estimator.predict_proba(x_valid)
            predictions = probabilities.argmax(axis=1)
            from .evaluation import compute_metrics
            metrics = compute_metrics(y_valid, predictions, probabilities)
            run_dir = output / "runs" / f"{name}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "model.joblib"
            d["joblib"].dump({
                "model": estimator, "classes": CLASSES, "feature_names": feature_names,
                "seed": seed, "model_name": name,
            }, checkpoint)
            with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                fields = ["image_id", "true_class", "predicted_class"] + [f"p_{c}" for c in CLASSES]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for image_id, truth, pred, probs in zip(valid_ids, y_valid, predictions, probabilities):
                    writer.writerow({
                        "image_id": image_id, "true_class": CLASSES[truth],
                        "predicted_class": CLASSES[pred],
                        **{f"p_{cls}": float(probs[i]) for i, cls in enumerate(CLASSES)},
                    })
            record = {
                "family": "objective_features",
                "model": name,
                "seed": seed,
                "selection_split": "valid",
                "test_used": False,
                "feature_count": len(feature_names),
                "training_seconds": elapsed,
                "metrics": metrics,
                "checkpoint": str(checkpoint),
            }
            (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            runs.append(record)
    leaderboard = []
    for name in models:
        selected = [run for run in runs if run["model"] == name]
        leaderboard.append({
            "model": name,
            "seeds": list(seeds),
            "mean_macro_f1": float(np.mean([r["metrics"]["macro_f1"] for r in selected])),
            "std_macro_f1": float(np.std([r["metrics"]["macro_f1"] for r in selected])),
            "mean_balanced_accuracy": float(np.mean([r["metrics"]["balanced_accuracy"] for r in selected])),
            "mean_ece": float(np.mean([r["metrics"]["ece"] for r in selected])),
            "mean_training_seconds": float(np.mean([r["training_seconds"] for r in selected])),
        })
    leaderboard.sort(key=lambda row: (
        -row["mean_macro_f1"], -row["mean_balanced_accuracy"], row["mean_ece"],
        row["std_macro_f1"], row["mean_training_seconds"],
    ))
    summary = {
        "primary_selection_metric": "mean validation macro F1 across seeds",
        "test_used": False,
        "features_csv": str(Path(features_csv).resolve()),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "sklearn": d["sklearn"].__version__,
        },
        "leaderboard": leaderboard,
        "runs": runs,
    }
    (output / "validation_leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "validation_leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0]) if leaderboard else ["model"])
        writer.writeheader()
        writer.writerows(leaderboard)
    return summary
