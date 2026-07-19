from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from ..dataset import CLASSES
from . import PRIMARY_METHODS


def _deps():
    try:
        import joblib
        import sklearn
        from sklearn.compose import ColumnTransformer
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError('Install fusion dependencies with: pip install -e ".[ml]"') from exc
    return locals()


def _load(features_csv: str | Path, embeddings_npz: str | Path):
    with open(features_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    feature_names = [name for name in rows[0] if name not in {"image_id", "path", "split", "class"}]
    feature_by_id = {
        row["image_id"]: (
            np.asarray([float(row[name]) if row[name] not in ("", "nan", "NaN") else np.nan
                        for name in feature_names], dtype=np.float32),
            row["split"], row["class"],
        )
        for row in rows
    }
    cache = np.load(embeddings_npz)
    joined = []
    for image_id, embedding, split, label in zip(
        cache["image_ids"], cache["embeddings"], cache["splits"], cache["labels"]
    ):
        key = str(image_id)
        if key not in feature_by_id:
            continue
        features, feature_split, feature_label = feature_by_id[key]
        if str(split) != feature_split or str(label) != feature_label:
            raise ValueError(f"Feature/embedding metadata mismatch for image_id {key}")
        joined.append((key, features, embedding.astype(np.float32), feature_split, feature_label))
    if not joined:
        raise ValueError("No aligned image IDs between features and embeddings")
    return feature_names, joined


def _model(method: str, seed: int, objective_dim: int, embedding_dim: int, d: dict):
    if method not in PRIMARY_METHODS:
        raise ValueError(f"Unknown primary fusion method {method!r}")
    separate_scaler = d["ColumnTransformer"]([
        ("objective", d["make_pipeline"](d["SimpleImputer"](strategy="median"),
                                         d["StandardScaler"]()), slice(0, objective_dim)),
        ("embedding", d["StandardScaler"](), slice(objective_dim, objective_dim + embedding_dim)),
    ])
    if method == "pca_early_fusion":
        transformer = d["ColumnTransformer"]([
            ("objective", d["make_pipeline"](d["SimpleImputer"](strategy="median"),
                                             d["StandardScaler"]()), slice(0, objective_dim)),
            ("embedding", d["make_pipeline"](d["StandardScaler"](),
                                             d["PCA"](n_components=.95, random_state=seed)),
             slice(objective_dim, objective_dim + embedding_dim)),
        ])
        return d["make_pipeline"](transformer, d["LogisticRegression"](
            max_iter=2000, class_weight="balanced", random_state=seed))
    if method == "mlp_early_fusion":
        return d["make_pipeline"](separate_scaler, d["MLPClassifier"](
            hidden_layer_sizes=(256, 64), early_stopping=True, max_iter=500, random_state=seed))
    return d["make_pipeline"](separate_scaler, d["LogisticRegression"](
        max_iter=2000, class_weight="balanced", random_state=seed))


def train_primary_fusion(
    features_csv: str | Path, embeddings_npz: str | Path, output: str | Path,
    methods: list[str] | None = None, seeds: tuple[int, ...] = (42, 123, 2026),
    configuration_hash: str | None = None,
) -> dict:
    d = _deps()
    embedding_metadata_path = Path(embeddings_npz).with_name("embedding_metadata.json")
    if not embedding_metadata_path.exists():
        raise ValueError("Embedding metadata is required beside embeddings.npz")
    embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
    methods = methods or list(PRIMARY_METHODS)
    feature_names, joined = _load(features_csv, embeddings_npz)
    objective_dim = len(feature_names)
    embedding_dim = len(joined[0][2])
    arrays = {}
    for split in ("train", "valid"):
        selected = [item for item in joined if item[3] == split]
        if not selected:
            raise ValueError(f"No aligned rows for required split {split!r}")
        arrays[split] = (
            np.stack([np.concatenate((item[1], item[2])) for item in selected]),
            np.asarray([CLASSES.index(item[4]) for item in selected]),
            [item[0] for item in selected],
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for method in methods:
        for seed in seeds:
            model = _model(method, seed, objective_dim, embedding_dim, d)
            started = time.perf_counter()
            model.fit(arrays["train"][0], arrays["train"][1])
            probabilities = model.predict_proba(arrays["valid"][0])
            predictions = probabilities.argmax(1)
            metrics = {
                "accuracy": float(d["accuracy_score"](arrays["valid"][1], predictions)),
                "macro_f1": float(d["f1_score"](arrays["valid"][1], predictions, average="macro")),
                "weighted_f1": float(d["f1_score"](arrays["valid"][1], predictions, average="weighted")),
                "balanced_accuracy": float(d["balanced_accuracy_score"](arrays["valid"][1], predictions)),
                "log_loss": float(d["log_loss"](arrays["valid"][1], probabilities,
                                                 labels=list(range(len(CLASSES))))),
            }
            run_dir = output / "runs" / f"{method}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "fusion.joblib"
            d["joblib"].dump({
                "model": model, "model_family": "primary_multimodal_fusion",
                "method": method, "classes": CLASSES, "seed": seed,
                "objective_feature_names": feature_names,
                "embedding_dimension": embedding_dim,
                "embedding_backbone": embedding_metadata["backbone"],
                "embedding_preprocessing_hash": embedding_metadata["preprocessing_hash"],
                "checkpoint_type": "doar_fusion_bundle_v1",
                "feature_extraction_version": "3.1.0",
                "segmentation_version": "segmentation_ensemble_v1",
                "class_order": CLASSES,
                "calibration": {"method": "none", "status": "uncalibrated"},
                "training_configuration": {
                    "selection_split": "valid", "seed": seed, "method": method,
                },
                "dependency_versions": {"sklearn": d["sklearn"].__version__},
                "configuration_sha256": configuration_hash,
                "psychologist_rules_used": False,
                "concern_profiles_used": False,
            }, checkpoint)
            record = {
                "method": method, "seed": seed, "metrics": metrics,
                "training_seconds": time.perf_counter() - started,
                "selection_split": "valid", "test_used": False,
                "checkpoint": str(checkpoint),
            }
            (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            runs.append(record)
    leaderboard = []
    for method in methods:
        selected = [run for run in runs if run["method"] == method]
        leaderboard.append({
            "method": method,
            "mean_macro_f1": float(np.mean([r["metrics"]["macro_f1"] for r in selected])),
            "std_macro_f1": float(np.std([r["metrics"]["macro_f1"] for r in selected])),
            "mean_balanced_accuracy": float(np.mean(
                [r["metrics"]["balanced_accuracy"] for r in selected])),
        })
    leaderboard.sort(key=lambda row: (
        -row["mean_macro_f1"], -row["mean_balanced_accuracy"], row["std_macro_f1"]
    ))
    summary = {
        "thesis_layer": "Layer 1 - Primary Emotion Classifier",
        "inputs": ["deep_embeddings", "objective_features"],
        "excluded_inputs": ["psychologist_rules", "concern_profiles"],
        "primary_metric": "mean validation macro F1 across seeds",
        "seeds": list(seeds), "leaderboard": leaderboard, "runs": runs,
        "environment": {"python": platform.python_version(), "sklearn": d["sklearn"].__version__},
    }
    (output / "fusion_leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "fusion_leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0]))
        writer.writeheader()
        writer.writerows(leaderboard)
    return summary
