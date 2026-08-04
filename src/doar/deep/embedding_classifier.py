"""
embedding_classifier.py -- Experiment C (RQ3): how effective are frozen
embeddings (DINOv2 or any other backbone already supported by
embeddings.py) on this dataset, using only a classical classifier on top?

Deliberately separate from fusion/embedding_comparison.py (which fixes ONE
logistic-regression pipeline per representation-vs-fusion configuration) --
this module compares MULTIPLE classifier families on a SINGLE embedding
source, mirroring experiments.py::run_feature_experiment's protocol
exactly (same classifier menu, same metrics, same leaderboard shape) so
Experiment A and Experiment C results are directly comparable. The
encoder is never fine-tuned here -- embeddings are extracted once
(deep/embeddings.py::extract_embeddings, label-free) and cached to
embeddings.npz; this module only ever reads that cache.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from ..dataset import CLASSES
from ..experiments import DEFAULT_SEEDS, _deps, _model

# A small, explicitly-declared subset of experiments.py's classifier menu --
# "linear probe, or a small regularized MLP only if justified" per the task
# spec (RQ3). rbf_svm/random_forest/extra_trees/hist_gradient_boosting are
# still selectable via `models=` for a fuller comparison; this default
# favors the two forms the research question actually asks about.
DEFAULT_MODELS = ("logistic_regression", "mlp_small")


def _load_embeddings(path: str | Path):
    payload = np.load(path, allow_pickle=False)
    embeddings, ids = payload["embeddings"], payload["image_ids"]
    splits, labels = payload["splits"], payload["labels"]
    if embeddings.shape[0] != len(ids):
        raise ValueError("embeddings/image_ids length mismatch in cache")
    data = {}
    for split in ("train", "valid"):
        mask = splits == split
        if not mask.any():
            raise ValueError(f"No cached embeddings for split {split!r}")
        x = embeddings[mask]
        y = np.asarray([CLASSES.index(str(c)) for c in labels[mask]])
        split_ids = [str(v) for v in ids[mask]]
        data[split] = (x, y, split_ids)
    return data


def _classifier(name: str, seed: int, d: dict):
    if name == "mlp_small":
        # A deliberately small, regularized MLP -- see module docstring;
        # early_stopping guards against overfitting the small training set.
        from sklearn.neural_network import MLPClassifier
        return d["make_pipeline"](
            d["StandardScaler"](),
            MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True,
                          max_iter=500, random_state=seed),
        )
    if name == "logistic_regression":
        return d["make_pipeline"](d["StandardScaler"](), d["LogisticRegression"](
            max_iter=2000, class_weight="balanced", random_state=seed))
    # Fall back to the shared classifier menu for anything else requested
    # (rbf_svm, random_forest, ...) -- note these do NOT need SimpleImputer
    # since embeddings never contain missing values (label-free extraction
    # either succeeds for an image or the image is excluded, never partial).
    return _model(name, seed, d)


def run_embedding_classifier_experiment(
    embeddings_npz: str | Path, output: str | Path,
    models: list[str] | None = None, seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict:
    d = _deps()
    models = list(models) if models else list(DEFAULT_MODELS)
    data = _load_embeddings(embeddings_npz)
    x_train, y_train, _ = data["train"]
    x_valid, y_valid, valid_ids = data["valid"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(embeddings_npz).with_name("embedding_metadata.json")
    backbone = "unknown"
    embedding_dimension = int(x_train.shape[1])
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        backbone = meta.get("backbone", backbone)
        embedding_dimension = meta.get("embedding_dimension", embedding_dimension)

    from ..evaluation import compute_metrics
    runs = []
    for name in models:
        for seed in seeds:
            estimator = _classifier(name, seed, d)
            started = time.perf_counter()
            estimator.fit(x_train, y_train)
            elapsed = time.perf_counter() - started
            probabilities = estimator.predict_proba(x_valid)
            predictions = probabilities.argmax(axis=1)
            metrics = compute_metrics(y_valid, predictions, probabilities)
            run_dir = output / "runs" / f"{name}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "model.joblib"
            d["joblib"].dump({
                "model": estimator, "classes": CLASSES, "backbone": backbone,
                "embedding_dimension": embedding_dimension, "seed": seed, "model_name": name,
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
                "family": "frozen_embedding_classifier", "backbone": backbone, "model": name,
                "seed": seed, "selection_split": "valid", "test_used": False,
                "embedding_dimension": embedding_dimension, "training_seconds": elapsed,
                "metrics": metrics, "checkpoint": str(checkpoint),
            }
            (run_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            runs.append(record)

    leaderboard = []
    for name in models:
        selected = [r for r in runs if r["model"] == name]
        leaderboard.append({
            "model": name, "backbone": backbone, "seeds": list(seeds),
            "mean_macro_f1": float(np.mean([r["metrics"]["macro_f1"] for r in selected])),
            "std_macro_f1": float(np.std([r["metrics"]["macro_f1"] for r in selected])),
            "mean_balanced_accuracy": float(np.mean([r["metrics"]["balanced_accuracy"] for r in selected])),
            "mean_ece": float(np.mean([r["metrics"]["ece"] for r in selected])),
            "mean_training_seconds": float(np.mean([r["training_seconds"] for r in selected])),
        })
    leaderboard.sort(key=lambda row: (-row["mean_macro_f1"], row["std_macro_f1"]))

    summary = {
        "primary_selection_metric": "mean validation macro F1 across seeds",
        "test_used": False, "backbone": backbone, "embedding_dimension": embedding_dimension,
        "embeddings_npz": str(Path(embeddings_npz).resolve()),
        "backbone_fine_tuned": False,
        "leaderboard": leaderboard, "runs": runs,
    }
    (output / "embedding_classifier_leaderboard.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "embedding_classifier_leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0]) if leaderboard else ["model"])
        writer.writeheader()
        writer.writerows(leaderboard)
    return summary
