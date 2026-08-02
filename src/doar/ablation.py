"""
ablation.py — objective-feature ablation experiments (Item 17).

Feature-family ablations on the objective features table, using IDENTICAL splits
and seeds so differences are attributable to the removed family:
  all_features, without_colour, without_composition, without_stroke,
  without_segmentation, without_shape.
For each: mean/std validation macro-F1 across seeds, ECE, runtime, and the delta
vs all_features.

Representation-level ablations (deep-only / objective-only / generic-embedding /
finetuned-embedding / fusion) are provided by fusion.embedding_comparison — this
module does not duplicate them.

The column-selection logic is pure-numpy and CPU-tested; training uses sklearn.
Every result is written as JSON + CSV (matching source data for figures).
"""

from __future__ import annotations
import csv
import json
import time
from pathlib import Path

import numpy as np

from .dataset import CLASSES

META = {"image_id", "path", "split", "class"}

FAMILY_PREFIXES = {
    "colour": "colour.",
    "composition": "composition.",
    "stroke": "stroke.",
    "segmentation": "segmentation.",
    "shape": "shape.",
    "quality": "quality.",
}


def ablation_configs() -> list[dict]:
    return [
        {"name": "all_features", "drop_families": []},
        {"name": "without_colour", "drop_families": ["colour"]},
        {"name": "without_composition", "drop_families": ["composition"]},
        {"name": "without_stroke", "drop_families": ["stroke"]},
        {"name": "without_segmentation", "drop_families": ["segmentation"]},
        {"name": "without_shape", "drop_families": ["shape"]},
    ]


def select_columns(feature_names: list[str], drop_families: list[str]) -> list[int]:
    """Return indices of columns to KEEP after dropping the given families (pure)."""
    prefixes = [FAMILY_PREFIXES[f] for f in drop_families]
    return [i for i, name in enumerate(feature_names)
            if not any(name.startswith(p) for p in prefixes)]


def _load(features_csv):
    with open(features_csv, newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    names = [n for n in rows[0] if n not in META]
    data = {}
    for split in ("train", "valid"):
        sel = [r for r in rows if r["split"] == split]
        if not sel:
            raise ValueError(f"No rows for split {split!r}")
        X = np.asarray([[float(r[n]) if r[n] not in ("", "nan", "NaN") else np.nan
                         for n in names] for r in sel])
        y = np.asarray([CLASSES.index(r["class"]) for r in sel])
        data[split] = (X, y)
    return names, data


def run_feature_ablation(features_csv, output, seeds=(42, 123, 2026)) -> dict:
    """Train one consistent classifier per ablation config across seeds. sklearn."""
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - needs [ml]
        raise RuntimeError('Install ML support with: pip install -e ".[ml]"') from exc
    from .evaluation import compute_metrics

    names, data = _load(features_csv)
    Xtr, ytr = data["train"]
    Xva, yva = data["valid"]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for cfg in ablation_configs():
        keep = select_columns(names, cfg["drop_families"])
        f1s, eces, secs = [], [], []
        for seed in seeds:
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=seed))
            t0 = time.perf_counter()
            clf.fit(Xtr[:, keep], ytr)
            secs.append(time.perf_counter() - t0)
            proba = clf.predict_proba(Xva[:, keep])
            m = compute_metrics(yva, proba.argmax(1), proba)
            f1s.append(m["macro_f1"])
            eces.append(m["ece"])
        results.append({
            "configuration": cfg["name"], "dropped_families": cfg["drop_families"],
            "n_features": len(keep), "selection_split": "valid", "test_used": False,
            "mean_macro_f1": float(np.mean(f1s)), "std_macro_f1": float(np.std(f1s)),
            "mean_ece": float(np.mean(eces)), "mean_runtime_seconds": float(np.mean(secs)),
        })

    full = next(r for r in results if r["configuration"] == "all_features")
    for r in results:
        r["delta_macro_f1_vs_full"] = round(r["mean_macro_f1"] - full["mean_macro_f1"], 6)

    summary = {"seeds": list(seeds), "primary_metric": "mean validation macro-F1",
               "test_used": False, "results": results}
    (out / "ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "ablation.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=[
            "configuration", "n_features", "mean_macro_f1", "std_macro_f1",
            "mean_ece", "mean_runtime_seconds", "delta_macro_f1_vs_full"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    return summary
