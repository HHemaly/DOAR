"""
embedding_comparison.py — compare objective / generic / fine-tuned representations
(Item 5), aligned by sample_id.

Five configurations, all on the SAME leak-safe split, selected on validation
macro-F1 (test never used):
  1. objective features only
  2. generic embeddings only
  3. fine-tuned embeddings only
  4. objective + generic
  5. objective + fine-tuned

The assembly (aligning blocks by sample_id and concatenating) is pure-numpy and
CPU-tested. Training uses sklearn (lazy) — that path needs the [ml] extra and
real extracted features/embeddings from the dataset.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

from ..dataset import CLASSES

META = {"image_id", "path", "split", "class"}


# ---------------------------------------------------------------------------
# Indexed loaders: {split: {sample_id: (vector, label_index)}}
# ---------------------------------------------------------------------------

def load_features_indexed(features_csv) -> dict:
    with open(features_csv, newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    if not rows:
        raise ValueError("Empty feature table")
    names = [n for n in rows[0] if n not in META]
    out: dict = {}
    for row in rows:
        vec = [float(row[n]) if row[n] not in ("", "nan", "NaN") else np.nan for n in names]
        out.setdefault(row["split"], {})[row["image_id"]] = (
            np.asarray(vec, dtype=float), CLASSES.index(row["class"]))
    return out


def load_embeddings_indexed(npz_path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]
    ids = data["image_ids"].astype(str)
    splits = data["splits"].astype(str)
    labels = data["labels"].astype(str)
    out: dict = {}
    for i, sid in enumerate(ids):
        out.setdefault(splits[i], {})[sid] = (emb[i].astype(float), CLASSES.index(labels[i]))
    return out


# ---------------------------------------------------------------------------
# Assembly (pure) — align blocks by sample_id, concatenate feature vectors
# ---------------------------------------------------------------------------

def assemble_config(blocks: list[dict], split: str):
    """blocks: list of {split: {sample_id: (vec, label)}}. Returns
    (sample_ids, X_concat, y) aligned by sample_id intersection across blocks."""
    per_block = [b.get(split, {}) for b in blocks]
    if any(not pb for pb in per_block):
        raise ValueError(f"A block has no rows for split {split!r}")
    common = set(per_block[0])
    for pb in per_block[1:]:
        common &= set(pb)
    if not common:
        raise ValueError(f"No shared sample_ids across blocks for split {split!r}")
    sample_ids = sorted(common)
    y = None
    columns = []
    for pb in per_block:
        mat = np.vstack([pb[sid][0] for sid in sample_ids])
        columns.append(mat)
        labels = np.asarray([pb[sid][1] for sid in sample_ids])
        if y is None:
            y = labels
        elif not np.array_equal(y, labels):
            raise ValueError("Label disagreement across blocks for the same sample_id")
    return sample_ids, np.hstack(columns), y


CONFIGURATIONS = {
    "objective_only": ("features",),
    "generic_embeddings_only": ("generic",),
    "finetuned_embeddings_only": ("finetuned",),
    "objective_plus_generic": ("features", "generic"),
    "objective_plus_finetuned": ("features", "finetuned"),
}


def run_embedding_comparison(features_csv, generic_npz, finetuned_npz, output,
                             seed: int = 42) -> dict:
    """Train + validation-select a classifier for each of the 5 configurations.
    Requires sklearn."""
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - needs [ml]
        raise RuntimeError('Install ML support with: pip install -e ".[ml]"') from exc
    from ..evaluation import compute_metrics

    sources = {
        "features": load_features_indexed(features_csv),
        "generic": load_embeddings_indexed(generic_npz),
        "finetuned": load_embeddings_indexed(finetuned_npz),
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for name, blocks in CONFIGURATIONS.items():
        block_list = [sources[b] for b in blocks]
        _, x_train, y_train = assemble_config(block_list, "train")
        ids_valid, x_valid, y_valid = assemble_config(block_list, "valid")
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=seed))
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_valid)
        metrics = compute_metrics(y_valid, proba.argmax(1), proba)
        results.append({"configuration": name, "blocks": list(blocks),
                        "selection_split": "valid", "test_used": False,
                        "n_valid": len(ids_valid),
                        "validation_macro_f1": metrics["macro_f1"],
                        "metrics": metrics})

    results.sort(key=lambda r: -r["validation_macro_f1"])
    summary = {
        "primary_selection_metric": "validation_macro_f1",
        "test_used": False,
        "winner_configuration": results[0]["configuration"] if results else None,
        "configurations": results,
    }
    (out / "embedding_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
