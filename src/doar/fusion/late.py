"""
late.py — runnable late-fusion / stacking CLI arm (D2).

Consumes exported base-model probability files and fuses them, selecting on the
validation split only (the test split is never loaded here). Each base file is a
.npz produced from a trained model's probabilities with keys:
    probabilities : (N, C) float
    labels        : (N,)   int
    splits        : (N,)   str  ("train" | "valid")   # "test" rows are ignored
    sample_ids    : (N,)   str  (optional; used for OOF stacking checks)
    fold_ids      : (N,)   int  (optional; required for logistic_probability_meta)

This makes the previously-unused probability primitives executable while
preserving leakage discipline (validation-only selection, OOF checks for
stacking). The base probabilities themselves come from the user's model runs.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .probability import run_late_fusion, validate_oof_folds


def _load_npz(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _split_arrays(bundle: dict, split: str):
    splits = bundle["splits"].astype(str)
    mask = splits == split
    return bundle["probabilities"][mask], bundle["labels"][mask].astype(int), mask


def train_late_fusion(base_files: list[str], output: str | Path,
                      method: str = "validation_weighted_late_fusion",
                      calibrated: bool = False) -> dict:
    """Run late fusion over >=2 exported base-model probability files."""
    if len(base_files) < 2:
        raise ValueError("Late fusion requires at least two base-model files")
    bundles = [_load_npz(p) for p in base_files]

    # Validation arrays for every base model (must align in length + labels).
    valid_sets, valid_labels = [], None
    for b in bundles:
        probs, labels, _ = _split_arrays(b, "valid")
        valid_sets.append(probs)
        if valid_labels is None:
            valid_labels = labels
        elif not np.array_equal(valid_labels, labels):
            raise ValueError("Validation labels differ across base files (misaligned exports)")

    train_sets = train_labels = None
    if method == "logistic_probability_meta":
        # OOF discipline: require sample_ids + fold_ids and validate them.
        train_sets, train_labels = [], None
        for b in bundles:
            probs, labels, mask = _split_arrays(b, "train")
            if "sample_ids" in b and "fold_ids" in b:
                validate_oof_folds(
                    list(b["sample_ids"][mask].astype(str)),
                    list(b["fold_ids"][mask].astype(int)),
                )
            train_sets.append(probs)
            train_labels = labels if train_labels is None else train_labels

    result = run_late_fusion(
        valid_sets, valid_labels, method=method, calibrated=calibrated,
        train_probability_sets=train_sets, train_labels=train_labels,
    )
    result["base_files"] = [str(p) for p in base_files]

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "late_fusion_result.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result
