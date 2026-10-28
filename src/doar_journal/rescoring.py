"""X1: re-score SAVED validation predictions on source-shared vs independent images.

No training, no test access. Inputs are the existing E1 run folders on Drive
(each containing `best_valid_predictions.csv` with columns image_id,
true_class, predicted_class, prob_<class>) and the T0 manifest that maps
image_id to filename.

Caveat recorded with every output: shared-source and independent validation
images may differ in difficulty for reasons other than leakage (class mix,
source family, acquisition). Macro-F1 weights classes equally, and per-class
recall plus a source-family breakdown are reported so that this alternative
explanation can be inspected rather than assumed away.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .leakage import path_of, source_key, split_of
from .stats import CLASSES, bootstrap_metric, macro_f1, unpaired_bootstrap_diff


def shared_source_valid_ids(rows: list[dict], evidence: str = "rf_stem") -> tuple[set[str], set[str]]:
    """(shared, independent) validation image ids. `shared` = validation images
    whose Roboflow stem (or, if evidence='any', plain name key too) occurs in train."""
    train_rf = {source_key(path_of(r)).rf_stem for r in rows if split_of(r) == "train"}
    train_names = {source_key(path_of(r)).name_key for r in rows if split_of(r) == "train"}
    shared, indep = set(), set()
    for r in rows:
        if split_of(r) != "valid":
            continue
        k = source_key(path_of(r))
        hit = (k.rf_stem is not None and k.rf_stem in train_rf)
        if evidence == "any":
            hit = hit or k.name_key in train_names
        (shared if hit else indep).add(r["image_id"])
    return shared, indep


def source_family(path: str) -> str:
    """Coarse filename family (proxy for the contributing source dataset)."""
    import re
    k = source_key(path)
    if k.is_roboflow:
        if re.match(r"(happy|sad|fear|angry)_\d+_\d+$", k.name_key):
            return "rf_class_n_n"
        if re.match(r"[hsfa]\d+$", k.name_key):
            return "rf_letter_n"
        if k.name_key.startswith("photo"):
            return "rf_photo"
        return "rf_other"
    if re.match(r"[hsfa]\d+$", k.name_key):
        return "plain_letter_n"
    return "plain_other"


def read_predictions(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rescore_run(pred_rows: list[dict], shared: set[str], indep: set[str], n_boot: int = 10_000,
                groups: dict[str, str] | None = None) -> dict:
    """`groups` (image_id -> group) switches all intervals to a cluster bootstrap."""
    ids = {r["image_id"] for r in pred_rows}
    if not ids <= (shared | indep):
        unknown = sorted(ids - shared - indep)[:5]
        raise ValueError(f"predictions contain non-validation image ids, e.g. {unknown}")
    by = {r["image_id"]: r for r in pred_rows}
    def arr(sub):
        sub = sorted(sub & ids)
        g = None if groups is None else [groups.get(i, i) for i in sub]
        return (np.array([by[i]["true_class"] for i in sub]),
                np.array([by[i]["predicted_class"] for i in sub]), g)
    ya, pa, ga = arr(shared)
    yb, pb, gb = arr(indep)
    yall, pall, gall = arr(shared | indep)
    out = {
        "n_all": len(yall), "n_shared": len(ya), "n_independent": len(yb),
        "n_groups_all": len(set(gall)) if gall is not None else len(yall),
        "macro_f1_all": bootstrap_metric(yall, pall, n_boot=n_boot, groups=gall).as_dict(),
        "macro_f1_shared": bootstrap_metric(ya, pa, n_boot=n_boot, groups=ga).as_dict(),
        "macro_f1_independent": bootstrap_metric(yb, pb, n_boot=n_boot, groups=gb).as_dict(),
        "diff_shared_minus_independent": unpaired_bootstrap_diff(ya, pa, yb, pb, n_boot=n_boot,
                                                                 groups_a=ga, groups_b=gb),
        "interpretation": ("exploratory association: shared-source and independent validation images may "
                           "also differ in class mix, source family and difficulty"),
        "accuracy_shared": float((ya == pa).mean()) if len(ya) else None,
        "accuracy_independent": float((yb == pb).mean()) if len(yb) else None,
    }
    for name, (y, p) in {"shared": (ya, pa), "independent": (yb, pb)}.items():
        out[f"recall_by_class_{name}"] = {c: (float((p[y == c] == c).mean()) if (y == c).any() else None)
                                          for c in CLASSES}
        out[f"class_counts_{name}"] = dict(Counter(y.tolist()))
    return out


def find_runs(root: str | Path) -> list[Path]:
    """All `best_valid_predictions.csv` below `root` (E1-A2 probe outputs, E1-B runs, E1-C runs).

    Walks the tree once and does not descend into folders that only hold
    embeddings, figures or qualitative images: listing those on a mounted
    Google Drive is slow and they never contain prediction files."""
    import os
    skip = {"embeddings", "figures", "qualitative", "REPORTS", "panels", "contact_sheets"}
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        if "best_valid_predictions.csv" in filenames:
            found.append(Path(dirpath) / "best_valid_predictions.csv")
    return sorted(found)


def macro_f1_from_rows(rows: list[dict]) -> float:
    return macro_f1([r["true_class"] for r in rows], [r["predicted_class"] for r in rows])
