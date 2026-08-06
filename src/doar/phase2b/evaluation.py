"""Phase 2B honest evaluation: per-class support counts, precision/recall/F1,
threshold sensitivity, and a same-detector-vs-random-ranking check (a
simple, dependency-free stand-in for AUC that only needs the True/False
labels and a ranking, not a full sklearn ROC implementation).

Ground truth statuses `uncertain` and `not_assessable` are excluded from
precision/recall computation (they are not usable positive/negative
labels) but their counts are still reported -- silently dropping them
without recording how many were dropped would misrepresent the real
support available.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

MIN_POSITIVE_SUPPORT = 5  # documented in docs/PHASE2B_ANNOTATION_PROTOCOL.md


@dataclass
class ClassMetrics:
    class_name: str
    n_present: int
    n_absent: int
    n_uncertain: int
    n_not_assessable: int
    n_usable: int  # present + absent, the only ones usable for P/R/F1
    sufficient_support: bool
    precision: float | None
    recall: float | None
    f1: float | None
    tp: int
    fp: int
    fn: int
    tn: int


def _usable_pairs(ground_truth: dict[str, str], predicted_positive: dict[str, bool]) -> list[tuple[bool, bool]]:
    """Returns [(is_positive_truth, is_positive_pred), ...] for pilot_ids
    whose ground truth is present/absent (usable) and that have a
    prediction. Skips uncertain/not_assessable ground truth and any
    pilot_id missing a prediction."""
    pairs = []
    for pilot_id, truth in ground_truth.items():
        if truth not in ("present", "absent"):
            continue
        if pilot_id not in predicted_positive:
            continue
        pairs.append((truth == "present", predicted_positive[pilot_id]))
    return pairs


def compute_class_metrics(class_name: str, ground_truth: dict[str, str],
                           predicted_positive: dict[str, bool]) -> ClassMetrics:
    n_present = sum(1 for v in ground_truth.values() if v == "present")
    n_absent = sum(1 for v in ground_truth.values() if v == "absent")
    n_uncertain = sum(1 for v in ground_truth.values() if v == "uncertain")
    n_na = sum(1 for v in ground_truth.values() if v == "not_assessable")
    pairs = _usable_pairs(ground_truth, predicted_positive)
    tp = sum(1 for truth, pred in pairs if truth and pred)
    fp = sum(1 for truth, pred in pairs if not truth and pred)
    fn = sum(1 for truth, pred in pairs if truth and not pred)
    tn = sum(1 for truth, pred in pairs if not truth and not pred)
    sufficient = n_present >= MIN_POSITIVE_SUPPORT
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)
    return ClassMetrics(
        class_name=class_name, n_present=n_present, n_absent=n_absent,
        n_uncertain=n_uncertain, n_not_assessable=n_na, n_usable=len(pairs),
        sufficient_support=sufficient, precision=precision, recall=recall, f1=f1,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def ranking_separation(ground_truth: dict[str, str], similarity: dict[str, float]) -> float | None:
    """A dependency-free stand-in for AUC: the fraction of (positive,
    negative) pairs where the positive example's similarity score is
    strictly higher than the negative's (ties count as 0.5). 0.5 = no
    better than random ranking; 1.0 = perfect separation. Returns None if
    there are no usable positive or negative examples."""
    positives = [similarity[k] for k, v in ground_truth.items() if v == "present" and k in similarity]
    negatives = [similarity[k] for k, v in ground_truth.items() if v == "absent" and k in similarity]
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def threshold_sensitivity(ground_truth: dict[str, str], similarity: dict[str, float],
                           thresholds: list[float]) -> list[dict]:
    """Recomputes precision/recall/F1 at each candidate threshold, so a
    single chosen operating point never hides how unstable the result is
    at nearby thresholds."""
    rows = []
    for t in thresholds:
        predicted_positive = {k: (v >= t) for k, v in similarity.items()}
        m = compute_class_metrics("_", ground_truth, predicted_positive)
        rows.append({"threshold": t, "precision": m.precision, "recall": m.recall,
                     "f1": m.f1, "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn})
    return rows


def write_csv(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
