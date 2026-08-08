"""Phase 2C.2 evaluation: wraps Phase 2B's own, unchanged
compute_class_metrics/ranking_separation (src/doar/phase2b/evaluation.py)
-- never reimplements them -- and adds what Phase 2C.2 specifically needs:
specificity, false-positive/false-negative pilot_id lists, per-cohort
evaluation (full/dev-eligible/locked-test), classical-CV evaluation for
`circle` (Phase 2B computed this ad hoc and never scripted it -- this is
the first reusable, tested version), and a comparison against Phase 2B's
original 20-image findings.

No threshold is tuned anywhere in this module -- `predicted_positive_from_raw`
reads the `__status == "detected"` field Phase 2B's own fixed
detect_threshold/uncertain_margin already produced; nothing here searches
for a better threshold.
"""
from __future__ import annotations

from ..phase2b.evaluation import compute_class_metrics, ranking_separation
from ..phase2b.ontology import CLASS_NAMES
from .cohorts import DEV_ELIGIBLE, FULL, LOCKED_TEST, split_pilot_ids_by_cohort
from .ground_truth import genuine_human_ground_truth

CLIP_BASELINE = "clip_zero_shot"
CLASSICAL_CV_BASELINE = "classical_cv_circularity"
CLASSICAL_CV_CLASS = "circle"  # the only class the classical-CV baseline covers


def predicted_positive_from_clip(raw_predictions: dict[str, dict], class_name: str) -> dict[str, bool]:
    return {pid: row[f"{class_name}__status"] == "detected" for pid, row in raw_predictions.items()}


def similarity_from_clip(raw_predictions: dict[str, dict], class_name: str) -> dict[str, float]:
    return {pid: float(row[f"{class_name}__similarity"]) for pid, row in raw_predictions.items()}


def predicted_positive_from_classical_cv(raw_predictions: dict[str, dict]) -> dict[str, bool]:
    return {pid: str(row["circle_classical__detected"]).strip().lower() == "true"
            for pid, row in raw_predictions.items()}


def similarity_from_classical_cv(raw_predictions: dict[str, dict]) -> dict[str, float]:
    """Circularity score (0..1, 1.0 = perfect circle) used as the
    ranking-independent variable for classical-CV -- not a probability,
    but the same role `similarity` plays for CLIP: a higher value should
    rank more circle-like."""
    return {pid: float(row["circle_classical__circularity"]) for pid, row in raw_predictions.items()}


def compute_specificity(tn: int, fp: int) -> float | None:
    return tn / (tn + fp) if (tn + fp) else None


def find_fp_fn_pilot_ids(ground_truth: dict[str, str], predicted_positive: dict[str, bool]) -> dict:
    false_positives, false_negatives = [], []
    for pid, truth in ground_truth.items():
        if truth not in ("present", "absent") or pid not in predicted_positive:
            continue
        pred = predicted_positive[pid]
        if truth == "absent" and pred:
            false_positives.append(pid)
        elif truth == "present" and not pred:
            false_negatives.append(pid)
    return {"false_positive_pilot_ids": sorted(false_positives),
            "false_negative_pilot_ids": sorted(false_negatives)}


def evaluate_one(class_name: str, baseline: str, ground_truth_full: dict[str, str],
                  predicted_positive_full: dict[str, bool], similarity_full: dict[str, float],
                  cohort_name: str, cohort_pilot_ids: list[str]) -> dict:
    """Restricts the full-cohort ground truth/predictions to `cohort_pilot_ids`
    and computes the complete metric set for that slice only."""
    cohort_set = set(cohort_pilot_ids)
    gt = {pid: s for pid, s in ground_truth_full.items() if pid in cohort_set}
    pred = {pid: p for pid, p in predicted_positive_full.items() if pid in cohort_set}
    sim = {pid: v for pid, v in similarity_full.items() if pid in cohort_set}

    m = compute_class_metrics(class_name, gt, pred)
    sep = ranking_separation(gt, sim)
    specificity = compute_specificity(m.tn, m.fp)
    fp_fn = find_fp_fn_pilot_ids(gt, pred)

    return {
        "class_name": class_name,
        "baseline": baseline,
        "cohort": cohort_name,
        "n_images_in_cohort": len(cohort_pilot_ids),
        "n_present": m.n_present,
        "n_absent": m.n_absent,
        "n_uncertain": m.n_uncertain,
        "n_not_assessable": m.n_not_assessable,
        "n_usable_for_metrics": m.n_usable,
        "sufficient_support": m.sufficient_support,
        "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "specificity": specificity,
        "ranking_separation": sep,
        **fp_fn,
    }


def evaluate_all(store: dict, raw_predictions: dict[str, dict], mapping_rows: list[dict]) -> list[dict]:
    """Top-level orchestration: every ontology class x CLIP baseline x 3
    cohorts, plus the classical-CV baseline x 3 cohorts for `circle` only.
    Ground truth is always genuine Phase 2C.1 human annotation
    (ground_truth.genuine_human_ground_truth) -- never the legacy Phase 2B
    provisional labels."""
    cohorts = split_pilot_ids_by_cohort(mapping_rows)
    results = []
    for cls in CLASS_NAMES:
        gt = genuine_human_ground_truth(store, cls)
        clip_pred = predicted_positive_from_clip(raw_predictions, cls)
        clip_sim = similarity_from_clip(raw_predictions, cls)
        for cohort_name in (FULL, DEV_ELIGIBLE, LOCKED_TEST):
            results.append(evaluate_one(cls, CLIP_BASELINE, gt, clip_pred, clip_sim,
                                         cohort_name, cohorts[cohort_name]))
        if cls == CLASSICAL_CV_CLASS:
            cv_pred = predicted_positive_from_classical_cv(raw_predictions)
            cv_sim = similarity_from_classical_cv(raw_predictions)
            for cohort_name in (FULL, DEV_ELIGIBLE, LOCKED_TEST):
                results.append(evaluate_one(cls, CLASSICAL_CV_BASELINE, gt, cv_pred, cv_sim,
                                             cohort_name, cohorts[cohort_name]))
    return results


def compare_to_phase2b_20(new_results: list[dict], old_per_class_rows: list[dict]) -> list[dict]:
    """Compares this round's full_80-cohort CLIP results against Phase 2B's
    original 20-image `per_class_metrics.csv` rows, per class. Both sides
    are already-computed metrics -- this performs no new evaluation, only a
    side-by-side comparison, and never touches the locked-test subset
    specially (Phase 2B's original 20 were themselves not split by
    original_split at all)."""
    old_by_class = {r["class"]: r for r in old_per_class_rows}
    new_by_class = {r["class_name"]: r for r in new_results
                     if r["baseline"] == CLIP_BASELINE and r["cohort"] == FULL}
    comparisons = []
    for cls in CLASS_NAMES:
        old, new = old_by_class.get(cls), new_by_class.get(cls)
        if old is None or new is None:
            continue

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        comparisons.append({
            "class_name": cls,
            "old_n_present_20": int(old["n_present"]),
            "new_n_present_80": new["n_present"],
            "old_precision": _f(old["precision"]),
            "new_precision": new["precision"],
            "old_recall": _f(old["recall"]),
            "new_recall": new["recall"],
            "old_ranking_separation": _f(old["ranking_separation_vs_random_0.5"]),
            "new_ranking_separation": new["ranking_separation"],
            "old_sufficient_support": str(old["sufficient_support"]).strip().lower() == "true",
            "new_sufficient_support": new["sufficient_support"],
        })
    return comparisons
