"""Phase 2C.4: generalizes Phase 2C.2's evaluation machinery
(src/doar/phase2c2/evaluation.py) from "CLIP + classical-CV circle only" to
"any model that produces a per-class {pilot_id: bool} presence prediction
and, optionally, a {pilot_id: float} ranking score." Reuses, never
duplicates, phase2b.evaluation.compute_class_metrics/ranking_separation,
phase2c1.ground_truth-equivalent extraction (via phase2c2.ground_truth),
and phase2c2.cohorts's exact three-cohort split.

No threshold tuning anywhere in this module: `predicted_positive` is always
supplied by the caller, already computed from each model's own documented
default operating point -- this module only scores an already-made
decision, never searches for a better one.
"""
from __future__ import annotations

from ..phase2b.evaluation import compute_class_metrics, ranking_separation
from ..phase2b.ontology import CLASS_NAMES
from ..phase2c2.cohorts import DEV_ELIGIBLE, FULL, LOCKED_TEST, split_pilot_ids_by_cohort
from ..phase2c2.evaluation import compute_specificity, find_fp_fn_pilot_ids
from ..phase2c2.ground_truth import genuine_human_ground_truth

COHORT_NAMES = (FULL, DEV_ELIGIBLE, LOCKED_TEST)


def balanced_accuracy(recall: float | None, specificity: float | None) -> float | None:
    """(recall + specificity) / 2 -- the requested class-imbalance-robust
    metric absent from phase2b.evaluation.ClassMetrics. None if either
    input is unavailable (mirrors every other metric's own None-propagation
    convention rather than silently substituting a default)."""
    if recall is None or specificity is None:
        return None
    return (recall + specificity) / 2


def evaluate_model_class_cohort(model_name: str, class_name: str,
                                 ground_truth_full: dict[str, str],
                                 predicted_positive_full: dict[str, bool],
                                 similarity_full: dict[str, float] | None,
                                 cohort_name: str, cohort_pilot_ids: list[str]) -> dict:
    cohort_set = set(cohort_pilot_ids)
    gt = {pid: s for pid, s in ground_truth_full.items() if pid in cohort_set}
    pred = {pid: p for pid, p in predicted_positive_full.items() if pid in cohort_set}
    sim = ({pid: v for pid, v in similarity_full.items() if pid in cohort_set}
           if similarity_full is not None else {})

    m = compute_class_metrics(class_name, gt, pred)
    sep = ranking_separation(gt, sim) if sim else None
    specificity = compute_specificity(m.tn, m.fp)
    bal_acc = balanced_accuracy(m.recall, specificity)
    fp_fn = find_fp_fn_pilot_ids(gt, pred)

    return {
        "model": model_name,
        "class_name": class_name,
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
        "balanced_accuracy": bal_acc,
        "ranking_separation": sep,
        **fp_fn,
    }


def evaluate_model(model_name: str, store: dict, mapping_rows: list[dict],
                    predicted_positive_by_class: dict[str, dict[str, bool]],
                    similarity_by_class: dict[str, dict[str, float]] | None = None) -> list[dict]:
    """Evaluates one model across all 10 ontology classes x 3 cohorts.
    `predicted_positive_by_class`/`similarity_by_class`: {class_name: {pilot_id: value}},
    already derived from that model's own fixed/documented operating point --
    this function performs no thresholding of its own."""
    cohorts = split_pilot_ids_by_cohort(mapping_rows)
    similarity_by_class = similarity_by_class or {}
    results = []
    for cls in CLASS_NAMES:
        if cls not in predicted_positive_by_class:
            continue
        gt = genuine_human_ground_truth(store, cls)
        pred = predicted_positive_by_class[cls]
        sim = similarity_by_class.get(cls)
        for cohort_name in COHORT_NAMES:
            results.append(evaluate_model_class_cohort(
                model_name, cls, gt, pred, sim, cohort_name, cohorts[cohort_name]))
    return results
