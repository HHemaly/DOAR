"""Phase 2C.4A secondary calibration experiment.

The original Phase 2C.4 zero-shot default-threshold results
(`artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv`, committed at
`39e65b8`) remain frozen and are never overwritten by this module. This is a
SEPARATE pass using only the raw per-class confidence scores already
recorded in `artifacts/phase2c4/raw_predictions_{owlv2,grounding_dino}.csv`
-- no new inference, no new downloads, no model re-run.

Hard constraint on what a threshold sweep can validly recover from that
data: a row's recorded score is exactly 0.0 whenever a class was NOT
detected at the model's original operating threshold --
`post_process_grounded_object_detection` discards below-threshold
detections before a score is ever written, so an undetected class's true
confidence is only known to be *below* the original threshold, never the
exact value. A candidate threshold t is only faithfully re-derivable from
this data when t >= the model's original operating threshold: a row scored
0.0 is correctly "not detected" at any such t (0.0 < t). Candidate
thresholds below the original threshold cannot be recovered from this data
and are never attempted here -- which happens to align with the requested
precision-oriented (not recall-maximizing) calibration goal: this module
can only ever raise the bar, never lower it.

No threshold selection in this module ever looks at the locked-test
cohort. `freeze_operating_points` is the single point where a decision
becomes final; only `apply_frozen_thresholds` (called afterward, on the
already-frozen dict) may touch the locked-test cohort, and only to report
descriptively -- never to revise a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..phase2b.evaluation import compute_class_metrics
from ..phase2c2.evaluation import compute_specificity
from .evaluation import balanced_accuracy

OWLV2_ORIGINAL_THRESHOLD = 0.1
GROUNDING_DINO_ORIGINAL_THRESHOLD = 0.25

# Predeclared before any dev-cohort calibration result was computed, and
# never adjusted after seeing per-class outcomes (see module docstring and
# PHASE2C4A_VALIDATION_CORRECTION_REPORT.md section 5).
CANDIDATE_THRESHOLDS = {
    "owlv2": (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    "grounding_dino": (0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
}

# Predeclared minimum precision for a class's detector output to be
# considered a candidate evidence source for a psychological rule -- a
# false "circle present" is not a neutral error in this system, so the
# bar favors precision over recall by design, per instruction.
MIN_PRECISION_FOR_RULE_EVIDENCE = 0.6


def sweep_thresholds(class_name: str, ground_truth: dict[str, str], scores: dict[str, float],
                      thresholds: tuple[float, ...]) -> list[dict]:
    """Recomputes full metrics at each candidate threshold from the same
    recorded `scores` dict -- no new inference."""
    rows = []
    for t in thresholds:
        predicted_positive = {pid: (score >= t) for pid, score in scores.items()}
        m = compute_class_metrics(class_name, ground_truth, predicted_positive)
        specificity = compute_specificity(m.tn, m.fp)
        bal_acc = balanced_accuracy(m.recall, specificity)
        rows.append({
            "class_name": class_name, "threshold": t,
            "n_present": m.n_present, "n_absent": m.n_absent,
            "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
            "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "specificity": specificity, "balanced_accuracy": bal_acc,
        })
    return rows


def select_operating_point(threshold_rows: list[dict],
                            min_precision: float = MIN_PRECISION_FOR_RULE_EVIDENCE) -> dict:
    """Predeclared, fixed selection rule -- identical for every class and
    model, applied mechanically: among candidate thresholds whose precision
    clears `min_precision`, pick the SMALLEST such threshold (the most
    recall-preserving choice that still clears the precision bar); ties
    broken by higher balanced accuracy, then higher recall. If no candidate
    threshold clears the bar, the class abstains -- no calibrated operating
    point is recommended for it at this precision bar, rather than forcing
    a low-precision choice onto a rule-evidence pipeline."""
    candidates = [r for r in threshold_rows if r["precision"] is not None and r["precision"] >= min_precision]
    if not candidates:
        return {"decision": "abstain", "threshold": None,
                "reason": f"no candidate threshold reached precision >= {min_precision}"}
    candidates = sorted(candidates, key=lambda r: (
        r["threshold"], -(r["balanced_accuracy"] or 0.0), -(r["recall"] or 0.0)))
    chosen = candidates[0]
    return {"decision": "calibrated", "threshold": chosen["threshold"], "row": chosen}


@dataclass
class FrozenOperatingPoint:
    model: str
    class_name: str
    original_threshold: float
    decision: str  # "calibrated" | "abstain"
    frozen_threshold: float | None
    dev_precision: float | None
    dev_recall: float | None
    dev_balanced_accuracy: float | None
    dev_n_present: int
    low_support: bool
    reason: str | None = field(default=None)


def freeze_operating_points(model: str, original_threshold: float,
                             ground_truth_by_class: dict[str, dict[str, str]],
                             scores_by_class: dict[str, dict[str, float]],
                             thresholds: tuple[float, ...],
                             min_positive_support: int,
                             min_precision: float = MIN_PRECISION_FOR_RULE_EVIDENCE) -> list[FrozenOperatingPoint]:
    """Runs the full sweep + predeclared selection for every class in
    `ground_truth_by_class`, using ONLY the supplied (dev-cohort) ground
    truth and scores. This is the single freeze point: once this function
    returns, every threshold in its output is final for this correction
    pass -- `apply_frozen_thresholds` below may only ever consume this
    output, never recompute a selection."""
    out = []
    for cls, gt in ground_truth_by_class.items():
        scores = scores_by_class[cls]
        rows = sweep_thresholds(cls, gt, scores, thresholds)
        pick = select_operating_point(rows, min_precision)
        n_present = sum(1 for v in gt.values() if v == "present")
        if pick["decision"] == "calibrated":
            r = pick["row"]
            out.append(FrozenOperatingPoint(
                model=model, class_name=cls, original_threshold=original_threshold,
                decision="calibrated", frozen_threshold=r["threshold"],
                dev_precision=r["precision"], dev_recall=r["recall"],
                dev_balanced_accuracy=r["balanced_accuracy"], dev_n_present=n_present,
                low_support=n_present < min_positive_support))
        else:
            out.append(FrozenOperatingPoint(
                model=model, class_name=cls, original_threshold=original_threshold,
                decision="abstain", frozen_threshold=None,
                dev_precision=None, dev_recall=None, dev_balanced_accuracy=None,
                dev_n_present=n_present, low_support=n_present < min_positive_support,
                reason=pick["reason"]))
    return out


def apply_frozen_thresholds(frozen: list[FrozenOperatingPoint],
                             locked_ground_truth_by_class: dict[str, dict[str, str]],
                             locked_scores_by_class: dict[str, dict[str, float]]) -> list[dict]:
    """Descriptive-only: scores the already-frozen thresholds against the
    locked-test cohort. Never selects, adjusts, or re-derives a threshold
    -- any class whose `decision == "abstain"` stays abstained here too,
    reported as such rather than silently defaulting to some threshold."""
    rows = []
    for fop in frozen:
        if fop.decision != "calibrated":
            rows.append({"model": fop.model, "class_name": fop.class_name, "decision": fop.decision,
                         "frozen_threshold": None, "n_present": None, "n_absent": None,
                         "precision": None, "recall": None, "balanced_accuracy": None})
            continue
        gt = locked_ground_truth_by_class[fop.class_name]
        scores = locked_scores_by_class[fop.class_name]
        predicted_positive = {pid: (score >= fop.frozen_threshold) for pid, score in scores.items()}
        m = compute_class_metrics(fop.class_name, gt, predicted_positive)
        specificity = compute_specificity(m.tn, m.fp)
        bal_acc = balanced_accuracy(m.recall, specificity)
        rows.append({"model": fop.model, "class_name": fop.class_name, "decision": "calibrated",
                     "frozen_threshold": fop.frozen_threshold, "n_present": m.n_present, "n_absent": m.n_absent,
                     "precision": m.precision, "recall": m.recall, "balanced_accuracy": bal_acc})
    return rows
