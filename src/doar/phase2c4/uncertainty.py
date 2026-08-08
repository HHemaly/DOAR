"""Phase 2C.4A uncertainty reporting: image-level bootstrap confidence
intervals for macro balanced accuracy / macro F1 on the dev-eligible
cohort only (80 pilot drawings and, per class, single-digit-to-low-tens
positive counts is a small enough sample that a single point estimate
overstates precision -- see PHASE2C4A_VALIDATION_CORRECTION_REPORT.md
section 4).

Resampling is at the image (pilot_id) level, not the per-class-pair level,
so a drawing's correlated per-class labels move together across classes in
each resample -- the same structure the real evaluation has. Uses
`random.Random(seed)` for reproducibility; the seed is recorded in every
output so the interval can be regenerated exactly.
"""
from __future__ import annotations

import random

from ..phase2b.ontology import CLASS_NAMES
from .macro_metrics import balanced_accuracy_for_macro, f1_for_macro

DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_SEED = 20260807  # date this correction pass was written, for traceability


def _class_metrics_for_ids(ground_truth: dict[str, str], predicted: dict[str, bool],
                            pilot_ids: list[str]) -> dict:
    tp = fp = fn = tn = 0
    for pid in pilot_ids:
        truth = ground_truth.get(pid)
        if truth not in ("present", "absent") or pid not in predicted:
            continue
        pred = predicted[pid]
        if truth == "present" and pred:
            tp += 1
        elif truth == "absent" and pred:
            fp += 1
        elif truth == "present" and not pred:
            fn += 1
        else:
            tn += 1
    n_present, n_absent = tp + fn, tn + fp
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)
    specificity = tn / (tn + fp) if (tn + fp) else None
    bal_acc = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {"n_present": n_present, "n_absent": n_absent, "f1": f1, "balanced_accuracy": bal_acc}


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, round(p * (len(sorted_values) - 1))))
    return sorted_values[idx]


def bootstrap_macro_ci(cohort_pilot_ids: list[str],
                        ground_truth_by_class: dict[str, dict[str, str]],
                        predicted_by_class: dict[str, dict[str, bool]],
                        class_names: tuple[str, ...] = CLASS_NAMES,
                        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                        seed: int = DEFAULT_SEED) -> dict:
    """95% bootstrap CI (2.5th/97.5th percentile of the resample
    distribution) for macro F1 and macro balanced accuracy, using the same
    zero-division policy as `macro_metrics.macro_average_for_model_cohort`
    so the interval is comparable to the point estimate it brackets."""
    rng = random.Random(seed)
    n = len(cohort_pilot_ids)
    f1_samples, bal_samples = [], []
    for _ in range(n_bootstrap):
        sample_ids = [cohort_pilot_ids[rng.randrange(n)] for _ in range(n)]
        f1_vals, bal_vals = [], []
        for cls in class_names:
            m = _class_metrics_for_ids(ground_truth_by_class[cls], predicted_by_class[cls], sample_ids)
            f1m = f1_for_macro(m["f1"], m["n_present"])
            balm = balanced_accuracy_for_macro(m["balanced_accuracy"], m["n_present"], m["n_absent"])
            if f1m is not None:
                f1_vals.append(f1m)
            if balm is not None:
                bal_vals.append(balm)
        if f1_vals:
            f1_samples.append(sum(f1_vals) / len(f1_vals))
        if bal_vals:
            bal_samples.append(sum(bal_vals) / len(bal_vals))
    f1_samples.sort()
    bal_samples.sort()
    return {
        "n_bootstrap": n_bootstrap, "seed": seed, "n_images_in_cohort": n,
        "macro_f1_ci_low": _percentile(f1_samples, 0.025),
        "macro_f1_ci_high": _percentile(f1_samples, 0.975),
        "macro_balanced_accuracy_ci_low": _percentile(bal_samples, 0.025),
        "macro_balanced_accuracy_ci_high": _percentile(bal_samples, 0.975),
    }
