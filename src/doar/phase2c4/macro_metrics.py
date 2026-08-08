"""Phase 2C.4A correction: macro-averaging with an explicit, documented
zero-division policy, and a fixed 10-class denominator for every model.

Bug this module fixes (found on audit): `phase2b.evaluation.compute_class_metrics`
correctly returns `f1=None` when F1 is mathematically undefined (no usable
positive prediction was made at all, or precision and recall are both
exactly 0.0). The Phase 2C.4 report's macro-F1 table was computed by
averaging only the classes where `f1` was not None -- silently dropping
classes a detector never fired on. A detector that never predicts a class
positive gets zero macro-F1 credit *and* zero penalty under that scheme,
which inflates its macro score relative to a detector that fires but is
simply wrong. This module never touches `compute_class_metrics` itself
(other phases rely on its existing None-propagation convention); it only
defines how *this* module's macro aggregation treats an undefined F1.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..phase2b.evaluation import MIN_POSITIVE_SUPPORT
from ..phase2b.ontology import CLASS_NAMES


def f1_for_macro(f1: float | None, n_present: int) -> float | None:
    """Zero-division policy for macro-F1 averaging only (does not mutate the
    per-class-per-cohort `f1` field anywhere else).

    - `f1` already defined -> used as-is.
    - `f1` undefined (None) and `n_present > 0` -> the class has genuine
      positive ground truth in this cohort but the detector produced no
      usable positive prediction (or predicted positive with precision and
      recall both exactly 0.0) -- scored 0.0, a real miss, not dropped.
    - `f1` undefined and `n_present == 0` -> no genuine positive ground
      truth exists for this class in this cohort at all; there is no
      detection task to score, so the class is excluded from the macro
      average (returns None, caller must exclude, not average-as-zero).
    """
    if f1 is not None:
        return f1
    return 0.0 if n_present > 0 else None


def balanced_accuracy_for_macro(balanced_accuracy: float | None, n_present: int, n_absent: int) -> float | None:
    """Same policy, generalized to balanced accuracy. Undefined only when
    `n_present == 0` (recall undefined) or `n_absent == 0` (specificity
    undefined) in this cohort -- both mean there is no full detection task
    to score for that class in that cohort, so it is excluded (None), never
    silently zeroed (unlike F1, a missing negative or positive population
    is not the detector's fault)."""
    if balanced_accuracy is not None:
        return balanced_accuracy
    return None if (n_present == 0 or n_absent == 0) else balanced_accuracy


@dataclass
class MacroSummary:
    model: str
    cohort: str
    n_classes_total: int
    n_classes_included_f1: int
    n_classes_excluded_f1: int
    excluded_f1_classes: list[str]
    macro_f1: float | None
    n_classes_included_bal_acc: int
    n_classes_excluded_bal_acc: int
    excluded_bal_acc_classes: list[str]
    macro_balanced_accuracy: float | None
    low_support_classes: list[str]


def macro_average_for_model_cohort(rows: list[dict], class_names: tuple[str, ...] = CLASS_NAMES) -> MacroSummary:
    """`rows`: the subset of `phase2c4_model_class_cohort_metrics.csv` rows
    for exactly one (model, cohort) pair. Requires all `class_names` present
    (fixed 10-class denominator) -- raises if any class is missing, rather
    than silently averaging over fewer classes than the ontology defines."""
    by_class = {r["class_name"]: r for r in rows}
    missing = [c for c in class_names if c not in by_class]
    if missing:
        raise ValueError(f"macro_average_for_model_cohort: missing classes {missing} "
                          f"-- every model must report all {len(class_names)} ontology classes")
    if not rows:
        raise ValueError("macro_average_for_model_cohort: no rows given")
    model, cohort = rows[0]["model"], rows[0]["cohort"]

    f1_values, excluded_f1 = [], []
    bal_values, excluded_bal = [], []
    low_support = []
    for cls in class_names:
        r = by_class[cls]
        n_present = int(r["n_present"])
        n_absent = int(r["n_absent"])
        if n_present < MIN_POSITIVE_SUPPORT:
            low_support.append(cls)

        f1 = f1_for_macro(_as_float_or_none(r["f1"]), n_present)
        if f1 is None:
            excluded_f1.append(cls)
        else:
            f1_values.append(f1)

        bal = balanced_accuracy_for_macro(_as_float_or_none(r["balanced_accuracy"]), n_present, n_absent)
        if bal is None:
            excluded_bal.append(cls)
        else:
            bal_values.append(bal)

    return MacroSummary(
        model=model, cohort=cohort, n_classes_total=len(class_names),
        n_classes_included_f1=len(f1_values), n_classes_excluded_f1=len(excluded_f1),
        excluded_f1_classes=excluded_f1,
        macro_f1=(sum(f1_values) / len(f1_values) if f1_values else None),
        n_classes_included_bal_acc=len(bal_values), n_classes_excluded_bal_acc=len(excluded_bal),
        excluded_bal_acc_classes=excluded_bal,
        macro_balanced_accuracy=(sum(bal_values) / len(bal_values) if bal_values else None),
        low_support_classes=low_support,
    )


def _as_float_or_none(v) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def macro_summaries_by_model(all_rows: list[dict], cohort: str,
                              class_names: tuple[str, ...] = CLASS_NAMES) -> list[MacroSummary]:
    """One `MacroSummary` per model, restricted to `cohort`, sorted by
    macro_balanced_accuracy descending (None sorts last) -- the ranking
    this module exists to make correct."""
    models = sorted({r["model"] for r in all_rows})
    summaries = []
    for model in models:
        rows = [r for r in all_rows if r["model"] == model and r["cohort"] == cohort]
        if not rows:
            continue
        summaries.append(macro_average_for_model_cohort(rows, class_names))
    summaries.sort(key=lambda s: (s.macro_balanced_accuracy is None, -(s.macro_balanced_accuracy or 0.0)))
    return summaries
