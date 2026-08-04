"""Deterministic evidence-to-rule engine (Phase 1).

Evaluates `RuleV2` entries (`rule_schema.py`) against an `EvidenceSet`
(`evidence_schema.py`) and returns one `RuleResult` per rule, using the
full outcome vocabulary the task requires:

    triggered, not_triggered, insufficient_evidence, not_applicable,
    extractor_unavailable, requires_manual_review, conflicting_evidence

This is a new, additive module. It does not replace or modify
`rules.py::evaluate_rules`, which remains the tested, live path for the
existing `analyze_image` pipeline (unchanged, still produces
`weak_support`/`not_matched`/`missing_detector`/`not_evaluated`). This
engine is the traceable Phase-1 path: same 19 rules, same underlying
measurements, a stricter and more explicit outcome/evidence contract.

Hard invariant (task requirement, mechanically enforced here): **a
missing extractor must never produce a passing or negative rule result.**
Any required feature whose evidence item is `unavailable` or `failed`
resolves to `extractor_unavailable` -- never `not_triggered`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .evidence_schema import CONCLUSIVE_STATUSES, EvidenceItem, EvidenceSet
from .rule_schema import RuleV2

OUTCOMES = frozenset({
    "triggered",
    "not_triggered",
    "insufficient_evidence",
    "not_applicable",
    "extractor_unavailable",
    "requires_manual_review",
    "conflicting_evidence",
})


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    evidence_ids_used: list[str]
    actual_measured_values: dict[str, Any]
    conditions_evaluated: str
    threshold_provenance: str
    outcome: str
    source_citation: str
    limitations: list[str]
    validation_status: str
    confidence_ceiling: float
    rule_confidence: float
    professional_reasoning: str | None
    parent_safe_wording: str | None
    requires_clinician_review: bool = True

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"RuleResult {self.rule_id!r}: outcome {self.outcome!r} not in {sorted(OUTCOMES)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gather(evidence: EvidenceSet, feature_ids: list[str]) -> dict[str, list[EvidenceItem]]:
    return {fid: evidence.by_feature(fid) for fid in feature_ids}


def _resolve_availability(items_by_feature: dict[str, list[EvidenceItem]]) -> str | None:
    """Returns an early-exit outcome if evidence isn't usable, else None to
    continue to condition evaluation."""
    for fid, items in items_by_feature.items():
        if not items:
            return "extractor_unavailable"
        for item in items:
            if item.status in ("unavailable", "failed"):
                return "extractor_unavailable"
            if item.status == "insufficient_evidence":
                return "insufficient_evidence"
            if item.status == "requires_manual_review":
                return "requires_manual_review"
        conclusive = [item for item in items if item.status in CONCLUSIVE_STATUSES]
        if len(conclusive) >= 2:
            values = {repr(item.value) for item in conclusive}
            if len(values) > 1:
                return "conflicting_evidence"
    return None


_COVERAGE_OBSERVABLES = {
    "coverage_about_half": (0.40, 0.60, "0.40 <= bounding_box_coverage <= 0.60"),
    "coverage_full": (0.90, None, "bounding_box_coverage >= 0.90"),
    "coverage_small": (0.0, 0.20, "0 < bounding_box_coverage <= 0.20"),
}
# placement centroid split points -- identical to analysis.py::_composition,
# duplicated here deliberately so this engine's condition text is
# self-contained and independently auditable rather than importing a
# private helper from the analysis module.
_PLACEMENT_SPLIT = 0.4, 0.6


def _evaluate_condition(observable: str, items_by_feature: dict[str, list[EvidenceItem]]) -> tuple[str, str, dict[str, Any]]:
    """Returns (outcome, conditions_evaluated_str, actual_measured_values).
    Only called once evidence is confirmed usable (conclusive, non-conflicting)."""
    if observable in _COVERAGE_OBSERVABLES:
        lo, hi, cond = _COVERAGE_OBSERVABLES[observable]
        item = items_by_feature["composition.bounding_box_coverage"][0]
        value = float(item.value)
        matched = (lo < value <= hi) if observable == "coverage_small" else (
            value >= lo if hi is None else lo <= value <= hi
        )
        return ("triggered" if matched else "not_triggered", cond, {"bounding_box_coverage": value})

    if observable.startswith("placement_"):
        item = items_by_feature["composition.centroid_normalized"][0]
        cx, cy = (float(v) for v in item.value)
        lo, hi = _PLACEMENT_SPLIT
        which = observable.removeprefix("placement_")
        conditions = {
            "top": (cy < lo, f"centroid_y < {lo}"),
            "left": (cx < lo, f"centroid_x < {lo}"),
            "right": (cx > hi, f"centroid_x > {hi}"),
        }
        matched, cond = conditions[which]
        return ("triggered" if matched else "not_triggered", cond, {"centroid_normalized": [cx, cy]})

    # Every non-composition observable in the current registry is tier-2
    # (content-conditional) and its required feature is always unavailable
    # today, so `_resolve_availability` short-circuits before this branch is
    # ever reached in practice. This raise exists so a future rule whose
    # observable isn't wired here fails loudly instead of silently
    # fabricating an outcome.
    raise ValueError(f"No condition evaluator wired for observable {observable!r}.")


# Base confidence assigned to a *triggered* rule before the registry's
# per-rule confidence_ceiling caps it -- mirrors rules.py's _WEAK_SUPPORT_BASE
# so the two engines agree on numeric confidence even though their outcome
# vocabularies differ.
_TRIGGERED_BASE_CONFIDENCE = 0.5


def evaluate_rule_v2(rule: RuleV2, evidence: EvidenceSet) -> RuleResult:
    items_by_feature = _gather(evidence, rule.required_feature_ids)
    used_ids = [item.evidence_id for items in items_by_feature.values() for item in items]
    citation = f"{rule.source_document}, page {rule.source_page}, \"{rule.source_section}\""

    early = _resolve_availability(items_by_feature)
    if early is not None:
        return RuleResult(
            rule_id=rule.rule_id, evidence_ids_used=used_ids, actual_measured_values={},
            conditions_evaluated="not evaluated (evidence not usable)",
            threshold_provenance=rule.threshold_source, outcome=early,
            source_citation=citation, limitations=list(rule.limitations),
            validation_status=rule.activation_status, confidence_ceiling=rule.confidence_ceiling,
            rule_confidence=0.0, professional_reasoning=None, parent_safe_wording=None,
        )

    if rule.tier == "tier_3_prompt_or_age_dependent":
        return RuleResult(
            rule_id=rule.rule_id, evidence_ids_used=used_ids, actual_measured_values={},
            conditions_evaluated="not evaluated (prompt/age context required)",
            threshold_provenance=rule.threshold_source, outcome="not_applicable",
            source_citation=citation, limitations=list(rule.limitations),
            validation_status=rule.activation_status, confidence_ceiling=rule.confidence_ceiling,
            rule_confidence=0.0, professional_reasoning=None, parent_safe_wording=None,
        )

    outcome, condition_str, measured = _evaluate_condition(rule.observable, items_by_feature)
    triggered = outcome == "triggered"
    rule_confidence = round(min(_TRIGGERED_BASE_CONFIDENCE, rule.confidence_ceiling), 4) if triggered else 0.0
    return RuleResult(
        rule_id=rule.rule_id, evidence_ids_used=used_ids, actual_measured_values=measured,
        conditions_evaluated=condition_str, threshold_provenance=rule.threshold_source,
        outcome=outcome, source_citation=citation, limitations=list(rule.limitations),
        validation_status=rule.activation_status, confidence_ceiling=rule.confidence_ceiling,
        rule_confidence=rule_confidence,
        professional_reasoning=rule.professional_reasoning if triggered else None,
        parent_safe_wording=rule.parent_safe_wording if triggered else None,
    )


def evaluate_all_rules_v2(rules: list[RuleV2], evidence: EvidenceSet) -> list[RuleResult]:
    return [evaluate_rule_v2(rule, evidence) for rule in rules]


# ---------------------------------------------------------------------------
# Omission logic (task requirement: an omitted component must not be
# reported unless the parent object was reliably detected, the region was
# visible, the component extractor was applicable/reliable, and the
# absence criterion was genuinely evaluated). No rule in the current
# 19-rule registry is an omission-type rule (e.g. "hands are missing"), so
# this is not wired into evaluate_rule_v2 above -- it is provided as a
# ready, independently-tested primitive for the first component-omission
# rule that gets a real body-part/component detector (Phase 2).
# ---------------------------------------------------------------------------

def evaluate_omission(
    *,
    parent_object_reliably_detected: bool,
    region_visible: bool,
    component_extractor_applicable_and_reliable: bool,
    absence_criterion_evaluated: bool,
) -> str:
    """Returns "insufficient_evidence" unless every precondition for
    reporting a genuine omission is met, in which case the caller may
    proceed to its own triggered/not_triggered logic. Never returns
    triggered/not_triggered itself -- it only gates whether omission may be
    assessed at all."""
    if not (
        parent_object_reliably_detected
        and region_visible
        and component_extractor_applicable_and_reliable
        and absence_criterion_evaluated
    ):
        return "insufficient_evidence"
    return "evaluable"
