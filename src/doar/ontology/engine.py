"""Three-valued rule engine (DOAR V2).

Evaluates a declarative `Rule` (`rule.py`) against a `FeatureStore`
(`store.py`) and a `CaseContext` (`context.py`), and returns one
`RuleEvaluation` carrying both the outcome and a full machine-readable
trace of how it was reached.

Hard invariants this module enforces mechanically (task requirements):

- A rule that requires context (age/task/narrative) the case does not
  have never "evaluates normally" -- it resolves to `MISSING_PREREQUISITE`
  before its premise is even inspected.
- A feature whose ontology-declared prerequisites are not satisfied is
  never treated as `FALSE` just because it was "not detected" -- its
  condition resolves to `TriState.UNKNOWN`, which can only ever produce
  `NOT_MATCHED` if the *rest* of the premise is independently `FALSE`,
  never on its own.
- `UNKNOWN` never silently becomes `FALSE` or `TRUE` when combined via
  `AllOf`/`AnyOf`/`NotOf` -- see `tristate.py`'s truth tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .context import CaseContext
from .feature_schema import FeatureOntology
from .rule import AllOf, AnyOf, FeatureCondition, NotOf, Premise, Rule
from .store import EffectiveObservation, FeatureStore, ObservationStatus
from .tristate import TriState, tri_and, tri_not, tri_or


class RuleOutcome:
    MATCHED = "MATCHED"
    MATCHED_PROVISIONAL = "MATCHED_PROVISIONAL"
    NOT_MATCHED = "NOT_MATCHED"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"
    NOT_APPLICABLE_TO_TASK = "NOT_APPLICABLE_TO_TASK"
    NOT_APPLICABLE_TO_AGE = "NOT_APPLICABLE_TO_AGE"
    MISSING_PREREQUISITE = "MISSING_PREREQUISITE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"

    ALL = (
        MATCHED, MATCHED_PROVISIONAL, NOT_MATCHED, NOT_ASSESSABLE,
        NOT_APPLICABLE_TO_TASK, NOT_APPLICABLE_TO_AGE, MISSING_PREREQUISITE,
        CONFLICTING_EVIDENCE,
    )

# Statuses whose effective value is trusted enough that a MATCH built
# entirely from them is reported as MATCHED rather than MATCHED_PROVISIONAL.
_STRONG_STATUSES = frozenset({
    ObservationStatus.CORROBORATED, ObservationStatus.HUMAN_CONFIRMED, ObservationStatus.HUMAN_CORRECTED,
})

_BASE_MATCH_CONFIDENCE = 0.5


@dataclass(frozen=True)
class ConditionTrace:
    feature_id: str
    operator: str
    expected: Any
    effective_status: str
    effective_value: Any
    result: str  # TriState.value
    prerequisite_blocked_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PremiseTrace:
    node_type: str  # "condition" | "all_of" | "any_of" | "not_of"
    result: str
    condition: ConditionTrace | None = None
    children: tuple["PremiseTrace", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = {"node_type": self.node_type, "result": self.result}
        if self.condition is not None:
            data["condition"] = self.condition.to_dict()
        if self.children:
            data["children"] = [c.to_dict() for c in self.children]
        return data

    def leaf_conditions(self) -> list[ConditionTrace]:
        if self.condition is not None:
            return [self.condition]
        return [c for child in self.children for c in child.leaf_conditions()]


@dataclass(frozen=True)
class RuleEvaluation:
    rule_id: str
    outcome: str
    rule_confidence: float
    requested_feature_ids: tuple[str, ...]
    feature_snapshots: dict[str, dict[str, Any]]
    premise_trace: PremiseTrace | None
    missing_prerequisites: tuple[str, ...]
    context_note: str | None
    citation: dict[str, Any]
    framework: str
    evidence_grade: str
    limitations: tuple[str, ...]
    professional_reasoning: str | None
    parent_safe_wording: str | None
    requires_clinician_review: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "outcome": self.outcome,
            "rule_confidence": self.rule_confidence,
            "requested_feature_ids": list(self.requested_feature_ids),
            "feature_snapshots": self.feature_snapshots,
            "premise_trace": self.premise_trace.to_dict() if self.premise_trace else None,
            "missing_prerequisites": list(self.missing_prerequisites),
            "context_note": self.context_note,
            "citation": self.citation,
            "framework": self.framework,
            "evidence_grade": self.evidence_grade,
            "limitations": list(self.limitations),
            "professional_reasoning": self.professional_reasoning,
            "parent_safe_wording": self.parent_safe_wording,
            "requires_clinician_review": self.requires_clinician_review,
        }


def _prerequisites_satisfied(
    feature_id: str, store: FeatureStore, ontology: FeatureOntology | None,
) -> tuple[bool, tuple[str, ...]]:
    """Returns (satisfied, unmet_prerequisite_feature_ids). A feature with
    no ontology entry, or no declared requirements, has none to check."""
    if ontology is None or feature_id not in ontology:
        return True, ()
    unmet: list[str] = []
    for req in ontology.requirements_of(feature_id):
        eff = store.effective(req)
        satisfied = eff.assessable and eff.value is True
        if not satisfied:
            unmet.append(req)
    return (not unmet), tuple(unmet)


def _apply_operator(operator: str, value: Any, expected: Any) -> bool:
    if operator == "equals":
        return value == expected
    if operator == "not_equals":
        return value != expected
    if operator == "gte":
        return value >= expected
    if operator == "lte":
        return value <= expected
    if operator == "gt":
        return value > expected
    if operator == "lt":
        return value < expected
    if operator == "in":
        return value in expected
    if operator == "contains":
        return expected in value
    raise ValueError(f"Unknown operator {operator!r}")  # pragma: no cover - guarded by rule.py


def evaluate_condition(
    condition: FeatureCondition, store: FeatureStore, ontology: FeatureOntology | None,
) -> tuple[TriState, ConditionTrace]:
    ok, unmet = _prerequisites_satisfied(condition.feature_id, store, ontology)
    eff: EffectiveObservation = store.effective(condition.feature_id)

    if not ok:
        trace = ConditionTrace(
            feature_id=condition.feature_id, operator=condition.operator, expected=condition.expected,
            effective_status=eff.status, effective_value=eff.value, result=TriState.UNKNOWN.value,
            prerequisite_blocked_by=unmet,
        )
        return TriState.UNKNOWN, trace

    if eff.status in (ObservationStatus.NOT_ASSESSABLE, ObservationStatus.CONFLICT) or not eff.assessable:
        trace = ConditionTrace(
            feature_id=condition.feature_id, operator=condition.operator, expected=condition.expected,
            effective_status=eff.status, effective_value=eff.value, result=TriState.UNKNOWN.value,
        )
        return TriState.UNKNOWN, trace

    matched = _apply_operator(condition.operator, eff.value, condition.expected)
    result = TriState.TRUE if matched else TriState.FALSE
    trace = ConditionTrace(
        feature_id=condition.feature_id, operator=condition.operator, expected=condition.expected,
        effective_status=eff.status, effective_value=eff.value, result=result.value,
    )
    return result, trace


def evaluate_premise(
    premise: Premise, store: FeatureStore, ontology: FeatureOntology | None,
) -> tuple[TriState, PremiseTrace]:
    if isinstance(premise, FeatureCondition):
        result, cond_trace = evaluate_condition(premise, store, ontology)
        return result, PremiseTrace(node_type="condition", result=result.value, condition=cond_trace)

    if isinstance(premise, AllOf):
        results, traces = zip(*(evaluate_premise(c, store, ontology) for c in premise.conditions))
        combined = tri_and(*results)
        return combined, PremiseTrace(node_type="all_of", result=combined.value, children=traces)

    if isinstance(premise, AnyOf):
        results, traces = zip(*(evaluate_premise(c, store, ontology) for c in premise.conditions))
        combined = tri_or(*results)
        return combined, PremiseTrace(node_type="any_of", result=combined.value, children=traces)

    if isinstance(premise, NotOf):
        result, child_trace = evaluate_premise(premise.condition, store, ontology)
        combined = tri_not(result)
        return combined, PremiseTrace(node_type="not_of", result=combined.value, children=(child_trace,))

    raise TypeError(f"Unknown premise node type: {type(premise)!r}")  # pragma: no cover


def _context_gate(rule: Rule, context: CaseContext) -> str | None:
    """Returns an outcome string if context requirements block evaluation
    before the premise is even inspected, else None to proceed."""
    req = rule.requirements
    if req.age:
        if context.age is None:
            return RuleOutcome.MISSING_PREREQUISITE
        if req.age_range is not None:
            lo, hi = req.age_range
            if not (lo <= context.age <= hi):
                return RuleOutcome.NOT_APPLICABLE_TO_AGE
    if req.task:
        if context.drawing_task is None:
            return RuleOutcome.MISSING_PREREQUISITE
        if req.allowed_tasks is not None and context.drawing_task not in req.allowed_tasks:
            return RuleOutcome.NOT_APPLICABLE_TO_TASK
    if req.narrative and context.narrative is None:
        return RuleOutcome.MISSING_PREREQUISITE
    return None


def _feature_snapshots(feature_ids: tuple[str, ...], store: FeatureStore) -> dict[str, dict[str, Any]]:
    snapshots = {}
    for fid in feature_ids:
        eff = store.effective(fid)
        snapshots[fid] = {
            "status": eff.status, "value": eff.value, "assessable": eff.assessable,
            "reason": eff.reason, "contributing_sources": list(eff.contributing_sources),
        }
    return snapshots


def evaluate_rule(
    rule: Rule, store: FeatureStore, context: CaseContext, ontology: FeatureOntology | None = None,
) -> RuleEvaluation:
    feature_ids = rule.required_feature_ids()
    citation = rule.citation.to_dict()

    def _finish(outcome: str, *, confidence: float = 0.0, trace: PremiseTrace | None = None,
                missing: tuple[str, ...] = (), context_note: str | None = None) -> RuleEvaluation:
        return RuleEvaluation(
            rule_id=rule.rule_id, outcome=outcome, rule_confidence=confidence,
            requested_feature_ids=feature_ids, feature_snapshots=_feature_snapshots(feature_ids, store),
            premise_trace=trace, missing_prerequisites=missing, context_note=context_note,
            citation=citation, framework=rule.framework, evidence_grade=rule.evidence_grade,
            limitations=rule.limitations,
            professional_reasoning=rule.professional_reasoning if outcome in (RuleOutcome.MATCHED, RuleOutcome.MATCHED_PROVISIONAL) else None,
            parent_safe_wording=rule.parent_safe_wording if outcome in (RuleOutcome.MATCHED, RuleOutcome.MATCHED_PROVISIONAL) else None,
            requires_clinician_review=rule.requires_clinician_review,
        )

    gated = _context_gate(rule, context)
    if gated is not None:
        note = {
            RuleOutcome.MISSING_PREREQUISITE: "Required case context (age/task/narrative) is not available.",
            RuleOutcome.NOT_APPLICABLE_TO_AGE: f"Case age {context.age} is outside this rule's declared age_range {rule.requirements.age_range}.",
            RuleOutcome.NOT_APPLICABLE_TO_TASK: f"Case drawing_task {context.drawing_task!r} is not in this rule's allowed_tasks {rule.requirements.allowed_tasks}.",
        }[gated]
        return _finish(gated, context_note=note)

    result, trace = evaluate_premise(rule.premise, store, ontology)
    leaves = trace.leaf_conditions()
    missing_prereqs = tuple(dict.fromkeys(fid for leaf in leaves for fid in leaf.prerequisite_blocked_by))
    has_conflict = any(leaf.effective_status == ObservationStatus.CONFLICT for leaf in leaves)

    if result is TriState.TRUE:
        contributing_statuses = {
            store.effective(leaf.feature_id).status for leaf in leaves
        }
        strong = contributing_statuses <= _STRONG_STATUSES
        outcome = RuleOutcome.MATCHED if strong else RuleOutcome.MATCHED_PROVISIONAL
        confidence = min(_BASE_MATCH_CONFIDENCE, rule.confidence_ceiling)
        if outcome == RuleOutcome.MATCHED_PROVISIONAL:
            confidence = round(confidence * 0.5, 4)
        return _finish(outcome, confidence=confidence, trace=trace)

    if result is TriState.FALSE:
        return _finish(RuleOutcome.NOT_MATCHED, trace=trace)

    # TriState.UNKNOWN
    if has_conflict:
        return _finish(RuleOutcome.CONFLICTING_EVIDENCE, trace=trace, missing=missing_prereqs)
    if missing_prereqs:
        return _finish(RuleOutcome.MISSING_PREREQUISITE, trace=trace, missing=missing_prereqs)
    return _finish(RuleOutcome.NOT_ASSESSABLE, trace=trace)


def evaluate_all_rules(
    rules: list[Rule], store: FeatureStore, context: CaseContext, ontology: FeatureOntology | None = None,
) -> list[RuleEvaluation]:
    return [evaluate_rule(rule, store, context, ontology) for rule in rules]
