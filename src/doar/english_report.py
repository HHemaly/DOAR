"""Traceable English report generator (Phase 1 Section 7).

Deterministically renders `EvidenceItem`s and `RuleResult`s
(`evidence_schema.py`, `evidence_rule_engine.py`) into plain-English
Markdown. No language model is involved -- every sentence is built from a
fixed template filled with real, already-computed values, so the report
can never say more than the evidence and rule engine actually support.

Every triggered-rule observation states, in this order: the plain-English
observation, where to find it, the extracted feature/value, its
confidence and validation status, the linked rule, the rule outcome, the
source PDF+page, a cautious literature association, limitations, and the
clinician-review-required flag -- matching the task's own worked example
format. Non-triggered/blocked rules are reported too (never silently
dropped), each with a plain reason evaluators can double-check.
"""

from __future__ import annotations

from typing import Any

from .evidence_rule_engine import RuleResult
from .evidence_schema import EvidenceSet
from .judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC
from .rule_schema import RuleV2

DISCLAIMER = (
    "This report describes measured drawing evidence and unvalidated psychologist-"
    "supplied hypotheses. It is not a diagnosis. Only a qualified clinician may "
    "form a clinical conclusion, and only after independent review."
)

_FEATURE_DESCRIPTIONS = {
    "composition.bounding_box_coverage": "the drawing's bounding box occupies {value:.1%} of the usable page area",
    "composition.centroid_normalized": "the drawing's visual centroid sits at normalized position ({x:.2f}, {y:.2f}) on the page",
}


def _describe_measured_values(feature_id: str, values: dict[str, Any]) -> str:
    if feature_id == "composition.bounding_box_coverage" and "bounding_box_coverage" in values:
        return _FEATURE_DESCRIPTIONS[feature_id].format(value=values["bounding_box_coverage"])
    if feature_id == "composition.centroid_normalized" and "centroid_normalized" in values:
        x, y = values["centroid_normalized"]
        return _FEATURE_DESCRIPTIONS[feature_id].format(x=x, y=y)
    return f"measured value(s): {values}"


def _rule_by_id(rules: list[RuleV2]) -> dict[str, RuleV2]:
    return {rule.rule_id: rule for rule in rules}


def _render_triggered(rule: RuleV2, result: RuleResult, evidence: EvidenceSet) -> str:
    ev_ids = ", ".join(result.evidence_ids_used) or "none"
    primary_feature = rule.required_feature_ids[0]
    measured_desc = _describe_measured_values(primary_feature, result.actual_measured_values)
    where = "Evidence " + ev_ids + " (see the highlighted overlay for the exact region)."
    threshold_note = {
        "directly_sourced": "The threshold used here is stated directly in the source.",
        "sourced_center_invented_band": "The threshold's center value is stated in the source; the tolerance band around it was chosen by this implementation, not the source.",
        "invented_numeric_stand_in": "The source gives only a qualitative description here; the numeric threshold is an implementation choice, not stated in the source.",
        "invented_no_anchor": "The source gives no numeric boundary at all for this observation; the threshold is entirely an implementation choice.",
        "not_applicable": "This rule has no numeric threshold.",
    }[result.threshold_provenance]

    lines = [
        f"**Observation**: {measured_desc.capitalize()}, and this is highlighted as {where}",
        f"**Extracted feature/value**: `{primary_feature}` = {result.actual_measured_values}",
        f"**Confidence / validation status**: rule confidence {result.rule_confidence:.2f} "
        f"(ceiling {result.confidence_ceiling:.2f}), registry status `{result.validation_status}`.",
        f"**Linked rule**: `{rule.rule_id}` — condition evaluated: `{result.conditions_evaluated}`.",
        f"**Rule outcome**: `{result.outcome}`. {threshold_note}",
        f"**Source**: {result.source_citation}.",
        f"**Cautious literature association**: {rule.professional_reasoning} "
        f"(scientific support status: `{rule.scientific_support}`; this is a clinician-supplied "
        "hypothesis, not an established finding).",
        f"**Limitations / alternatives**: {'; '.join(result.limitations) if result.limitations else 'none recorded'}.",
        "**Clinician review required**: yes — a clinician may review this observation using the cited source. "
        "This system does not draw a conclusion on its own.",
    ]
    return "\n".join(lines)


def _render_blocked(rule: RuleV2, result: RuleResult) -> str:
    reason_by_outcome = {
        "extractor_unavailable": "the extractor this rule requires does not exist in this release, so no measurement was ever attempted",
        "insufficient_evidence": "the extractor ran but could not produce a reliable measurement for this image",
        "requires_manual_review": "the underlying evidence is not yet trusted enough for automatic evaluation",
        "not_applicable": "this rule requires drawing-task context this system does not have",
        "conflicting_evidence": "two independent measurements disagreed and could not be reconciled automatically",
        "not_triggered": "the rule's condition was evaluated against real measured evidence and was not met",
    }
    reason = reason_by_outcome.get(result.outcome, result.outcome)
    return (
        f"`{rule.rule_id}` ({rule.title_english}) — outcome `{result.outcome}`: {reason}. "
        f"Source: {result.source_citation}. No claim is made for this rule on this image."
    )


def generate_traceable_report(
    rules: list[RuleV2], results: list[RuleResult], evidence: EvidenceSet, image_path: str,
) -> str:
    rule_lookup = _rule_by_id(rules)
    triggered = [r for r in results if r.outcome == "triggered"]
    not_triggered = [r for r in results if r.outcome == "not_triggered"]
    blocked = [r for r in results if r.outcome not in ("triggered", "not_triggered")]

    parts = [
        f"# Traceable Evidence Report\n\n**Image**: `{image_path}`\n\n**{DISCLAIMER}**\n",
    ]

    parts.append(f"## Observations with a matching rule ({len(triggered)})\n")
    if triggered:
        for result in triggered:
            parts.append(_render_triggered(rule_lookup[result.rule_id], result, evidence) + "\n")
    else:
        parts.append("None on this image — no rule's condition was met by the measured evidence.\n")

    parts.append(f"## Rules evaluated and not matched ({len(not_triggered)})\n")
    for result in not_triggered:
        parts.append("- " + _render_blocked(rule_lookup[result.rule_id], result))
    if not not_triggered:
        parts.append("None.")
    parts.append("")

    parts.append(f"## Rules that could not be evaluated ({len(blocked)})\n")
    for result in blocked:
        parts.append("- " + _render_blocked(rule_lookup[result.rule_id], result))
    if not blocked:
        parts.append("None.")
    parts.append("")

    parts.append(
        "## Evidence inventory\n\n"
        f"{len(evidence.items)} evidence items were considered "
        f"({sum(1 for i in evidence.items if i.status in ('available', 'experimental'))} usable, "
        f"{sum(1 for i in evidence.items if i.status == 'unavailable')} from unimplemented extractors, "
        f"{sum(1 for i in evidence.items if i.status == 'insufficient_evidence')} insufficient). "
        "Full machine-readable detail is in the accompanying evidence JSON."
    )
    text = "\n".join(parts)
    hit = _diagnostic_scan(text)
    if hit:
        raise ValueError(
            f"generate_traceable_report produced text matching a diagnostic-language pattern: {hit!r}. "
            "This must never happen — refusing to return the report."
        )
    return text


def _diagnostic_scan(text: str) -> str | None:
    if _ARABIC_DIAGNOSTIC.search(text):
        return _ARABIC_DIAGNOSTIC.pattern
    for pattern in DIAGNOSTIC_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None
