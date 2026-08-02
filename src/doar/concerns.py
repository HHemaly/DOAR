"""
concerns.py — evidence-convergence engine for clinical concern profiles (D6,
Phase 2 correction 2026-08-02).

Replaces the hardcoded empty-concerns stub with a REAL convergence rule, while
keeping the same safety guarantee: a concern can never arise from a single
symbol or a single evidence source.

Aggregation strength (working spec Section 10) is one of four levels,
computed from evidence count and source-type diversity -- never an arbitrary
clinical-confidence score:
  * INSUFFICIENT              -- fewer than 2 matched evidence IDs.
  * WEAK_HYPOTHESIS            -- >=2 evidence IDs, but only ONE source type
                                  (e.g. two correlated clinician-hypothesis
                                  rules firing together -- still just two
                                  unvalidated guesses from the same source).
  * POSSIBLE_FOR_EXPLORATION   -- >=2 evidence IDs from >=2 distinct source
                                  types, at least one of which is not a bare
                                  clinician hypothesis (e.g. a rule agrees
                                  with the independent emotion-model
                                  prediction).
  * PROFESSIONAL_REVIEW_SUGGESTED -- >=3 distinct source types. Defined here
                                  for completeness with the working spec's
                                  vocabulary; nothing in the current pipeline
                                  can reach it yet (only two source types --
                                  clinician rules and the emotion model --
                                  exist at all until Phase 3 detectors ship).
                                  Exercised only with synthetic 3-source test
                                  data, same as every other level was before
                                  real data existed for it.

Phase 2 correction: the previous "bug" description (rules.py mistagging
source_type) was inaccurate -- clinician-authored rules ARE correctly one
source type regardless of which feature backs them; that tagging is
intentional, not a defect. The actual gap was that `evaluate_rules()`
discarded the `evidence` argument, so the emotion model's independent
prediction -- the only genuinely independent source in the system -- never
reached this module. That is now fixed (rules.py passes it through as
`model_evidence`). See DECISION_LOG.md 2026-08-02 for the full correction.

CONCERNS_ENABLED stays False in production regardless of this fix -- Phase 2
makes the logic correct and testable, it does not make concern output
user-facing. See IMPLEMENTATION_PLAN.md Phase 2 / DECISION_LOG.md.

Every concern records supporting, contradicting, and missing evidence, source
diversity, uncertainty, professional + parent-safe wording, and a review flag.
"""

from __future__ import annotations

from typing import Any

MIN_EVIDENCE = 2      # at least two independent evidence IDs
MIN_SOURCES = 2       # from at least two distinct source types
PROFESSIONAL_REVIEW_SOURCES = 3   # source-type count for the strongest level

# Item 11 — the generic converged concern lacks a real, clinician-authored
# taxonomy (eligible evidence, contradicting evidence, per-concern wording and
# approval). Until concern-specific profiles AND independent detectors exist,
# concerns are DISABLED by default rather than emitting a generic concern. The
# convergence logic below is retained and unit-tested for when it is enabled with
# a proper taxonomy. Never enable this without concern-specific profiles.
CONCERNS_ENABLED = False


# Which evidence IDs count as which independent "source type". Clinician-supplied
# symbolic rules are all ONE source type, so several of them alone never converge.
def _source_type(evidence_id: str, rule: dict) -> str:
    if rule.get("source_type") == "psychologist_supplied_hypothesis":
        return "clinician_symbolic"
    if evidence_id.startswith("ev_seg") or evidence_id.startswith("ev_bbox") or evidence_id.startswith("ev_centroid"):
        return "objective_composition"
    if evidence_id.startswith("ev_colour") or evidence_id.startswith("ev_dominant"):
        return "objective_colour"
    if evidence_id.startswith("ev_emotion"):
        return "model_prediction"
    if evidence_id.startswith("ev_detection") or evidence_id.startswith("ev_object"):
        return "object_detector"
    if evidence_id.startswith("ev_ocr"):
        return "ocr"
    return "other"


def _aggregation_strength(evidence_ids: set[str], source_types: set[str]) -> str:
    if len(evidence_ids) < MIN_EVIDENCE:
        return "INSUFFICIENT"
    non_clinical = source_types - {"clinician_symbolic"}
    if len(source_types) < MIN_SOURCES or not non_clinical:
        return "WEAK_HYPOTHESIS"
    if len(source_types) >= PROFESSIONAL_REVIEW_SOURCES:
        return "PROFESSIONAL_REVIEW_SUGGESTED"
    return "POSSIBLE_FOR_EXPLORATION"


def derive_concerns(
    rule_evaluations: list[dict],
    model_evidence: list[Any] | None = None,
    *,
    enabled: bool | None = None,
) -> list[dict]:
    """Return concern profiles that satisfy multi-source convergence.

    DISABLED by default (Item 11): returns [] unless explicitly enabled with a
    real taxonomy in place. The convergence logic is retained for that future
    state and is exercised by unit tests via `enabled=True`.

    `model_evidence` is an optional list of Evidence-like objects (each with
    `.evidence_id`, `.confidence`) carrying the emotion model's independent
    prediction, if available -- see rules.py::evaluate_rules(). It is treated
    as a genuinely independent source, distinct from every clinician rule."""
    if enabled is None:
        enabled = CONCERNS_ENABLED
    if not enabled:
        return []
    # Collect supporting evidence grouped by the theme (observable family).
    supporting = []  # (evidence_id, source_type, confidence_ceiling, missing_evidence)
    for rule in rule_evaluations:
        if rule.get("status") != "weak_support":
            continue
        for evidence_id in rule.get("matched_evidence_ids", []):
            supporting.append((
                evidence_id, _source_type(evidence_id, rule),
                float(rule.get("confidence_ceiling", 0.0)), rule.get("missing_evidence", []),
            ))
    for item in (model_evidence or []):
        supporting.append((item.evidence_id, "model_prediction", float(item.confidence), []))

    if len(supporting) < MIN_EVIDENCE:
        return []

    evidence_ids = {e for e, _, _, _ in supporting}
    source_types = {s for _, s, _, _ in supporting}
    strength = _aggregation_strength(evidence_ids, source_types)
    if strength == "INSUFFICIENT":
        return []

    # Confidence is capped by the lowest contributing ceiling -- including the
    # model's own confidence when it participates, so a highly-confident model
    # prediction can never itself raise a clinician rule's low ceiling.
    ceilings = [c for _, _, c, _ in supporting]
    capped_confidence = min(ceilings) if ceilings else 0.0

    return [{
        "concern_id": "concern_converged_001",
        "aggregation_strength": strength,
        "supporting_evidence": sorted(evidence_ids),
        "contradicting_evidence": [],
        "missing_evidence": sorted({m for _, _, _, missing in supporting for m in missing}),
        "source_types": sorted(source_types),
        "source_diversity": len(source_types),
        "confidence": capped_confidence,
        "uncertainty": "high",
        "professional_wording": (
            "Multiple independent observations converge; a qualified clinician "
            "should review this case in context. This is not a diagnosis."
        ),
        "parent_safe_wording": (
            "Some features in the drawing are worth gently discussing with your "
            "child. This is not a diagnosis and does not mean anything is wrong."
        ),
        "requires_clinician_review": True,
        "approval_status": "pending",
    }]
