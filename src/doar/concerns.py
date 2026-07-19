"""
concerns.py — evidence-convergence engine for clinical concern profiles (D6).

Replaces the hardcoded empty-concerns stub with a REAL convergence rule, while
keeping the same safety guarantee: a concern can never arise from a single
symbol or a single evidence source.

A concern profile is emitted only when ALL hold:
  * >= MIN_EVIDENCE independent matched evidence IDs, AND
  * those evidence IDs come from >= MIN_SOURCES distinct source types, AND
  * none of the supporting rules is a single clinician-supplied symbolic rule
    acting alone (source diversity requirement), AND
  * confidence stays capped by the lowest contributing confidence ceiling.

With the current release (only weak, single-symbol clinician rules and no object
detectors) this correctly yields an EMPTY concern list — but now by convergence
logic, not by a hardcoded `return []`. As independent detectors are added, real
converging evidence will populate concerns automatically.

Every concern records supporting, contradicting, and missing evidence, source
diversity, uncertainty, professional + parent-safe wording, and a review flag.
"""

from __future__ import annotations

MIN_EVIDENCE = 2      # at least two independent evidence IDs
MIN_SOURCES = 2       # from at least two distinct source types


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


def derive_concerns(rule_evaluations: list[dict]) -> list[dict]:
    """Return concern profiles that satisfy multi-source convergence. Usually
    empty in the current release (no independent detectors)."""
    # Collect supporting evidence grouped by the theme (observable family).
    supporting = []  # (evidence_id, source_type, rule)
    for rule in rule_evaluations:
        if rule.get("status") != "weak_support":
            continue
        for evidence_id in rule.get("matched_evidence_ids", []):
            supporting.append((evidence_id, _source_type(evidence_id, rule), rule))

    if len(supporting) < MIN_EVIDENCE:
        return []

    evidence_ids = {e for e, _, _ in supporting}
    source_types = {s for _, s, _ in supporting}

    # Source diversity: clinician_symbolic alone can never converge.
    non_clinical_sources = source_types - {"clinician_symbolic"}
    if len(source_types) < MIN_SOURCES or not non_clinical_sources:
        return []

    if len(evidence_ids) < MIN_EVIDENCE:
        return []

    # Confidence is capped by the lowest contributing ceiling.
    ceilings = [float(r.get("confidence_ceiling", 0.0)) for _, _, r in supporting]
    capped_confidence = min(ceilings) if ceilings else 0.0

    return [{
        "concern_id": "concern_converged_001",
        "supporting_evidence": sorted(evidence_ids),
        "contradicting_evidence": [],
        "missing_evidence": sorted({
            m for _, _, r in supporting for m in r.get("missing_evidence", [])
        }),
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
