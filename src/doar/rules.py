from __future__ import annotations

import json
from pathlib import Path

from .schemas import Evidence


REGISTRY = Path(__file__).resolve().parents[2] / "resources" / "psychology_sources" / "rules_registry.json"

# The only observables rules.py can actually evaluate against a real,
# implemented feature today (Phase 0 audit, RULE_COVERAGE_MATRIX.csv). Every
# other Tier-2 observable requires a detector (face/eye, animal, shape,
# symbol) that does not exist yet -- see IMPLEMENTATION_PLAN.md Phase 3.
_TIER1_DISPATCH = {"coverage_about_half", "coverage_full", "coverage_small"}


def _tier1_status(key: str, composition: dict) -> tuple[str, list[str], list[str]]:
    if key == "coverage_about_half":
        value = composition["bounding_box_coverage"]
        return ("weak_support" if 0.40 <= value <= 0.60 else "not_matched",
                ["ev_bbox_coverage"] if 0.40 <= value <= 0.60 else [], [])
    if key == "coverage_full":
        value = composition["bounding_box_coverage"]
        return ("weak_support" if value >= 0.90 else "not_matched",
                ["ev_bbox_coverage"] if value >= 0.90 else [], [])
    if key == "coverage_small":
        value = composition["bounding_box_coverage"]
        return ("weak_support" if 0 < value <= 0.20 else "not_matched",
                ["ev_bbox_coverage"] if 0 < value <= 0.20 else [], [])
    # placement_* observables
    if composition["placement"] == "unavailable":
        return "not_evaluated", [], ["ev_centroid"]
    wanted = key.removeprefix("placement_")
    matched = wanted in composition["placement"]
    return ("weak_support" if matched else "not_matched",
            ["ev_centroid"] if matched else [], [])


def _status_for(rule: dict, composition: dict) -> tuple[str, list[str], list[str]]:
    """Tier-aware dispatch (Phase 2). Behavior-identical to the pre-Phase-2
    implicit fallthrough for every rule in the current registry: the same 6
    coverage/placement rules still evaluate against real composition
    measurements, and the same 13 eyes/animals/shapes/symbols rules still
    return `missing_detector` unconditionally -- this refactor only makes the
    *reason* explicit and tier-checked instead of an implicit default branch,
    per PROPOSED_ARCHITECTURE.md Section 3."""
    tier = rule.get("tier")
    key = rule["observable"]
    if tier == "tier_3_prompt_or_age_dependent":
        # No rule in the current registry is Tier 3. This branch exists for
        # future HTP/DAP/KFD-style prompt- or age-dependent rules, which must
        # never silently apply to an ordinary free drawing (working spec
        # Section 8 / SCIENTIFIC_LIMITATIONS.md Section 4).
        return "not_assessable_context_unknown", [], []
    if tier == "tier_2_content_conditional":
        # Object/figure/eye/animal detectors do not exist in this release.
        # Missing evidence is NOT negative evidence, so this never becomes
        # "not_matched".
        return "missing_detector", [], [f"detector_absent:{key}"]
    if tier == "tier_1_prompt_independent" and (key in _TIER1_DISPATCH or key.startswith("placement_")):
        return _tier1_status(key, composition)
    # A rule claims tier_1 but its observable isn't one of the ones this
    # module actually knows how to evaluate -- fail loudly rather than
    # silently falling through to missing_detector, so a registry/tier
    # mismatch is caught immediately instead of masquerading as "no detector".
    raise ValueError(
        f"Rule {rule.get('rule_id')!r} is tagged tier_1_prompt_independent with "
        f"observable {key!r}, which has no implemented Tier-1 dispatch."
    )


# Nominal confidence assigned to a weak-support match BEFORE the per-rule ceiling
# is applied. The registry ceilings (0.05-0.25) always bind, so the emitted
# confidence is min(base, ceiling) — the ceiling is now numerically enforced.
_WEAK_SUPPORT_BASE = 0.5


def evaluate_rules(
    composition: dict, colour: dict, evidence: list[Evidence]
) -> tuple[list[dict], list[dict]]:
    del colour
    rules = json.loads(REGISTRY.read_text(encoding="utf-8"))
    evaluations = []
    for rule in rules["rules"]:
        status, matched, missing = _status_for(rule, composition)
        ceiling = float(rule["confidence_ceiling"])
        # Enforce the ceiling: only weak-support matches carry any confidence,
        # and it can never exceed the registry ceiling for that rule.
        rule_confidence = round(min(_WEAK_SUPPORT_BASE, ceiling), 4) if status == "weak_support" else 0.0
        evaluations.append({
            "rule_id": rule["rule_id"],
            "tier": rule.get("tier"),
            "activation_status": rule.get("activation_status"),
            "status": status,
            "matched_evidence_ids": matched,
            "missing_evidence": missing,
            # Every rule in this registry is a clinician-supplied hypothesis,
            # regardless of whether the underlying feature is objectively
            # measurable (Tier 1) -- the *interpretation* attached to it is
            # not independently derived. Two Tier-1 rules matching together
            # are still two unvalidated guesses from the same source, not two
            # independent confirmations. This is intentional, not a bug --
            # see DECISION_LOG.md 2026-08-02 correction.
            "source_type": "psychologist_supplied_hypothesis",
            "scientific_support": rule["scientific_support"],
            "confidence_ceiling": ceiling,
            "rule_confidence": rule_confidence,
            "confidence_ceiling_enforced": True,
            "professional_reasoning": rule["professional_reasoning"] if status == "weak_support" else None,
            "parent_safe_wording": rule["parent_safe_wording"] if status == "weak_support" else None,
            "original_arabic": rule["arabic"],
            "english_translation": rule["english"],
            "references": rule["references"],
            "limitations": rule["limitations"],
            "requires_clinician_review": True,
        })
    # Concern profiles come from a real convergence engine (D6, corrected
    # Phase 2): a concern requires >=2 independent evidence IDs from >=2
    # distinct source types, and clinician-supplied rule hypotheses alone
    # (however many fire) are exactly ONE source type -- they can never
    # converge with each other. The only other genuinely independent source
    # in the system is the emotion model's own prediction; it is passed
    # through here (it was previously discarded) so the diversity check has
    # real data to evaluate against. CONCERNS_ENABLED stays False in
    # production regardless -- see concerns.py.
    from .concerns import derive_concerns
    model_evidence = [item for item in evidence if item.kind == "model_prediction"]
    concerns = derive_concerns(evaluations, model_evidence)
    return evaluations, concerns
