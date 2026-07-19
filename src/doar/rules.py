from __future__ import annotations

import json
from pathlib import Path

from .schemas import Evidence


REGISTRY = Path(__file__).resolve().parents[2] / "resources" / "psychology_sources" / "rules_registry.json"


def _status_for(rule: dict, composition: dict) -> tuple[str, list[str], list[str]]:
    key = rule["observable"]
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
    if key.startswith("placement_"):
        if composition["placement"] == "unavailable":
            return "not_evaluated", [], ["ev_centroid"]
        wanted = key.removeprefix("placement_")
        matched = wanted in composition["placement"]
        return ("weak_support" if matched else "not_matched",
                ["ev_centroid"] if matched else [], [])
    # Object/figure/eye/animal detectors do not exist in this release. This is
    # "missing_detector" (no capability), distinct from "not_evaluated" (a
    # detector exists but could not run, e.g. blank page). Missing evidence is
    # NOT negative evidence, so this never becomes "not_matched".
    return "missing_detector", [], [f"detector_absent:{key}"]


# Nominal confidence assigned to a weak-support match BEFORE the per-rule ceiling
# is applied. The registry ceilings (0.05-0.25) always bind, so the emitted
# confidence is min(base, ceiling) — the ceiling is now numerically enforced.
_WEAK_SUPPORT_BASE = 0.5


def evaluate_rules(
    composition: dict, colour: dict, evidence: list[Evidence]
) -> tuple[list[dict], list[dict]]:
    del colour, evidence
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
            "status": status,
            "matched_evidence_ids": matched,
            "missing_evidence": missing,
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
    # Concern profiles now come from a real convergence engine (D6): a concern
    # requires >=2 independent evidence IDs from >=2 distinct source types and can
    # never arise from a single clinician-supplied symbol. With the current
    # release this still yields [] — but by convergence logic, not a hardcode.
    from .concerns import derive_concerns
    concerns = derive_concerns(evaluations)
    return evaluations, concerns
