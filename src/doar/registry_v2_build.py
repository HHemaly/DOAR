"""Builds `resources/psychology_sources/rules_registry_v2.json` (DOAR-TRACE
4C) from the production registry (`rules_registry.json`) plus the
provenance layer already built this session (`rule_schema.py`).

This module does not read the production registry at import time and
silently regenerate it on every run -- `rules_registry_v2.json` is a
committed, human-reviewable file (same convention as `rules_registry.json`
itself); this module is how it was produced and how it can be
regenerated/tested, not a live code path any pipeline imports.

**Every classification field below (observability_class,
allowed_output_level, evidence_level, target_construct, evidence_family,
dependency_group, direction, alternative_explanations) is a new judgment
call made this session, grounded in the rule's own wording and the real
extractor/detector inventory -- not extracted from any paper.** No
citation is fabricated: `references` is copied verbatim from the
production registry (6 real external papers, all cautionary, see
`rules_registry.json`'s own `references` block), and no new reference is
invented here.

`registry_v2_status` is `already_in_production_registry` for all 19 rows,
never the task's default `candidate_unreviewed` label -- because, per
`docs/RULE_PDF_COVERAGE_AUDIT.md`, the one PDF actually available in this
repository maps 1:1 onto the 19 rules already in `rules_registry.json`.
There is no genuinely new candidate row to mark unreviewed. Claiming
`candidate_unreviewed` for a rule that is literally already the
production registry would be a false downgrade, not a caution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .rule_schema import RULE_PROVENANCE, load_rules_v2

REGISTRY_V2_PATH = Path(__file__).resolve().parents[2] / "resources" / "psychology_sources" / "rules_registry_v2.json"

# ---------------------------------------------------------------------------
# Observability class (DOAR_TRACE_MASTER_SPEC.md Section 2)
# ---------------------------------------------------------------------------
OBSERVABILITY_CLASS: dict[str, str] = {
    # Tier-1: real, direct geometric measurements of the drawing's extent/position.
    "PSY_AR_SIZE_HALF_014": "static_direct",
    "PSY_AR_SIZE_FULL_015": "static_direct",
    "PSY_AR_SIZE_SMALL_016": "static_direct",
    "PSY_AR_PLACE_TOP_017": "static_direct",
    "PSY_AR_PLACE_LEFT_018": "static_direct",
    "PSY_AR_PLACE_RIGHT_019": "static_direct",
    # Tier-2, single static image, needs a detector that doesn't exist yet.
    "PSY_AR_EYES_WIDE_001": "static_detector",
    "PSY_AR_EYES_STERN_002": "static_detector",
    "PSY_AR_EYES_CLOSED_003": "static_detector",
    "PSY_AR_ANIMAL_TIGER_WOLF_004": "static_detector",
    "PSY_AR_ANIMAL_FOX_005": "static_detector",
    "PSY_AR_ANIMAL_SQUIRREL_006": "static_detector",
    "PSY_AR_ANIMAL_LION_007": "static_detector",
    "PSY_AR_GEOMETRY_008": "static_detector",
    "PSY_AR_STARS_009": "static_detector",
    "PSY_AR_CIRCLES_011": "static_detector",
    "PSY_AR_TRANSPORT_012": "static_detector",
    "PSY_AR_HEARTS_013": "static_detector",
    # Precondition ("while distracted/absent-minded") describes the drawing
    # PROCESS, not the finished image -- not observable from a static photo
    # of the result even with a perfect symbol detector.
    "PSY_AR_FLOWERS_CLOUDS_SUN_010": "process_required",
}

ALLOWED_OUTPUT_LEVEL: dict[str, str] = {
    rule_id: ("question_generating" if cls == "static_direct" else "disabled")
    for rule_id, cls in OBSERVABILITY_CLASS.items()
}

EVIDENCE_LEVEL: dict[str, str] = {
    rule_id: (
        "weak_quantitative_unvalidated" if cls == "static_direct"
        else "not_assessable" if cls == "process_required"
        else "speculative_symbolic"
    )
    for rule_id, cls in OBSERVABILITY_CLASS.items()
}

TARGET_CONSTRUCT: dict[str, str] = {
    "PSY_AR_EYES_WIDE_001": "outgoing_personality",
    "PSY_AR_EYES_STERN_002": "anger",
    "PSY_AR_EYES_CLOSED_003": "self_reflection_avoidance",
    "PSY_AR_ANIMAL_TIGER_WOLF_004": "anger",
    "PSY_AR_ANIMAL_FOX_005": "malicious_intent",
    "PSY_AR_ANIMAL_SQUIRREL_006": "need_for_protection",
    "PSY_AR_ANIMAL_LION_007": "superiority_belief",
    "PSY_AR_GEOMETRY_008": "goal_orientation_stubbornness",
    "PSY_AR_STARS_009": "attention_seeking",
    "PSY_AR_FLOWERS_CLOUDS_SUN_010": "positive_outlook",
    "PSY_AR_CIRCLES_011": "loneliness",
    "PSY_AR_TRANSPORT_012": "sociability_adventurousness",
    "PSY_AR_HEARTS_013": "affectionate_personality",
    "PSY_AR_SIZE_HALF_014": "situational_extraversion_introversion",
    "PSY_AR_SIZE_FULL_015": "self_esteem",
    "PSY_AR_SIZE_SMALL_016": "instability_fear",
    "PSY_AR_PLACE_TOP_017": "dreaminess_difficulty_adapting",
    "PSY_AR_PLACE_LEFT_018": "introversion",
    "PSY_AR_PLACE_RIGHT_019": "extraversion",
}

EVIDENCE_FAMILY: dict[str, str] = {
    "PSY_AR_EYES_WIDE_001": "facial_feature_style",
    "PSY_AR_EYES_STERN_002": "facial_feature_style",
    "PSY_AR_EYES_CLOSED_003": "facial_feature_style",
    "PSY_AR_ANIMAL_TIGER_WOLF_004": "animal_symbolism",
    "PSY_AR_ANIMAL_FOX_005": "animal_symbolism",
    "PSY_AR_ANIMAL_SQUIRREL_006": "animal_symbolism",
    "PSY_AR_ANIMAL_LION_007": "animal_symbolism",
    "PSY_AR_GEOMETRY_008": "shape_symbolism",
    "PSY_AR_STARS_009": "shape_symbolism",
    "PSY_AR_CIRCLES_011": "shape_symbolism",
    "PSY_AR_FLOWERS_CLOUDS_SUN_010": "object_symbolism",
    "PSY_AR_TRANSPORT_012": "object_symbolism",
    "PSY_AR_HEARTS_013": "object_symbolism",
    "PSY_AR_SIZE_HALF_014": "size_composition",
    "PSY_AR_SIZE_FULL_015": "size_composition",
    "PSY_AR_SIZE_SMALL_016": "size_composition",
    "PSY_AR_PLACE_TOP_017": "spatial_placement",
    "PSY_AR_PLACE_LEFT_018": "spatial_placement",
    "PSY_AR_PLACE_RIGHT_019": "spatial_placement",
}

# Generic, honest alternative explanations per evidence family -- not
# invented per rule, since the underlying uncertainty is the same across
# every rule in a family (copying, skill/development, task/context, style).
ALTERNATIVE_EXPLANATIONS: dict[str, list[str]] = {
    "facial_feature_style": [
        "May reflect an intentionally depicted expression (e.g. drawing an angry character on purpose), not the child's own state.",
        "May reflect copying a reference image, drawing style, motor/fine-skill limitations, or artistic convention.",
    ],
    "animal_symbolism": [
        "May reflect recent exposure to media, books, or a favourite character featuring the animal, not a psychological statement.",
        "May reflect the drawing prompt or immediate context (e.g. 'draw an animal') rather than a stable trait.",
    ],
    "shape_symbolism": [
        "May reflect current developmental drawing stage (geometric shapes are a common, age-typical drawing stage), practiced skill, or simple preference.",
        "May reflect what the child has recently been taught or practiced drawing at school.",
    ],
    "object_symbolism": [
        "May reflect a specific recent event, request, or interest (e.g. an upcoming trip, a favourite object) rather than a general personality trait.",
        "May reflect the drawing prompt rather than a spontaneous choice.",
    ],
    "size_composition": [
        "May reflect available paper space, drawing order, or how much detail the child chose to add, not emotional significance.",
        "May reflect perceived depth/proximity or task instructions rather than self-image (see REF_SIZE_DEPTH_2013).",
    ],
    "spatial_placement": [
        "May reflect handedness, where the child started drawing, or incidental page layout rather than a stable psychological trait.",
        "Placement effects were found weak and easily masked in the cited literature (REF_PLACEMENT_1992), not validated as fixed meanings.",
    ],
}

PSYCHOLOGIST_REVIEW_STATUS = "supplied_by_named_source_not_independently_reviewed"

REGISTRY_V2_SCHEMA_VERSION = "rules_registry_v2_draft_1"


def _direction(_rule_id: str) -> str:
    # No rule in the single available source PDF is phrased as contradicting
    # another rule's construct -- every rule states "observable -> supports
    # this trait." Cross-theme contradiction (e.g. one rule's construct
    # conflicting with another's) is a downstream aggregation-time question
    # (Section 4D), not a static per-rule attribute.
    return "supports_target_construct"


def build_registry_v2() -> dict[str, Any]:
    raw = json.loads((REGISTRY_V2_PATH.parent / "rules_registry.json").read_text(encoding="utf-8"))
    rules_v2 = {r.rule_id: r for r in load_rules_v2()}
    raw_by_id = {r["rule_id"]: r for r in raw["rules"]}

    entries = []
    for rule_id, provenance in RULE_PROVENANCE.items():
        v2 = rules_v2[rule_id]
        raw_rule = raw_by_id[rule_id]
        family = EVIDENCE_FAMILY[rule_id]
        entries.append({
            "rule_id": rule_id,
            "registry_v2_status": "already_in_production_registry",
            "source_document": v2.source_document,
            "source_page": v2.source_page,
            "source_section": v2.source_section,
            "faithful_source_quote": provenance.get("source_quote"),
            "observable": v2.observable,
            "possible_interpretation": {"english": v2.title_english, "arabic": v2.arabic},
            "observability_class": OBSERVABILITY_CLASS[rule_id],
            "allowed_output_level": ALLOWED_OUTPUT_LEVEL[rule_id],
            "evidence_level": EVIDENCE_LEVEL[rule_id],
            "target_construct": TARGET_CONSTRUCT[rule_id],
            "evidence_family": family,
            "dependency_group": list(v2.required_feature_ids),
            "direction": _direction(rule_id),
            "required_detector_or_metadata": (
                None if OBSERVABILITY_CLASS[rule_id] == "static_direct"
                else v2.required_feature_ids[0]
            ),
            "limitations": list(raw_rule["limitations"]),
            "alternative_explanations": list(ALTERNATIVE_EXPLANATIONS[family]),
            "parent_safe_wording": raw_rule["parent_safe_wording"],
            "professional_wording": raw_rule["professional_reasoning"],
            "reference_ids": list(raw_rule["references"]),
            "psychologist_review_status": PSYCHOLOGIST_REVIEW_STATUS,
            "validation_status": raw_rule["activation_status"],
            "confidence_ceiling": raw_rule["confidence_ceiling"],
            "scientific_support": raw_rule["scientific_support"],
            "threshold_source": v2.threshold_source,
            "operational_threshold": provenance.get("operational_threshold"),
        })

    return {
        "schema_version": REGISTRY_V2_SCHEMA_VERSION,
        "status": "draft -- NOT the production registry; rules.py/rules_registry.json are unchanged and remain authoritative",
        "source_pdf": "التحليل النفسي للصور.pdf",
        "note": (
            "child_drawing_rules_compiled.pdf, named in the task that "
            "produced this file, does not exist in this repository. The "
            "one PDF actually present was used instead; see "
            "docs/RULE_PDF_COVERAGE_AUDIT.md."
        ),
        "rule_count": len(entries),
        "references": raw["references"],
        "rules": entries,
    }


def write_registry_v2() -> Path:
    document = build_registry_v2()
    REGISTRY_V2_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return REGISTRY_V2_PATH


if __name__ == "__main__":
    path = write_registry_v2()
    print(f"wrote {path}")
