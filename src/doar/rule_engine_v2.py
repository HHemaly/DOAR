"""Parallel v2 rule evaluator for newly-operational static-direct/
static-proxy rules (DOAR-TRACE Phase 2A, Section 7).

This is **additive and parallel** to `rules.py::evaluate_rules` -- it
does not modify, replace, or delete the historical production rule
engine, and it never touches `rules_registry.json`. Its output uses the
exact same status vocabulary as `rules.py`
(`weak_support`/`not_matched`/`missing_detector`) plus one new status,
`not_assessable`, for page-frame-gated rules -- so its results merge
directly into the same `rule_evaluations` list
`structured_report.py` already consumes.

Exactly 4 rules are activated here, each satisfying every one of the 7
required conditions (feature exists, definition matches the source
observable, input/page-frame requirements satisfied, threshold policy
recorded, proxy-safe wording, tests exist, still labeled psychologically
unvalidated):

  EN_COMPILED_PLACEMENT_CENTER_029    -- composition.centroid_normalized (page-frame-gated)
  EN_COMPILED_LINE_HEAVY_PRESSURE_030 -- stroke.intensity_proxy, top quartile (real, non-test-data-derived)
  EN_COMPILED_LINE_LIGHT_PRESSURE_031 -- stroke.intensity_proxy, bottom quartile
  EN_COMPILED_LINE_SHAKY_BROKEN_032   -- stroke.fragmentation, top quartile

`EN_COMPILED_EXCESSIVE_DETAIL_040`/`EN_COMPILED_NEGLECT_BACKGROUND_041`
were audited and deliberately NOT activated -- see
docs/STATIC_PROXY_RULE_POLICY.md for the full reasoning (no sufficiently
precise existing feature maps to either observable without risking a
misleading claim).

Threshold provenance: `empirically_exploratory` -- quartile cut points
computed from a real, non-test, class-balanced 80-image sample
(`docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md`), not expert-defined and
not sourced from either PDF (neither source discusses stroke
pressure/fragmentation numerically at all). Heavy/light pressure use the
top/bottom quartile of the SAME `stroke.intensity_proxy` distribution,
which makes them structurally mutually exclusive (a value cannot be both
>= the 75th percentile and <= the 25th percentile) -- not just
conventionally, but by construction.

This module also exports `apply_page_frame_gating`, which post-processes
(never replaces) the historical `rules.py::evaluate_rules` output so its
6 original page-coverage/placement rules become `not_assessable` -- not
`not_matched` -- when the page is not visible in the image, per Section 3.
"""

from __future__ import annotations

from typing import Any

from .page_frame import ASSESSABLE_STATUSES

# Empirically-derived from a real, non-test, class-balanced 80-image
# sample (phase2a_threshold_sensitivity-style sampling, seed=11) --
# see docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md for the full
# distribution (min/p25/p50/p75/max) this was drawn from.
INTENSITY_PROXY_HEAVY_THRESHOLD = 0.6488   # p75
INTENSITY_PROXY_LIGHT_THRESHOLD = 0.4300   # p25
FRAGMENTATION_SHAKY_THRESHOLD = 0.2573     # p75
PLACEMENT_CENTER_BAND = (0.40, 0.60)       # matches the existing top/left/right convention exactly

V2_RULE_IDS = frozenset({
    "EN_COMPILED_PLACEMENT_CENTER_029", "EN_COMPILED_LINE_HEAVY_PRESSURE_030",
    "EN_COMPILED_LINE_LIGHT_PRESSURE_031", "EN_COMPILED_LINE_SHAKY_BROKEN_032",
})

# The original 6 tier-1 rules dispatched by the HISTORICAL production engine
# (rules.py::evaluate_rules / rules_registry.json) are page-coverage or
# placement rules: their underlying measurement (composition.bounding_box_
# coverage / composition.centroid_normalized) is only meaningful relative to
# the full page. rules.py itself is never edited (per Phase 2A working-style
# constraints); instead this function POST-PROCESSES its output, applied by
# analysis.py right after evaluate_rules() returns -- additive, not a
# replacement of the historical engine.
PAGE_GATED_HISTORICAL_RULE_IDS = frozenset({
    "PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_FULL_015", "PSY_AR_SIZE_SMALL_016",
    "PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019",
})


def apply_page_frame_gating(
    rule_evaluations: list[dict[str, Any]], page_frame: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rewrites the historical engine's page-coverage/placement rule
    evaluations to `not_assessable` (never `not_matched`) when the page
    is not visible/assessable in the uploaded image (Phase 2A, Section 3).
    Leaves every other rule (including the 13 missing_detector rules and
    any not_evaluated placement rows already produced by a missing
    centroid) exactly as rules.py produced it."""
    if page_frame.get("page_frame_status") in ASSESSABLE_STATUSES:
        return rule_evaluations
    gated = []
    for rule_eval in rule_evaluations:
        if rule_eval["rule_id"] in PAGE_GATED_HISTORICAL_RULE_IDS and rule_eval["status"] in ("weak_support", "not_matched"):
            gated.append({
                **rule_eval,
                "status": "not_assessable",
                "matched_evidence_ids": [],
                "missing_evidence": ["page_frame_status_not_assessable"],
                "rule_confidence": 0.0,
                "professional_reasoning": None,
                "parent_safe_wording": None,
            })
        else:
            gated.append(rule_eval)
    return gated


def _base_eval(rule: dict[str, Any], status: str, matched: list[str], missing: list[str]) -> dict[str, Any]:
    ceiling = float(rule["confidence_ceiling"])
    rule_confidence = round(min(0.5, ceiling), 4) if status == "weak_support" else 0.0
    return {
        "rule_id": rule["rule_id"],
        "tier": None,  # v2 rules use observability_class instead of the legacy tier vocabulary
        "activation_status": "IMPLEMENTED_UNVALIDATED",
        "status": status,
        "matched_evidence_ids": matched,
        "missing_evidence": missing,
        "source_type": "psychologist_supplied_hypothesis",
        "scientific_support": rule["scientific_support"],
        "confidence_ceiling": ceiling,
        "rule_confidence": rule_confidence,
        "confidence_ceiling_enforced": True,
        "professional_reasoning": rule["professional_wording"] if status == "weak_support" else None,
        "parent_safe_wording": rule["parent_safe_wording"] if status == "weak_support" else None,
        "original_arabic": None,
        "english_translation": rule["possible_interpretation"],
        "references": rule["reference_ids"],
        "limitations": rule["limitations"],
        "requires_clinician_review": True,
    }


def evaluate_v2_rules(
    rules_v2_by_id: dict[str, dict[str, Any]], objective_features: dict[str, Any], page_frame: dict[str, Any],
) -> list[dict[str, Any]]:
    """`objective_features` is `Analysis.to_dict()["objective_features"]`
    (feature_id -> asdict(FeatureValue)). `page_frame` is
    `Analysis.to_dict()["page_frame"]` (page_frame.py's assessment)."""
    evaluations = []
    page_assessable = page_frame.get("page_frame_status") in ASSESSABLE_STATUSES

    # --- EN_COMPILED_PLACEMENT_CENTER_029: page-frame-gated ------------------
    rule = rules_v2_by_id["EN_COMPILED_PLACEMENT_CENTER_029"]
    cx_fv = objective_features.get("composition.centroid_x")
    cy_fv = objective_features.get("composition.centroid_y")
    if not page_assessable:
        evaluations.append(_base_eval(rule, "not_assessable", [], ["page_frame_status_not_assessable"]))
    elif cx_fv is None or cy_fv is None or cx_fv.get("missing") or cy_fv.get("missing"):
        evaluations.append(_base_eval(rule, "not_matched", [], ["centroid_unavailable"]))
    else:
        cx, cy = cx_fv["value"], cy_fv["value"]
        lo, hi = PLACEMENT_CENTER_BAND
        matched = lo <= cx <= hi and lo <= cy <= hi
        status = "weak_support" if matched else "not_matched"
        evaluations.append(_base_eval(rule, status, ["ev_centroid"] if matched else [], []))

    # --- Heavy/light line-pressure-appearance: mutually exclusive by construction ---
    intensity_fv = objective_features.get("stroke.intensity_proxy")
    for rule_id, threshold, comparator in (
        ("EN_COMPILED_LINE_HEAVY_PRESSURE_030", INTENSITY_PROXY_HEAVY_THRESHOLD, "ge"),
        ("EN_COMPILED_LINE_LIGHT_PRESSURE_031", INTENSITY_PROXY_LIGHT_THRESHOLD, "le"),
    ):
        rule = rules_v2_by_id[rule_id]
        if intensity_fv is None or intensity_fv.get("missing"):
            evaluations.append(_base_eval(rule, "not_matched", [], ["stroke_intensity_proxy_unavailable"]))
            continue
        value = intensity_fv["value"]
        matched = value >= threshold if comparator == "ge" else value <= threshold
        evidence_id = "ev_feature_stroke_intensity_proxy"
        evaluations.append(_base_eval(rule, "weak_support" if matched else "not_matched",
                                       [evidence_id] if matched else [], []))

    # --- Shaky/broken lines: fragmentation proxy -----------------------------
    rule = rules_v2_by_id["EN_COMPILED_LINE_SHAKY_BROKEN_032"]
    frag_fv = objective_features.get("stroke.fragmentation")
    if frag_fv is None or frag_fv.get("missing"):
        evaluations.append(_base_eval(rule, "not_matched", [], ["stroke_fragmentation_unavailable"]))
    else:
        matched = frag_fv["value"] >= FRAGMENTATION_SHAKY_THRESHOLD
        evaluations.append(_base_eval(rule, "weak_support" if matched else "not_matched",
                                       ["ev_feature_stroke_fragmentation"] if matched else [], []))

    return evaluations
