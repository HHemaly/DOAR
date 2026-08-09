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

**Phase 2A.1, Section 4**: `redefine_coverage_full` post-processes
`PSY_AR_SIZE_FULL_015` (`coverage_full`) specifically, replacing its
image-relative `bounding_box_coverage >= 0.90` trigger condition with a
margin-based "approaches all four page margins" definition -- see
`docs/PAGE_COVERAGE_DEFINITION_DECISION.md` for the full comparison of
5 candidate definitions and why this one was chosen. `rules.py` itself
still computes its own (unused-downstream) `bounding_box_coverage`-based
verdict first; this function overrides it, exactly like
`apply_page_frame_gating` overrides status without ever editing
`rules.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Empirically-derived from a real, non-test, class-balanced 80-image
# sample (phase2a_threshold_sensitivity-style sampling, seed=11) --
# see docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md for the full
# distribution (min/p25/p50/p75/max) this was drawn from.
INTENSITY_PROXY_HEAVY_THRESHOLD = 0.6488   # p75
INTENSITY_PROXY_LIGHT_THRESHOLD = 0.4300   # p25
FRAGMENTATION_SHAKY_THRESHOLD = 0.2573     # p75
PLACEMENT_CENTER_BAND = (0.40, 0.60)       # matches the existing top/left/right convention exactly

# Phase 2A.1, Section 4: `coverage_full`'s redefined trigger condition --
# content must come within this fraction of EVERY page margin (left,
# top, right, bottom). `invented_operational_standin`: neither source
# PDF gives a number for "covers the whole page" at all (same honest
# status the original >=0.90 area threshold carried); NOT tuned to
# trigger frequency -- see docs/PAGE_COVERAGE_DEFINITION_DECISION.md.
COVERAGE_FULL_MARGIN_THRESHOLD = 0.15

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

# All page-relative rules across BOTH engines -- the full set page_frame_judge
# (judge_schemas.py) checks are never left ungated. EN_COMPILED_PLACEMENT_CENTER_029
# is already gated internally by evaluate_v2_rules above; included here so the
# judge has one authoritative list to check against.
ALL_PAGE_GATED_RULE_IDS = PAGE_GATED_HISTORICAL_RULE_IDS | {"EN_COMPILED_PLACEMENT_CENTER_029"}


def apply_page_frame_gating(
    rule_evaluations: list[dict[str, Any]], page_reference: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rewrites the historical engine's page-coverage/placement rule
    evaluations to `not_assessable` (never `not_matched`) when no page
    reference is assessable (Phase 2A, Section 3; Phase 2A.1, Section 3
    -- gates on the RESOLVED `page_reference.page_relative_features_assessable`,
    not the raw automatic `page_frame` status directly, so an explicit
    `user_confirmed_full_frame`/`user_defined_page_corners` declaration
    correctly overrides an automatic `cropped_or_content_only` reading).
    Leaves every other rule (including the 13 missing_detector rules and
    any not_evaluated placement rows already produced by a missing
    centroid) exactly as rules.py produced it."""
    if page_reference.get("page_relative_features_assessable"):
        return rule_evaluations
    gated = []
    for rule_eval in rule_evaluations:
        if rule_eval["rule_id"] in PAGE_GATED_HISTORICAL_RULE_IDS and rule_eval["status"] in ("weak_support", "not_matched"):
            gated.append({
                **rule_eval,
                "status": "not_assessable",
                "matched_evidence_ids": [],
                "missing_evidence": ["page_reference_not_assessable"],
                "rule_confidence": 0.0,
                "professional_reasoning": None,
                "parent_safe_wording": None,
            })
        else:
            gated.append(rule_eval)
    return gated


def _base_eval(rule: dict[str, Any], status: str, matched: list[str], missing: list[str]) -> dict[str, Any]:
    # confidence_ceiling is None for every not-yet-curator-reviewed
    # static_detector rule (rules_registry_v2.json) -- the 4 original
    # callers (V2_RULE_IDS) always have a real ceiling; this guard only
    # matters for evaluate_visual_object_presence_rules's broader,
    # registry-gated static_detector sweep, most of whose rows are the
    # gate-closed 'missing_detector' status anyway.
    raw_ceiling = rule.get("confidence_ceiling")
    ceiling = float(raw_ceiling) if raw_ceiling is not None else None
    rule_confidence = (round(min(0.5, ceiling), 4) if status == "weak_support" and ceiling is not None
                        else 0.0)
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
    rules_v2_by_id: dict[str, dict[str, Any]], objective_features: dict[str, Any], page_reference: dict[str, Any],
) -> list[dict[str, Any]]:
    """`objective_features` is `Analysis.to_dict()["objective_features"]`
    (feature_id -> asdict(FeatureValue)). `page_reference` is
    `Analysis.to_dict()["page_reference"]` (page_reference.py's resolved
    reference -- Phase 2A.1, Section 3; gates on
    `page_relative_features_assessable`, which correctly reflects an
    explicit user declaration overriding the automatic page_frame
    assessment, not the raw automatic status directly)."""
    evaluations = []
    page_assessable = bool(page_reference.get("page_relative_features_assessable"))

    # --- EN_COMPILED_PLACEMENT_CENTER_029: page-reference-gated --------------
    rule = rules_v2_by_id["EN_COMPILED_PLACEMENT_CENTER_029"]
    cx_fv = objective_features.get("composition.centroid_x")
    cy_fv = objective_features.get("composition.centroid_y")
    if not page_assessable:
        evaluations.append(_base_eval(rule, "not_assessable", [], ["page_reference_not_assessable"]))
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


def evaluate_visual_object_presence_rules(
    rules_v2_by_id: dict[str, dict[str, Any]], visual_evidence: list[Any],
) -> list[dict[str, Any]]:
    """DOAR MVP visual-evidence integration: extends this SAME real engine
    (not a parallel one) to `static_detector` rules whose `observable`
    exactly names a target this session's frozen visual-detector policy
    can validate for presence. Output shape/vocabulary is identical to
    every other v2 rule row (`_base_eval`) -- merges directly into the
    same `rule_evaluations` list `structured_report.py`/`parent_view.py`
    already consume, exactly like `evaluate_v2_rules`'s own rows.

    TWO independent safety gates, both required, neither bypassable here:
      1. `allowed_output_level == "individual_heuristic_only"` in the
         CURRENT `rules_registry_v2.json` -- this project's own existing
         governance decision. As of this session every `static_detector`
         rule is `disabled` there (curated when `DETECTOR_UNAVAILABLE`
         was still true) -- this function does NOT flip that gate; it
         only respects it. A rule with a real matching visual finding
         still reports `missing_detector` (never a fabricated
         `weak_support`) until the registry itself is curator-updated.
      2. `evidence.value["rule_eligible"]` on each visual Evidence record
         (set by `visual_evidence.py`'s adapter -- True ONLY for a
         finding whose `evidence_status == "validated_evidence"`).
         Experimental/unknown/disabled visual findings are structurally
         invisible to this function (filtered out before the loop below).

    "Not detected" is NEVER read as a matched negative/absence claim --
    this function only ever emits `weak_support` (a real, validated match
    found) or `missing_detector` (nothing evaluable to say, whether
    because the registry gate is closed, no match exists, or the only
    matching finding was experimental/unknown). A genuine absence claim
    would require established detector RECALL, which this project has not
    validated for any target (see Phase 2C.7's own explicit hand-detector
    caveat: low recall means absence-of-detection must never be read as
    absence-of-object) -- so `not_matched` is deliberately never produced
    here, unlike the historical engines' composition/placement rules."""
    evaluations = []
    by_label: dict[str, list[Any]] = {}
    for ev in visual_evidence:
        if ev.kind == "visual_detection" and ev.value.get("rule_eligible"):
            by_label.setdefault(ev.value["label"], []).append(ev)

    seen_rule_ids: set[str] = set()
    for label, matches in by_label.items():
        for rule_id, rule in rules_v2_by_id.items():
            if rule.get("observability_class") != "static_detector" or rule.get("observable") != label:
                continue
            if rule_id in seen_rule_ids:
                continue
            seen_rule_ids.add(rule_id)
            if rule.get("allowed_output_level") != "individual_heuristic_only":
                row = _base_eval(rule, "missing_detector", [],
                                  [f"rule_disabled_pending_registry_review:{rule_id}"])
            else:
                best = max(matches, key=lambda e: e.confidence)
                row = _base_eval(rule, "weak_support", [best.evidence_id], [])
            evaluations.append({**row, "visual_evidence_sourced": True})
    return evaluations


_PRODUCTION_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "resources" / "psychology_sources" / "rules_registry.json"


def redefine_coverage_full(
    rule_evaluations: list[dict[str, Any]], composition: dict[str, Any], page_reference: dict[str, Any],
) -> list[dict[str, Any]]:
    """DOAR-TRACE Phase 2A.1, Section 4: overrides `PSY_AR_SIZE_FULL_015`'s
    (`coverage_full`) status with the margin-based "approaches all four
    page margins" definition -- see docs/PAGE_COVERAGE_DEFINITION_DECISION.md
    for why this replaced the original image-relative
    `bounding_box_coverage >= 0.90` condition. Applies AFTER
    `apply_page_frame_gating`: any evaluation already `not_assessable`
    (page not visible) is left untouched -- this function only
    re-evaluates the trigger condition for cases where the page WAS
    confirmed/detected, using the confirmed page's own margins
    (`composition.margins_normalized`, currently identical to the
    image's margins for the two whole-image page-reference modes; see
    page_reference.py for when this would differ)."""
    if not page_reference.get("page_relative_features_assessable"):
        return rule_evaluations  # nothing to redefine -- already not_assessable

    production = json.loads(_PRODUCTION_REGISTRY_PATH.read_text(encoding="utf-8"))
    rule_def = next(r for r in production["rules"] if r["rule_id"] == "PSY_AR_SIZE_FULL_015")
    margins = composition.get("margins_normalized")

    out = []
    for rule_eval in rule_evaluations:
        if rule_eval["rule_id"] != "PSY_AR_SIZE_FULL_015" or rule_eval["status"] == "not_assessable":
            out.append(rule_eval)
            continue
        if margins is None:
            out.append({
                **rule_eval, "status": "not_matched", "matched_evidence_ids": [],
                "missing_evidence": ["margins_unavailable"], "rule_confidence": 0.0,
                "professional_reasoning": None, "parent_safe_wording": None,
            })
            continue
        matched = all(m <= COVERAGE_FULL_MARGIN_THRESHOLD for m in margins)
        status = "weak_support" if matched else "not_matched"
        ceiling = float(rule_eval["confidence_ceiling"])
        out.append({
            **rule_eval,
            "status": status,
            "matched_evidence_ids": ["ev_bbox_coverage"] if matched else [],
            "missing_evidence": [],
            "rule_confidence": round(min(0.5, ceiling), 4) if status == "weak_support" else 0.0,
            "professional_reasoning": rule_def["professional_reasoning"] if status == "weak_support" else None,
            "parent_safe_wording": rule_def["parent_safe_wording"] if status == "weak_support" else None,
        })
    return out
