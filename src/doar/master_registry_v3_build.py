"""Phase G0 -- MASTER_RULE_FEATURE_REGISTRY_V3.json builder.

Builds a RESEARCH-ONLY registry that reconciles the existing, frozen DOAR
rule corpus (RULE_EVIDENCE_MATRIX.csv / rules_registry_v2.json / CONCERN_
DOMAIN_MAP.json -- read here, never written) with the new candidate source
families named in DOAR_Master_Rules_and_Features_Inventory_V3.pdf (FEATS,
Koppitz, the 2023 HTP meta-analysis, the 2026 tree-imagery update, the 2022
family-drawing attachment literature, and the new objective graphic/formal
measurements proposed after the psychologist meeting).

This module NEVER modifies rules_registry_v2.json, RULE_EVIDENCE_MATRIX.csv,
CONCERN_DOMAIN_MAP.json, the aggregator, or the rule engine -- it only reads
them (for the 41 existing rules, verbatim) and writes ONE new, separate,
additive file. Mirrors registry_v2_build.py's own build/write-function
pattern exactly.

Every entry newly introduced by the PDF (i.e. every entry NOT one of the 41
existing rules) has `activation_allowed=False` -- by construction, not by a
per-row flag anyone could accidentally flip here. `existing_41_rules()`
alone can ever set it to a real production status, and only by re-reading
what is ALREADY enabled in RULE_EVIDENCE_MATRIX.csv/CONCERN_DOMAIN_MAP.json.

Field list per Section 8, Step 2 of the task:
id, canonical_observable, display_name, source_family, source_reference,
source_document, source_page_or_section_if_known, source_interpretation,
safe_doar_interpretation, evidence_strength, scientific_support_status,
significant_or_not_if_meta_analysis, effect_size_if_available, task_type,
free_vs_instructed, age_or_development_limits,
culture_material_motor_confounders, static_image_detectable,
requires_process_data, requires_longitudinal_data, measurement_spec,
candidate_detector, existing_feature_id_if_any, evidence_family,
dedup_group, alternative_explanations, parent_safe_wording,
clinician_wording, maturity_status, expert_review_status,
activation_allowed.

PHASE G0.1 -- THREE EXPLICIT, NEVER-CONFLATED HIERARCHY LEVELS (previously
the registry only reported one ambiguous "dedup group" count, which was
actually just the evidence-family count):

  1. SOURCE ENTRY        -- one row `id` per literature/registry mention
                             (e.g. Koppitz "Tiny figure", HTP "Very small
                             person" are TWO source entries).
  2. CANONICAL OBSERVABLE -- `canonical_observable_id` -- the distinct
                             MEASURABLE CONCEPT after semantic dedup.
                             Koppitz "Tiny figure" and HTP "Very small
                             person" merge into ONE canonical observable
                             (`person_relative_size_small`); HTP "Very
                             small house"/"Very small tree" do NOT merge
                             into it -- they remain separate, object-
                             specific canonical observables
                             (`house_relative_size_small`,
                             `tree_relative_size_small`), even though all
                             three share the same evidence family.
  3. EVIDENCE FAMILY      -- `evidence_family` -- the broad group used
                             later to prevent double-counting independent
                             evidence (reuses RULE_EVIDENCE_MATRIX.csv's
                             15 existing family names wherever a new
                             observable genuinely belongs to one of them,
                             plus a small number of new formal-dimension
                             families for concepts the existing 15 don't
                             cover at all).

`dedup_source_group` == `canonical_observable_id` (kept as its own field
per the task's literal field list; it names which source entries were
merged together, discoverable via `canonical_observables[id]
["source_entry_ids"]` in the built document). Merges below are ONLY ever
asserted where verified BY HAND against the real PDF text quoted in
Section B-G's row tables -- every entry NOT listed in
`_CANONICAL_OBSERVABLE_OVERRIDES` keeps its own 1:1 canonical observable
(a plain slug of its own `canonical_observable` text), which is the
scientifically honest default: no merge is claimed unless verified.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MASTER_REGISTRY_V3_PATH = ROOT / "MASTER_RULE_FEATURE_REGISTRY_V3.json"
MASTER_REGISTRY_V3_SCHEMA_VERSION = "master_rule_feature_registry_v3_v2"

_SOURCE_PDF_NAME = "DOAR_Master_Rules_and_Features_Inventory_V3.pdf"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", slug)


# ---------------------------------------------------------------------------
# Hand-verified canonical-observable merges across source families. Key is
# (source_family, canonical_observable) -- `canonical_observable` on each
# hand-authored row below is set to that row's own `name`/text VERBATIM, so
# this key reliably matches without depending on any auto-generated id
# scheme. Value is (canonical_observable_id, canonical_observable_label).
# ---------------------------------------------------------------------------
_CANONICAL_OBSERVABLE_MERGES: dict[tuple[str, str], tuple[str, str]] = {
    # --- SIZE (evidence family: size_composition) --------------------------
    ("koppitz_hfd_indicator", "Tiny figure"): ("person_relative_size_small", "Person: relatively small size"),
    ("htp_2023_meta_analysis", "Very small person"): ("person_relative_size_small", "Person: relatively small size"),
    ("family_drawing_2022_marker", "Unusually small figures"): ("person_relative_size_small", "Person: relatively small size"),
    ("koppitz_hfd_indicator", "Big figure"): ("person_relative_size_large", "Person: relatively large size"),
    ("family_drawing_2022_marker", "Unusually large figures"): ("person_relative_size_large", "Person: relatively large size"),
    ("htp_2023_meta_analysis", "Very small house"): ("house_relative_size_small", "House: relatively small size"),
    ("htp_2023_meta_analysis", "Very small tree"): ("tree_relative_size_small", "Tree: relatively small size"),
    ("htp_2023_meta_analysis", "Small drawing size"): ("whole_drawing_relative_size_small", "Whole drawing: relatively small size"),
    ("existing_doar_rule", "coverage_small"): ("whole_drawing_relative_size_small", "Whole drawing: relatively small size"),
    # NOTE: `very_small_drawing_absolute` (EN_COMPILED_VERY_SMALL_DRAWING_026)
    # is deliberately NOT merged here -- it is an ABSOLUTE-physical-size
    # measurement (requires_absolute_scale_or_context=True in the frozen
    # matrix), a genuinely different measurement type from `coverage_small`'s
    # relative/proportional page-coverage measurement, even though both
    # describe "small size" in the same evidence family.
    ("existing_doar_rule", "coverage_full"): ("whole_drawing_relative_size_large", "Whole drawing: full/expansive page use"),
    ("koppitz_hfd_indicator", "Tiny head"): ("head_relative_size_small", "Head: relatively small size"),
    ("family_drawing_2022_marker", "Exaggeration of heads"): ("head_relative_size_large", "Head: relatively large size"),
    ("koppitz_hfd_indicator", "Big hands"): ("hands_relative_size_large", "Hands: relatively large size"),
    ("koppitz_hfd_indicator", "Short arms"): ("arms_relative_length_short", "Arms: relatively short length"),
    ("koppitz_hfd_indicator", "Long arms"): ("arms_relative_length_long", "Arms: relatively long length"),
    ("htp_2023_meta_analysis", "Inappropriate body proportions"): (
        "body_proportion_anomaly_generic", "Body: atypical proportions (part unspecified)"),
    ("new_objective_graphic_feature", "Body-part proportion anomalies"): (
        "body_proportion_anomaly_generic", "Body: atypical proportions (part unspecified)"),
    ("new_objective_graphic_feature", "Figure/object size ratios"): (
        "figure_object_size_ratio_generic", "Figure/object: relative size ratio (generic measurement)"),
    # --- MISSING BODY PARTS (evidence family: missing_body_part) -----------
    ("existing_doar_rule", "missing_hands"): ("missing_hands", "Hands: absent"),
    ("koppitz_hfd_indicator", "Hands cut off / absent"): ("missing_hands", "Hands: absent"),
    ("existing_doar_rule", "missing_mouth"): ("missing_mouth", "Mouth: absent"),
    ("koppitz_hfd_indicator", "No mouth"): ("missing_mouth", "Mouth: absent"),
    ("koppitz_hfd_indicator", "No eyes"): ("missing_eyes", "Eyes: absent"),
    ("koppitz_hfd_indicator", "No nose"): ("missing_nose", "Nose: absent"),
    ("koppitz_hfd_indicator", "No body"): ("missing_body_trunk", "Body/trunk: absent"),
    ("koppitz_hfd_indicator", "No arms"): ("missing_arms", "Arms: absent"),
    ("koppitz_hfd_indicator", "No legs"): ("missing_legs", "Legs: absent"),
    ("koppitz_hfd_indicator", "No feet"): ("missing_feet", "Feet: absent"),
    ("koppitz_hfd_indicator", "No neck"): ("missing_neck", "Neck: absent"),
    ("htp_2023_meta_analysis", "Incomplete person"): ("incomplete_person_generic", "Person: incomplete (part unspecified)"),
    ("family_drawing_2022_marker", "Incomplete figures"): ("incomplete_person_generic", "Person: incomplete (part unspecified)"),
    ("htp_2023_meta_analysis", "Complete/partial loss of limbs"): (
        "missing_limb_generic", "Limb(s): partial/complete absence (arm or leg unspecified)"),
    ("family_drawing_2022_marker", "Omission of mother or child"): (
        "family_member_omitted", "Family member: omitted from scene"),
    ("htp_2023_meta_analysis", "Omitted house/tree/person"): (
        "whole_scene_object_omitted", "Whole scene: house/tree/person omitted"),
    ("htp_2023_meta_analysis", "No door"): ("house_door_missing", "House: door absent"),
    ("htp_2023_meta_analysis", "No window"): ("house_window_missing", "House: window absent"),
    ("new_objective_graphic_feature", "Part omissions"): ("part_omission_generic", "Object part: omission (generic measurement)"),
    # --- LINE DARKNESS / PRESSURE (evidence family: line_intensity_quality) -
    ("existing_doar_rule", "heavy_line_pressure_appearance"): ("line_darkness_intensity", "Line darkness/pressure appearance"),
    ("existing_doar_rule", "light_line_pressure_appearance"): ("line_darkness_intensity", "Line darkness/pressure appearance"),
    ("new_objective_graphic_feature", "Line darkness"): ("line_darkness_intensity", "Line darkness/pressure appearance"),
    ("new_objective_graphic_feature", "Darkness variability"): ("line_darkness_variability", "Line darkness/pressure variability"),
    # --- SHADING / AREA DARKNESS (evidence family: colour_mood_flags) ------
    ("htp_2023_meta_analysis", "Shaded/blackened drawing"): ("whole_drawing_darkness", "Whole drawing: shaded/blackened area"),
    ("new_objective_graphic_feature", "Blackened-area ratio"): ("whole_drawing_darkness", "Whole drawing: shaded/blackened area"),
    ("new_objective_graphic_feature", "Shading density"): ("whole_drawing_shading_density", "Whole drawing: local shading density"),
    ("htp_2023_meta_analysis", "Shaded/blackened wall"): ("house_wall_darkness", "House wall: shaded/blackened area"),
    ("htp_2023_meta_analysis", "Shaded/blackened tree"): ("tree_darkness", "Tree: shaded/blackened area"),
    ("tree_imagery_2026_update", "Blackened tree (example OR≈2.01)"): ("tree_darkness", "Tree: shaded/blackened area"),
    # --- TREE SHAPE (evidence family: object_symbolism) -- Phase G0.2
    # Step 2 audit: the 2023 HTP meta-analysis and its own 2026 tree-
    # imagery update both report the identical tree-crown-shape and
    # tree-size observables, just from two different (overlapping)
    # meta-analyses -- unlike the "weak lines"/"overly simple" tree2026
    # rows (deliberately kept distinct below), these two are the SAME
    # measurable concept under different citations, not merely similar.
    ("htp_2023_meta_analysis", "Flattened crown"): ("tree_crown_flattened", "Tree: flattened crown shape"),
    ("tree_imagery_2026_update", "Flattened crown (example OR≈3.10)"): ("tree_crown_flattened", "Tree: flattened crown shape"),
    ("tree_imagery_2026_update", "Very small tree (example OR≈3.93)"): ("tree_relative_size_small", "Tree: relatively small size"),
    ("htp_2023_meta_analysis", "Shaded/blackened person"): ("person_darkness", "Person: shaded/blackened area"),
    ("koppitz_hfd_indicator", "Shading of face"): ("face_shading", "Face: shading"),
    ("koppitz_hfd_indicator", "Shading of body/limbs"): ("body_limb_shading", "Body/limbs: shading"),
    ("koppitz_hfd_indicator", "Shading of hands/neck"): ("hands_neck_shading", "Hands/neck: shading"),
    # --- EYE APPEARANCE (evidence family: facial_feature_style) -----------
    ("existing_doar_rule", "wide_eyes"): ("eyes_wide", "Eyes: wide appearance"),
    ("existing_doar_rule", "stern_eyes"): ("eyes_stern", "Eyes: stern/intense appearance"),
    ("existing_doar_rule", "closed_eyes"): ("eyes_closed", "Eyes: closed"),
    ("existing_doar_rule", "eyes_missing_or_undetailed"): ("eyes_missing_or_undetailed", "Eyes: missing or undetailed"),
    ("koppitz_hfd_indicator", "Crossed eyes"): ("eyes_crossed", "Eyes: crossed/inward"),
    # --- FACIAL EXPRESSION/AFFECT (evidence family: facial_feature_style) --
    ("existing_doar_rule", "face_expression"): ("facial_expression_generic", "Facial expression (generic depicted affect)"),
    ("new_objective_graphic_feature", "Facial affect in drawing"): (
        "facial_expression_generic", "Facial expression (generic depicted affect)"),
    ("htp_2023_meta_analysis", "Negative expression"): ("facial_expression_negative", "Facial expression: negative"),
    ("family_drawing_2022_marker", "Neutral/negative facial affect"): (
        "facial_expression_negative", "Facial expression: negative"),
    ("htp_2023_meta_analysis", "Poker face"): ("facial_expression_neutral", "Facial expression: neutral/flat"),
    # --- COLOUR PRESENCE (evidence family: colour_mood_flags) --------------
    ("family_drawing_2022_marker", "Lack of color"): ("colour_diversity_low", "Low colour diversity/use"),
    ("new_objective_graphic_feature", "Colour diversity"): ("colour_diversity_low", "Low colour diversity/use"),
    # --- ISOLATION THEME (evidence family: isolation_theme) ----------------
    ("existing_doar_rule", "repeated_isolation_themes"): ("isolation_theme_generic", "Isolation theme (generic)"),
    ("family_drawing_2022_global_scale", "Emotional Distance / Isolation"): (
        "isolation_theme_generic", "Isolation theme (generic)"),
}


def _base_entry(**kwargs: Any) -> dict[str, Any]:
    """Every field defaults to None/[]/False so a hand-authored row below
    only needs to set what actually applies to it -- an omitted field is
    honestly absent, never silently fabricated."""
    entry = {
        "id": None, "canonical_observable": None, "display_name": None,
        "source_family": None, "source_reference": None, "source_document": _SOURCE_PDF_NAME,
        "source_page_or_section_if_known": None, "source_interpretation": None,
        "safe_doar_interpretation": None, "evidence_strength": None,
        "scientific_support_status": "not_scientifically_validated",
        "significant_or_not_if_meta_analysis": None, "effect_size_if_available": None,
        "task_type": None, "free_vs_instructed": None, "age_or_development_limits": None,
        "culture_material_motor_confounders": None, "static_image_detectable": None,
        "requires_process_data": False, "requires_longitudinal_data": False,
        "measurement_spec": None, "candidate_detector": None, "existing_feature_id_if_any": None,
        "evidence_family": None, "dedup_group": None,
        # Phase G0.1, Step 5: WHOLE_DRAWING / OBJECT_REGION / RELATIONSHIP /
        # PROCESS_REQUIRED / LONGITUDINAL_REQUIRED. Defaults to
        # WHOLE_DRAWING (the common case); overridden explicitly for rows
        # that are genuinely process-/longitudinal-/region-/relationship-
        # scoped (see requires_process_data/requires_longitudinal_data for
        # the first two, and _OBJECTIVE_FEATURE_IMAGE_SCOPE for the rest).
        "image_scope": "WHOLE_DRAWING",
        # Phase G0.2, Step 4-6 -- filled in by _enrich_scope_task_and_
        # readiness() during assembly; left as None here only so a
        # missing enrichment pass is loud (KeyError-free but visibly
        # None), never silently defaulted to a specific value.
        "primary_scope": None, "secondary_scope": None, "task_requirement": None, "technical_readiness": None,
        "canonical_observable_id": None, "dedup_source_group": None, "alternative_explanations": [],
        "parent_safe_wording": None, "clinician_wording": None,
        "maturity_status": None, "expert_review_status": "pending_expert_review",
        "activation_allowed": False,
    }
    entry.update(kwargs)
    return entry


def _default_safe_interpretation(display_name: str) -> str:
    return (f"'{display_name}' is recorded as a research candidate observation only -- not a validated "
            "psychological rule, and not currently used by DOAR's concern-domain aggregation.")


def _default_parent_wording(display_name: str) -> str:
    return (f"DOAR recorded a formal/graphic pattern related to '{display_name}'. This is a research "
            "observation, not a finding about your child -- it is not currently part of DOAR's conclusions.")


def _default_clinician_wording(display_name: str, source_interpretation: str, source_reference: str) -> str:
    return (f"Observable '{display_name}'. Literature-reported association (as written in the source, "
            f"NOT a DOAR finding): {source_interpretation} (source: {source_reference}). Research candidate "
            "only -- not wired into concern-domain support.")


def _finalize(raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fills every field a hand-authored row left as None with a safe,
    formulaic default derived from the row's own display_name/source_
    interpretation/source_reference -- never a fabricated clinical claim,
    and activation_allowed is forced to False here unconditionally for
    every entry this module adds (belt-and-braces on top of _base_entry's
    own default)."""
    out = []
    for e in raw_entries:
        entry = dict(e)
        # Step 5: a row explicitly requiring process/longitudinal data is
        # never left at the generic WHOLE_DRAWING default, even if its own
        # section builder didn't set image_scope explicitly.
        if entry["image_scope"] == "WHOLE_DRAWING":
            if entry["requires_longitudinal_data"]:
                entry["image_scope"] = "LONGITUDINAL_REQUIRED"
            elif entry["requires_process_data"]:
                entry["image_scope"] = "PROCESS_REQUIRED"
        display_name = entry["display_name"] or entry["canonical_observable"] or entry["id"]
        if entry["safe_doar_interpretation"] is None:
            entry["safe_doar_interpretation"] = _default_safe_interpretation(display_name)
        if entry["parent_safe_wording"] is None:
            entry["parent_safe_wording"] = _default_parent_wording(display_name)
        if entry["clinician_wording"] is None:
            entry["clinician_wording"] = _default_clinician_wording(
                display_name, entry["source_interpretation"] or "", entry["source_reference"] or "")
        entry["activation_allowed"] = False
        out.append(entry)
    return out


def _assign_canonical_observable(entry: dict[str, Any]) -> dict[str, Any]:
    """LEVEL 2 of the hierarchy. Looks up (source_family, canonical_
    observable) in the hand-verified merge table; if no merge was
    verified, the entry keeps its OWN 1:1 canonical observable (a plain
    slug of its own text) -- never force-merged by keyword guessing.
    `dedup_source_group` is set equal to `canonical_observable_id` (the
    task's literal field list keeps both names; they carry the same
    grouping key here)."""
    key = (entry["source_family"], entry["canonical_observable"])
    merge = _CANONICAL_OBSERVABLE_MERGES.get(key)
    if merge:
        canonical_id, canonical_label = merge
        entry["canonical_observable_id"] = canonical_id
        entry["canonical_observable"] = canonical_label
    else:
        raw = entry["canonical_observable"] or entry["display_name"] or entry["id"]
        entry["canonical_observable_id"] = _slugify(raw)
    entry["dedup_source_group"] = entry["canonical_observable_id"]
    return entry


# ---------------------------------------------------------------------------
# Phase G0.2, Step 4-6 -- explicit scope/task/readiness classification,
# replacing G0.1's coarse 5-value `image_scope` with the fuller controlled
# vocabulary. Keyword-matched against `canonical_observable_id` (already
# hand-verified, object-specific naming from Step 2's merge table) plus
# `evidence_family`/`source_family`/the G0.1 `image_scope` signal where
# that was already precise (OBJECT_REGION/RELATIONSHIP/PROCESS_REQUIRED/
# LONGITUDINAL_REQUIRED rows keep that classification). This is
# TECHNICAL READINESS/SCOPE metadata only -- it never changes
# activation_allowed, evidence_family, or any aggregation-facing field.
# ---------------------------------------------------------------------------

PRIMARY_SCOPE_VALUES = (
    "WHOLE_DRAWING", "OBJECT_REGION", "HUMAN_FIGURE", "FACE_REGION", "HOUSE_REGION",
    "TREE_REGION", "COLOUR_REGION", "RELATIONSHIP", "SCENE_SEMANTIC", "TASK_SPECIFIC",
    "PROCESS_REQUIRED", "LONGITUDINAL_REQUIRED", "CONTEXT_REQUIRED", "FRAMEWORK_ONLY",
)
TASK_REQUIREMENT_VALUES = (
    "ANY_DRAWING", "FREE_DRAWING", "HUMAN_FIGURE_DRAWING", "HTP", "FAMILY_DRAWING",
    "PPAT_OR_FEATS_TASK", "PROCESS_SESSION", "LONGITUDINAL_SERIES",
)
TECHNICAL_READINESS_VALUES = (
    "READY_CURRENT_DETERMINISTIC", "READY_EXISTING_SEMANTIC_PROVIDER", "NEEDS_OBJECT_DETECTOR_VALIDATION",
    "NEEDS_PART_DETECTOR", "NEEDS_SEGMENTATION_MASK", "NEEDS_VECTOR_STROKE_METHOD",
    "NEEDS_EXPERT_SEMANTIC_REVIEW", "PROCESS_DATA_REQUIRED", "LONGITUDINAL_DATA_REQUIRED",
    "CONTEXT_REQUIRED", "NOT_CURRENTLY_MEASURABLE",
)

_HOUSE_KEYWORDS = ("house", "door", "window", "chimney", "roof", "_wall")
_TREE_KEYWORDS = ("tree", "root", "crown", "branch", "trunk")
_FACE_KEYWORDS = ("eye", "face", "mouth", "nose", "teeth")
_HUMAN_FIGURE_KEYWORDS = (
    "person", "figure", "body", "head_", "hands_", "arms_", "limb", "neck", "feet",
    "genital", "missing_hands", "missing_arms", "missing_legs",
)
_RELATIONSHIP_KEYWORDS = ("separat", "distance", "apart", "crowd", "overlap", "barrier")
_COLOUR_KEYWORDS = ("colour", "color", "shading", "darkness", "bright", "satur", "chromatic")
_SCENE_SEMANTIC_FAMILIES = ("isolation_theme", "task_engagement", "SEMANTIC_THEME_CANDIDATE")
_TASK_SPECIFIC_TASK_TYPES = {
    "house_tree_person": "HTP", "family_drawing_protocol": "FAMILY_DRAWING",
    "any_drawing_including_ppat": "PPAT_OR_FEATS_TASK", "human_figure_drawing": "HUMAN_FIGURE_DRAWING",
}


def _classify_scope(entry: dict[str, Any]) -> tuple[str, str | None]:
    if entry["source_family"] == "framework_source":
        return "FRAMEWORK_ONLY", None
    if entry["requires_longitudinal_data"]:
        return "LONGITUDINAL_REQUIRED", "SCENE_SEMANTIC"
    if entry["requires_process_data"]:
        return "PROCESS_REQUIRED", None
    # G0.1 already precisely classified these via _OBJECTIVE_FEATURE_IMAGE_SCOPE.
    if entry["image_scope"] in ("OBJECT_REGION", "RELATIONSHIP"):
        return entry["image_scope"], "TASK_SPECIFIC" if entry["task_type"] in _TASK_SPECIFIC_TASK_TYPES else None

    cid = (entry["canonical_observable_id"] or "").lower()
    family = entry["evidence_family"] or ""
    secondary = "TASK_SPECIFIC" if entry["task_type"] in _TASK_SPECIFIC_TASK_TYPES else None

    if any(k in cid for k in _HOUSE_KEYWORDS):
        return "HOUSE_REGION", secondary
    if any(k in cid for k in _TREE_KEYWORDS):
        return "TREE_REGION", secondary
    if any(k in cid for k in _FACE_KEYWORDS):
        return "FACE_REGION", "HUMAN_FIGURE"
    if any(k in cid for k in _HUMAN_FIGURE_KEYWORDS):
        return "HUMAN_FIGURE", secondary
    if any(k in cid for k in _RELATIONSHIP_KEYWORDS):
        return "RELATIONSHIP", secondary
    if family in _SCENE_SEMANTIC_FAMILIES:
        return "SCENE_SEMANTIC", secondary
    if any(k in cid for k in _COLOUR_KEYWORDS) or family == "colour_mood_flags":
        return "COLOUR_REGION", "WHOLE_DRAWING"
    if family == "object_symbolism":
        return "OBJECT_REGION", secondary
    return "WHOLE_DRAWING", secondary


def _classify_task_requirement(entry: dict[str, Any]) -> str:
    if entry["source_family"] == "framework_source":
        return "ANY_DRAWING"
    if entry["requires_longitudinal_data"]:
        return "LONGITUDINAL_SERIES"
    if entry["requires_process_data"]:
        return "PROCESS_SESSION"
    task = entry["task_type"] or ""
    if task in _TASK_SPECIFIC_TASK_TYPES:
        return _TASK_SPECIFIC_TASK_TYPES[task]
    if entry["source_family"] == "koppitz_hfd_indicator":
        return "HUMAN_FIGURE_DRAWING"
    if entry["free_vs_instructed"] == "free":
        return "FREE_DRAWING"
    return "ANY_DRAWING"


def _classify_technical_readiness(entry: dict[str, Any]) -> str:
    if entry["requires_longitudinal_data"]:
        return "LONGITUDINAL_DATA_REQUIRED"
    if entry["requires_process_data"]:
        return "PROCESS_DATA_REQUIRED"
    if entry["source_family"] == "framework_source":
        return "CONTEXT_REQUIRED"
    if entry["maturity_status"] == "EXPERIMENT_CANDIDATE":
        return "NEEDS_SEGMENTATION_MASK"
    if entry["existing_feature_id_if_any"]:
        # A real, implemented deterministic CV measurement already exists
        # for this canonical observable (RULE_EVIDENCE_MATRIX.csv's static-
        # direct rules, or one of the 22 formal_features.py measurements).
        return "READY_CURRENT_DETERMINISTIC"
    if entry["source_family"] == "existing_doar_rule":
        # Every existing rule already has SOME detector story today
        # (static_image_detectable proxy features, or an entity/relation
        # semantic match) even where not yet individually enabled.
        return "READY_EXISTING_SEMANTIC_PROVIDER" if entry["static_image_detectable"] else "NEEDS_PART_DETECTOR"
    if entry["maturity_status"] == "ETHICS_REVIEW_LIKELY_REJECT":
        return "NEEDS_EXPERT_SEMANTIC_REVIEW"
    primary_scope = entry.get("primary_scope")
    if primary_scope in ("HOUSE_REGION", "TREE_REGION", "HUMAN_FIGURE", "FACE_REGION"):
        # Needs a validated object/part detector to even locate the region
        # before any measurement inside it is possible.
        return "NEEDS_PART_DETECTOR" if primary_scope in ("FACE_REGION",) else "NEEDS_OBJECT_DETECTOR_VALIDATION"
    if primary_scope == "SCENE_SEMANTIC":
        return "NEEDS_EXPERT_SEMANTIC_REVIEW"
    if not entry["static_image_detectable"]:
        return "NOT_CURRENTLY_MEASURABLE"
    return "NEEDS_EXPERT_SEMANTIC_REVIEW"


def _enrich_scope_task_and_readiness(entry: dict[str, Any]) -> dict[str, Any]:
    primary_scope, secondary_scope = _classify_scope(entry)
    entry["primary_scope"] = primary_scope
    entry["secondary_scope"] = secondary_scope
    entry["task_requirement"] = _classify_task_requirement(entry)
    entry["technical_readiness"] = _classify_technical_readiness(entry)
    return entry


# ---------------------------------------------------------------------------
# Section A -- the 41 EXISTING DOAR rules, read verbatim from the frozen
# RULE_EVIDENCE_MATRIX.csv (never re-typed from the PDF's own paraphrase of
# them) -- the only section where activation_allowed reflects a REAL
# existing production status, never a new grant.
# ---------------------------------------------------------------------------

def existing_41_rules() -> list[dict[str, Any]]:
    matrix_path = ROOT / "RULE_EVIDENCE_MATRIX.csv"
    with open(matrix_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    entries = []
    for row in rows:
        rule_id = row["rule_id"]
        enabled = row["allowed_output_level"] == "individual_heuristic_only"
        entries.append(_base_entry(
            id=rule_id, canonical_observable=row["observable_feature"],
            display_name=row["observable_feature"].replace("_", " "),
            source_family="existing_doar_rule",
            source_reference="DOAR rules_registry_v2.json / RULE_EVIDENCE_MATRIX.csv",
            source_document="child_drawing_rules_compiled.pdf / التحليل النفسي للصور.pdf",
            source_page_or_section_if_known=row["source_pdf_page_section"] or None,
            source_interpretation=row["possible_interpretation_as_written"],
            safe_doar_interpretation=row["possible_interpretation_as_written"],
            evidence_strength=row["evidence_strength_as_written"] or None,
            scientific_support_status="not_scientifically_validated",
            task_type="free_drawing", free_vs_instructed="free",
            # Phase G0.1 audit fix: RULE_EVIDENCE_MATRIX.csv actually stores
            # these as lowercase "yes"/"no" (never "True"/"False") --
            # the original Phase G0 comparison against "True" silently
            # never matched anything, so every existing rule's
            # static_image_detectable/requires_process_data/requires_
            # longitudinal_data was always False in the registry
            # regardless of the real, frozen matrix content. Fixed here by
            # reading the actual real values (still read-only -- the CSV
            # itself is untouched).
            static_image_detectable=row["static_image_detectable"] != "no",
            requires_process_data=row["requires_process_data"] == "yes",
            requires_longitudinal_data=row["requires_longitudinal_data"] == "yes",
            image_scope=("LONGITUDINAL_REQUIRED" if row["requires_longitudinal_data"] == "yes"
                         else "PROCESS_REQUIRED" if row["requires_process_data"] == "yes"
                         else "WHOLE_DRAWING"),
            existing_feature_id_if_any=rule_id,
            evidence_family=row["evidence_family"],
            dedup_group=row["evidence_family"],
            alternative_explanations=(row["alternative_explanations"].split(" | ")
                                       if row["alternative_explanations"] else []),
            maturity_status="ENABLED_HEURISTIC" if enabled else "DISABLED_UNVALIDATED",
            expert_review_status="already_in_production_registry",
            # The ONLY entries in this file allowed to be True -- and only
            # because they already are, in the real frozen registry.
            activation_allowed=enabled,
        ))
    return entries


# ---------------------------------------------------------------------------
# Section B -- FEATS 14 formal-element scales (Appendix B). Gantt & Tabone /
# Gantt & Bucciarelli are NOT yet legally-obtained/read manuals -- only the
# publicly documented scale NAMES/definitions are recorded, per the PDF's
# own explicit caution.
# ---------------------------------------------------------------------------

_FEATS_REF = "Gantt & Bucciarelli (2025), Wiley Handbook of Art Therapy ch.70; Gantt & Tabone Rating Manual (not yet obtained)"

_FEATS_ROWS = [
    ("Prominence of Color", "Amount/prominence of colour", "Formal quantity of colour; not a diagnosis",
     "colour_mood_flags", "OBJECTIVE_ONLY"),
    ("Color Fit", "How colours fit depicted objects/task", "Contextual appropriateness, not mood by itself",
     "SEMANTIC_THEME_CANDIDATE", "CLINICIAN_CANDIDATE"),
    ("Implied Energy", "Perceived energy/activity in the drawing", "Formal energy/activity dimension",
     "STROKE_DENSITY_TEXTURE_FORMAL", "LITERATURE_CANDIDATE"),
    ("Space", "Use of picture plane", "Extent/distribution of page use", "size_composition", "OBJECTIVE_ONLY"),
    ("Integration", "Overall cohesiveness/organization", "Integrated vs fragmented/chaotic composition",
     "SCENE_ORGANIZATION_FORMAL", "LITERATURE_CANDIDATE"),
    ("Logic", "Whether components fit the task/scene", "Scene/task coherence",
     "SEMANTIC_THEME_CANDIDATE", "CLINICIAN_CANDIDATE"),
    ("Realism", "Recognizability/plausibility of forms", "Degree of representational realism",
     "SEMANTIC_THEME_CANDIDATE", "LITERATURE_CANDIDATE"),
    ("Problem Solving", "How the task is solved", "Task-specific coping/solution quality, not general trait",
     "SEMANTIC_THEME_CANDIDATE", "EXPERIMENT_CANDIDATE"),
    ("Developmental Level", "Formal drawing sophistication", "Age/development-related sophistication; must be age-normalized",
     "SCENE_ORGANIZATION_FORMAL", "EXPERIMENT_CANDIDATE"),
    ("Details of Objects/Environment", "Amount of detail", "Detail richness, not pathology alone",
     "detail_fixation", "OBJECTIVE_ONLY"),
    ("Line Quality", "Control/continuity/steadiness of line", "Formal line-control dimension",
     "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Person", "Quality/completeness of person in PPAT", "Task-specific human-figure representation",
     "missing_body_part", "EXPERIMENT_CANDIDATE"),
    ("Rotation", "Tilt relative to expected vertical axis", "Formal rotation/slant",
     "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Perseveration", "Repeated marks/elements beyond task need", "Repetitive graphic response",
     "SYMMETRY_REGULARITY_FORMAL", "LITERATURE_CANDIDATE"),
]


def feats_scales() -> list[dict[str, Any]]:
    entries = []
    for i, (name, observed, interp, dedup, maturity) in enumerate(_FEATS_ROWS, start=1):
        entries.append(_base_entry(
            id=f"FEATS_{i:02d}_{name.upper().replace(' ', '_').replace('/', '_')}",
            canonical_observable=name, display_name=name, source_family="feats_formal_element_scale",
            source_reference=_FEATS_REF, source_interpretation=interp,
            task_type="any_drawing_including_ppat", free_vs_instructed="either",
            static_image_detectable=maturity in ("OBJECTIVE_ONLY", "LITERATURE_CANDIDATE"),
            evidence_family=dedup, dedup_group=dedup, maturity_status=maturity,
        ))
    return entries


# ---------------------------------------------------------------------------
# Section C -- Koppitz 30 historical HFD indicators (Appendix C).
# ---------------------------------------------------------------------------

_KOPPITZ_REF = "Koppitz (1966) J. Clin. Psychol. 22(3); Catte & Cox (1999) Eur. Child Adolesc. Psychiatry 8(2)"

_KOPPITZ_ROWS = [
    ("Poor integration of parts", "Historically grouped with emotional disturbance/impulsivity; also motor/developmental", "SCENE_ORGANIZATION_FORMAL"),
    ("Shading of face", "Historical anxiety/self-concept interpretation; nonspecific", "colour_mood_flags"),
    ("Shading of body/limbs", "Historical anxiety/body concern interpretation; nonspecific", "colour_mood_flags"),
    ("Shading of hands/neck", "Historical tension/anxiety interpretation; nonspecific", "colour_mood_flags"),
    ("Gross asymmetry of limbs", "Historical instability/impulsivity; developmental/motor confound", "SYMMETRY_REGULARITY_FORMAL"),
    ("Slanting figure", "Historical insecurity/instability candidate", "LINE_GEOMETRY_FORMAL"),
    ("Tiny figure", "Historical insecurity/withdrawal candidate", "size_composition"),
    ("Big figure", "Historical expansiveness/impulsivity candidate", "size_composition"),
    ("Transparencies", "Historical impulsivity/immaturity/disturbance candidate", "SEMANTIC_THEME_CANDIDATE"),
    ("Tiny head", "Historical insecurity/body-image candidate", "size_composition"),
    ("Crossed eyes", "Historical emotional indicator; could also be style", "facial_feature_style"),
    ("Teeth", "Historical aggression/tension candidate", "facial_feature_style"),
    ("Short arms", "Historical inadequacy/withdrawal candidate", "size_composition"),
    ("Long arms", "Historical reach/control/aggression interpretations", "size_composition"),
    ("Arms clinging to body", "Historical inhibition/rigidity candidate", "SYMMETRY_REGULARITY_FORMAL"),
    ("Big hands", "Historical acting-out/aggression/insecurity candidate", "size_composition"),
    ("Hands cut off / absent", "Historical insecurity/acting difficulty; developmental confound", "missing_body_part"),
    ("Legs pressed together", "Historical anxiety/insecurity candidate", "SYMMETRY_REGULARITY_FORMAL"),
    ("Genitals", "Historical concern indicator; must be handled with extreme contextual caution", "SEMANTIC_THEME_CANDIDATE"),
    ("Monster/grotesque figure", "Historical insecurity/distress candidate; fantasy/media confound", "SEMANTIC_THEME_CANDIDATE"),
    ("Three or more figures spontaneously", "Historical indicator only in specific instructed context", "SCENE_ORGANIZATION_FORMAL"),
    ("Clouds/rain/snow/flying birds", "Historical anxiety/environmental tension candidate; weak specificity", "colour_mood_flags"),
    ("No eyes", "Historical withdrawal/anxiety candidate; developmental confound", "missing_body_part"),
    ("No nose", "Historical indicator; developmental confound", "missing_body_part"),
    ("No mouth", "Historical communication/affect interpretation; developmental confound", "missing_body_part"),
    ("No body", "Historical severe immaturity/disturbance indicator; age critical", "missing_body_part"),
    ("No arms", "Historical insecurity/inefficacy candidate; development confound", "missing_body_part"),
    ("No legs", "Historical insecurity/instability candidate; development confound", "missing_body_part"),
    ("No feet", "Historical insecurity candidate; age norms matter", "missing_body_part"),
    ("No neck", "Historical impulsivity/immaturity candidate; age norms changed in later work", "missing_body_part"),
]


def koppitz_indicators() -> list[dict[str, Any]]:
    entries = []
    for i, (name, interp, dedup) in enumerate(_KOPPITZ_ROWS, start=1):
        entries.append(_base_entry(
            id=f"KOPPITZ_{i:02d}_{name.upper().replace(' ', '_').replace('/', '_')}",
            canonical_observable=name, display_name=name, source_family="koppitz_hfd_indicator",
            source_reference=_KOPPITZ_REF, source_interpretation=interp,
            task_type="human_figure_drawing", free_vs_instructed="either",
            age_or_development_limits="Strong developmental confounding reported by modern studies -- age norms essential",
            static_image_detectable=True, evidence_family=dedup, dedup_group=dedup,
            maturity_status="LITERATURE_CANDIDATE" if name != "Genitals" else "CLINICIAN_CANDIDATE",
        ))
    return entries


# ---------------------------------------------------------------------------
# Section D -- 2023 HTP meta-analysis, all 50 recurrent characteristics
# (Appendix D) -- 39 significant, 11 explicitly non-significant, ALL kept
# (never silently dropped, per instruction not to let unsupported folklore
# quietly re-enter via omission of the negative findings).
# ---------------------------------------------------------------------------

_HTP_REF = "Guo et al. (2023) Front. Psychiatry 13:1041770 (30 studies, n=6295)"

# (level, characteristic, result_or_None_if_not_significant, dedup_group)
_HTP_ROWS = [
    ("Whole", "Emphasis on straight lines", "OR≈11.75", "LINE_GEOMETRY_FORMAL"),
    ("Whole", "Simplified drawing", "OR≈9.64", "SCENE_ORGANIZATION_FORMAL"),
    ("Whole", "No theme", "OR≈9.36", "SEMANTIC_THEME_CANDIDATE"),
    ("Whole", "Small drawing size", "OR≈5.71", "size_composition"),
    ("Whole", "Excessive separation among items", "OR≈3.84", "SCENE_ORGANIZATION_FORMAL"),
    ("Whole", "Weak or intermittent lines", "OR≈3.19", "LINE_GEOMETRY_FORMAL"),
    ("Whole", "No motion", "OR≈2.96", "SEMANTIC_THEME_CANDIDATE"),
    ("Whole", "Shadow", "OR≈2.88", "colour_mood_flags"),
    ("Whole", "Omitted house/tree/person", "OR≈2.81", "missing_body_part"),
    ("Whole", "Shaded/blackened drawing", "OR≈2.72", "colour_mood_flags"),
    ("Whole", "No additional decoration", "OR≈2.59", "detail_fixation"),
    ("Whole", "Scribbled drawing", "OR≈2.56", "STROKE_DENSITY_TEXTURE_FORMAL"),
    ("Whole", "Emphasized horizon", None, "SCENE_ORGANIZATION_FORMAL"),
    ("Whole", "Weighted/repeated lines", None, "LINE_GEOMETRY_FORMAL"),
    ("Whole", "Fence", None, "object_symbolism"),
    ("House", "Bizarre house", "OR≈4.64", "object_symbolism"),
    ("House", "No door", "OR≈4.52", "missing_body_part"),
    ("House", "Very small house", "OR≈4.24", "size_composition"),
    ("House", "No window", "high reported OR", "missing_body_part"),
    ("House", "Leaning house", "OR≈2.68", "LINE_GEOMETRY_FORMAL"),
    ("House", "Decorated roof", "OR≈2.32", "detail_fixation"),
    ("House", "Smoking chimney", "OR≈2.27", "object_symbolism"),
    ("House", "Two-dimensional house", "OR≈1.76", "SCENE_ORGANIZATION_FORMAL"),
    ("House", "Shaded/blackened wall", "OR≈1.66", "colour_mood_flags"),
    ("House", "Chimney present", None, "object_symbolism"),
    ("House", "Closed door", None, "object_symbolism"),
    ("House", "Closed window", None, "object_symbolism"),
    ("Tree", "Roots", "OR≈4.35", "object_symbolism"),
    ("Tree", "Truncated tree", "OR≈2.90", "object_symbolism"),
    ("Tree", "Flattened crown", "OR≈2.82", "object_symbolism"),
    ("Tree", "Bizarre tree", "OR≈2.78", "object_symbolism"),
    ("Tree", "Dead tree", "OR≈2.67", "object_symbolism"),
    ("Tree", "Very small tree", "OR≈2.65", "size_composition"),
    ("Tree", "Sharp branch", "OR≈2.35", "LINE_GEOMETRY_FORMAL"),
    ("Tree", "Tree scars", None, "object_symbolism"),
    ("Tree", "Shaded/blackened tree", "not significant in 2023; 2026 update reports a significant pooled signal (evolving)", "colour_mood_flags"),
    ("Person", "Incomplete person", "OR≈4.90", "missing_body_part"),
    ("Person", "Shaded/blackened person", "OR≈4.63", "colour_mood_flags"),
    ("Person", "Fist", "OR≈3.66", "SEMANTIC_THEME_CANDIDATE"),
    ("Person", "Negative expression", "OR≈3.59", "facial_feature_style"),
    ("Person", "Bizarre person", "OR≈3.18", "SEMANTIC_THEME_CANDIDATE"),
    ("Person", "Very small person", "OR≈3.02", "size_composition"),
    ("Person", "Loss of facial features", "OR≈2.71", "facial_feature_style"),
    ("Person", "Poker face", "OR≈2.09", "facial_feature_style"),
    ("Person", "Inappropriate body proportions", "OR≈1.99", "size_composition"),
    ("Person", "Single-line limbs", "OR≈1.93", "LINE_GEOMETRY_FORMAL"),
    ("Person", "Complete/partial loss of limbs", "OR≈1.82", "missing_body_part"),
    ("Person", "Simple person", None, "detail_fixation"),
    ("Person", "Fingers", None, "detail_fixation"),
    ("Person", "Not-frontal portrait", None, "LINE_GEOMETRY_FORMAL"),
]


def htp_characteristics() -> list[dict[str, Any]]:
    entries = []
    for i, (level, name, result, dedup) in enumerate(_HTP_ROWS, start=1):
        significant = result is not None
        entries.append(_base_entry(
            id=f"HTP_{i:02d}_{level.upper()}_{name.upper().replace(' ', '_').replace('/', '_')}",
            canonical_observable=name, display_name=f"{level}: {name}", source_family="htp_2023_meta_analysis",
            source_reference=_HTP_REF, source_interpretation=(
                f"Association signal in HTP studies (not a standalone diagnosis): {result}" if significant
                else "Not significant overall in this meta-analysis -- must not be used as a positive rule from it"),
            significant_or_not_if_meta_analysis="significant" if significant else "not_significant",
            effect_size_if_available=result if significant else None,
            task_type="house_tree_person", free_vs_instructed="instructed",
            static_image_detectable=True, evidence_family=dedup, dedup_group=dedup,
            maturity_status="LITERATURE_CANDIDATE",
        ))
    return entries


# ---------------------------------------------------------------------------
# Section E -- 2026 tree-imagery meta-analysis evidence update (Appendix E).
# ---------------------------------------------------------------------------

_TREE_2026_REF = "Guo et al. (2026) Depression and Anxiety 2026:9571222"

_TREE_2026_ROWS = [
    ("Blackened-out", "Blackened tree (example OR≈2.01)", "General pooled mental-disorder association; not diagnosis"),
    ("Scribbled lines", "Weak lines (example OR≈2.82)", "Low-vitality/line-quality signal; nonspecific"),
    ("Oddly shaped", "Flattened crown (example OR≈3.10)", "Shape abnormality signal"),
    ("No vitality", "Very small tree (example OR≈3.93)", "No-vitality/smallness category"),
    ("Overly simple", "Simplified drawing (example OR≈7.07)", "Low-complexity/simplification category"),
    ("Affective subgroup", "Blackened tree / no motion / excessive separation",
     "Associated with affective-disorder subgroup in this meta-analysis; still adjunctive evidence only"),
    ("Thought-disorder subgroup", "Roots", "Reported unique subgroup association; must not be used as a standalone disorder claim"),
]


def tree_imagery_2026() -> list[dict[str, Any]]:
    entries = []
    for i, (category, feature, interp) in enumerate(_TREE_2026_ROWS, start=1):
        entries.append(_base_entry(
            id=f"TREE2026_{i:02d}_{category.upper().replace(' ', '_').replace('-', '_')}",
            canonical_observable=feature, display_name=f"{category}: {feature}", source_family="tree_imagery_2026_update",
            source_reference=_TREE_2026_REF, source_interpretation=interp,
            task_type="tree_drawing_or_htp", free_vs_instructed="either",
            static_image_detectable=True, evidence_family="object_symbolism", dedup_group="object_symbolism",
            maturity_status="LITERATURE_CANDIDATE",
        ))
    return entries


# ---------------------------------------------------------------------------
# Section F -- Family Drawing attachment-based candidates: 24 markers + 8
# global scales (Appendix F). Task-specific: a family-drawing protocol, not
# arbitrary free drawings. Two historical markers (#19, #20) encode
# gender-norm assumptions -- kept as historical source entries, flagged
# ETHICS_REVIEW, never operationalized.
# ---------------------------------------------------------------------------

_FAMILY_2022_REF = "Pace, Muzi & Vizzino (2022) Front. Psychol. 13:980129; Pace et al. (2022) Attach. Hum. Dev. 24(4)"

_FAMILY_MARKER_ROWS = [
    ("Lack of color", "Avoidant-marker system", "Possible emotional distance/avoidant attachment representation", "colour_mood_flags", False),
    ("Child far apart from mother", "Avoidant-marker system", "Relational distance in family representation", "isolation_theme", False),
    ("Omission of mother or child", "Avoidant-marker system", "Relational omission in family representation", "missing_body_part", False),
    ("Lack of individuation of family members", "Avoidant-marker system", "Low differentiation among represented family members", "facial_feature_style", False),
    ("Arms downward/close to body", "Avoidant-marker system", "Inhibited/closed pose in coding system", "SYMMETRY_REGULARITY_FORMAL", False),
    ("Exaggeration of heads", "Avoidant-marker system", "Head-size anomaly in coding system", "size_composition", False),
    ("Disguised family members", "Avoidant-marker system", "Family figures represented indirectly/disguised", "SEMANTIC_THEME_CANDIDATE", False),
    ("Figures separated by barriers", "Ambivalent-marker system", "Barrier/separation in family representation", "SCENE_ORGANIZATION_FORMAL", False),
    ("Figures crowded or overlapping", "Ambivalent-marker system", "Crowding/overlap relational pattern", "SCENE_ORGANIZATION_FORMAL", False),
    ("Unusually small figures", "Ambivalent-marker system", "Vulnerability/smallness pattern in coding system", "size_composition", False),
    ("Unusually large figures", "Ambivalent-marker system", "Size/vulnerability pattern in coding system", "size_composition", False),
    ("Exaggeration of body part", "Ambivalent-marker system", "Body-part emphasis/vulnerability pattern", "size_composition", False),
    ("Exaggeration of hands/arms", "Ambivalent-marker system", "Limb emphasis in coding system", "size_composition", False),
    ("Exaggeration of facial features", "Ambivalent-marker system", "Facial-feature emphasis", "facial_feature_style", False),
    ("Figures at corner of page", "Ambivalent-marker system", "Peripheral placement/vulnerability pattern", "spatial_placement", False),
    ("Lack of background detail", "General insecurity marker", "Sparse environmental context", "detail_fixation", False),
    ("Figures not grounded / floating", "General insecurity marker", "Lack of baseline/grounding in scene", "SCENE_ORGANIZATION_FORMAL", False),
    ("Incomplete figures", "General insecurity marker", "Incomplete human representation", "missing_body_part", False),
    ("Mother not feminized", "Historical coding marker", "Gender-norm dependent historical marker", "SEMANTIC_THEME_CANDIDATE", True),
    ("Males/females undifferentiated by gender", "Historical coding marker", "Gender-norm dependent historical marker", "SEMANTIC_THEME_CANDIDATE", True),
    ("Neutral/negative facial affect", "General insecurity marker", "Depicted affect within family scene", "facial_feature_style", False),
    ("False starts", "Disorganized-marker system", "Process disorganization in original coding system", "task_engagement", False),
    ("Scrunched figures", "Disorganized-marker system", "Visually compressed/distorted figures", "size_composition", False),
    ("Unusual signs/symbols/scenes", "Disorganized-marker system", "Bizarreness/unusual-scene pattern", "SEMANTIC_THEME_CANDIDATE", False),
]

_FAMILY_GLOBAL_SCALE_ROWS = [
    ("Vitality / Creativity", "Emotional investment expressed through creativity, detail, embellishment", "STROKE_DENSITY_TEXTURE_FORMAL"),
    ("Family Pride / Happiness", "Sense of belonging / positive family representation", "SEMANTIC_THEME_CANDIDATE"),
    ("Emotional Distance / Isolation", "Loneliness/distance reflected in affect and mother-child distance", "isolation_theme"),
    ("Tension / Anger", "Tension/anger in family representation; cited examples include no colour/detail or scribbling/cross-outs", "colour_mood_flags"),
    ("Vulnerability", "Placement/size/exaggeration pattern", "size_composition"),
    ("Role Reversal", "Unusual size/role relationships", "SEMANTIC_THEME_CANDIDATE"),
    ("Bizarreness / Dissociation", "Unusual symbols/signs/fantasy themes", "SEMANTIC_THEME_CANDIDATE"),
    ("Global Pathology (renamed 'Global family-drawing organization')",
     "Overall organization using completeness, colour, detail, affect, background", "SCENE_ORGANIZATION_FORMAL"),
]


def family_drawing_candidates() -> list[dict[str, Any]]:
    entries = []
    for i, (name, grouping, interp, dedup, ethics_review) in enumerate(_FAMILY_MARKER_ROWS, start=1):
        entries.append(_base_entry(
            id=f"FAMILY_MARKER_{i:02d}_{name.upper().replace(' ', '_').replace('/', '_')}",
            canonical_observable=name, display_name=name, source_family="family_drawing_2022_marker",
            source_reference=f"{_FAMILY_2022_REF} ({grouping})", source_interpretation=interp,
            task_type="family_drawing_protocol", free_vs_instructed="instructed",
            static_image_detectable=True, evidence_family=dedup, dedup_group=dedup,
            maturity_status="ETHICS_REVIEW_LIKELY_REJECT" if ethics_review else "LITERATURE_CANDIDATE",
        ))
    for i, (name, interp, dedup) in enumerate(_FAMILY_GLOBAL_SCALE_ROWS, start=1):
        entries.append(_base_entry(
            id=f"FAMILY_GLOBAL_SCALE_{i:02d}_{name.upper().replace(' ', '_').replace('/', '_').split('(')[0].strip()}",
            canonical_observable=name, display_name=name, source_family="family_drawing_2022_global_scale",
            source_reference=_FAMILY_2022_REF, source_interpretation=interp,
            task_type="family_drawing_protocol", free_vs_instructed="instructed",
            static_image_detectable=True, evidence_family=dedup, dedup_group=dedup,
            maturity_status="LITERATURE_CANDIDATE",
        ))
    return entries


# ---------------------------------------------------------------------------
# Section G -- new objective graphic/formal features proposed after the
# psychologist meeting (Appendix G). `maturity_status` here is taken
# directly from the PDF's own "Maturity" column -- OBJECTIVE_ONLY features
# are exactly the ones implemented this phase (see formal_features.py);
# CLINICIAN_CANDIDATE ones remain descriptive/unactivated.
# ---------------------------------------------------------------------------

# (feature, interpretation_allowed_now, dedup_group, maturity)
_OBJECTIVE_FEATURE_ROWS = [
    ("Mean line width", "Formal line thickness only", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Line-width variability", "Consistency vs irregularity of mark width", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Line darkness", "Faint vs dark appearance; not true pressure", "colour_mood_flags", "OBJECTIVE_ONLY"),
    ("Darkness variability", "Consistency of mark intensity", "colour_mood_flags", "OBJECTIVE_ONLY"),
    ("Line continuity", "Formal continuity", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Fragmentation rate", "Formal fragmentation", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Line jitter/shakiness", "Possible tension/motor difficulty only after validation", "LINE_GEOMETRY_FORMAL", "LITERATURE_CANDIDATE"),
    ("Straightness", "Formal geometry", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Curvature", "Formal geometry", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Angularity", "Formal angularity", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Dominant orientation", "Formal directional tendency", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Orientation entropy", "High = many competing directions", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Local orientation coherence", "Low = locally multidirectional/irregular", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Direction-cluster count", "Formal multidirectionality", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Crossing density", "Formal complexity/crossing", "STROKE_DENSITY_TEXTURE_FORMAL", "OBJECTIVE_ONLY"),
    ("Scribble density", "Scribble/disorganization observation; psych meaning later", "STROKE_DENSITY_TEXTURE_FORMAL", "LITERATURE_CANDIDATE"),
    ("Hatching density", "Formal shading technique", "STROKE_DENSITY_TEXTURE_FORMAL", "OBJECTIVE_ONLY"),
    ("Cross-hatching", "Formal shading technique", "STROKE_DENSITY_TEXTURE_FORMAL", "OBJECTIVE_ONLY"),
    ("Retracing/overdraw appearance", "Possible control/repetition pattern; cannot equal true drawing-process count",
     "STROKE_DENSITY_TEXTURE_FORMAL", "OBJECTIVE_PROXY"),
    ("Sharp-turn density", "Formal irregularity/angularity", "LINE_GEOMETRY_FORMAL", "OBJECTIVE_ONLY"),
    ("Stroke-density map", "Crowded vs sparse drawing areas", "STROKE_DENSITY_TEXTURE_FORMAL", "OBJECTIVE_ONLY"),
    ("Colour coverage", "Formal colour prominence", "colour_mood_flags", "OBJECTIVE_ONLY"),
    ("Colour diversity", "Palette diversity", "colour_mood_flags", "OBJECTIVE_ONLY"),
    ("Brightness/darkness palette", "Formal palette characteristic", "colour_mood_flags", "OBJECTIVE_ONLY"),
    ("Saturation", "Formal palette characteristic", "colour_mood_flags", "OBJECTIVE_ONLY"),
    ("Blackened-area ratio", "Formal darkening/shading", "colour_mood_flags", "LITERATURE_CANDIDATE"),
    ("Shading density", "Formal shading", "colour_mood_flags", "LITERATURE_CANDIDATE"),
    ("Fill completeness", "Formal task execution", "FILL_QUALITY_FORMAL", "OBJECTIVE_ONLY"),
    ("Boundary overflow", "Formal boundary-control clue", "FILL_QUALITY_FORMAL", "OBJECTIVE_ONLY"),
    ("Fill uniformity", "Formal organization", "FILL_QUALITY_FORMAL", "OBJECTIVE_ONLY"),
    ("Region colouring-direction coherence", "Doctor's 'neat vs messy directional colouring' observation",
     "FILL_QUALITY_FORMAL", "CLINICIAN_CANDIDATE"),
    ("Region colouring-orientation entropy", "High = multidirectional filling", "FILL_QUALITY_FORMAL", "CLINICIAN_CANDIDATE"),
    ("Bilateral symmetry", "Formal regularity/control observation", "SYMMETRY_REGULARITY_FORMAL", "OBJECTIVE_ONLY"),
    ("Local symmetry", "Formal regularity", "SYMMETRY_REGULARITY_FORMAL", "OBJECTIVE_ONLY"),
    ("Spacing regularity", "Formal regularity", "SYMMETRY_REGULARITY_FORMAL", "OBJECTIVE_ONLY"),
    ("Repetition/perseveration score", "Formal repetitive pattern; clinical meaning context-dependent",
     "SYMMETRY_REGULARITY_FORMAL", "LITERATURE_CANDIDATE"),
    ("Graphic integration score", "Integrated vs fragmented/chaotic formal pattern", "SCENE_ORGANIZATION_FORMAL", "LITERATURE_CANDIDATE"),
    ("Detail density", "Sparse vs detailed formal output", "detail_fixation", "OBJECTIVE_ONLY"),
    ("Background-detail score", "Sparse vs elaborated background", "SCENE_ORGANIZATION_FORMAL", "OBJECTIVE_ONLY"),
    ("Scene crowding", "Crowded vs open composition", "SCENE_ORGANIZATION_FORMAL", "OBJECTIVE_ONLY"),
    ("Element separation", "Relational separation", "spatial_placement", "OBJECTIVE_ONLY"),
    ("Figure/object size ratios", "Relative prominence/smallness", "size_composition", "OBJECTIVE_ONLY"),
    ("Part omissions", "Task-specific incompleteness", "missing_body_part", "LITERATURE_CANDIDATE"),
    ("Body-part proportion anomalies", "Task-specific distortion", "size_composition", "LITERATURE_CANDIDATE"),
    ("Facial affect in drawing", "Depicted emotion only, not child's diagnosis", "facial_feature_style", "LITERATURE_CANDIDATE"),
    ("Implied motion/activity", "Formal/semantic energy", "STROKE_DENSITY_TEXTURE_FORMAL", "LITERATURE_CANDIDATE"),
    ("Semantic conflict/action theme", "Narrative theme only; real-life behavior cannot be inferred directly",
     "SEMANTIC_THEME_CANDIDATE", "CLINICIAN_CANDIDATE"),
    ("Ownership/permission/boundary theme", "Possible boundary/permission theme to explore; never label child 'thief'",
     "SEMANTIC_THEME_CANDIDATE", "CLINICIAN_CANDIDATE"),
    ("Global regularity/control pattern", "Possible over-control/rigidity only after validation; never OCD from drawing alone",
     "SYMMETRY_REGULARITY_FORMAL", "CLINICIAN_CANDIDATE"),
    ("Global graphic disorganization", "Possible dysregulation/tension/environmental-instability relevance only after "
     "expert validation", "SCENE_ORGANIZATION_FORMAL", "CLINICIAN_CANDIDATE"),
]

# Objective features actually IMPLEMENTED this phase in formal_features.py.
# Every "OBJECTIVE_ONLY" row above that also appears here gets
# existing_feature_id_if_any populated so the registry traces straight to
# real code, not just a proposal. Phase G0.1, Step 4: four of these were
# RENAMED to accurately describe what is actually measured (never keep a
# name implying more precision/validation than the implementation has):
# "Crossing density" -> junction_corner_density_proxy (corner/junction
# RESPONSE, not true skeleton crossings); "Colour coverage" ->
# chromatic_coverage (genuine chromatic measurement, not foreground_
# coverage reuse); "Fill uniformity" -> global_spatial_density_uniformity
# (whole-drawing grid occupancy, not per-object region fill); a "Scribble
# density" mapping is deliberately NOT added here -- it stays an
# unimplemented-by-this-id EXPLORATORY composite
# (stroke.scribble_candidate_score), not claimed as a validated detector.
IMPLEMENTED_FORMAL_FEATURE_IDS = {
    "Mean line width": "line.mean_width", "Line-width variability": "line.width_variability",
    "Line darkness": "line.darkness_mean", "Darkness variability": "line.darkness_variability",
    "Line continuity": "line.continuity", "Fragmentation rate": "line.fragmentation_rate",
    "Straightness": "line.straightness", "Dominant orientation": "line.dominant_orientation_code",
    "Orientation entropy": "line.orientation_entropy", "Local orientation coherence": "line.orientation_coherence",
    "Direction-cluster count": "line.direction_cluster_count",
    "Crossing density": "stroke.junction_corner_density_proxy",
    "Stroke-density map": "stroke.density_mean", "Colour coverage": "colour.chromatic_coverage",
    "Colour diversity": "colour.diversity", "Brightness/darkness palette": "colour.brightness",
    "Saturation": "colour.saturation", "Fill uniformity": "fill.global_spatial_density_uniformity",
    "Bilateral symmetry": "symmetry.bilateral", "Spacing regularity": "symmetry.spacing_regularity",
    "Detail density": "scene.detail_density",
}

# Phase G0.1, Step 5 -- explicit image-scope classification, one of:
# WHOLE_DRAWING / OBJECT_REGION / RELATIONSHIP / PROCESS_REQUIRED /
# LONGITUDINAL_REQUIRED. DOAR currently has only bbox entities (no
# per-object pixel masks), so every OBJECT_REGION row here is also
# EXPERIMENT_CANDIDATE maturity (see needs_region_masks below) --
# never approximated by running a whole-mask measurement over a bounding
# box and calling it region-specific.
_OBJECTIVE_FEATURE_IMAGE_SCOPE: dict[str, str] = {
    "Retracing/overdraw appearance": "PROCESS_REQUIRED",
    "Fill completeness": "OBJECT_REGION", "Boundary overflow": "OBJECT_REGION",
    "Region colouring-direction coherence": "OBJECT_REGION", "Region colouring-orientation entropy": "OBJECT_REGION",
    "Local symmetry": "OBJECT_REGION",
    "Element separation": "RELATIONSHIP", "Figure/object size ratios": "RELATIONSHIP",
    "Semantic conflict/action theme": "RELATIONSHIP", "Ownership/permission/boundary theme": "RELATIONSHIP",
}


def new_objective_features() -> list[dict[str, Any]]:
    entries = []
    for i, (name, interp, dedup, maturity) in enumerate(_OBJECTIVE_FEATURE_ROWS, start=1):
        implemented_id = IMPLEMENTED_FORMAL_FEATURE_IDS.get(name)
        needs_region_masks = "Region " in name or name in ("Local symmetry", "Fill completeness", "Boundary overflow")
        entries.append(_base_entry(
            id=f"OBJFEAT_{i:02d}_{name.upper().replace(' ', '_').replace('/', '_').replace('-', '_')}",
            canonical_observable=name, display_name=name, source_family="new_objective_graphic_feature",
            image_scope=_OBJECTIVE_FEATURE_IMAGE_SCOPE.get(name, "WHOLE_DRAWING"),
            source_reference="Psychologist-meeting proposal, DOAR_Master_Rules_and_Features_Inventory_V3.pdf Appendix G",
            source_interpretation=interp,
            task_type="any_drawing", free_vs_instructed="either",
            static_image_detectable=maturity in ("OBJECTIVE_ONLY", "OBJECTIVE_PROXY"),
            requires_process_data=maturity == "OBJECTIVE_PROXY",
            existing_feature_id_if_any=implemented_id,
            candidate_detector=("OpenCV/numpy (formal_features.py)" if implemented_id
                                 else "EXPERIMENT_CANDIDATE -- requires per-object segmentation mask (SAM2-class) "
                                      "not yet in the DOAR pipeline" if needs_region_masks
                                 else "OpenCV/numpy -- deferred this phase"),
            evidence_family=dedup, dedup_group=dedup,
            maturity_status="EXPERIMENT_CANDIDATE" if needs_region_masks else maturity,
        ))
    return entries


# ---------------------------------------------------------------------------
# Section H -- Malchiodi: a FRAMEWORK_SOURCE, not a rule list (Appendix H).
# ---------------------------------------------------------------------------

def malchiodi_framework() -> list[dict[str, Any]]:
    return [_base_entry(
        id="MALCHIODI_FRAMEWORK_SOURCE",
        canonical_observable="contextual_interpretation_framework", display_name="Malchiodi interpretation framework",
        source_family="framework_source", source_reference="Malchiodi (1998), Understanding Children's Drawings, Guilford Press",
        source_interpretation=(
            "Always interpret in context (age/development, materials, task, environment, clinician relationship, "
            "child's own narrative); use the drawing as a communication springboard, not a standalone diagnostic "
            "test; separate developmental from emotional content; consider interpersonal/family aspects separately; "
            "ask open questions; include ethical safeguards against overinterpretation."),
        safe_doar_interpretation=(
            "Governs question generation, context requirements, alternative explanations, and parent wording -- "
            "not a per-item observable/rule."),
        task_type="any", free_vs_instructed="either", static_image_detectable=False,
        evidence_family=None, dedup_group=None,
        maturity_status="FRAMEWORK_SOURCE",
        parent_safe_wording="(not applicable -- this entry governs HOW DOAR phrases things, not a finding itself.)",
        clinician_wording="(not applicable -- this entry governs HOW DOAR phrases things, not a finding itself.)",
    )]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_master_registry_v3() -> dict[str, Any]:
    existing = existing_41_rules()
    new_sections = (
        feats_scales() + koppitz_indicators() + htp_characteristics() + tree_imagery_2026()
        + family_drawing_candidates() + new_objective_features() + malchiodi_framework()
    )
    new_sections = _finalize(new_sections)
    all_entries = [_enrich_scope_task_and_readiness(_assign_canonical_observable(e))
                   for e in (existing + new_sections)]

    ids = [e["id"] for e in all_entries]
    duplicate_ids = sorted({i for i in ids if ids.count(i) > 1})

    # LEVEL 3 -- evidence families (the `dedup_group`/`evidence_family`
    # field every row-builder already sets to the SAME family name; this is
    # what the earlier report mislabeled "canonical_dedup_group_count").
    evidence_family_groups: dict[str, list[str]] = {}
    for e in all_entries:
        if e["evidence_family"]:
            evidence_family_groups.setdefault(e["evidence_family"], []).append(e["id"])

    # LEVEL 2 -- canonical observables (distinct measurable concepts after
    # verified semantic dedup -- see _CANONICAL_OBSERVABLE_MERGES). Several
    # source entries can share one canonical_observable_id; every canonical
    # observable belongs to exactly one evidence_family.
    canonical_observables: dict[str, dict[str, Any]] = {}
    for e in all_entries:
        cid = e["canonical_observable_id"]
        bucket = canonical_observables.setdefault(cid, {
            "canonical_observable": e["canonical_observable"], "evidence_family": e["evidence_family"],
            "source_entry_ids": [],
        })
        bucket["source_entry_ids"].append(e["id"])

    merged_canonical_observables = {
        cid: info for cid, info in canonical_observables.items() if len(info["source_entry_ids"]) > 1
    }

    return {
        "schema_version": MASTER_REGISTRY_V3_SCHEMA_VERSION,
        "status": (
            "RESEARCH-ONLY registry -- does NOT replace or modify rules_registry_v2.json / "
            "RULE_EVIDENCE_MATRIX.csv / CONCERN_DOMAIN_MAP.json, which remain the sole production "
            "sources. Every entry outside the 41 existing-rule section has activation_allowed=false."
        ),
        "source_specification_document": _SOURCE_PDF_NAME,
        "existing_41_rules_accounted_for": len(existing) == 41,
        "existing_rule_count": len(existing),
        "existing_rule_count_enabled": sum(1 for e in existing if e["activation_allowed"]),
        "new_source_entry_count": len(new_sections),
        "total_entry_count": len(all_entries),
        "duplicate_ids_found": duplicate_ids,
        # --- THREE EXPLICIT, NEVER-CONFLATED HIERARCHY LEVELS -------------
        "source_entry_count": len(all_entries),
        "canonical_observable_count": len(canonical_observables),
        "evidence_family_count": len(evidence_family_groups),
        "canonical_observables_that_merge_multiple_source_entries": len(merged_canonical_observables),
        "canonical_observables": {k: canonical_observables[k] for k in sorted(canonical_observables)},
        "evidence_family_groups": {k: sorted(v) for k, v in sorted(evidence_family_groups.items())},
        "new_source_entry_count_by_family": {
            fam: sum(1 for e in new_sections if e["source_family"] == fam)
            for fam in sorted({e["source_family"] for e in new_sections})
        },
        # Phase G0.2, Step 4-6 distributions.
        "primary_scope_distribution": {
            v: sum(1 for e in all_entries if e["primary_scope"] == v) for v in PRIMARY_SCOPE_VALUES
            if any(e["primary_scope"] == v for e in all_entries)
        },
        "task_requirement_distribution": {
            v: sum(1 for e in all_entries if e["task_requirement"] == v) for v in TASK_REQUIREMENT_VALUES
            if any(e["task_requirement"] == v for e in all_entries)
        },
        "technical_readiness_distribution": {
            v: sum(1 for e in all_entries if e["technical_readiness"] == v) for v in TECHNICAL_READINESS_VALUES
            if any(e["technical_readiness"] == v for e in all_entries)
        },
        "entries": all_entries,
    }


def write_master_registry_v3() -> Path:
    document = build_master_registry_v3()
    MASTER_REGISTRY_V3_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return MASTER_REGISTRY_V3_PATH


if __name__ == "__main__":
    path = write_master_registry_v3()
    print(f"wrote {path}")
