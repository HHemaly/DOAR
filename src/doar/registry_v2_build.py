"""Builds `resources/psychology_sources/rules_registry_v2.json` (DOAR-TRACE
Phase 1.5, Section 4) from the full `source_rule_catalog.json` (both PDFs)
and `construct_registry.json`.

This is a full rewrite of the Phase 1 version of this module, which only
ever iterated over the original 19 rule IDs (`rule_schema.py::RULE_PROVENANCE`)
-- see `docs/RULE_PDF_COVERAGE_AUDIT_V2.md` Section "Phase 1 audit findings"
for the exact before/after. The original 19 production rule IDs
(`PSY_AR_*`) are preserved unchanged; 22 new rule IDs
(`EN_COMPILED_*`) are added for observables that exist only in
`child_drawing_rules_compiled.pdf`. `rules_registry.json` (the production
registry `rules.py` actually dispatches against) is never edited or
replaced by this module.

**Phase 1.5**: no new detector, proxy wiring, or evaluator was built.
Every rule beyond the original 6 tier-1 composition/placement rules was
`allowed_output_level: disabled` -- including rules whose
`observability_class` is `static_direct`/`static_proxy` (i.e. a real
underlying feature technically exists) but which were simply not wired
into any evaluator. `observability_class` answers "could this be
computed in principle from what already exists in this codebase";
`allowed_output_level` answers "is it actually computed and shown today"
-- these are two different questions, both answered honestly here rather
than conflated.

**Phase 2A, Section 7**: exactly 4 more rules were audited, found to
satisfy every one of the 7 required activation conditions, and wired to
a real evaluator (`rule_engine_v2.py`, additive/parallel -- never
replacing `rules.py`/`rules_registry.json`): `EN_COMPILED_PLACEMENT_CENTER_029`,
`EN_COMPILED_LINE_HEAVY_PRESSURE_030`, `EN_COMPILED_LINE_LIGHT_PRESSURE_031`,
`EN_COMPILED_LINE_SHAKY_BROKEN_032`. These now carry
`allowed_output_level: individual_heuristic_only`. All other
`EN_COMPILED_*` rules remain `disabled` -- see
docs/STATIC_PROXY_RULE_POLICY.md for the full per-rule audit, including
why `EXCESSIVE_DETAIL_040`/`NEGLECT_BACKGROUND_041` were deliberately
NOT activated despite superficially similarly-named features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .source_catalog_build import build_source_catalog

REGISTRY_V2_PATH = Path(__file__).resolve().parents[2] / "resources" / "psychology_sources" / "rules_registry_v2.json"
REGISTRY_V2_SCHEMA_VERSION = "rules_registry_v2_v2"  # Phase 1.5 -- distinct from Phase 1's "rules_registry_v2_draft_1"

PSYCHOLOGIST_REVIEW_STATUS = "supplied_by_named_source_not_independently_reviewed"

# ---------------------------------------------------------------------------
# Evidence-family -> generic, honest alternative explanations (extended from
# Phase 1's set with the new families this phase introduces).
# ---------------------------------------------------------------------------
ALTERNATIVE_EXPLANATIONS: dict[str, list[str]] = {
    "facial_feature_style": [
        "May reflect an intentionally depicted expression, not the child's own state.",
        "May reflect copying a reference image, drawing style, motor/fine-skill limitations, or artistic convention.",
    ],
    "animal_symbolism": [
        "May reflect recent exposure to media, books, or a favourite character featuring the animal, not a psychological statement.",
        "May reflect the drawing prompt or immediate context rather than a stable trait.",
    ],
    "shape_symbolism": [
        "May reflect current developmental drawing stage, practiced skill, or simple preference.",
        "May reflect what the child has recently been taught or practiced drawing at school.",
    ],
    "object_symbolism": [
        "May reflect a specific recent event, request, or interest rather than a general personality trait.",
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
    "line_quality": [
        "Line appearance is strongly affected by the drawing tool, paper, and motor development, not only emotional state.",
        "This is a visual appearance proxy only -- it does not measure actual physical pencil pressure, which cannot be recovered from a flat scan/photo.",
    ],
    "line_intensity_quality": [
        "Line darkness/thickness appearance is strongly affected by the drawing tool (marker vs. pencil), paper, scan/photo exposure, and motor development, not only emotional state.",
        "This is an extracted line-darkness proxy only -- it does not measure actual physical pencil pressure, which cannot be recovered from a flat scan/photo.",
    ],
    "line_fragmentation_quality": [
        "Line continuity/fragmentation appearance is strongly affected by drawing speed, tool, fine-motor development, and image resolution, not only emotional state.",
        "This is an extracted line-fragmentation proxy only -- it does not measure the child's felt anxiety or intent.",
    ],
    "missing_body_part": [
        "Omitted or simplified body parts are common at many normal developmental stages and skill levels.",
        "May reflect running out of time or space, not an emotional signal.",
    ],
    "isolation_theme": [
        "May reflect the drawing prompt or a specific story the child had in mind, not a general social state.",
    ],
    "colour_mood_flags": [
        "Colour choice is strongly affected by which colours/pens were available, not only mood.",
        "Colour-based mood inference has mixed, non-diagnostic literature support.",
    ],
    "composition_omission": [
        "Background/detail omission commonly reflects limited time or a developmental focus on the main subject, not distress.",
    ],
    "detail_fixation": [
        "May reflect genuine interest or enjoyment in a specific detail, not anxiety.",
    ],
    "task_engagement": [
        "May reflect fatigue, disinterest in the specific task, or many unrelated situational factors.",
    ],
    "global_expressive_content_model": [
        "The model's output is a similarity judgment against a dataset of drawings, not a validated psychological or diagnostic instrument.",
    ],
}

# ---------------------------------------------------------------------------
# Rule definitions. One entry per rule_id. `source_entry_ids` link back to
# source_rule_catalog.json (built by source_catalog_build.py) -- every field
# here that echoes source content (page/section/interpretation) is derived
# from those catalog rows, not re-invented.
# ---------------------------------------------------------------------------
_EXECUTABLE_STATIC_DIRECT_FEATURE_IDS = {
    "coverage_about_half": ["composition.bounding_box_coverage"],
    "coverage_full": ["composition.bounding_box_coverage"],
    "coverage_small": ["composition.bounding_box_coverage"],
    "placement_top": ["composition.centroid_normalized"],
    "placement_left": ["composition.centroid_normalized"],
    "placement_right": ["composition.centroid_normalized"],
    # Phase 2A, Section 7 -- 4 rules newly wired to rule_engine_v2.py. The
    # dict name predates static_proxy rules being added here; despite the
    # name, this dict now also gates the 2 static_proxy line-quality rules
    # below (their `observability_class` remains "static_proxy" -- see
    # docs/STATIC_PROXY_RULE_POLICY.md for why proxy rules are still safe
    # to wire, unlike EXCESSIVE_DETAIL_040/NEGLECT_BACKGROUND_041, which
    # were deliberately left unwired).
    "placement_center": ["composition.centroid_normalized"],
    "heavy_line_pressure_appearance": ["stroke.intensity_proxy"],
    "light_line_pressure_appearance": ["stroke.intensity_proxy"],
    "shaky_or_broken_lines": ["stroke.fragmentation"],
}

# rule_id -> definition. Fields not repeated per-rule (professional_wording/
# parent_safe_wording/question_template) are generated from `interpretation`
# + `possible_interpretation` text by `_default_wordings()` below, to avoid
# hand-authoring near-duplicate prose 41 times; a handful of rules override
# these with a more specific `parent_safe_wording_override`.
_RULE_DEFS: list[dict[str, Any]] = [
    # ===== Original 19 (production rule_ids, unchanged) =====
    {"rule_id": "PSY_AR_EYES_WIDE_001", "source_entry_ids": ["SRC_AR_001", "SRC_EN_001"],
     "observable": "wide_eyes", "evidence_family": "facial_feature_style", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "openness/alertness (per the Arabic source: outgoing personality); in some emotion-drawing work, wide eyes can also fit fear/surprise -- the two English/Arabic readings point in different directions, so this rule is not mapped to a single construct."},
    {"rule_id": "PSY_AR_EYES_STERN_002", "source_entry_ids": ["SRC_AR_002", "SRC_EN_002"],
     "observable": "stern_eyes", "evidence_family": "facial_feature_style", "observability_class": "static_detector",
     "target_construct": "tension_or_anger_pattern", "direction": "supports",
     "interpretation": "anger, tension, suspicion, or defensiveness"},
    {"rule_id": "PSY_AR_EYES_CLOSED_003", "source_entry_ids": ["SRC_AR_003", "SRC_EN_003"],
     "observable": "closed_eyes", "evidence_family": "facial_feature_style", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "avoidance, shutting out the world, or refusing to look inward -- a specific, idiosyncratic reading that does not cleanly fit any of the 12 broad constructs"},
    {"rule_id": "PSY_AR_ANIMAL_TIGER_WOLF_004", "source_entry_ids": ["SRC_AR_004", "SRC_EN_006"],
     "observable": "tiger_or_wolf", "evidence_family": "animal_symbolism", "observability_class": "static_detector",
     "target_construct": "tension_or_anger_pattern", "direction": "supports",
     "interpretation": "anger, threat, or aggression"},
    {"rule_id": "PSY_AR_ANIMAL_FOX_005", "source_entry_ids": ["SRC_AR_005", "SRC_EN_007"],
     "observable": "fox", "evidence_family": "animal_symbolism", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "sneakiness, cunning, or (per the Arabic source) thinking of doing something mean/malicious"},
    {"rule_id": "PSY_AR_ANIMAL_SQUIRREL_006", "source_entry_ids": ["SRC_AR_006", "SRC_EN_008"],
     "observable": "squirrel", "evidence_family": "animal_symbolism", "observability_class": "static_detector",
     "target_construct": "protection_or_coping", "direction": "supports",
     "interpretation": "need for protection, nervous energy, or seeking care"},
    {"rule_id": "PSY_AR_ANIMAL_LION_007", "source_entry_ids": ["SRC_AR_007", "SRC_EN_009"],
     "observable": "lion", "evidence_family": "animal_symbolism", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "dominance, pride, strength, or a belief in one's own superiority -- a personality-trait reading, not a drawing-level emotional/social pattern, so left unmapped"},
    {"rule_id": "PSY_AR_GEOMETRY_008", "source_entry_ids": ["SRC_AR_008", "SRC_EN_011"],
     "observable": "repeated_geometric_shapes", "evidence_family": "shape_symbolism", "observability_class": "static_detector",
     "target_construct": "repetition_or_rigidity", "direction": "supports",
     "interpretation": "planning, control, persistence, or stubbornness"},
    {"rule_id": "PSY_AR_STARS_009", "source_entry_ids": ["SRC_AR_009", "SRC_EN_012"],
     "observable": "stars", "evidence_family": "shape_symbolism", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "wanting to be the center of attention, admiration, or visibility"},
    {"rule_id": "PSY_AR_FLOWERS_CLOUDS_SUN_010", "source_entry_ids": ["SRC_AR_010", "SRC_EN_014"],
     "observable": "flowers_clouds_sun", "evidence_family": "shape_symbolism", "observability_class": "process_required",
     "target_construct": "positive_affective_tone", "direction": "supports",
     "interpretation": "optimism, imagination, positive mood -- Arabic source requires the child to have been distracted/absent-minded while drawing; English source softens this to 'used in a dreamy context'. Either way this precondition describes the drawing PROCESS, not the finished image."},
    {"rule_id": "PSY_AR_CIRCLES_011", "source_entry_ids": ["SRC_AR_011", "SRC_EN_013"],
     "observable": "circles", "evidence_family": "shape_symbolism", "observability_class": "static_detector",
     "target_construct": "social_distance_or_isolation", "direction": "supports",
     "interpretation": "unity, wholeness, or need for closeness; per the Arabic source specifically, loneliness and need for support"},
    {"rule_id": "PSY_AR_TRANSPORT_012", "source_entry_ids": ["SRC_AR_012", "SRC_EN_016"],
     "observable": "vehicles", "evidence_family": "object_symbolism", "observability_class": "static_detector",
     "target_construct": "affiliation_or_connection", "direction": "supports",
     "interpretation": "loves travel/vacations, social, outgoing, playful, adventurous"},
    {"rule_id": "PSY_AR_HEARTS_013", "source_entry_ids": ["SRC_AR_013", "SRC_EN_015"],
     "observable": "hearts", "evidence_family": "object_symbolism", "observability_class": "static_detector",
     "target_construct": "affiliation_or_connection", "direction": "supports",
     "interpretation": "affectionate/loving personality, social, romantic, able to attract others"},
    {"rule_id": "PSY_AR_SIZE_HALF_014", "source_entry_ids": ["SRC_AR_014", "SRC_EN_021"],
     "observable": "coverage_about_half", "evidence_family": "size_composition", "observability_class": "static_direct",
     "target_construct": None, "direction": "supports",
     "interpretation": "outgoing at times and introverted at other times -- bidirectional by the source's own wording, so not mapped to a single-direction construct",
     "threshold_source": "source_centre_with_invented_band"},
    {"rule_id": "PSY_AR_SIZE_FULL_015", "source_entry_ids": ["SRC_AR_015", "SRC_EN_020"],
     "observable": "coverage_full", "evidence_family": "size_composition", "observability_class": "static_direct",
     "target_construct": "visual_dominance_or_prominence", "direction": "supports",
     "interpretation": "high self-respect/self-esteem; English source frames this as expansiveness/strong energy/self-assertion",
     "threshold_source": "invented_operational_standin"},
    {"rule_id": "PSY_AR_SIZE_SMALL_016", "source_entry_ids": ["SRC_AR_016", "SRC_EN_023"],
     "observable": "coverage_small", "evidence_family": "size_composition", "observability_class": "static_direct",
     "target_construct": "fear_or_insecurity_pattern", "direction": "supports",
     "interpretation": "instability or fear; English source frames the page-area-fraction version as caution/uncertainty/low confidence",
     "threshold_source": "directly_sourced"},
    {"rule_id": "PSY_AR_PLACE_TOP_017", "source_entry_ids": ["SRC_AR_017", "SRC_EN_024"],
     "observable": "placement_top", "evidence_family": "spatial_placement", "observability_class": "static_direct",
     "target_construct": None, "direction": "supports",
     "interpretation": "dreamy personality, fantasy world, difficulty adapting -- does not cleanly fit any of the 12 broad constructs",
     "threshold_source": "invented_operational_standin"},
    {"rule_id": "PSY_AR_PLACE_LEFT_018", "source_entry_ids": ["SRC_AR_018", "SRC_EN_025"],
     "observable": "placement_left", "evidence_family": "spatial_placement", "observability_class": "static_direct",
     "target_construct": "social_distance_or_isolation", "direction": "supports",
     "interpretation": "introverted personality; English source adds past orientation/dependence",
     "threshold_source": "invented_operational_standin"},
    {"rule_id": "PSY_AR_PLACE_RIGHT_019", "source_entry_ids": ["SRC_AR_019", "SRC_EN_026"],
     "observable": "placement_right", "evidence_family": "spatial_placement", "observability_class": "static_direct",
     "target_construct": "affiliation_or_connection", "direction": "supports",
     "interpretation": "outgoing personality; English source adds future orientation/outward movement",
     "threshold_source": "invented_operational_standin"},
    # ===== New rules from the compiled English PDF only =====
    {"rule_id": "EN_COMPILED_EYES_MISSING_DETAIL_020", "source_entry_ids": ["SRC_EN_004", "SRC_EN_035"],
     "observable": "eyes_missing_or_undetailed", "evidence_family": "facial_feature_style", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "withdrawal, avoidance, low interest in contact, or simply limited drawing skill -- two near-duplicate source rows merged into one rule; see source_rule_catalog.json relationships"},
    {"rule_id": "EN_COMPILED_FACE_EXPRESSION_021", "source_entry_ids": ["SRC_EN_005"],
     "observable": "face_expression", "evidence_family": "facial_feature_style", "observability_class": "static_detector",
     "target_construct": "low_mood_or_emotional_distress_pattern", "direction": "supports",
     "interpretation": "positive tone (smiling) or distress (sad/tense/frightened) -- the PDF's own 'Moderate for basic emotion only' grade is the strongest evidence level anywhere in either source"},
    {"rule_id": "EN_COMPILED_ANIMAL_CHOICE_GENERAL_022", "source_entry_ids": ["SRC_EN_010"],
     "observable": "animal_choice_general", "evidence_family": "animal_symbolism", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "interests, story play, fear, identification, or a projective theme -- too broad/non-specific to map to one construct"},
    {"rule_id": "EN_COMPILED_HOUSE_023", "source_entry_ids": ["SRC_EN_017"],
     "observable": "house", "evidence_family": "object_symbolism", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "family life, safety, privacy, or emotional openness"},
    {"rule_id": "EN_COMPILED_TREE_024", "source_entry_ids": ["SRC_EN_018"],
     "observable": "tree", "evidence_family": "object_symbolism", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "stability, growth, roots, or vulnerability"},
    {"rule_id": "EN_COMPILED_REPEATED_MONSTERS_DANGER_025", "source_entry_ids": ["SRC_EN_019", "SRC_EN_037"],
     "observable": "repeated_monsters_danger_injury", "evidence_family": "isolation_theme", "observability_class": "longitudinal_required",
     "target_construct": "fear_or_insecurity_pattern", "direction": "supports",
     "interpretation": "fear, distress, or preoccupation with threat -- the word 'repeated' in the source means this requires comparing multiple drawings over time, not one image"},
    {"rule_id": "EN_COMPILED_VERY_SMALL_DRAWING_026", "source_entry_ids": ["SRC_EN_022"],
     "observable": "very_small_drawing_absolute", "evidence_family": "size_composition", "observability_class": "not_operational",
     "target_construct": "fear_or_insecurity_pattern", "direction": "supports",
     "interpretation": "insecurity, withdrawal, fear, or low confidence -- distinct from the page-fraction-based coverage_small rule; absolute figure size cannot be measured without a physical scale reference, which a scanned/photographed drawing does not carry"},
    {"rule_id": "EN_COMPILED_REPEATED_FAMILY_CONFLICT_027", "source_entry_ids": ["SRC_EN_038"],
     "observable": "repeated_family_conflict", "evidence_family": "isolation_theme", "observability_class": "longitudinal_required",
     "target_construct": "low_mood_or_emotional_distress_pattern", "direction": "supports",
     "interpretation": "tension, perceived conflict, or concern in the home"},
    {"rule_id": "EN_COMPILED_REPEATED_ISOLATION_THEMES_028", "source_entry_ids": ["SRC_EN_039"],
     "observable": "repeated_isolation_themes", "evidence_family": "isolation_theme", "observability_class": "longitudinal_required",
     "target_construct": "social_distance_or_isolation", "direction": "supports",
     "interpretation": "loneliness or withdrawal"},
    {"rule_id": "EN_COMPILED_PLACEMENT_CENTER_029", "source_entry_ids": ["SRC_EN_027"],
     "observable": "placement_center", "evidence_family": "spatial_placement", "observability_class": "static_direct",
     "target_construct": "visual_dominance_or_prominence", "direction": "supports",
     "interpretation": "balance, security, or comfort; a real feature (composition.centroid_normalized) exists and is now wired via rule_engine_v2.py, gated on page-frame assessability (see docs/PAGE_FRAME_ASSESSABILITY.md)",
     "threshold_source": "invented_operational_standin",
     # Conservative, explicitly-invented ceiling (Phase 2A) -- no
     # clinician-assigned value exists for this rule (it is not in
     # rules_registry.json). Matches the lowest ceiling already used for
     # the 3 comparable, similarly-unvalidated placement rules
     # (PSY_AR_PLACE_TOP/LEFT/RIGHT, all 0.10) rather than inventing a new
     # number. Not tuned to any agreement metric.
     "confidence_ceiling": 0.10,
     "scientific_support": "placement_effects_weak_not_specific"},
    {"rule_id": "EN_COMPILED_LINE_HEAVY_PRESSURE_030", "source_entry_ids": ["SRC_EN_028"],
     "observable": "heavy_line_pressure_appearance", "evidence_family": "line_intensity_quality", "observability_class": "static_proxy",
     "target_construct": "tension_or_anger_pattern", "direction": "supports",
     "interpretation": "tension, force, anger, or high energy -- a real proxy feature (stroke.intensity_proxy) is now wired via rule_engine_v2.py; must never be described as actual physical pencil pressure",
     "threshold_source": "empirically_exploratory",
     "parent_safe_wording_override": (
         "The lines appear relatively dark or thick in the uploaded image. Some sources associate this visual "
         "appearance with tension, force, anger, or high energy -- but line darkness/thickness is also strongly "
         "affected by the drawing tool, paper, and scan/photo exposure. This is not a diagnosis, and it does not "
         "measure actual physical pencil pressure."
     ),
     "professional_wording_override": (
         "The extracted line-intensity proxy (stroke.intensity_proxy) is elevated (>= the 75th percentile of a "
         "real, non-test reference sample), consistent with a relatively dark/thick line appearance in the image. "
         "Source interpretation: tension, force, anger, or high energy. This is an appearance-based proxy only, "
         "not a measurement of physical pencil pressure."
     ),
     # Conservative, explicitly-invented ceiling (Phase 2A) -- same
     # rationale as EN_COMPILED_PLACEMENT_CENTER_029 above: no
     # clinician-assigned value exists, so this uses the lowest ceiling
     # already established in rules_registry.json rather than a new number.
     "confidence_ceiling": 0.10,
     "scientific_support": "line_pressure_appearance_proxy_not_independently_validated"},
    {"rule_id": "EN_COMPILED_LINE_LIGHT_PRESSURE_031", "source_entry_ids": ["SRC_EN_029"],
     "observable": "light_line_pressure_appearance", "evidence_family": "line_intensity_quality", "observability_class": "static_proxy",
     "target_construct": "caution_or_low_visual_energy", "direction": "supports",
     "interpretation": "hesitation, shyness, low energy, or delicacy -- same stroke.intensity_proxy feature at the opposite end",
     "threshold_source": "empirically_exploratory",
     "parent_safe_wording_override": (
         "The lines appear relatively light or thin in the uploaded image. Some sources associate this visual "
         "appearance with hesitation, shyness, or low energy -- but line darkness/thickness is also strongly "
         "affected by the drawing tool, paper, and scan/photo exposure. This is not a diagnosis, and it does not "
         "measure actual physical pencil pressure."
     ),
     "professional_wording_override": (
         "The extracted line-intensity proxy (stroke.intensity_proxy) is low (<= the 25th percentile of a real, "
         "non-test reference sample), consistent with a relatively light/thin line appearance in the image. Source "
         "interpretation: hesitation, shyness, low energy, or delicacy. This is an appearance-based proxy only, not "
         "a measurement of physical pencil pressure."
     ),
     "confidence_ceiling": 0.10,
     "scientific_support": "line_pressure_appearance_proxy_not_independently_validated"},
    {"rule_id": "EN_COMPILED_LINE_SHAKY_BROKEN_032", "source_entry_ids": ["SRC_EN_030"],
     "observable": "shaky_or_broken_lines", "evidence_family": "line_fragmentation_quality", "observability_class": "static_proxy",
     "target_construct": "disorganisation_or_fragmentation", "direction": "supports",
     "interpretation": "anxiety, uncertainty, distress, or motor difficulty -- a real proxy feature (stroke.fragmentation) is now wired via rule_engine_v2.py",
     "threshold_source": "empirically_exploratory",
     "parent_safe_wording_override": (
         "The extracted line-fragmentation proxy is elevated for this drawing. Some sources associate shaky or "
         "broken-looking lines with anxiety, uncertainty, or motor difficulty -- but line fragmentation is also "
         "strongly affected by drawing speed, tool, fine-motor development, and image resolution. This is not a "
         "diagnosis."
     ),
     "professional_wording_override": (
         "The extracted line-fragmentation proxy (stroke.fragmentation) is elevated (>= the 75th percentile of a "
         "real, non-test reference sample). Source interpretation: anxiety, uncertainty, distress, or motor "
         "difficulty. This is an appearance-based proxy only."
     ),
     "confidence_ceiling": 0.10,
     "scientific_support": "line_fragmentation_appearance_proxy_not_independently_validated"},
    {"rule_id": "EN_COMPILED_LINE_ZIGZAG_033", "source_entry_ids": ["SRC_EN_031"],
     "observable": "zigzag_lines", "evidence_family": "line_quality", "observability_class": "static_detector",
     "target_construct": "tension_or_anger_pattern", "direction": "supports",
     "interpretation": "agitation, violence, unpredictability, or conflict -- no existing feature proxies a zigzag pattern specifically"},
    {"rule_id": "EN_COMPILED_LINE_OVER_ERASING_034", "source_entry_ids": ["SRC_EN_032"],
     "observable": "over_erasing_appearance", "evidence_family": "line_quality", "observability_class": "process_required",
     "target_construct": None, "direction": "supports",
     "interpretation": "perfectionism, anxiety, frustration, or uncertainty -- erasure frequency describes the drawing PROCESS over time, not visible in a single flattened final scan"},
    {"rule_id": "EN_COMPILED_MISSING_HANDS_035", "source_entry_ids": ["SRC_EN_033"],
     "observable": "missing_hands", "evidence_family": "missing_body_part", "observability_class": "static_detector",
     "target_construct": None, "direction": "supports",
     "interpretation": "difficulty acting, social difficulty, or avoidance"},
    {"rule_id": "EN_COMPILED_MISSING_MOUTH_036", "source_entry_ids": ["SRC_EN_034"],
     "observable": "missing_mouth", "evidence_family": "missing_body_part", "observability_class": "static_detector",
     "target_construct": "low_mood_or_emotional_distress_pattern", "direction": "supports",
     "interpretation": "difficulty expressing emotion or communication issues"},
    {"rule_id": "EN_COMPILED_EXAGGERATED_BODY_PARTS_037", "source_entry_ids": ["SRC_EN_036"],
     "observable": "exaggerated_body_parts", "evidence_family": "missing_body_part", "observability_class": "static_detector",
     "target_construct": "fear_or_insecurity_pattern", "direction": "supports",
     "interpretation": "aggression, fear, or hypervigilance -- the source itself grades this 'Very tentative'"},
    {"rule_id": "EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038", "source_entry_ids": ["SRC_EN_040"],
     "observable": "dark_colors_sad_faces_isolation", "evidence_family": "colour_mood_flags", "observability_class": "static_detector",
     "target_construct": "low_mood_or_emotional_distress_pattern", "direction": "supports",
     "interpretation": "possible sadness, low mood, or distress -- a compound row: the colour component (colour.dark_ratio) is genuinely static_direct, but 'sad faces' and 'isolation' both require detectors that do not exist, so the whole row is blocked at the weakest-link level"},
    {"rule_id": "EN_COMPILED_REFUSAL_TO_DRAW_039", "source_entry_ids": ["SRC_EN_041"],
     "observable": "refusal_to_draw_figures", "evidence_family": "task_engagement", "observability_class": "process_required",
     "target_construct": None, "direction": "supports",
     "interpretation": "distress, resistance, perfectionism, or lack of task engagement -- describes something the child DID NOT draw during the session, not visible in any finished image"},
    {"rule_id": "EN_COMPILED_EXCESSIVE_DETAIL_040", "source_entry_ids": ["SRC_EN_042"],
     "observable": "excessive_detail_or_fixation", "evidence_family": "detail_fixation", "observability_class": "static_detector",
     "target_construct": "repetition_or_rigidity", "direction": "supports",
     "interpretation": "anxiety, obsession with parts of the task, or strong interest"},
    {"rule_id": "EN_COMPILED_NEGLECT_BACKGROUND_041", "source_entry_ids": ["SRC_EN_043"],
     "observable": "neglect_of_background", "evidence_family": "composition_omission", "observability_class": "static_detector",
     "target_construct": "disorganisation_or_fragmentation", "direction": "supports",
     "interpretation": "focus on the main subject, limited time, or developmental style -- the source itself notes this is 'not a pathology sign by itself'"},
]


def _allowed_output_level(observable: str) -> str:
    # The 6 original tier-1 composition/placement rules plus the 4 rules
    # newly wired in Phase 2A, Section 7 (see rule_engine_v2.py) are the
    # only rules actually dispatched by any evaluator today -- every other
    # rule, regardless of its observability_class, remains disabled in
    # practice (see module docstring).
    if observable in _EXECUTABLE_STATIC_DIRECT_FEATURE_IDS:
        return "individual_heuristic_only"
    return "disabled"


def _required_detector_or_metadata(rule_id: str, observable: str, observability_class: str) -> str | None:
    if observable in _EXECUTABLE_STATIC_DIRECT_FEATURE_IDS:
        return None
    return f"detector_or_context_absent:{observability_class}"


def build_registry_v2() -> dict[str, Any]:
    catalog = build_source_catalog()
    entries_by_id = {e["source_entry_id"]: e for e in catalog["entries"]}
    production = json.loads((REGISTRY_V2_PATH.parent / "rules_registry.json").read_text(encoding="utf-8"))
    production_by_id = {r["rule_id"]: r for r in production["rules"]}
    # Real, existing academic references (never fabricated here) -- copied
    # verbatim from the production registry, the only place they are
    # defined. New EN_COMPILED_* rules cite none of these directly (the
    # compiled PDF's own "Notes/literature" column is not a formal
    # citation and is preserved instead in each rule's `limitations`).
    academic_references = production["references"]

    rules = []
    for defn in _RULE_DEFS:
        rule_id = defn["rule_id"]
        source_entries = [entries_by_id[sid] for sid in defn["source_entry_ids"]]
        primary = source_entries[0]
        family = defn["evidence_family"]
        allowed_output = _allowed_output_level(defn["observable"])
        evidence_level_terms = sorted({e["evidence_level_as_written"] for e in source_entries if e["evidence_level_as_written"]})
        reference_ids = list(production_by_id[rule_id]["references"]) if rule_id in production_by_id else []
        # confidence_ceiling/scientific_support: for the 6 already-in-production
        # rules, reuse the real clinician-set values from rules_registry.json
        # (the same source `reference_ids` above already reuses). For the 4
        # newly-activated EN_COMPILED_* rules (Phase 2A, Section 7), no such
        # value exists anywhere -- `_RULE_DEFS` carries an explicit,
        # conservative, non-silent value instead (see per-rule comments).
        # Every other (still-disabled) rule has neither, which is fine: only
        # executed rules are ever run through rule_engine_v2.py's _base_eval.
        if rule_id in production_by_id:
            confidence_ceiling = production_by_id[rule_id]["confidence_ceiling"]
            scientific_support = production_by_id[rule_id]["scientific_support"]
        else:
            confidence_ceiling = defn.get("confidence_ceiling")
            scientific_support = defn.get("scientific_support")

        rules.append({
            "rule_id": rule_id,
            "registry_v2_status": "already_in_production_registry" if rule_id.startswith("PSY_AR_") else "new_candidate_from_compiled_pdf_unreviewed",
            "source_entry_ids": defn["source_entry_ids"],
            "source_document": primary["source_document"],
            "source_page": primary["source_page"],
            "source_section": primary["source_section"],
            "faithful_source_quote": primary["original_text"],
            "observable": defn["observable"],
            "possible_interpretation": defn["interpretation"],
            "evidence_level_as_written": evidence_level_terms or None,
            "confidence_ceiling": confidence_ceiling,
            "scientific_support": scientific_support,
            "observability_class": defn["observability_class"],
            "required_feature_ids": _EXECUTABLE_STATIC_DIRECT_FEATURE_IDS.get(defn["observable"], []),
            "required_detector_or_metadata": _required_detector_or_metadata(rule_id, defn["observable"], defn["observability_class"]),
            "evidence_family": family,
            "dependency_group": _EXECUTABLE_STATIC_DIRECT_FEATURE_IDS.get(defn["observable"], []),
            "target_construct": defn["target_construct"],
            "direction": defn["direction"] if defn["target_construct"] else None,
            "allowed_output_level": allowed_output,
            "alternative_explanations": list(ALTERNATIVE_EXPLANATIONS.get(family, [])),
            "reference_ids": reference_ids,
            "limitations": sorted({n for e in source_entries if (n := e.get("notes_as_written"))}),
            "parent_safe_wording": defn.get("parent_safe_wording_override") or (
                f"The drawing includes a feature ({defn['observable'].replace('_', ' ')}) some sources associate with "
                f"{defn['interpretation'].split(' -- ')[0].split(';')[0]}. This is not a diagnosis, and the same feature "
                f"has other common, non-clinical explanations."
            ),
            "professional_wording": defn.get("professional_wording_override") or (
                f"Observable '{defn['observable']}' matched. Source interpretation: {defn['interpretation']}. "
                f"Evidence grade as written in source: "
                f"{', '.join(evidence_level_terms) or 'not stated (Arabic source has no explicit evidence-level column)'}."
            ),
            "question_template": f"Would you be willing to ask your child about the {defn['observable'].replace('_', ' ')} in this drawing?",
            "psychologist_review_status": PSYCHOLOGIST_REVIEW_STATUS,
            # BUG FIX (Phase 2A): this previously checked `rule_id in
            # _EXECUTABLE_STATIC_DIRECT_FEATURE_IDS`, but that dict is keyed
            # by `observable`, not `rule_id` -- so the condition was always
            # False and no rule's validation_status was ever correctly set.
            "validation_status": "IMPLEMENTED_UNVALIDATED" if allowed_output == "individual_heuristic_only" else "DETECTOR_UNAVAILABLE",
            "threshold_source": defn.get("threshold_source", "not_applicable"),
            "version": REGISTRY_V2_SCHEMA_VERSION,
        })

    return {
        "schema_version": REGISTRY_V2_SCHEMA_VERSION,
        "status": "draft -- NOT the production registry; rules.py/rules_registry.json are unchanged and remain authoritative",
        "source_documents": catalog["sources"],
        "note": (
            "Phase 1.5 full rewrite: consumes source_rule_catalog.json (both PDFs, 67 source rows) "
            "and construct_registry.json. Supersedes the Phase 1 draft, which only ever covered "
            "the Arabic PDF's 19 rows. See docs/RULE_PDF_COVERAGE_AUDIT_V2.md."
        ),
        "rule_count": len(rules),
        "rule_count_original_production": sum(1 for r in rules if r["rule_id"].startswith("PSY_AR_")),
        "rule_count_new_from_compiled_pdf": sum(1 for r in rules if r["rule_id"].startswith("EN_COMPILED_")),
        "references": academic_references,
        "rules": rules,
    }


def write_registry_v2() -> Path:
    document = build_registry_v2()
    REGISTRY_V2_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return REGISTRY_V2_PATH


if __name__ == "__main__":
    path = write_registry_v2()
    print(f"wrote {path}")
