"""Phase 2C.4: rule visual-coverage classification.

Classifies each of the 41 rules in resources/psychology_sources/rules_registry_v2.json
into exactly one primary blocking-reason bucket, grounded in
PHASE2C3_AUDIT_AND_DESIGN.md's own traceability audit (Section 4) and this
phase's real detector benchmark results (which classes the benchmarked
detectors actually cover: the existing 10-class ontology). Classification
only -- no rule's activation_status or allowed_output_level is read from or
written to rules_registry_v2.json by this module, and nothing here is
capable of activating a rule.
"""
from __future__ import annotations

import csv
from pathlib import Path

PRIMARY_CLASSIFICATIONS = frozenset({
    "already_measurable_by_objective_features",
    "potentially_measurable_by_benchmarked_detector",
    "requires_new_object_or_part_class",
    "requires_object_attribute",
    "requires_bbox_localization",
    "requires_process_information",
    "requires_longitudinal_information",
    "remains_not_operational",
})

# One row per rule in rules_registry_v2.json (41 total, verified count).
# `benchmarked_class` names the current 10-class ontology member this rule's
# `observable` maps to, if any -- None if no current class applies.
RULE_COVERAGE = [
    {"rule_id": "PSY_AR_EYES_WIDE_001", "observable": "wide_eyes", "benchmarked_class": None,
     "primary_classification": "requires_new_object_or_part_class",
     "rationale": "No 'eye' part class exists in Ontology V1/V2-candidate yet; a state judgment "
                  "(wide) cannot even be attempted before the part itself is detectable."},
    {"rule_id": "PSY_AR_EYES_STERN_002", "observable": "stern_eyes", "benchmarked_class": None,
     "primary_classification": "requires_new_object_or_part_class",
     "rationale": "Same eye-part gap as 001; state (stern) is a second, separate gap on top."},
    {"rule_id": "PSY_AR_EYES_CLOSED_003", "observable": "closed_eyes", "benchmarked_class": None,
     "primary_classification": "requires_new_object_or_part_class",
     "rationale": "Same eye-part gap as 001."},
    {"rule_id": "PSY_AR_ANIMAL_TIGER_WOLF_004", "observable": "tiger_or_wolf", "benchmarked_class": "animal",
     "primary_classification": "requires_object_attribute",
     "rationale": "'animal' (general) is benchmarked, but this rule needs species -- an attribute "
                  "on top of presence. Also explicitly forbidden to activate even if detected "
                  "(Phase 2B Section 3 policy, unchanged)."},
    {"rule_id": "PSY_AR_ANIMAL_FOX_005", "observable": "fox", "benchmarked_class": "animal",
     "primary_classification": "requires_object_attribute",
     "rationale": "Same species-attribute gap as 004; same forbidden-to-activate policy."},
    {"rule_id": "PSY_AR_ANIMAL_SQUIRREL_006", "observable": "squirrel", "benchmarked_class": "animal",
     "primary_classification": "requires_object_attribute",
     "rationale": "Same species-attribute gap as 004; same forbidden-to-activate policy."},
    {"rule_id": "PSY_AR_ANIMAL_LION_007", "observable": "lion", "benchmarked_class": "animal",
     "primary_classification": "requires_object_attribute",
     "rationale": "Same species-attribute gap as 004; same forbidden-to-activate policy."},
    {"rule_id": "PSY_AR_GEOMETRY_008", "observable": "repeated_geometric_shapes", "benchmarked_class": "circle",
     "primary_classification": "requires_object_attribute",
     "rationale": "'circle' presence is benchmarked, but this rule needs repetition COUNT across "
                  "shapes generally, not single-class presence."},
    {"rule_id": "PSY_AR_STARS_009", "observable": "stars", "benchmarked_class": "star",
     "primary_classification": "potentially_measurable_by_benchmarked_detector",
     "rationale": "Plain presence of an already-benchmarked class."},
    {"rule_id": "PSY_AR_FLOWERS_CLOUDS_SUN_010", "observable": "flowers_clouds_sun", "benchmarked_class": None,
     "primary_classification": "requires_process_information",
     "rationale": "observability_class=process_required in the registry -- no detector, however good, "
                  "can supply drawing-process information from a static image."},
    {"rule_id": "PSY_AR_CIRCLES_011", "observable": "circles", "benchmarked_class": "circle",
     "primary_classification": "potentially_measurable_by_benchmarked_detector",
     "rationale": "Plain presence of an already-benchmarked class."},
    {"rule_id": "PSY_AR_TRANSPORT_012", "observable": "vehicles", "benchmarked_class": "vehicle",
     "primary_classification": "potentially_measurable_by_benchmarked_detector",
     "rationale": "Plain presence of an already-benchmarked class."},
    {"rule_id": "PSY_AR_HEARTS_013", "observable": "hearts", "benchmarked_class": "heart",
     "primary_classification": "potentially_measurable_by_benchmarked_detector",
     "rationale": "Plain presence of an already-benchmarked class."},
    {"rule_id": "PSY_AR_SIZE_HALF_014", "observable": "coverage_about_half", "benchmarked_class": None,
     "primary_classification": "already_measurable_by_objective_features",
     "rationale": "static_direct, allowed_output_level=individual_heuristic_only -- already executable "
                  "via composition.bounding_box_coverage, no detector involved."},
    {"rule_id": "PSY_AR_SIZE_FULL_015", "observable": "coverage_full", "benchmarked_class": None,
     "primary_classification": "already_measurable_by_objective_features",
     "rationale": "Same as 014."},
    {"rule_id": "PSY_AR_SIZE_SMALL_016", "observable": "coverage_small", "benchmarked_class": None,
     "primary_classification": "already_measurable_by_objective_features",
     "rationale": "Same as 014."},
    {"rule_id": "PSY_AR_PLACE_TOP_017", "observable": "placement_top", "benchmarked_class": None,
     "primary_classification": "already_measurable_by_objective_features",
     "rationale": "static_direct via composition.centroid_normalized, already executable."},
    {"rule_id": "PSY_AR_PLACE_LEFT_018", "observable": "placement_left", "benchmarked_class": None,
     "primary_classification": "already_measurable_by_objective_features",
     "rationale": "Same as 017."},
    {"rule_id": "PSY_AR_PLACE_RIGHT_019", "observable": "placement_right", "benchmarked_class": None,
     "primary_classification": "already_measurable_by_objective_features",
     "rationale": "Same as 017."},
    {"rule_id": "EN_COMPILED_EYES_MISSING_DETAIL_020", "observable": "eyes_missing_or_undetailed",
     "benchmarked_class": None, "primary_classification": "requires_new_object_or_part_class",
     "rationale": "Same eye-part gap as 001; this rule additionally needs an omission/detail judgment "
                  "once the part exists."},
    {"rule_id": "EN_COMPILED_FACE_EXPRESSION_021", "observable": "face_expression", "benchmarked_class": "face",
     "primary_classification": "requires_object_attribute",
     "rationale": "'face' presence is benchmarked, but expression classification is a new attribute "
                  "with no existing labeled taxonomy -- the harder technical problem, not solved by "
                  "better presence detection alone."},
    {"rule_id": "EN_COMPILED_ANIMAL_CHOICE_GENERAL_022", "observable": "animal_choice_general",
     "benchmarked_class": "animal", "primary_classification": "potentially_measurable_by_benchmarked_detector",
     "rationale": "General (non-species) animal presence -- exactly what 'animal' already covers."},
    {"rule_id": "EN_COMPILED_HOUSE_023", "observable": "house", "benchmarked_class": "house",
     "primary_classification": "potentially_measurable_by_benchmarked_detector",
     "rationale": "Plain presence of an already-benchmarked class."},
    {"rule_id": "EN_COMPILED_TREE_024", "observable": "tree", "benchmarked_class": "tree",
     "primary_classification": "potentially_measurable_by_benchmarked_detector",
     "rationale": "Plain presence of an already-benchmarked class."},
    {"rule_id": "EN_COMPILED_REPEATED_MONSTERS_DANGER_025", "observable": "repeated_monsters_danger_injury",
     "benchmarked_class": None, "primary_classification": "requires_longitudinal_information",
     "rationale": "observability_class=longitudinal_required -- needs repetition across multiple "
                  "drawings over time; a single-image detector cannot supply this."},
    {"rule_id": "EN_COMPILED_VERY_SMALL_DRAWING_026", "observable": "very_small_drawing_absolute",
     "benchmarked_class": None, "primary_classification": "remains_not_operational",
     "rationale": "observability_class=not_operational -- needs an absolute physical size reference "
                  "no scan/photo carries; no detector or annotation change fixes this."},
    {"rule_id": "EN_COMPILED_REPEATED_FAMILY_CONFLICT_027", "observable": "repeated_family_conflict",
     "benchmarked_class": None, "primary_classification": "requires_longitudinal_information",
     "rationale": "Same as 025."},
    {"rule_id": "EN_COMPILED_REPEATED_ISOLATION_THEMES_028", "observable": "repeated_isolation_themes",
     "benchmarked_class": None, "primary_classification": "requires_longitudinal_information",
     "rationale": "Same as 025."},
    {"rule_id": "EN_COMPILED_PLACEMENT_CENTER_029", "observable": "placement_center", "benchmarked_class": None,
     "primary_classification": "already_measurable_by_objective_features",
     "rationale": "static_direct via composition.centroid_normalized, already executable."},
    {"rule_id": "EN_COMPILED_LINE_HEAVY_PRESSURE_030", "observable": "heavy_line_pressure_appearance",
     "benchmarked_class": None, "primary_classification": "already_measurable_by_objective_features",
     "rationale": "static_proxy via stroke.intensity_proxy, already executable."},
    {"rule_id": "EN_COMPILED_LINE_LIGHT_PRESSURE_031", "observable": "light_line_pressure_appearance",
     "benchmarked_class": None, "primary_classification": "already_measurable_by_objective_features",
     "rationale": "Same as 030."},
    {"rule_id": "EN_COMPILED_LINE_SHAKY_BROKEN_032", "observable": "shaky_or_broken_lines",
     "benchmarked_class": None, "primary_classification": "already_measurable_by_objective_features",
     "rationale": "static_proxy via stroke.fragmentation, already executable."},
    {"rule_id": "EN_COMPILED_LINE_ZIGZAG_033", "observable": "zigzag_lines", "benchmarked_class": None,
     "primary_classification": "requires_new_object_or_part_class",
     "rationale": "Flagged by Phase 2B's own ontology doc as arguably not a real 'object' at all -- a "
                  "line-pattern attribute closer in kind to the existing stroke proxies (030-032) than "
                  "to an object class. Borderline: could plausibly become 'already_measurable_by_"
                  "objective_features' via a classical-CV stroke-pattern analyzer instead of any "
                  "open-vocabulary detector; not attempted in this phase."},
    {"rule_id": "EN_COMPILED_LINE_OVER_ERASING_034", "observable": "over_erasing_appearance",
     "benchmarked_class": None, "primary_classification": "requires_process_information",
     "rationale": "observability_class=process_required -- needs to observe erasure during drawing, "
                  "invisible in the finished artifact."},
    {"rule_id": "EN_COMPILED_MISSING_HANDS_035", "observable": "missing_hands", "benchmarked_class": "hand",
     "primary_classification": "requires_bbox_localization",
     "rationale": "'hand' presence is benchmarked, but a rigorous omission judgment needs to confirm "
                  "the expected hand REGION was visible (not merely cropped out of frame) -- "
                  "evidence_rule_engine.py's existing evaluate_omission() primitive gates on "
                  "'region_visible', which needs spatial/bbox information. A lower-rigor presence-only "
                  "proxy (person detected + hand absent) is possible but was not what that primitive "
                  "was designed to certify."},
    {"rule_id": "EN_COMPILED_MISSING_MOUTH_036", "observable": "missing_mouth", "benchmarked_class": None,
     "primary_classification": "requires_new_object_or_part_class",
     "rationale": "No 'mouth' class exists yet (postponed from Phase 2B); omission reasoning is a "
                  "second gap on top, same pattern as 035 once the part exists."},
    {"rule_id": "EN_COMPILED_EXAGGERATED_BODY_PARTS_037", "observable": "exaggerated_body_parts",
     "benchmarked_class": "hand", "primary_classification": "requires_bbox_localization",
     "rationale": "'hand' is benchmarked, but 'exaggerated' is inherently a relative-SIZE judgment "
                  "(this part vs. a reference, e.g. the whole figure) -- needs bounding boxes on both, "
                  "not just presence."},
    {"rule_id": "EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038", "observable": "dark_colors_sad_faces_isolation",
     "benchmarked_class": "face", "primary_classification": "requires_bbox_localization",
     "rationale": "Compound rule: colour (already measurable) + face (benchmarked) + isolation "
                  "(a spatial relationship between detected figures -- needs bboxes, not yet built, "
                  "Phase 2C.3 Stage D). Blocked at its weakest link."},
    {"rule_id": "EN_COMPILED_REFUSAL_TO_DRAW_039", "observable": "refusal_to_draw_figures",
     "benchmarked_class": None, "primary_classification": "requires_process_information",
     "rationale": "observability_class=process_required -- a behavioral/process observation, not "
                  "visible in a static image."},
    {"rule_id": "EN_COMPILED_EXCESSIVE_DETAIL_040", "observable": "excessive_detail_or_fixation",
     "benchmarked_class": None, "primary_classification": "requires_object_attribute",
     "rationale": "'Excessive' has no natural detector output -- needs a count/density attribute with "
                  "a defensible threshold, not a presence detection."},
    {"rule_id": "EN_COMPILED_NEGLECT_BACKGROUND_041", "observable": "neglect_of_background",
     "benchmarked_class": None, "primary_classification": "requires_new_object_or_part_class",
     "rationale": "'Background' is a region, not any of the 10 benchmarked object classes -- needs a "
                  "region-level presence/extent judgment (segmentation-adjacent), plus a figure-vs-"
                  "background spatial relationship on top (Phase 2C.3 Stage D) once the region exists."},
]


def write_csv(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rule_id", "observable", "benchmarked_class", "primary_classification", "rationale"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in RULE_COVERAGE:
            writer.writerow({k: row[k] for k in fields})
    return output_path


def summary_counts() -> dict[str, int]:
    out = {c: 0 for c in sorted(PRIMARY_CLASSIFICATIONS)}
    for row in RULE_COVERAGE:
        out[row["primary_classification"]] += 1
    return out
