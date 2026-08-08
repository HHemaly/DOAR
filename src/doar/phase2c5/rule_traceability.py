"""Phase 2C.5 Stage 1: exact rule-to-annotation-target traceability.

Every row below was read directly from
resources/psychology_sources/rules_registry_v2.json this session (rule_id,
observable, faithful_source_quote) -- not inferred from memory. A test
(`tests/test_phase2c5_rule_traceability.py`) verifies every rule_id
referenced here actually exists in the live registry file with the exact
`observable` recorded.

IMPORTANT, restated from artifacts/phase2c4/RULE_VISUAL_COVERAGE_AFTER_DETECTOR_BENCHMARK.csv
and every rule row's own `allowed_output_level` field: all 9 rules below
currently have `allowed_output_level == "disabled"` in the live registry.
Nothing in this phase changes that. Collecting the evidence these rules
would need if enabled is not the same as enabling them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AnnotationTargetTraceability:
    target_name: str
    supported_rule_ids: tuple[str, ...]
    required_observation: str
    presence_sufficient: bool
    bbox_required: bool
    count_required: bool
    attribute_required: str  # "" if none
    derivable_after_annotation: str
    suitable_this_phase: bool
    rationale: str


RULE_ANNOTATION_TRACEABILITY: list[AnnotationTargetTraceability] = [
    AnnotationTargetTraceability(
        target_name="eye",
        supported_rule_ids=(
            "PSY_AR_EYES_CLOSED_003", "EN_COMPILED_EYES_MISSING_DETAIL_020",
        ),
        required_observation="presence, bbox per eye, instance count, open/closed state, "
                              "detail state (detailed/undetailed/missing)",
        presence_sufficient=False,
        bbox_required=True,
        count_required=True,
        attribute_required="eye_state (open/closed/uncertain/not_assessable), "
                            "eye_detail (detailed/undetailed/missing/uncertain/not_assessable)",
        derivable_after_annotation="none new -- open/closed and detail are annotated directly "
                                    "because they are not reliably recoverable from bbox geometry alone",
        suitable_this_phase=True,
        rationale="Directly supports PSY_AR_EYES_CLOSED_003 (closed_eyes) and "
                   "EN_COMPILED_EYES_MISSING_DETAIL_020 (eyes_missing_or_undetailed) with a "
                   "narrow, operationally-defined state vocabulary -- no psychological "
                   "interpretation is annotated, only the drawn state of the eye itself.",
    ),
    AnnotationTargetTraceability(
        target_name="eye_wide_geometry_future_work",
        supported_rule_ids=("PSY_AR_EYES_WIDE_001",),
        required_observation="none new this phase -- eye bbox (above) is sufficient input",
        presence_sufficient=False,
        bbox_required=True,
        count_required=False,
        attribute_required="",
        derivable_after_annotation="'wide' can later be DERIVED as eye-bbox height/width or "
                                    "eye-bbox-area/face-bbox-area, compared deterministically "
                                    "against a population distribution once enough eye+face "
                                    "bboxes exist -- not annotated as a subjective label",
        suitable_this_phase=False,
        rationale="No new annotation target: PSY_AR_EYES_WIDE_001 (wide_eyes) is left as a "
                   "future DERIVED geometric computation on top of the eye bbox this phase "
                   "already collects, once enough support exists for a population-relative "
                   "threshold. Listed separately so it is not silently forgotten.",
    ),
    AnnotationTargetTraceability(
        target_name="eye_stern_postponed",
        supported_rule_ids=("PSY_AR_EYES_STERN_002",),
        required_observation="unresolved -- no operational definition identified",
        presence_sufficient=False,
        bbox_required=False,
        count_required=False,
        attribute_required="",
        derivable_after_annotation="none identified",
        suitable_this_phase=False,
        rationale="'stern_eyes' has no defensible geometric or drawn-appearance proxy "
                   "identified this phase (unlike open/closed, which is a binary drawn "
                   "state). Postponed, not annotated, not derived -- would need a separate "
                   "operational-definition design pass before any annotation is attempted.",
    ),
    AnnotationTargetTraceability(
        target_name="mouth",
        supported_rule_ids=("EN_COMPILED_MISSING_MOUTH_036",),
        required_observation="presence, bbox (region_visible for the omission judgment)",
        presence_sufficient=False,
        bbox_required=True,
        count_required=False,
        attribute_required="",
        derivable_after_annotation="a future geometric 'mouth_curve_direction' "
                                    "(upturned/downturned/straight/absent) could be DERIVED "
                                    "later via classical-CV curve analysis on the mouth bbox "
                                    "region, without a new annotation pass -- see "
                                    "EN_COMPILED_FACE_EXPRESSION_021 entry below for why this "
                                    "is not annotated as a label this phase",
        suitable_this_phase=True,
        rationale="'mouth' does not exist in the Phase 2B/2C.1 object ontology. Presence+bbox "
                   "is the minimum evidence-rule_engine.py's evaluate_omission() primitive "
                   "needs (region_visible) -- no attribute is annotated this phase.",
    ),
    AnnotationTargetTraceability(
        target_name="hand",
        supported_rule_ids=("EN_COMPILED_MISSING_HANDS_035", "EN_COMPILED_EXAGGERATED_BODY_PARTS_037"),
        required_observation="existing Phase 2C.1 presence (reused, not re-litigated), NEW: "
                              "bbox per instance, instance count",
        presence_sufficient=False,
        bbox_required=True,
        count_required=True,
        attribute_required="",
        derivable_after_annotation="relative body-part size (see 'body_part_relative_size' "
                                    "below) is fully derived from this bbox + the person bbox, "
                                    "never annotated as 'exaggerated' directly",
        suitable_this_phase=True,
        rationale="'hand' is already a benchmarked, presence-annotated Phase 2C.1 class. "
                   "Bbox is the only new evidence needed: it satisfies "
                   "evaluate_omission()'s region_visible requirement for "
                   "EN_COMPILED_MISSING_HANDS_035, and supplies one half of the geometry "
                   "EN_COMPILED_EXAGGERATED_BODY_PARTS_037 needs (the other half is the "
                   "person bbox, below).",
    ),
    AnnotationTargetTraceability(
        target_name="face",
        supported_rule_ids=("EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038",),
        required_observation="existing Phase 2C.1 presence (reused), NEW: bbox per instance",
        presence_sufficient=False,
        bbox_required=True,
        count_required=True,
        attribute_required="",
        derivable_after_annotation="isolation (a spatial relationship between multiple "
                                    "detected figures) is derived later from face+person "
                                    "bboxes across a drawing -- Phase 2C.3 Stage D scope, "
                                    "explicitly out of scope this phase",
        suitable_this_phase=True,
        rationale="EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038 is a compound rule blocked at "
                   "its weakest link (the 'isolation' spatial-relationship component, which "
                   "this phase does not build). Face bbox is collected now as forward-looking "
                   "infrastructure -- cheap given 'face' is already presence-annotated -- but "
                   "does NOT by itself unlock this rule this phase.",
    ),
    AnnotationTargetTraceability(
        target_name="person",
        supported_rule_ids=(),  # reference structure, not tied to one rule_id directly
        required_observation="existing Phase 2C.1 presence (reused), NEW: bbox per instance "
                              "(whole-figure box)",
        presence_sufficient=False,
        bbox_required=True,
        count_required=True,
        attribute_required="",
        derivable_after_annotation="serves as the reference denominator for "
                                    "body_part_relative_size below and any future body-part-"
                                    "size rule",
        suitable_this_phase=True,
        rationale="Not gated on a single rule_id -- it is the REQUIRED REFERENCE STRUCTURE "
                   "for EN_COMPILED_EXAGGERATED_BODY_PARTS_037's relative-size denominator "
                   "(a part bbox alone cannot express 'relative to what'). Listed as its own "
                   "target because it needs its own bbox pass, not because it answers a rule "
                   "on its own.",
    ),
    AnnotationTargetTraceability(
        target_name="body_part_relative_size",
        supported_rule_ids=("EN_COMPILED_EXAGGERATED_BODY_PARTS_037",),
        required_observation="NONE new -- computed deterministically from hand bbox + person "
                              "bbox, both already collected above",
        presence_sufficient=False,
        bbox_required=True,  # consumes existing bboxes, does not add its own
        count_required=False,
        attribute_required="",
        derivable_after_annotation="area_ratio = hand_bbox_area / person_bbox_area, a plain "
                                    "float, computed post-hoc -- never annotated as "
                                    "'exaggerated' by a human (explicit instruction: annotate "
                                    "geometry, derive the judgment)",
        suitable_this_phase=True,
        rationale="NOT an annotation target -- a derived computation over the hand and person "
                   "bboxes above. Listed explicitly so the traceability table shows exactly "
                   "how EN_COMPILED_EXAGGERATED_BODY_PARTS_037 gets its evidence without a "
                   "human ever labeling a part 'exaggerated'.",
    ),
    AnnotationTargetTraceability(
        target_name="face_expression_postponed",
        supported_rule_ids=("EN_COMPILED_FACE_EXPRESSION_021",),
        required_observation="none annotated this phase",
        presence_sufficient=False,
        bbox_required=False,
        count_required=False,
        attribute_required="",
        derivable_after_annotation="a geometric 'mouth_curve_direction' proxy (see 'mouth' "
                                    "above) could later be DERIVED from the mouth bbox via "
                                    "classical-CV curve-shape analysis, without any new "
                                    "annotation pass",
        suitable_this_phase=False,
        rationale="POSTPONED. The source rule's own text ('Smiling can suggest positive "
                   "tone; sad, tense, or frightened faces can suggest distress') is framed in "
                   "emotion-adjacent language that sits too close to the project's explicitly "
                   "forbidden Angry/Fear/Happy/Sad vocabulary for a defensible operational "
                   "annotation protocol to be designed safely this phase -- even a "
                   "'geometric-only' taxonomy risks priming an annotator toward affect "
                   "judgment. A fully geometric alternative (mouth curve direction, derived "
                   "post-hoc from pixels, never labeled by a human) exists as future work and "
                   "does not require a new annotation pass at all.",
    ),
]


def to_rows() -> list[dict]:
    rows = []
    for t in RULE_ANNOTATION_TRACEABILITY:
        d = asdict(t)
        d["supported_rule_ids"] = ";".join(d["supported_rule_ids"])
        rows.append(d)
    return rows
