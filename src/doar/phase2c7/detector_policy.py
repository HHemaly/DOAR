"""Phase 2C.7 Stage 5: class-by-class visual detector policy, combining
Phase 2C.4/2C.4A's frozen object-detector benchmark (never re-run, never
modified) with this phase's own new eye evaluation (§ eye_evaluation.py)
into ONE table every downstream consumer reads.

Three statuses (never a fourth, never blurred):

- VALIDATED_AUTOMATIC: real human-reviewed ground truth exists at or
  above `MIN_POSITIVE_SUPPORT` (not `low_support`-flagged), AND at least
  one candidate model reaches a calibrated (non-abstaining) operating
  point at >=0.6 precision, AND that model's balanced accuracy is >=0.6 --
  eligible to enter the validated evidence pipeline.
- EXPERIMENTAL_AUTOMATIC: the detector produces real signal (a default-
  threshold detection with above-chance balanced accuracy, OR a
  calibrated point exists but with too-thin support, OR only a small
  qualitative feasibility read exists) but does not clear the
  VALIDATED bar -- may be stored/displayed as experimental evidence, but
  MUST NOT activate a psychological conclusion (enforced downstream by
  `visual_detector.py` and tested).
- DISABLED: no model reaches above-chance balanced accuracy at any tested
  threshold -- not used at all.

Every object-class entry below (person/face/hand/animal/house/tree/
heart/star/circle/vehicle) is sourced directly from
`artifacts/phase2c4a/calibration_frozen_operating_points.csv` and
`artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv`
(dev_eligible_excl_test cohort only, never the locked-test split) --
exact figures cited in each entry's `rationale`, not invented. The eye
entry is populated by `build_eye_policy_entry` from this phase's own
dev-split evaluation (never the eye holdout). Mouth has no human-reviewed
ground truth in the new part-annotation schema at all yet (confirmed by
this phase's Stage 1 export inspection: 0 mouth rows) -- classified
EXPERIMENTAL_AUTOMATIC on the Phase 2C.5 15-image feasibility-pilot
qualitative read alone.
"""
from __future__ import annotations

from dataclasses import dataclass

VALIDATED_AUTOMATIC = "VALIDATED_AUTOMATIC"
EXPERIMENTAL_AUTOMATIC = "EXPERIMENTAL_AUTOMATIC"
DISABLED = "DISABLED"

MIN_BALANCED_ACCURACY_FOR_VALIDATION = 0.6
MIN_PRECISION_FOR_VALIDATION = 0.6  # same bar Phase 2C.4A's own calibration used


@dataclass(frozen=True)
class TargetPolicyEntry:
    target: str
    status: str
    best_model: str | None
    best_model_checkpoint: str | None
    prompt: str | None
    threshold: float | None
    precision: float | None
    recall: float | None
    balanced_accuracy: float | None
    n_ground_truth_present: int | None
    localization_validated: bool
    rationale: str
    related_rule_ids: tuple[str, ...]
    allowed_downstream_usage: str


# Object-class entries: sourced from Phase 2C.4A's frozen, unmodified
# artifacts (never re-evaluated or re-thresholded here).
OBJECT_CLASS_POLICY: dict[str, TargetPolicyEntry] = {
    "person": TargetPolicyEntry(
        target="person", status=VALIDATED_AUTOMATIC, best_model="grounding_dino_object_classes",
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny", prompt="person. face. hand. ...",
        threshold=0.25, precision=0.84, recall=0.808, balanced_accuracy=0.713,
        n_ground_truth_present=52, localization_validated=False,
        rationale="Phase 2C.4A calibration_frozen_operating_points.csv: both OWLv2 (prec=1.0, "
                   "bal_acc=0.654) and Grounding DINO (prec=0.84, bal_acc=0.713) calibrate to a "
                   "non-abstaining, precision>=0.6 operating point on dev_eligible_excl_test (n=52).",
        related_rule_ids=(), allowed_downstream_usage="validated evidence pipeline (presence only, "
        "no localization ground truth exists for this class)",
    ),
    "face": TargetPolicyEntry(
        target="face", status=VALIDATED_AUTOMATIC, best_model="grounding_dino_object_classes",
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny", prompt="person. face. hand. ...",
        threshold=0.25, precision=0.978, recall=0.703, balanced_accuracy=0.796,
        n_ground_truth_present=64, localization_validated=False,
        rationale="Both models calibrate at high precision (0.96-0.98); Grounding DINO bal_acc=0.796, "
                   "n=64, largest object-class support in Phase 2C.4A.",
        related_rule_ids=("EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038",),
        allowed_downstream_usage="validated evidence pipeline (presence only)",
    ),
    "hand": TargetPolicyEntry(
        target="hand", status=VALIDATED_AUTOMATIC, best_model="grounding_dino_object_classes",
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny", prompt="person. face. hand. ...",
        threshold=0.25, precision=0.909, recall=0.263, balanced_accuracy=0.617,
        n_ground_truth_present=38, localization_validated=False,
        rationale="Both models calibrate at high precision (0.80-0.91) but LOW recall (0.21-0.26) -- "
                   "validated as a precision-oriented 'when it fires, trust it' signal, not a "
                   "comprehensive detector; real misses are expected. Grounding DINO bal_acc=0.617 "
                   "clears the 0.6 floor; OWLv2's 0.577 does not (best-model rule applies).",
        related_rule_ids=("EN_COMPILED_MISSING_HANDS_035", "EN_COMPILED_EXAGGERATED_BODY_PARTS_037"),
        allowed_downstream_usage="validated evidence pipeline (presence only, precision-oriented -- "
        "low recall means absence-of-detection must NOT be read as absence-of-hand)",
    ),
    "tree": TargetPolicyEntry(
        target="tree", status=VALIDATED_AUTOMATIC, best_model="grounding_dino_object_classes",
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny", prompt="person. face. hand. ...",
        threshold=0.5, precision=0.636, recall=0.538, balanced_accuracy=0.735,
        n_ground_truth_present=13, localization_validated=False,
        rationale="Grounding DINO calibrates at threshold 0.5 (raised from 0.25), bal_acc=0.735, n=13.",
        related_rule_ids=("EN_COMPILED_TREE_024",), allowed_downstream_usage="validated evidence pipeline",
    ),
    "house": TargetPolicyEntry(
        target="house", status=VALIDATED_AUTOMATIC, best_model="grounding_dino_object_classes",
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny", prompt="person. face. hand. ...",
        threshold=0.4, precision=0.692, recall=0.75, balanced_accuracy=0.842,
        n_ground_truth_present=12, localization_validated=False,
        rationale="Grounding DINO calibrates at threshold 0.4 (raised from 0.25), bal_acc=0.842, best "
                   "of any object class; n=12.",
        related_rule_ids=("EN_COMPILED_HOUSE_023",), allowed_downstream_usage="validated evidence pipeline",
    ),
    "heart": TargetPolicyEntry(
        target="heart", status=EXPERIMENTAL_AUTOMATIC, best_model="owlv2_object_classes",
        best_model_checkpoint="google/owlv2-base-patch16-ensemble", prompt="a heart",
        threshold=0.35, precision=1.0, recall=0.25, balanced_accuracy=0.625,
        n_ground_truth_present=4, localization_validated=False,
        rationale="OWLv2 calibrates (prec=1.0) but Phase 2C.4A's own pipeline explicitly flags "
                   "low_support=True (n_present=4 < MIN_POSITIVE_SUPPORT=5); Grounding DINO abstains "
                   "entirely at every tested threshold. Too thin to trust generalization.",
        related_rule_ids=("PSY_AR_HEARTS_013",),
        allowed_downstream_usage="Technical View / experimental evidence only -- must not activate a "
        "psychological conclusion",
    ),
    "animal": TargetPolicyEntry(
        target="animal", status=EXPERIMENTAL_AUTOMATIC, best_model="grounding_dino_object_classes",
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny", prompt="person. face. hand. ...",
        threshold=0.25, precision=0.195, recall=0.8, balanced_accuracy=0.638,
        n_ground_truth_present=10, localization_validated=False,
        rationale="Grounding DINO's DEFAULT (uncalibrated) threshold shows real signal (bal_acc=0.638), "
                   "but Phase 2C.4A's calibration sweep found NO threshold reaching 0.6 precision (both "
                   "models abstain) -- real but not trustworthy enough for validated evidence.",
        related_rule_ids=("EN_COMPILED_ANIMAL_CHOICE_GENERAL_022",),
        allowed_downstream_usage="Technical View / experimental evidence only",
    ),
    "star": TargetPolicyEntry(
        target="star", status=EXPERIMENTAL_AUTOMATIC, best_model="grounding_dino_object_classes",
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny", prompt="person. face. hand. ...",
        threshold=0.25, precision=0.091, recall=0.6, balanced_accuracy=0.579,
        n_ground_truth_present=5, localization_validated=False,
        rationale="Both models abstain at calibration; default-threshold bal_acc sits just under chance "
                   "for OWLv2 (0.549) and modestly above for Grounding DINO (0.579) -- weak, unresolved.",
        related_rule_ids=("PSY_AR_STARS_009",), allowed_downstream_usage="Technical View / experimental "
        "evidence only",
    ),
    "circle": TargetPolicyEntry(
        target="circle", status=DISABLED, best_model=None, best_model_checkpoint=None, prompt=None,
        threshold=None, precision=None, recall=None, balanced_accuracy=0.32, n_ground_truth_present=4,
        localization_validated=False,
        rationale="Both models abstain at calibration AND both sit at or below chance on dev "
                   "(Grounding DINO bal_acc=0.321, OWLv2=0.391) -- the one object class with "
                   "below-chance performance from every model tried across Phase 2C.4/2C.4A.",
        related_rule_ids=("PSY_AR_CIRCLES_011",), allowed_downstream_usage="not used",
    ),
    "vehicle": TargetPolicyEntry(
        target="vehicle", status=DISABLED, best_model=None, best_model_checkpoint=None, prompt=None,
        threshold=None, precision=None, recall=None, balanced_accuracy=0.5, n_ground_truth_present=3,
        localization_validated=False,
        rationale="Both models abstain at calibration, zero recall from every model at every tested "
                   "threshold (bal_acc exactly 0.5, chance), and n_present=3 is below "
                   "MIN_POSITIVE_SUPPORT=5.",
        related_rule_ids=("PSY_AR_TRANSPORT_012",), allowed_downstream_usage="not used",
    ),
    "mouth": TargetPolicyEntry(
        target="mouth", status=EXPERIMENTAL_AUTOMATIC, best_model="grounding_dino_parts",
        best_model_checkpoint="IDEA-Research/grounding-dino-tiny", prompt="eye. mouth. hand. face. person.",
        threshold=0.25, precision=None, recall=None, balanced_accuracy=None, n_ground_truth_present=0,
        localization_validated=False,
        rationale="Zero human-reviewed mouth rows exist in the Phase 2C.5/2C.6 part-annotation store "
                   "(confirmed this phase's Stage 1 export inspection). Only the Phase 2C.5 15-image "
                   "feasibility pilot's qualitative read exists (both models excellent on 2 clean "
                   "drawings, false-positive on a hair ribbon / clothing pattern in a third) -- real "
                   "signal, no formal metric.",
        related_rule_ids=("EN_COMPILED_MISSING_MOUTH_036",),
        allowed_downstream_usage="Technical View / experimental evidence only",
    ),
    "person_part_reference": TargetPolicyEntry(
        target="person_part_reference", status=EXPERIMENTAL_AUTOMATIC, best_model=None,
        best_model_checkpoint=None, prompt=None, threshold=None, precision=None, recall=None,
        balanced_accuracy=None, n_ground_truth_present=0, localization_validated=False,
        rationale="'person' bbox as a reference structure for body-part relative-size rules -- presence "
                   "is validated (see 'person' entry) but no human-reviewed person BOX ground truth "
                   "exists yet to validate localization, which is what this specific downstream use "
                   "(a relative-size denominator) actually needs.",
        related_rule_ids=("EN_COMPILED_EXAGGERATED_BODY_PARTS_037",),
        allowed_downstream_usage="Technical View / experimental evidence only (localization, not presence)",
    ),
}


def build_eye_policy_entry(*, status: str, best_model: str, best_model_checkpoint: str, prompt: str,
                            threshold: float, precision: float | None, recall: float | None,
                            balanced_accuracy: float | None, n_ground_truth_present: int,
                            localization_validated: bool, rationale: str) -> TargetPolicyEntry:
    """Populated by scripts/phase2c7_generate_artifacts.py from this
    phase's own real dev-split evaluation (never the holdout) -- never a
    literal default here, so a caller cannot accidentally ship a
    placeholder eye policy."""
    return TargetPolicyEntry(
        target="eye", status=status, best_model=best_model, best_model_checkpoint=best_model_checkpoint,
        prompt=prompt, threshold=threshold, precision=precision, recall=recall,
        balanced_accuracy=balanced_accuracy, n_ground_truth_present=n_ground_truth_present,
        localization_validated=localization_validated, rationale=rationale,
        related_rule_ids=("PSY_AR_EYES_CLOSED_003", "EN_COMPILED_EYES_MISSING_DETAIL_020"),
        allowed_downstream_usage=(
            "validated evidence pipeline (presence and localization)" if status == VALIDATED_AUTOMATIC
            else "Technical View / experimental evidence only" if status == EXPERIMENTAL_AUTOMATIC
            else "not used"),
    )


def full_policy(eye_entry: TargetPolicyEntry) -> dict[str, TargetPolicyEntry]:
    policy = dict(OBJECT_CLASS_POLICY)
    policy["eye"] = eye_entry
    return policy


def to_rows(policy: dict[str, TargetPolicyEntry]) -> list[dict]:
    rows = []
    for entry in policy.values():
        rows.append({
            "target": entry.target, "status": entry.status, "best_model": entry.best_model or "",
            "best_model_checkpoint": entry.best_model_checkpoint or "", "prompt": entry.prompt or "",
            "threshold": entry.threshold, "precision": entry.precision, "recall": entry.recall,
            "balanced_accuracy": entry.balanced_accuracy,
            "n_ground_truth_present": entry.n_ground_truth_present,
            "localization_validated": entry.localization_validated,
            "related_rule_ids": ";".join(entry.related_rule_ids),
            "allowed_downstream_usage": entry.allowed_downstream_usage, "rationale": entry.rationale,
        })
    return rows
