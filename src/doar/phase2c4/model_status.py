"""Phase 2C.4A correction: records each benchmarked model's status as a
*configuration* finding, not a claim about the model architecture's
general capability -- specifically for Florence-2, whose
`<OPEN_VOCABULARY_DETECTION>` task returned a positive detection for
almost every queried class on almost every image (see
PHASE2C4A_VALIDATION_CORRECTION_REPORT.md section 3). That result means
"Florence-2-base + the <OPEN_VOCABULARY_DETECTION> task + this benchmark's
bare-noun-phrase prompt" produced no discriminative presence signal on
this dataset -- it does not mean "Florence-2 cannot do open-vocabulary
detection." A different task head (e.g. <CAPTION_TO_PHRASE_GROUNDING>) or
checkpoint was never tested and is recorded as open future work, not ruled
out.
"""
from __future__ import annotations

from dataclasses import dataclass

# Phrasing intentionally narrow: never "Florence-2 is poor/incapable",
# always "this configuration did not produce discriminative evidence" --
# a regression test greps for the forbidden broader phrasing.
FORBIDDEN_GENERALIZING_PHRASES = (
    "florence-2 is poor",
    "florence-2 is incapable",
    "florence-2 cannot",
    "florence-2 does not work",
    "florence-2 is not capable",
)


@dataclass(frozen=True)
class ModelConfigurationStatus:
    model: str
    configuration: str
    status: str
    rationale: str
    future_work: str


MODEL_CONFIGURATION_STATUS = [
    ModelConfigurationStatus(
        model="owlv2",
        configuration="google/owlv2-base-patch16-ensemble, threshold=0.1, prompt='a {cls}'",
        status="usable_this_session",
        rationale="Discriminative on 6/10 dev-cohort classes (balanced accuracy > 0.6); "
                   "highest dev-cohort macro balanced accuracy of the four candidates.",
        future_work="Calibration sweep (this correction pass) to raise precision on "
                     "over-firing classes (heart, star, circle).",
    ),
    ModelConfigurationStatus(
        model="grounding_dino",
        configuration="IDEA-Research/grounding-dino-tiny, threshold=0.25/0.25, "
                       "prompt='person. face. hand. ...'",
        status="usable_as_high_recall_candidate_generator",
        rationale="High recall on person/face/animal but precision collapses below 0.25 "
                   "on 6/10 classes -- not trustworthy as an unreviewed evidence source, "
                   "but a strong proposal generator for a human-reviewed annotation pass.",
        future_work="Model-assisted annotation workflow (design only, section 9 of the "
                     "original PHASE2C4_DETECTOR_BENCHMARK_REPORT.md).",
    ),
    ModelConfigurationStatus(
        model="florence2",
        configuration="microsoft/Florence-2-base, <OPEN_VOCABULARY_DETECTION> task, "
                       "prompt='{cls}' (bare noun, one call per class)",
        status="failed_inconclusive_for_current_use",
        rationale="This configuration returned a positive detection for nearly every "
                   "queried class on nearly every image (macro balanced accuracy ~= 0.50, "
                   "chance level) -- no discriminative presence/absence signal was "
                   "produced. This is a finding about this configuration, not a general "
                   "claim about the Florence-2 architecture's capability.",
        future_work="A small (5-10 dev-image, no locked-test) sanity check comparing "
                     "documented Florence-2 task formulations (e.g. "
                     "<CAPTION_TO_PHRASE_GROUNDING> instead of "
                     "<OPEN_VOCABULARY_DETECTION>) if scientifically justified -- not run "
                     "this session, not scheduled without separate approval.",
    ),
    ModelConfigurationStatus(
        model="yolo_world",
        configuration="yolov8s-worldv2.pt (Ultralytics packaging), conf=0.1, "
                       "prompt='{cls}' via set_classes(...)",
        status="usable_but_low_recall_this_configuration",
        rationale="Precise where it fires (0.8-1.0 on person/face/hand/tree) but confidence "
                   "scores sit far below the documented default threshold on this domain "
                   "(observed max ~0.05 at a diagnostic conf=0.001) -- most classes never "
                   "fire at conf=0.1.",
        future_work="A future recall-boost pass at a lower, separately-justified threshold "
                     "-- not attempted here (no new inference in this correction pass, and "
                     "the original protocol's fixed-threshold rule was never violated).",
    ),
]


def to_rows() -> list[dict]:
    return [vars(s) for s in MODEL_CONFIGURATION_STATUS]
