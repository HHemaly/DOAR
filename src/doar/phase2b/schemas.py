"""Phase 2B detector evidence record -- raw visual-detection evidence
only, never a psychological claim. See docs/PHASE2B_OBJECT_EVIDENCE_POLICY.md
for the hard constraints this schema exists to enforce mechanically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Mirrors the task's required status vocabulary -- deliberately distinct
# from evidence_schema.py's EVIDENCE_STATUSES (which is about trust in a
# measurement) because a detector's own semantics are different: "did the
# class appear" vs. "not_detected" (no evidence for it) vs. "uncertain"
# (score near the boundary) vs. "not_assessable" (image unsuitable) vs.
# "failed" (the detector itself errored).
DETECTOR_STATUSES = frozenset({"detected", "not_detected", "uncertain", "not_assessable", "failed"})

# A missing detection must never automatically mean the object is absent
# -- these statuses require a human or a stronger detector to resolve
# before being treated as a confirmed negative.
NON_CONCLUSIVE_DETECTOR_STATUSES = frozenset({"uncertain", "not_assessable", "failed"})


@dataclass(frozen=True)
class DetectorEvidenceRecord:
    evidence_id: str
    entity_id: str  # distinguishes multiple instances of the same class in one image
    class_name: str
    status: str
    confidence: float | None
    bounding_box: tuple[float, float, float, float] | None = None  # normalized x,y,w,h
    mask: str | None = None  # not used by either Phase 2B baseline; reserved for Phase 2C
    keypoints: list[tuple[float, float]] | None = None  # unused by either baseline; reserved
    source_model: str = "unspecified"
    model_version: str = "unspecified"
    checkpoint_hash: str | None = None  # None for zero-shot/classical (no fine-tuned checkpoint)
    preprocessing_version: str = "unspecified"
    limitations: list[str] = field(default_factory=list)
    validation_status: str = "pilot_unvalidated"
    human_review_status: str = "not_reviewed"

    def __post_init__(self) -> None:
        if self.status not in DETECTOR_STATUSES:
            raise ValueError(f"{self.evidence_id!r}: status {self.status!r} not in {sorted(DETECTOR_STATUSES)}")
        if self.status in NON_CONCLUSIVE_DETECTOR_STATUSES and self.confidence is not None:
            raise ValueError(
                f"{self.evidence_id!r}: status {self.status!r} is non-conclusive but carries a "
                f"confidence ({self.confidence!r}) -- this must never look like a real measurement.")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"{self.evidence_id!r}: confidence {self.confidence!r} out of [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_zero_shot_prediction(pilot_id: str, class_name: str, similarity: float, status: str,
                               *, model_name: str, model_version: str,
                               preprocessing_version: str) -> DetectorEvidenceRecord:
    """Builds one DetectorEvidenceRecord from a ZeroShotPrediction. Zero-shot
    similarity is reported as `confidence` only when status is `detected`
    or `not_detected` (conclusive) -- never for `uncertain`."""
    confidence = None if status in NON_CONCLUSIVE_DETECTOR_STATUSES else max(0.0, min(1.0, (similarity + 1) / 2))
    return DetectorEvidenceRecord(
        evidence_id=f"ev_p2b_{pilot_id}_{class_name}",
        entity_id=f"{pilot_id}_{class_name}_0",
        class_name=class_name,
        status=status,
        confidence=confidence,
        source_model="clip_zero_shot",
        model_version=f"{model_name}/{model_version}",
        checkpoint_hash=None,
        preprocessing_version=preprocessing_version,
        limitations=[
            "Zero-shot: never trained or fine-tuned on DOAR's drawing dataset.",
            "Evaluated on a 20-image pilot sample only -- not a validated detector.",
            "Presence-only signal; no relational, expression, or omission judgment.",
        ],
        validation_status="pilot_unvalidated",
        human_review_status="not_reviewed",
    )
