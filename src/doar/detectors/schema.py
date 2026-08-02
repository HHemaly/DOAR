"""
schema.py -- the structured contract any future Phase-3 detector must return.

Purely a data contract; no detector implementation lives here. Every field
that matters for validation and honest reporting is required, not optional,
so a detector cannot silently omit the information needed to judge it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectorResult:
    """One detected instance from one detector run on one image.

    `validated` MUST be False until the detector has cleared its documented
    acceptance threshold (PHASE3_DETECTOR_EVALUATION_PLAN.md Section 6) on the
    held-out annotation sample AND been explicitly approved -- see
    DECISION_LOG.md. A detector returning validated=True without both of
    those having happened is a bug, not a feature; rules.py must never trust
    this field without independently confirming it against the approval
    record.
    """

    detector_name: str
    detector_version: str
    domain: str                    # e.g. "photo", "sketch", "classical_cv"
    task_type: str                 # "detection" | "classification" | "keypoints" | "segmentation"
    class_label: str
    confidence: float
    bbox: tuple[float, float, float, float] | None   # (x_min, y_min, x_max, y_max), None for pure classification
    component_id: str | None       # links back to DOAR's existing connected-component extraction, if used
    validated: bool = False
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.validated and not self.limitations:
            raise ValueError(
                "A validated=True DetectorResult must still carry its known "
                "limitations -- 'validated' means 'cleared its documented "
                "acceptance bar', never 'perfect' or 'scientifically proven'."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "domain": self.domain,
            "task_type": self.task_type,
            "class_label": self.class_label,
            "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "component_id": self.component_id,
            "validated": self.validated,
            "limitations": self.limitations,
        }
