"""FeatureObservation: one source's claim about one feature on one case.

Deliberately vendor/model-neutral: nothing here names a specific detector,
VLM provider, or classifier. `source` is a free-form label
("existing_cv", "vlm_a", "human_reviewer", ...) so the same schema serves
a deterministic OpenCV measurement, a future VLM call, and a clinician's
manual correction without a schema change.

The one invariant this module exists to enforce, mechanically, in one
place: **`assessable=False` must never carry a value that looks like a
measurement.** `not detected` and `confirmed absent` are different
claims, and conflating them is exactly the failure mode the task calls
out for omission rules (missing hands/feet/eyes). A `FeatureObservation`
that violates this raises immediately in `__post_init__`, mirroring the
same-shaped invariant already enforced by `evidence_schema.EvidenceItem`
in the existing evidence pipeline.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    coordinate_space: str = "normalized_image"  # vs "pixel"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureObservation:
    feature_id: str
    value: Any
    value_type: str  # one of feature_schema.FeatureType.ALL
    assessable: bool
    source: str  # e.g. "existing_cv", "vlm_a", "human_reviewer"
    not_assessable_reason: str | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    prompt_version: str | None = None  # for prompt-driven (VLM) sources only
    raw_confidence: float | None = None
    calibrated_confidence: float | None = None
    bbox: BoundingBox | None = None
    mask_reference: str | None = None
    evidence_region: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.assessable:
            if self.value is not None:
                raise ValueError(
                    f"FeatureObservation {self.feature_id!r}: assessable=False but "
                    f"value is {self.value!r}. Unassessable evidence must never carry "
                    "a value that could be mistaken for a real measurement."
                )
            if not self.not_assessable_reason:
                raise ValueError(
                    f"FeatureObservation {self.feature_id!r}: assessable=False requires "
                    "a non-empty not_assessable_reason."
                )
        for name, conf in (("raw_confidence", self.raw_confidence), ("calibrated_confidence", self.calibrated_confidence)):
            if conf is not None and not (0.0 <= conf <= 1.0):
                raise ValueError(f"FeatureObservation {self.feature_id!r}: {name} {conf!r} out of [0, 1]")
        if not self.source:
            raise ValueError(f"FeatureObservation {self.feature_id!r}: source is required.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data


def not_assessable(
    feature_id: str,
    *,
    value_type: str,
    source: str,
    reason: str,
    extractor_name: str | None = None,
    extractor_version: str | None = None,
) -> FeatureObservation:
    """Construct an explicit not-assessable observation. Prefer this to
    hand-building `FeatureObservation(assessable=False, ...)` so the
    call site cannot forget `not_assessable_reason`."""
    return FeatureObservation(
        feature_id=feature_id,
        value=None,
        value_type=value_type,
        assessable=False,
        source=source,
        not_assessable_reason=reason,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
    )
