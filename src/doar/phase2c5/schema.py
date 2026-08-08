"""Phase 2C.5 part-annotation schema -- a NEW, additive schema/store,
versioned separately from src/doar/phase2c1/schema.py (SCHEMA_VERSION
below), which this module never imports for writing and never mutates.
Backward-compatible in spirit, not by sharing a file: same status
vocabulary (present/absent/uncertain/not_assessable), same
annotator_type/review_status conventions
(src/doar/phase2c1/schema.py::ANNOTATOR_TYPES/REVIEW_STATUSES, reused
directly), same atomic-write CSV store discipline (store.py).

One row = one (pilot_id, target_name, annotator_id) judgment, exactly
Phase 2C.1's own keying convention -- but because a target can have
multiple instances (two eyes, several hands), each judgment carries a
LIST of `PartInstance` boxes (serialized as one JSON cell,
`instances_json`) rather than a single bbox column. A `status != present`
row always carries zero instances, mirroring Phase 2C.1's
instance_count invariant exactly.

Every instance carries its own provenance (`bbox_source` +, when
model-proposed, the exact model/checkpoint/prompt/threshold/timestamp
that produced it) -- a proposal is data about a model, never ground
truth, until a human explicitly accepts or edits it (see BBOX_SOURCES
docstring below).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..phase2c1.schema import ANNOTATOR_TYPES, REVIEW_STATUSES
from .ontology import (
    ATTRIBUTE_ALLOWED_VALUES,
    BBOX_SOURCES,
    OBJECT_STATUSES,
    PART_ONTOLOGY_VERSION,
    PART_TARGETS,
    TARGET_ALLOWED_ATTRIBUTE_KEYS,
)

SCHEMA_VERSION = "phase2c5_part_annotation_schema_v1"

CSV_FIELDS = [
    "annotation_id", "pilot_id", "target_name", "status",
    "annotator_id", "annotator_type", "annotation_timestamp",
    "instances_json", "attributes_json", "uncertainty_reason",
    "review_status", "reviewer_id", "review_timestamp", "notes",
    "part_ontology_version", "schema_version",
]


def make_part_annotation_id(pilot_id: str, target_name: str, annotator_id: str) -> str:
    return f"{pilot_id}__{target_name}__{annotator_id}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PartInstance:
    """One drawn part instance and its provenance. `bbox` is normalized
    (x, y, w, h) in [0, 1], the same convention Phase 2C.1's own bbox
    column already uses."""
    instance_index: int
    bbox: tuple[float, float, float, float]
    bbox_source: str  # BBOX_SOURCES
    proposal_model: str = ""
    proposal_checkpoint: str = ""
    proposal_revision: str = ""
    proposal_prompt: str = ""
    proposal_threshold: float | None = None
    proposal_timestamp: str = ""

    def __post_init__(self) -> None:
        if self.bbox_source not in BBOX_SOURCES:
            raise ValueError(f"bbox_source {self.bbox_source!r} not in {sorted(BBOX_SOURCES)}")
        if len(self.bbox) != 4 or any(not (0.0 <= v <= 1.0) for v in self.bbox):
            raise ValueError(f"bbox {self.bbox!r} must be 4 normalized values in [0, 1]")
        x, y, w, h = self.bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"bbox {self.bbox!r} must have positive width/height")
        if self.bbox_source == "model_proposed" and not self.proposal_model:
            raise ValueError("bbox_source='model_proposed' requires a non-empty proposal_model")
        if self.bbox_source in ("human_drawn",) and self.proposal_model:
            raise ValueError(
                f"bbox_source={self.bbox_source!r} (never touched a model proposal) must not "
                f"carry a proposal_model ({self.proposal_model!r}) -- provenance would be false")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PartInstance":
        return PartInstance(
            instance_index=int(d["instance_index"]),
            bbox=tuple(float(v) for v in d["bbox"]),
            bbox_source=d["bbox_source"],
            proposal_model=d.get("proposal_model", "") or "",
            proposal_checkpoint=d.get("proposal_checkpoint", "") or "",
            proposal_revision=d.get("proposal_revision", "") or "",
            proposal_prompt=d.get("proposal_prompt", "") or "",
            proposal_threshold=(None if d.get("proposal_threshold") in (None, "")
                                 else float(d["proposal_threshold"])),
            proposal_timestamp=d.get("proposal_timestamp", "") or "",
        )


@dataclass(frozen=True)
class PartAnnotationRecord:
    pilot_id: str
    target_name: str
    status: str  # OBJECT_STATUSES
    annotator_id: str
    annotation_timestamp: str
    annotator_type: str = "human"
    instances: tuple[PartInstance, ...] = ()
    attributes: dict[str, str] = field(default_factory=dict)
    uncertainty_reason: str = ""
    review_status: str = "unreviewed"
    reviewer_id: str = ""
    review_timestamp: str = ""
    notes: str = ""
    part_ontology_version: str = PART_ONTOLOGY_VERSION
    schema_version: str = SCHEMA_VERSION

    @property
    def annotation_id(self) -> str:
        return make_part_annotation_id(self.pilot_id, self.target_name, self.annotator_id)

    def __post_init__(self) -> None:
        if self.target_name not in PART_TARGETS:
            raise ValueError(f"{self.target_name!r} is not a Phase 2C.5 part target {PART_TARGETS}")
        if self.status not in OBJECT_STATUSES:
            raise ValueError(f"status {self.status!r} not in {sorted(OBJECT_STATUSES)}")
        if self.status == "present" and len(self.instances) < 1:
            raise ValueError(
                f"{self.pilot_id}/{self.target_name}: status='present' requires >= 1 instance")
        if self.status != "present" and len(self.instances) != 0:
            raise ValueError(
                f"{self.pilot_id}/{self.target_name}: status={self.status!r} but "
                f"{len(self.instances)} instance(s) present -- a non-present status must never "
                "carry boxes (this would look like confirmed evidence instead of "
                "absence/uncertainty/not-assessable).")
        indices = [inst.instance_index for inst in self.instances]
        if indices != list(range(len(indices))):
            raise ValueError(f"instance_index values must be a contiguous 0..n-1 sequence, got {indices}")
        allowed_attrs = TARGET_ALLOWED_ATTRIBUTE_KEYS.get(self.target_name, frozenset())
        unknown = set(self.attributes) - allowed_attrs
        if unknown:
            raise ValueError(f"{self.target_name!r} does not allow attribute(s) {unknown} "
                              f"(allowed: {sorted(allowed_attrs)})")
        for key, value in self.attributes.items():
            allowed_values = ATTRIBUTE_ALLOWED_VALUES[key]
            if value not in allowed_values:
                raise ValueError(f"attribute {key}={value!r} not in {sorted(allowed_values)}")
        if self.annotator_type not in ANNOTATOR_TYPES:
            raise ValueError(f"annotator_type {self.annotator_type!r} not in {sorted(ANNOTATOR_TYPES)}")
        if not self.annotator_id.strip():
            raise ValueError("annotator_id must be non-empty -- an annotation can never be unattributed")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status {self.review_status!r} not in {sorted(REVIEW_STATUSES)}")

    def to_row(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "pilot_id": self.pilot_id,
            "target_name": self.target_name,
            "status": self.status,
            "annotator_id": self.annotator_id,
            "annotator_type": self.annotator_type,
            "annotation_timestamp": self.annotation_timestamp,
            "instances_json": json.dumps([inst.to_dict() for inst in self.instances]),
            "attributes_json": json.dumps(self.attributes, sort_keys=True),
            "uncertainty_reason": self.uncertainty_reason,
            "review_status": self.review_status,
            "reviewer_id": self.reviewer_id,
            "review_timestamp": self.review_timestamp,
            "notes": self.notes,
            "part_ontology_version": self.part_ontology_version,
            "schema_version": self.schema_version,
        }


def record_from_row(row: dict[str, str]) -> PartAnnotationRecord:
    instances_raw = json.loads(row.get("instances_json") or "[]")
    attributes_raw = json.loads(row.get("attributes_json") or "{}")
    return PartAnnotationRecord(
        pilot_id=row["pilot_id"],
        target_name=row["target_name"],
        status=row["status"],
        annotator_id=row["annotator_id"],
        annotation_timestamp=row["annotation_timestamp"],
        annotator_type=row.get("annotator_type") or "human",
        instances=tuple(PartInstance.from_dict(d) for d in instances_raw),
        attributes=dict(attributes_raw),
        uncertainty_reason=row.get("uncertainty_reason", "") or "",
        review_status=row.get("review_status") or "unreviewed",
        reviewer_id=row.get("reviewer_id", "") or "",
        review_timestamp=row.get("review_timestamp", "") or "",
        notes=row.get("notes", "") or "",
        part_ontology_version=row.get("part_ontology_version") or PART_ONTOLOGY_VERSION,
        schema_version=row.get("schema_version") or SCHEMA_VERSION,
    )
