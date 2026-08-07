"""Phase 2C.1 annotation schema -- versioned separately from Phase 2B's
artifacts/phase2b/annotation_manifest.csv, which this package never writes to
or reads for anything except one-time provisional migration (workspace.py).

One row = one (pilot_id, class_name, annotator_id) judgment. A second,
independent annotator (e.g. a reviewer working blind, see workspace docs)
produces its own row under its own annotator_id -- never overwrites the
first. Review/adjudication metadata (review_status, reviewer_id,
review_timestamp, adjudication_status) lives on the ORIGINAL annotator's row
and is updated in place by store.apply_review_outcome(), not by creating a
new row.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from ..phase2b.ontology import ANNOTATION_STATUSES, CLASS_NAMES

# Bumped only if the field list or semantics below change -- never silently.
SCHEMA_VERSION = "phase2c1_annotation_schema_v1"

# Mirrors docs/OBJECT_EVIDENCE_ONTOLOGY.md's 10-class list, unchanged by
# Phase 2C.1 (no new class was added this phase).
ONTOLOGY_VERSION = "phase2b_ontology_v1_10class"

OBJECT_STATUSES = ANNOTATION_STATUSES  # {"present", "absent", "uncertain", "not_assessable"}

ANNOTATOR_TYPES = frozenset({"human"})

REVIEW_STATUSES = frozenset({"unreviewed", "in_review", "reviewed"})

# "not_applicable" until a second, independent judgment has actually been
# recorded and compared -- never invented ahead of a real comparison.
ADJUDICATION_STATUSES = frozenset({
    "not_applicable", "agreement", "disagreement_unresolved", "disagreement_resolved",
})

CSV_FIELDS = [
    "annotation_id", "pilot_id", "image_id", "source_image_group", "class_name",
    "status", "instance_count", "bbox", "partial_or_occluded", "uncertainty_reason",
    "annotator_id", "annotator_type", "annotation_timestamp",
    "review_status", "reviewer_id", "review_timestamp", "adjudication_status",
    "notes", "ontology_version", "source_manifest_version",
]


def make_annotation_id(pilot_id: str, class_name: str, annotator_id: str) -> str:
    """Deterministic primary key -- the same (pilot_id, class_name,
    annotator_id) triple always maps to the same annotation_id, which is
    exactly what store.py's upsert relies on for duplicate-row prevention."""
    return f"{pilot_id}__{class_name}__{annotator_id}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AnnotationRecord:
    pilot_id: str
    image_id: str
    source_image_group: str
    class_name: str
    status: str  # one of OBJECT_STATUSES -- required, no default: can never be silently "absent"
    annotator_id: str
    annotation_timestamp: str
    instance_count: int = 0
    bbox: tuple[float, float, float, float] | None = None  # normalized (x, y, w, h)
    partial_or_occluded: bool = False
    uncertainty_reason: str = ""
    annotator_type: str = "human"
    review_status: str = "unreviewed"
    reviewer_id: str = ""
    review_timestamp: str = ""
    adjudication_status: str = "not_applicable"
    notes: str = ""
    ontology_version: str = ONTOLOGY_VERSION
    source_manifest_version: str = ""

    @property
    def annotation_id(self) -> str:
        return make_annotation_id(self.pilot_id, self.class_name, self.annotator_id)

    def __post_init__(self) -> None:
        if self.class_name not in CLASS_NAMES:
            raise ValueError(f"{self.class_name!r} is not a Phase 2B/2C.1 ontology class")
        if self.status not in OBJECT_STATUSES:
            raise ValueError(f"status {self.status!r} not in {sorted(OBJECT_STATUSES)}")
        if self.status == "present" and self.instance_count < 1:
            raise ValueError(
                f"{self.pilot_id}/{self.class_name}: status='present' requires "
                f"instance_count >= 1, got {self.instance_count}")
        if self.status != "present" and self.instance_count != 0:
            raise ValueError(
                f"{self.pilot_id}/{self.class_name}: status={self.status!r} but "
                f"instance_count={self.instance_count} -- a non-present status must "
                "never carry a positive count (this would look like a confirmed "
                "measurement instead of absence/uncertainty/not-assessable).")
        if self.bbox is not None:
            if self.status != "present":
                raise ValueError(
                    f"{self.pilot_id}/{self.class_name}: a bounding box requires "
                    f"status='present', got {self.status!r}")
            if len(self.bbox) != 4 or any(not (0.0 <= v <= 1.0) for v in self.bbox):
                raise ValueError(f"bbox {self.bbox!r} must be 4 normalized values in [0, 1]")
            x, y, w, h = self.bbox
            if w <= 0 or h <= 0:
                raise ValueError(f"bbox {self.bbox!r} must have positive width/height")
        if self.annotator_type not in ANNOTATOR_TYPES:
            raise ValueError(f"annotator_type {self.annotator_type!r} not in {sorted(ANNOTATOR_TYPES)}")
        if not self.annotator_id.strip():
            raise ValueError("annotator_id must be non-empty -- an annotation can never be unattributed")
        if self.review_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status {self.review_status!r} not in {sorted(REVIEW_STATUSES)}")
        if self.adjudication_status not in ADJUDICATION_STATUSES:
            raise ValueError(
                f"adjudication_status {self.adjudication_status!r} not in {sorted(ADJUDICATION_STATUSES)}")
        if self.adjudication_status != "not_applicable" and self.review_status == "unreviewed":
            raise ValueError(
                "adjudication_status implies a review happened, but review_status is "
                "still 'unreviewed' -- inconsistent state")

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["annotation_id"] = self.annotation_id
        d["bbox"] = "" if self.bbox is None else ",".join(f"{v:.6f}" for v in self.bbox)
        d["partial_or_occluded"] = str(self.partial_or_occluded)
        return {k: d[k] for k in CSV_FIELDS}


def record_from_row(row: dict[str, str]) -> AnnotationRecord:
    """Inverse of AnnotationRecord.to_row(). Ignores the row's own
    'annotation_id' column (recomputed from pilot_id/class_name/annotator_id,
    which is always the source of truth, not whatever happened to be on disk)."""
    bbox_raw = (row.get("bbox") or "").strip()
    bbox = None
    if bbox_raw:
        parts = [float(v) for v in bbox_raw.split(",")]
        if len(parts) != 4:
            raise ValueError(f"malformed bbox column: {bbox_raw!r}")
        bbox = tuple(parts)
    return AnnotationRecord(
        pilot_id=row["pilot_id"],
        image_id=row["image_id"],
        source_image_group=row["source_image_group"],
        class_name=row["class_name"],
        status=row["status"],
        annotator_id=row["annotator_id"],
        annotation_timestamp=row["annotation_timestamp"],
        instance_count=int(row.get("instance_count") or 0),
        bbox=bbox,
        partial_or_occluded=str(row.get("partial_or_occluded", "False")).strip().lower() == "true",
        uncertainty_reason=row.get("uncertainty_reason", "") or "",
        annotator_type=row.get("annotator_type") or "human",
        review_status=row.get("review_status") or "unreviewed",
        reviewer_id=row.get("reviewer_id", "") or "",
        review_timestamp=row.get("review_timestamp", "") or "",
        adjudication_status=row.get("adjudication_status") or "not_applicable",
        notes=row.get("notes", "") or "",
        ontology_version=row.get("ontology_version") or ONTOLOGY_VERSION,
        source_manifest_version=row.get("source_manifest_version", "") or "",
    )


def validate_image_complete(rows: list[AnnotationRecord], pilot_id: str, annotator_id: str) -> list[str]:
    """Returns the sorted list of missing class names for one (pilot_id,
    annotator_id) pair -- empty means every one of the 10 ontology classes
    has a row from this annotator. Never raises; callers decide what an
    incomplete image means for their own workflow (e.g. the app blocks
    'mark complete' but still allows an interrupted partial save)."""
    present = {
        r.class_name for r in rows
        if r.pilot_id == pilot_id and r.annotator_id == annotator_id
    }
    return sorted(set(CLASS_NAMES) - present)
