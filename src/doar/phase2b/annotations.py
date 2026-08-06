"""Phase 2B annotation manifest: schema, validation, and writer.

One row per (image, class) pair -- never one row per image -- so a
missing class for an image is structurally impossible to represent
silently; every annotated image has exactly `len(ontology.CLASSES)` rows.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .ontology import ANNOTATION_STATUSES, CLASS_NAMES

MANIFEST_FIELDS = [
    "pilot_id", "image_id", "source_image_group", "original_split", "class",
    "status", "bbox", "partial_or_occluded", "uncertain_reason",
    "instance_count", "annotator", "review_status", "notes",
]


@dataclass(frozen=True)
class AnnotationRow:
    pilot_id: str
    image_id: str
    source_image_group: str
    original_split: str
    class_name: str
    status: str  # one of ontology.ANNOTATION_STATUSES
    bbox: str = ""  # "" if not localized; "x,y,w,h" normalized 0-1 otherwise
    partial_or_occluded: bool = False
    uncertain_reason: str = ""
    instance_count: int = 0
    annotator: str = "single_annotator_session_2026-08-06"
    review_status: str = "unreviewed"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.class_name not in CLASS_NAMES:
            raise ValueError(f"{self.class_name!r} is not a Phase 2B ontology class")
        if self.status not in ANNOTATION_STATUSES:
            raise ValueError(f"status {self.status!r} not in {sorted(ANNOTATION_STATUSES)}")
        if self.status != "present" and self.instance_count != 0:
            raise ValueError(
                f"{self.pilot_id}/{self.class_name}: status={self.status!r} but "
                f"instance_count={self.instance_count} -- a non-present status must "
                "never carry a positive count (this would look like a confirmed "
                "measurement instead of absence/uncertainty).")

    def to_row(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("class_name")
        return {k: d[k] for k in MANIFEST_FIELDS}


def write_manifest(rows: list[AnnotationRow], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())
    return output_path


def load_manifest(path: Path) -> list[dict]:
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def validate_manifest_completeness(rows: list[dict]) -> None:
    """Every distinct pilot_id must have exactly one row per ontology class,
    and no (pilot_id, class) pair may repeat -- catches a partially-written
    or corrupted manifest immediately rather than downstream in evaluation."""
    seen: dict[str, set[str]] = {}
    for r in rows:
        seen.setdefault(r["pilot_id"], set())
        if r["class"] in seen[r["pilot_id"]]:
            raise ValueError(f"Duplicate ({r['pilot_id']}, {r['class']}) row in manifest")
        seen[r["pilot_id"]].add(r["class"])
    for pilot_id, classes in seen.items():
        missing = set(CLASS_NAMES) - classes
        if missing:
            raise ValueError(f"{pilot_id} is missing annotation rows for classes: {sorted(missing)}")
