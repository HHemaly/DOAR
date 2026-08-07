"""CSV-backed annotation store for Phase 2C.1 -- separate file from Phase
2B's committed artifacts/phase2b/annotation_manifest.csv, which this module
never writes to. Every write is an atomic replace (tempfile + os.replace,
same pattern as src/doar/human_review.py::_atomic_write_json) so a killed
process or closed browser tab can never leave a half-written store; loading
on startup is exactly how "resume" works -- there is no separate resume
mechanism to keep in sync.

Duplicate-row prevention is structural: the store is keyed by
`annotation_id` (schema.make_annotation_id), and `upsert` always replaces
the existing entry for that key rather than appending -- there is no code
path in this module that can produce two rows with the same
(pilot_id, class_name, annotator_id).
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from .schema import CSV_FIELDS, AnnotationRecord, record_from_row


def load_store(path: str | Path) -> dict[str, AnnotationRecord]:
    """Loads every row into {annotation_id: AnnotationRecord}. Returns an
    empty store (not an error) if the file does not exist yet -- the normal
    state before an annotator's first save."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    store: dict[str, AnnotationRecord] = {}
    for row in rows:
        rec = record_from_row(row)
        store[rec.annotation_id] = rec
    return store


def save_store(path: str | Path, store: dict[str, AnnotationRecord]) -> Path:
    """Atomic write of the full store, sorted by annotation_id for a
    deterministic, diff-friendly file (same discipline as the rest of this
    codebase's CSV writers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for key in sorted(store):
                writer.writerow(store[key].to_row())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return path


def upsert(store: dict[str, AnnotationRecord], record: AnnotationRecord) -> dict[str, AnnotationRecord]:
    """Inserts or replaces the entry for `record`'s (pilot_id, class_name,
    annotator_id) key. Returns the same dict, mutated in place, for
    call-site convenience (`store = upsert(store, rec)`)."""
    store[record.annotation_id] = record
    return store


def upsert_and_save(path: str | Path, store: dict[str, AnnotationRecord],
                     record: AnnotationRecord) -> dict[str, AnnotationRecord]:
    """The single call the Streamlit app's Save/Save & Next buttons use --
    upsert in memory, then autosave (write the whole store) immediately."""
    upsert(store, record)
    save_store(path, store)
    return store


def apply_review_outcome(path: str | Path, store: dict[str, AnnotationRecord], *,
                          pilot_id: str, class_name: str, primary_annotator_id: str,
                          reviewer_id: str, adjudication_status: str,
                          review_timestamp: str) -> dict[str, AnnotationRecord]:
    """Updates review metadata IN PLACE on the primary annotator's existing
    row -- never creates a new row and never touches the reviewer's own
    independent judgment row (that is a normal `upsert_and_save` call under
    the reviewer's own annotator_id, made separately, ideally BEFORE this
    function reveals/compares against the primary annotation, to avoid
    anchoring the reviewer on the original label)."""
    from .schema import make_annotation_id

    key = make_annotation_id(pilot_id, class_name, primary_annotator_id)
    if key not in store:
        raise KeyError(f"No primary annotation {key!r} to attach a review outcome to")
    old = store[key]
    new = AnnotationRecord(
        pilot_id=old.pilot_id, image_id=old.image_id, source_image_group=old.source_image_group,
        class_name=old.class_name, status=old.status, annotator_id=old.annotator_id,
        annotation_timestamp=old.annotation_timestamp, instance_count=old.instance_count,
        bbox=old.bbox, partial_or_occluded=old.partial_or_occluded,
        uncertainty_reason=old.uncertainty_reason, annotator_type=old.annotator_type,
        review_status="reviewed", reviewer_id=reviewer_id, review_timestamp=review_timestamp,
        adjudication_status=adjudication_status, notes=old.notes,
        ontology_version=old.ontology_version, source_manifest_version=old.source_manifest_version,
    )
    store[key] = new
    save_store(path, store)
    return store


def export_csv(store: dict[str, AnnotationRecord], out_path: str | Path) -> int:
    save_store(out_path, store)
    return len(store)


def export_json(store: dict[str, AnnotationRecord], out_path: str | Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [store[k].to_row() for k in sorted(store)]
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        os.replace(tmp_name, out_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return len(rows)


def records_for_pilot(store: dict[str, AnnotationRecord], pilot_id: str) -> list[AnnotationRecord]:
    return sorted(
        (r for r in store.values() if r.pilot_id == pilot_id),
        key=lambda r: (r.class_name, r.annotator_id),
    )


def annotated_pilot_ids(store: dict[str, AnnotationRecord], annotator_id: str | None = None) -> set[str]:
    """Pilot IDs with at least one saved row -- used for progress display.
    If `annotator_id` is given, restricts to that annotator's own rows
    (e.g. so one annotator's progress bar doesn't count another's work)."""
    return {
        r.pilot_id for r in store.values()
        if annotator_id is None or r.annotator_id == annotator_id
    }
