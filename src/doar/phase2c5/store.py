"""CSV-backed store for Phase 2C.5 part annotations -- mirrors
src/doar/phase2c1/store.py's atomic-write / resume-by-reload discipline
exactly (same tempfile+os.replace pattern), but a SEPARATE file and a
SEPARATE dict keyed by Phase 2C.5's own annotation_id
(schema.make_part_annotation_id). Never reads or writes
outputs/phase2c1/annotation_store.csv.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from .schema import CSV_FIELDS, PartAnnotationRecord, record_from_row


def load_store(path: str | Path) -> dict[str, PartAnnotationRecord]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    store: dict[str, PartAnnotationRecord] = {}
    for row in rows:
        rec = record_from_row(row)
        store[rec.annotation_id] = rec
    return store


def save_store(path: str | Path, store: dict[str, PartAnnotationRecord]) -> Path:
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


def upsert(store: dict[str, PartAnnotationRecord], record: PartAnnotationRecord) -> dict[str, PartAnnotationRecord]:
    store[record.annotation_id] = record
    return store


def upsert_and_save(path: str | Path, store: dict[str, PartAnnotationRecord],
                     record: PartAnnotationRecord) -> dict[str, PartAnnotationRecord]:
    upsert(store, record)
    save_store(path, store)
    return store


def records_for_pilot(store: dict[str, PartAnnotationRecord], pilot_id: str) -> list[PartAnnotationRecord]:
    return sorted(
        (r for r in store.values() if r.pilot_id == pilot_id),
        key=lambda r: (r.target_name, r.annotator_id),
    )


def annotated_pilot_ids(store: dict[str, PartAnnotationRecord], annotator_id: str | None = None) -> set[str]:
    return {
        r.pilot_id for r in store.values()
        if annotator_id is None or r.annotator_id == annotator_id
    }


def progress_for_annotator(store: dict[str, PartAnnotationRecord], pilot_ids: list[str],
                            annotator_id: str, targets: tuple[str, ...]) -> dict[str, int]:
    """{'done': n_images_with_all_targets_recorded, 'total': len(pilot_ids)}
    -- resume support: the annotation app calls this on load to show
    progress and pick up where a killed process / closed tab left off."""
    done = 0
    for pid in pilot_ids:
        recorded = {r.target_name for r in store.values()
                    if r.pilot_id == pid and r.annotator_id == annotator_id}
        if set(targets) <= recorded:
            done += 1
    return {"done": done, "total": len(pilot_ids)}
