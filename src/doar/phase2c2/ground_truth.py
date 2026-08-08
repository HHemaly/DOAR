"""Extracts genuine Phase 2C.1 human ground truth for evaluation --
NEVER the legacy Phase 2B provisional labels (annotator_type ==
"legacy_provisional_human" rows are always excluded here)."""
from __future__ import annotations

from ..phase2c1.schema import AnnotationRecord


def genuine_human_ground_truth(store: dict[str, AnnotationRecord], class_name: str) -> dict[str, str]:
    """{pilot_id: status} for one class, genuine human annotator_type only.
    Raises if more than one distinct genuine human annotator_id exists for
    the same (pilot_id, class_name) -- this function assumes exactly one
    genuine human judgment per (pilot_id, class), which is this round's
    real state; a future multi-annotator round should not silently pick
    one arbitrarily."""
    out: dict[str, str] = {}
    for r in store.values():
        if r.annotator_type != "human" or r.class_name != class_name:
            continue
        if r.pilot_id in out and out[r.pilot_id] != r.status:
            raise ValueError(
                f"Multiple disagreeing genuine-human judgments for "
                f"({r.pilot_id}, {class_name}) -- this function assumes exactly "
                "one genuine human annotator per (pilot_id, class) this round.")
        out[r.pilot_id] = r.status
    return out
