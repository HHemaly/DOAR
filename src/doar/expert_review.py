"""DOAR MVP: expert (psychologist) review workflow.

Lets a reviewer confirm/reject/rename a detected visual object, mark
missing visual evidence, or leave a free-text note. Writes to the case's
existing `clinician_review.json` (schema already created for every case by
`case_output.py::finalize_case`: `{"status": "not_submitted", "history":
[], "ai_output_preserved": True}`) -- this module only APPENDS review
actions to `history`; it never edits or removes the AI's own original
output (`ai_output_preserved` stays True always).

Expert corrections are kept SEPARATE from formal ground truth: this file
is never read by, or written into, any Phase 2C annotation/training
pipeline -- a review submitted here has no effect on any frozen detector
policy, benchmark, or annotation store.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .case_output import refresh_module_availability, write_versioned

REVIEW_ACTIONS = ("confirm", "reject", "rename", "mark_missing_evidence", "note")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_review(case_dir: str | Path) -> dict:
    path = Path(case_dir) / "clinician_review.json"
    if not path.exists():
        return {"status": "not_submitted", "history": [], "ai_output_preserved": True}
    return json.loads(path.read_text(encoding="utf-8"))


def submit_review(case_dir: str | Path, *, reviewer_name: str, action: str,
                   target_label: str | None = None, new_label: str | None = None,
                   note: str | None = None) -> dict:
    """Appends one review action to `clinician_review.json`'s history.
    `action` must be one of REVIEW_ACTIONS. Never mutates `detections.json`
    or any other AI-produced artifact -- expert input is recorded as a
    separate, clearly-attributed layer alongside the original output."""
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"unknown review action: {action!r} (expected one of {REVIEW_ACTIONS})")
    case_dir = Path(case_dir)
    review = load_review(case_dir)
    entry = {
        "reviewer_name": reviewer_name, "action": action, "target_label": target_label,
        "new_label": new_label, "note": note, "timestamp": _utc_now_iso(),
    }
    review = {
        "status": "submitted",
        "history": [*review.get("history", []), entry],
        "ai_output_preserved": True,
    }
    write_versioned(case_dir / "clinician_review.json", review)
    refresh_module_availability(case_dir, clinician_review="submitted", expert_review="submitted")
    return review
