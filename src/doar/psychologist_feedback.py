"""DOAR Milestone 2: psychologist feedback on DOAR's own interpretation.

Distinct from `expert_review.py` (which reviews individual DETECTED
VISUAL FINDINGS -- confirm/reject/rename a label). This module instead
records a professional's judgment of DOAR's overall case interpretation
(agree/partially agree/disagree/cannot assess + a free-text comment),
persisted per case for later thesis-level analysis. Never edits any
AI-produced artifact -- purely additive, append-only, like
`expert_review.py`'s own history log.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .case_output import write_versioned

FEEDBACK_VERDICTS = ("agree", "partially_agree", "disagree", "cannot_assess")

# Additive: the categories a psychologist can rate separately, rather than
# only one overall verdict. `submit_feedback` (single-verdict) is kept
# unchanged for backward compatibility; `submit_categorized_feedback`
# below is the smallest additive extension that captures all four.
FEEDBACK_CATEGORIES = (
    "overall_interpretation", "emotion_interpretation", "concern_domain_interpretation", "explanation_usefulness",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_feedback(case_dir: str | Path) -> dict:
    path = Path(case_dir) / "psychologist_feedback.json"
    if not path.exists():
        return {"entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def submit_feedback(case_dir: str | Path, *, reviewer_name: str, verdict: str,
                     comment: str | None = None) -> dict:
    """Appends one feedback entry. `verdict` must be one of
    FEEDBACK_VERDICTS. Never mutates detections.json/analysis.json/
    structured_analysis.json or any other AI-produced artifact."""
    if verdict not in FEEDBACK_VERDICTS:
        raise ValueError(f"unknown feedback verdict: {verdict!r} (expected one of {FEEDBACK_VERDICTS})")
    case_dir = Path(case_dir)
    feedback = load_feedback(case_dir)
    entry = {
        "reviewer_name": reviewer_name, "verdict": verdict, "comment": comment,
        "timestamp": _utc_now_iso(), "case_id": case_dir.name,
    }
    feedback = {"entries": [*feedback.get("entries", []), entry]}
    write_versioned(case_dir / "psychologist_feedback.json", feedback)
    return feedback


def submit_categorized_feedback(case_dir: str | Path, *, reviewer_name: str, ratings: dict[str, str],
                                comment: str | None = None) -> dict:
    """Additive: one feedback entry rating MULTIPLE categories at once
    (see FEEDBACK_CATEGORIES), each Agree/Partially agree/Disagree/Cannot
    assess. Append-only, same file/entries list as `submit_feedback` --
    entries from either function coexist; a consumer distinguishes them by
    whether "ratings" (this function) or "verdict" (the legacy single-
    verdict shape) is present. Never mutates any AI-produced artifact."""
    unknown_categories = set(ratings) - set(FEEDBACK_CATEGORIES)
    if unknown_categories:
        raise ValueError(f"unknown feedback categories: {sorted(unknown_categories)} "
                         f"(expected one of {FEEDBACK_CATEGORIES})")
    if not ratings:
        raise ValueError("ratings must include at least one category")
    for category, verdict in ratings.items():
        if verdict not in FEEDBACK_VERDICTS:
            raise ValueError(f"unknown feedback verdict {verdict!r} for category {category!r} "
                             f"(expected one of {FEEDBACK_VERDICTS})")
    case_dir = Path(case_dir)
    feedback = load_feedback(case_dir)
    entry = {
        "reviewer_name": reviewer_name, "ratings": dict(ratings), "comment": comment,
        "timestamp": _utc_now_iso(), "case_id": case_dir.name,
    }
    feedback = {"entries": [*feedback.get("entries", []), entry]}
    write_versioned(case_dir / "psychologist_feedback.json", feedback)
    return feedback


def export_all_feedback(cases_dir: str | Path) -> list[dict]:
    """Aggregates every case's psychologist_feedback.json under
    `cases_dir` into one flat list, for thesis-level analysis across
    cases -- read-only, computed fresh each call, never cached to disk
    itself (the per-case files remain the source of truth)."""
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        return []
    all_entries = []
    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        feedback = load_feedback(case_dir)
        all_entries.extend(feedback.get("entries", []))
    return all_entries
