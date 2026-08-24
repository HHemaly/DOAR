"""Phase G0, Step 7 -- append-only research feedback schema letting a
psychologist review each NEW candidate formal/graphic observation (from
formal_features.py / MASTER_RULE_FEATURE_REGISTRY_V3.json) per case.

Distinct from `expert_review.py` (reviews individual DETECTED VISUAL
FINDINGS -- confirm/reject/rename a label) and from
`psychologist_feedback.py` (reviews DOAR's overall case interpretation).
This module instead records a professional's judgment of ONE candidate
formal-feature OBSERVATION -- never promotes it into an enabled production
rule automatically; `expert_rule_decision` is recorded for future,
separately-reviewed registry changes, never applied here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .case_output import write_versioned

OBSERVATION_CORRECT_VALUES = ("yes", "partially", "no", "cannot_assess")
CLINICALLY_RELEVANT_VALUES = ("yes", "no", "only_in_combination", "uncertain")
EXPERT_SUPPORT_LEVEL_VALUES = ("none", "weak", "moderate", "strong", "cannot_assess")
EXPERT_RULE_DECISION_VALUES = ("keep_objective_only", "candidate", "experimental", "reject")
OVERALL_SYNTHESIS_AGREEMENT_VALUES = ("agree", "partially_agree", "disagree", "cannot_assess")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_formal_feature_review(case_dir: str | Path) -> dict:
    path = Path(case_dir) / "formal_feature_review.json"
    if not path.exists():
        return {"entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def submit_formal_feature_review(
        case_dir: str | Path, *, reviewer_name: str, feature_id: str,
        observation_correct: str, clinically_relevant: str, expert_support_level: str,
        expert_rule_decision: str, overall_synthesis_agreement: str,
        region: dict | None = None, raw_measurement: float | None = None, doar_observation: str | None = None,
        possible_meaning: str | None = None, requires_combination: bool | None = None,
        required_companion_features: list[str] | None = None, alternative_explanations: list[str] | None = None,
        age_limitations: str | None = None, task_limitations: str | None = None,
        material_tool_limitations: str | None = None, parent_wording: str | None = None,
        notes: str | None = None) -> dict:
    """Appends one review entry for ONE (drawing, feature) pair. Never
    mutates formal_features.json, MASTER_RULE_FEATURE_REGISTRY_V3.json, or
    any concern-domain/aggregation artifact -- append-only, exactly like
    expert_review.py/psychologist_feedback.py's own history logs.
    `expert_rule_decision` is recorded for a FUTURE, separately-reviewed
    registry change; it is never auto-applied by this function."""
    if observation_correct not in OBSERVATION_CORRECT_VALUES:
        raise ValueError(f"unknown observation_correct: {observation_correct!r} "
                         f"(expected one of {OBSERVATION_CORRECT_VALUES})")
    if clinically_relevant not in CLINICALLY_RELEVANT_VALUES:
        raise ValueError(f"unknown clinically_relevant: {clinically_relevant!r} "
                         f"(expected one of {CLINICALLY_RELEVANT_VALUES})")
    if expert_support_level not in EXPERT_SUPPORT_LEVEL_VALUES:
        raise ValueError(f"unknown expert_support_level: {expert_support_level!r} "
                         f"(expected one of {EXPERT_SUPPORT_LEVEL_VALUES})")
    if expert_rule_decision not in EXPERT_RULE_DECISION_VALUES:
        raise ValueError(f"unknown expert_rule_decision: {expert_rule_decision!r} "
                         f"(expected one of {EXPERT_RULE_DECISION_VALUES})")
    if overall_synthesis_agreement not in OVERALL_SYNTHESIS_AGREEMENT_VALUES:
        raise ValueError(f"unknown overall_synthesis_agreement: {overall_synthesis_agreement!r} "
                         f"(expected one of {OVERALL_SYNTHESIS_AGREEMENT_VALUES})")

    case_dir = Path(case_dir)
    review = load_formal_feature_review(case_dir)
    entry = {
        "drawing_id": case_dir.name, "feature_id": feature_id, "region": region,
        "raw_measurement": raw_measurement, "doar_observation": doar_observation,
        "observation_correct": observation_correct, "clinically_relevant": clinically_relevant,
        "possible_meaning": possible_meaning,
        "requires_combination": bool(requires_combination) if requires_combination is not None else None,
        "required_companion_features": list(required_companion_features or []),
        "alternative_explanations": list(alternative_explanations or []),
        "age_limitations": age_limitations, "task_limitations": task_limitations,
        "material_tool_limitations": material_tool_limitations, "parent_wording": parent_wording,
        "expert_support_level": expert_support_level, "expert_rule_decision": expert_rule_decision,
        "overall_synthesis_agreement": overall_synthesis_agreement, "notes": notes,
        "reviewer_name": reviewer_name, "timestamp": _utc_now_iso(),
    }
    review = {"entries": [*review.get("entries", []), entry]}
    write_versioned(case_dir / "formal_feature_review.json", review)
    return review


def export_all_formal_feature_reviews(cases_dir: str | Path) -> list[dict]:
    """Read-only aggregation across every case, mirroring
    psychologist_feedback.export_all_feedback exactly -- never itself
    cached to disk; the per-case files remain the source of truth."""
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        return []
    all_entries = []
    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        review = load_formal_feature_review(case_dir)
        all_entries.extend(review.get("entries", []))
    return all_entries
