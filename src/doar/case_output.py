from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .claims_pipeline import build_claims_and_verification
from .judge_schemas import run_all_judges_v2
from .judges import run_judges
from .objective_features_report import build_objective_features_document
from .registry_v2_build import build_registry_v2
from .reports import save_reports
from .structured_report import build_structured_analysis


def write_versioned(path: Path, value) -> None:
    """Write a case JSON artifact, archiving the previous version first if the
    content actually changed.

    Re-analyzing a case (e.g. re-running `analyze-image` on the same output
    directory) previously overwrote analysis.json/evidence.json/rules.json/
    concerns.json/judges.json/emotion.json/detections.json with no history
    retained -- found during the 2026-08-02 audit (CURRENT_STATE_AUDIT.md
    Section 6). This preserves every prior version under `<case>/versions/`
    and appends an entry to `<case>/versions/history.jsonl`, without changing
    the path any existing reader (Streamlit, qa.py, judges.py, reports.py)
    uses for the *current* version -- purely additive."""
    new_content = json.dumps(value, ensure_ascii=False, indent=2)
    if path.exists():
        old_content = path.read_text(encoding="utf-8")
        if old_content != new_content:
            _archive_previous_version(path, old_content)
    path.write_text(new_content, encoding="utf-8")


def _archive_previous_version(path: Path, old_content: str) -> None:
    versions_dir = path.parent / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(versions_dir.glob(f"{path.stem}.v*{path.suffix}"))
    next_version = len(existing) + 1
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived_name = f"{path.stem}.v{next_version}.{timestamp}{path.suffix}"
    (versions_dir / archived_name).write_text(old_content, encoding="utf-8")
    history_entry = {
        "file": path.name,
        "archived_as": archived_name,
        "version": next_version,
        "timestamp_utc": timestamp,
        "sha256": hashlib.sha256(old_content.encode("utf-8")).hexdigest(),
    }
    with (versions_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_entry, ensure_ascii=False) + "\n")


def _write(path: Path, value) -> None:
    write_versioned(path, value)


def refresh_module_availability(output: Path, **updates) -> None:
    """Patches specific `judges.json["module_availability"][key]` fields in
    place, e.g. after a real detection run or an expert-review submission --
    both happen strictly AFTER `finalize_case`'s own `run_judges(analysis)`
    snapshot, so without this patch `judges.json` would report stale values
    ("detection": "unavailable", "clinician_review": "not_submitted")
    forever, even once real detections/a real review exist. Narrowly scoped
    to the given keys only -- never rewrites judges.py's own logic. No-op if
    judges.json does not exist yet (case not finalized)."""
    judges_path = output / "judges.json"
    if not judges_path.exists():
        return
    judges = json.loads(judges_path.read_text(encoding="utf-8"))
    judges.setdefault("module_availability", {}).update(updates)
    write_versioned(judges_path, judges)


def finalize_case(analysis: dict, output: Path) -> None:
    judges = run_judges(analysis)
    _write(output / "evidence.json", analysis["evidence"])
    _write(output / "rules.json", analysis["rule_evaluations"])
    _write(output / "concerns.json", analysis["concerns"])
    _write(output / "judges.json", judges)
    _write(output / "detections.json", {"status": "unavailable", "detections": []})
    _write(output / "emotion.json", analysis["emotion"])
    # DOAR-TRACE 4B: objective features are now computed for every run
    # (analysis.py::analyze_image); persist them as their own document too,
    # not only nested inside analysis.json, so a consumer that only needs
    # features doesn't have to parse the whole case.
    _write(output / "objective_features.json",
           build_objective_features_document(analysis.get("objective_features", {})))
    # DOAR-TRACE 4D/Phase-1.5-6: deterministic structured report -- planner
    # only, no LLM. registry_v2 is built once here and reused for both the
    # structured analysis and claim verification below, so the two always
    # see the exact same registry snapshot within one case.
    registry_v2 = build_registry_v2()
    structured_analysis = build_structured_analysis(analysis, registry_v2=registry_v2)
    _write(output / "structured_analysis.json", structured_analysis)
    # DOAR-TRACE Phase 1.5 Section 8: automatic judges-v2 + claim
    # verification for EVERY case -- previously only callable ad hoc.
    judges_v2 = run_all_judges_v2(analysis, judges, structured_analysis=structured_analysis)
    _write(output / "judges_v2.json", {jid: v.to_dict() for jid, v in judges_v2.items()})
    generated_claims, verification_report = build_claims_and_verification(structured_analysis, analysis, registry_v2)
    _write(output / "generated_claims.json", generated_claims)
    _write(output / "verification_report.json", verification_report)
    review = output / "clinician_review.json"
    if not review.exists():
        _write(review, {"status": "not_submitted", "history": [], "ai_output_preserved": True})
    save_reports(analysis, judges, output / "reports")


def resynthesize_case_with_visual_evidence(analysis: dict, output: Path, *, registry_v2: dict) -> None:
    """The DOAR MVP rule-integration counterpart to `finalize_case`: visual
    findings only exist AFTER `analyze_image`'s synchronous call already
    ran (the real automatic visual scan is a separate, later step -- see
    `visual_evidence.py::run_and_persist_initial_scan`), so any
    rule_evaluation a validated visual finding legitimately produces can
    only be reflected in the case's synthesis by RE-RUNNING the same
    downstream tail `finalize_case` already runs (judges, structured
    analysis, judges_v2, generated claims/verification) against the
    UPDATED `analysis` dict (which by the time this is called already has
    the new visual Evidence + any new rule_evaluations merged in by
    `visual_evidence.py::integrate_visual_findings_into_case`).

    Deliberately does NOT touch `detections.json` (the real visual scan
    this function exists to react to -- `finalize_case` would overwrite it
    back to the honest stub) or unconditionally reset
    `clinician_review.json` (an expert review may already exist and must
    never be silently reset -- an existing 'submitted' status is read back
    and preserved in the refreshed judges.json). Every other artifact is
    written via the exact same functions `finalize_case` itself uses --
    no new synthesis logic, only re-orchestration with a later-arriving
    input."""
    judges = run_judges(analysis)
    availability = judges.setdefault("module_availability", {})
    availability["detection"] = "available"
    availability["visual_detection"] = "available"
    availability["open_world_search"] = "available"
    review_path = output / "clinician_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("status") == "submitted":
            availability["clinician_review"] = "submitted"
            availability["expert_review"] = "submitted"
    _write(output / "evidence.json", analysis["evidence"])
    _write(output / "rules.json", analysis["rule_evaluations"])
    _write(output / "concerns.json", analysis["concerns"])
    _write(output / "judges.json", judges)
    _write(output / "objective_features.json",
           build_objective_features_document(analysis.get("objective_features", {})))
    structured_analysis = build_structured_analysis(analysis, registry_v2=registry_v2)
    _write(output / "structured_analysis.json", structured_analysis)
    judges_v2 = run_all_judges_v2(analysis, judges, structured_analysis=structured_analysis)
    _write(output / "judges_v2.json", {jid: v.to_dict() for jid, v in judges_v2.items()})
    generated_claims, verification_report = build_claims_and_verification(structured_analysis, analysis, registry_v2)
    _write(output / "generated_claims.json", generated_claims)
    _write(output / "verification_report.json", verification_report)
    save_reports(analysis, judges, output / "reports")
