from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .judges import run_judges
from .objective_features_report import build_objective_features_document
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
    # DOAR-TRACE 4D: deterministic structured report -- planner only, no LLM.
    _write(output / "structured_analysis.json", build_structured_analysis(analysis))
    review = output / "clinician_review.json"
    if not review.exists():
        _write(review, {"status": "not_submitted", "history": [], "ai_output_preserved": True})
    save_reports(analysis, judges, output / "reports")
