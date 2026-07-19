from __future__ import annotations

import json
from pathlib import Path

from .judges import run_judges
from .reports import save_reports


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_case(analysis: dict, output: Path) -> None:
    judges = run_judges(analysis)
    _write(output / "evidence.json", analysis["evidence"])
    _write(output / "rules.json", analysis["rule_evaluations"])
    _write(output / "concerns.json", analysis["concerns"])
    _write(output / "judges.json", judges)
    _write(output / "detections.json", {"status": "unavailable", "detections": []})
    _write(output / "emotion.json", analysis["emotion"])
    review = output / "clinician_review.json"
    if not review.exists():
        _write(review, {"status": "not_submitted", "history": [], "ai_output_preserved": True})
    save_reports(analysis, judges, output / "reports")
