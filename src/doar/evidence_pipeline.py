"""Evidence-to-rule pipeline runner (Phase 1 Section 8).

Ties together the real, already-working `analyze_image` extractors with
the new canonical evidence schema, rule registry v2, deterministic rule
engine, and traceable English report generator, and writes real output
files for one image. This is the demonstration path Phase 1 requires: no
mocked results, no fabricated detections -- every output file is produced
by actually running the pipeline against a real, on-disk drawing.

Per-stage wall-clock timing is recorded. `analyze_image` itself is timed
as one block: its internal quality/segmentation/composition/colour stages
are not individually instrumented (a real, pre-existing gap noted in
`END_TO_END_INFERENCE_TRACE.md`, out of scope here since it would require
modifying the already-tested `analysis.py` internals) -- this pipeline
does not claim otherwise.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .analysis import analyze_image
from .english_report import generate_traceable_report
from .evidence_adapter import build_evidence_set
from .evidence_rule_engine import evaluate_all_rules_v2
from .rule_schema import load_rules_v2


def run_evidence_pipeline(
    image_path: str | Path, output_dir: str | Path, emotion_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    analysis = analyze_image(image_path, output_dir, emotion_checkpoint=emotion_checkpoint)
    timing["base_pipeline_seconds"] = round(time.perf_counter() - t0, 4)

    analysis_dict = analysis.to_dict()

    t0 = time.perf_counter()
    evidence = build_evidence_set(analysis_dict, artifacts=analysis_dict["artifacts"])
    timing["evidence_adapter_seconds"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    rules = load_rules_v2()
    results = evaluate_all_rules_v2(rules, evidence)
    timing["rule_engine_seconds"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    report = generate_traceable_report(rules, results, evidence, str(image_path))
    timing["report_generation_seconds"] = round(time.perf_counter() - t0, 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence_v2.json"
    rules_path = output_dir / "rule_evaluations_v2.json"
    report_path = output_dir / "report_en.md"
    timing_path = output_dir / "pipeline_timing.json"

    evidence_path.write_text(json.dumps(evidence.to_list(), indent=2, default=str), encoding="utf-8")
    rules_path.write_text(json.dumps([r.to_dict() for r in results], indent=2, default=str), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")

    outcome_counts: dict[str, int] = {}
    for result in results:
        outcome_counts[result.outcome] = outcome_counts.get(result.outcome, 0) + 1

    return {
        "image_path": str(image_path),
        "timing": timing,
        "evidence_count": len(evidence.items),
        "evidence_status_counts": {
            status: sum(1 for i in evidence.items if i.status == status)
            for status in {"available", "experimental", "unavailable", "failed", "insufficient_evidence", "requires_manual_review"}
        },
        "rule_outcome_counts": outcome_counts,
        "output_files": {
            "evidence_v2": str(evidence_path),
            "rule_evaluations_v2": str(rules_path),
            "report_en": str(report_path),
            "pipeline_timing": str(timing_path),
        },
    }
