"""
timed_analysis.py -- wraps analyze_image() with wall-clock timing.

END_TO_END_INFERENCE_TRACE.md found no processing-time field exists
anywhere in schemas.py::Analysis or the case output. Rather than modify
the core analysis.py pipeline (higher risk, touches every existing test
of analyze_image), this is a small additive wrapper that measures the
real call and persists a sidecar `timing.json` alongside the case.
"""
from __future__ import annotations

import time
from pathlib import Path

from .analysis import analyze_image
from .case_output import write_versioned
from .schemas import Analysis


def analyze_image_with_timing(
    image_path: str | Path, output_dir: str | Path, emotion_checkpoint: str | Path | None = None,
    user_page_declaration: dict | None = None,
) -> tuple[Analysis, dict]:
    start = time.perf_counter()
    result = analyze_image(image_path, output_dir, emotion_checkpoint, user_page_declaration=user_page_declaration)
    elapsed_seconds = time.perf_counter() - start
    timing = {
        "processing_seconds": round(elapsed_seconds, 4),
        "stage": "full_analyze_image_pipeline",
        "includes_case_output_and_reports": True,
    }
    write_versioned(Path(output_dir) / "timing.json", timing)
    return result, timing
