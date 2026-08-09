#!/usr/bin/env python
"""DOAR V1.1 stabilization acceptance check: runs the SAME real drawing
that revealed Problems A-G (outputs/prototype_cases/h38_1786305027/h38.jpg
-- confirmed by inspecting that case's own analysis.json: emotion
status="failed" with the exact "Checkpoint does not exist" reason,
artifacts.foreground_mask="artifacts/foreground_mask.png", and
page_frame confidence=1.0 on a cropped_or_content_only read) through the
ACTUAL application pipeline again, with the stabilization fixes applied,
and reports the real outcome honestly. Source image is read, never
modified.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.case_artifacts import resolve_analysis_artifacts  # noqa: E402
from doar.case_output import write_versioned  # noqa: E402
from doar.features import objective_feature_row, serialize_feature_row  # noqa: E402
from doar.production_config import resolve_production_config  # noqa: E402
from doar.timed_analysis import analyze_image_with_timing  # noqa: E402

SOURCE_IMAGE = ROOT / "outputs" / "prototype_cases" / "h38_1786305027" / "h38.jpg"
CASES_DIR = ROOT / "outputs" / "prototype_cases"


def main() -> None:
    t0 = time.monotonic()
    print(f"[1/13] Source image (the same real drawing from the manual bug report): {SOURCE_IMAGE}", flush=True)
    assert SOURCE_IMAGE.exists(), f"missing fixture image: {SOURCE_IMAGE}"

    case_name = f"v1_1_stabilization_check_{int(time.time())}"
    case_dir = CASES_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / SOURCE_IMAGE.name
    image_path.write_bytes(SOURCE_IMAGE.read_bytes())
    print(f"[2/13] Case created: {case_dir}", flush=True)

    print("[3/13] Resolving the automatic production configuration (no model choice)...", flush=True)
    production_config = resolve_production_config()
    print(f"       expressive_model_identifier={production_config.expressive_model_identifier!r} "
          f"available={production_config.expressive_model_available}", flush=True)
    if not production_config.expressive_model_available:
        print(f"       reason: {production_config.expressive_model_unavailable_reason}", flush=True)

    print("[4/13] Running the real analysis pipeline (analyze_image_with_timing)...", flush=True)
    analyze_image_with_timing(str(image_path), str(case_dir), production_config.expressive_model_checkpoint)
    write_versioned(case_dir / "production_config.json", production_config.to_dict())
    assert (case_dir / "analysis.json").exists()
    analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
    print("       analysis.json written.", flush=True)

    print("[5/13] Foreground mask artifact path check (Problem A)...", flush=True)
    assert (case_dir / analysis["artifacts"]["foreground_mask"]).exists(), "mask file missing on disk"
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(ROOT)  # simulate the app's real cwd (repo root, not case_dir)
        resolved = resolve_analysis_artifacts(analysis, case_dir)
        feature_row = objective_feature_row(str(image_path), resolved)
        serialized = serialize_feature_row(feature_row)
        print(f"       PASS: {len(serialized)} objective features computed from the app's own cwd "
              "with no [Errno 2] failure.", flush=True)
    finally:
        os.chdir(original_cwd)

    print("[6/13] Expressive-model branch honesty check (Problem B)...", flush=True)
    emotion = analysis["emotion"]
    if emotion["status"] == "available":
        print(f"       Genuinely available: top_class={emotion['top_class']!r}, "
              f"confidence={emotion['confidence']:.3f}.", flush=True)
    else:
        print(f"       Honestly unavailable: status={emotion['status']!r}, reason={emotion.get('reason')!r}. "
              "No probabilities fabricated.", flush=True)
        assert all(v is None for v in emotion["probabilities"].values()), "probabilities must not be fabricated"

    print("[7/13] Confirming NO user-facing model choice exists in the app source...", flush=True)
    app_source = (ROOT / "doar_prototype_app.py").read_text(encoding="utf-8")
    assert "KNOWN_CHECKPOINTS" not in app_source, "stale model-choice dict still present"
    assert "Emotion model\"" not in app_source, "stale emotion-model selectbox still present"
    print("       PASS: no KNOWN_CHECKPOINTS dict, no 'Emotion model' selectbox in app source.", flush=True)

    print("[8/13] Page-frame confidence honesty check (Problem F)...", flush=True)
    page_frame = analysis.get("page_frame") or {}
    print(f"       status={page_frame.get('page_frame_status')!r} confidence={page_frame.get('confidence')}", flush=True)
    assert page_frame.get("confidence", 0) < 1.0, "automatic page-frame confidence must never reach 1.0"
    page_reference = analysis.get("page_reference") or {}
    print(f"       page_reference: mode={page_reference.get('page_reference_mode')!r} "
          f"assessable={page_reference.get('page_relative_features_assessable')}", flush=True)

    print("[9/13] Capability-state check (Problem D)...", flush=True)
    judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
    availability = judges["module_availability"]
    print(f"       module_availability={availability}", flush=True)
    for key in ("detection", "visual_detection", "open_world_search", "objective_features",
                "expressive_model", "rules", "expert_review"):
        assert key in availability, f"missing canonical capability key: {key}"

    print("[10/13] Rendering Parent View summary sentences with live capability state...", flush=True)
    from doar.parent_view import build_overall_result_summary
    structured_path = case_dir / "structured_analysis.json"
    structured = json.loads(structured_path.read_text(encoding="utf-8")) if structured_path.exists() else {}
    sentences = build_overall_result_summary(structured, page_reference, "en", capabilities=availability)
    for s in sentences:
        print(f"       - {s}", flush=True)
    blob = " ".join(sentences)
    for leaked in ("checkpoint", "grounding_dino", "bbox", "rule_id"):
        assert leaked not in blob.lower(), f"{leaked!r} leaked into Parent View text"
    print("       PASS: no technical internals in Parent View summary.", flush=True)

    print("[11/13] Technical View smoke check (headless boot, real app file)...", flush=True)
    try:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
        at.session_state["case_dir"] = str(case_dir.resolve())
        at.run(timeout=60)
        assert len(at.exception) == 0, [str(e) for e in at.exception]
        error_texts = " ".join(e.value for e in at.error)
        assert "Feature computation failed" not in error_texts
        headers = [h.value for h in at.header]
        assert "11. Research / Validation" in headers
        print(f"       PASS: app rendered with 0 exceptions, {len(headers)} Technical View headers, "
              "no stale 'Feature computation failed' error.", flush=True)
    except ImportError:
        print("       streamlit.testing.v1.AppTest not available in this environment -- skipped.", flush=True)

    print("[12/13] Case reload check...", flush=True)
    reloaded = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
    assert reloaded["artifacts"]["foreground_mask"] == analysis["artifacts"]["foreground_mask"]
    reloaded_judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
    assert reloaded_judges["module_availability"] == availability
    print("       PASS: analysis.json/judges.json reload identically.", flush=True)

    elapsed = time.monotonic() - t0
    print(f"[13/13] DOAR V1.1 stabilization end-to-end check PASSED in {elapsed:.1f}s. Case: {case_dir}", flush=True)


if __name__ == "__main__":
    main()
