#!/usr/bin/env python
"""DOAR MVP acceptance check: runs ONE real child's drawing through the
complete, real pipeline end to end -- analysis, automatic visual scan,
grounded Q&A (existing evidence), on-demand visual search (Q&A about an
object NOT in the initial scan), expert review, and case reload -- using
real models throughout (no mocking). This is the CLI-equivalent of the
Streamlit app's 15-step acceptance test (the app itself needs a browser;
this script exercises the exact same underlying functions the app calls).

Not a new detector, not a new benchmark, not another annotation task --
purely an integration smoke test confirming the wired-together pipeline
actually works on a real image.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.chat import respond_to_chat  # noqa: E402
from doar.expert_review import load_review, submit_review  # noqa: E402
from doar.phase2c7 import detector_policy as pol  # noqa: E402
from doar.phase2c7 import runtime as rt  # noqa: E402
from doar.registry_v2_build import build_registry_v2  # noqa: E402
from doar.timed_analysis import analyze_image_with_timing  # noqa: E402
from doar.visual_evidence import load_detections, run_and_persist_initial_scan  # noqa: E402

SOURCE_IMAGE = ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0000.jpg"
CASES_DIR = ROOT / "outputs" / "prototype_cases"


def _load_eye_entry():
    policy_path = ROOT / "artifacts" / "phase2c7" / "visual_detector_policy.json"
    rows = json.loads(policy_path.read_text(encoding="utf-8"))
    eye_row = next(r for r in rows if r["target"] == "eye")
    return pol.build_eye_policy_entry(
        status=eye_row["status"], best_model=eye_row["best_model"],
        best_model_checkpoint=eye_row["best_model_checkpoint"], prompt=eye_row["prompt"],
        threshold=eye_row["threshold"], precision=eye_row["precision"], recall=eye_row["recall"],
        balanced_accuracy=eye_row["balanced_accuracy"],
        n_ground_truth_present=eye_row["n_ground_truth_present"],
        localization_validated=eye_row["localization_validated"], rationale=eye_row["rationale"])


def main() -> None:
    print(f"[1/9] Source image: {SOURCE_IMAGE}", flush=True)
    assert SOURCE_IMAGE.exists(), f"missing fixture image: {SOURCE_IMAGE}"

    case_name = f"e2e_check_{int(time.time())}"
    case_dir = CASES_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / SOURCE_IMAGE.name
    image_path.write_bytes(SOURCE_IMAGE.read_bytes())
    print(f"[2/9] Case dir: {case_dir}", flush=True)

    print("[3/9] Running the real analysis pipeline (analyze_image_with_timing)...", flush=True)
    analyze_image_with_timing(str(image_path), str(case_dir), None)
    assert (case_dir / "analysis.json").exists()
    assert (case_dir / "judges.json").exists()

    print("[4/9] Loading real models for the visual scan (grounding_dino x2, owlv2 x2, "
          "open-vocab query)...", flush=True)
    eye_entry = _load_eye_entry()
    registry_v2 = build_registry_v2()
    model_predict_fns = rt.build_real_model_predict_fns(eye_entry.best_model)
    model_predict_fns["open_vocab_query"] = rt.load_open_vocab_query_fn()

    print("[5/9] Running the automatic visual scan...", flush=True)
    findings = run_and_persist_initial_scan(
        case_dir, str(image_path), eye_entry=eye_entry, registry_v2=registry_v2,
        model_predict_fns=model_predict_fns)
    print(f"      {len(findings)} finding(s): "
          f"{[(f.label, f.validation_status) for f in findings]}", flush=True)
    assert findings, "expected at least one real detection on this image"
    detections_doc = json.loads((case_dir / "detections.json").read_text(encoding="utf-8"))
    assert detections_doc["status"] == "available"

    existing_label = findings[0].label
    print(f"[6/9] Q&A about an EXISTING finding ('{existing_label}')...", flush=True)
    resp_existing = respond_to_chat(
        case_dir, f"Is there a {existing_label} in the drawing?", "en",
        registry_v2=registry_v2, open_vocab_predict_fn=model_predict_fns["open_vocab_query"])
    print(f"      availability={resp_existing.availability!r} answer={resp_existing.answer!r}", flush=True)
    assert resp_existing.availability == "available"

    novel_target = "kite"
    print(f"[7/9] Q&A about an object NOT in initial evidence ('{novel_target}') -- "
          "triggers on-demand visual search...", flush=True)
    resp_novel = respond_to_chat(
        case_dir, f"Is there a {novel_target} in the drawing?", "en",
        registry_v2=registry_v2, open_vocab_predict_fn=model_predict_fns["open_vocab_query"])
    print(f"      availability={resp_novel.availability!r} answer={resp_novel.answer!r}", flush=True)
    assert resp_novel.availability in ("available", "not_found")
    after_search = load_detections(case_dir)
    novel_findings = [f for f in after_search if f.query == novel_target]
    if resp_novel.availability == "available":
        assert novel_findings, "on-demand hit should have been persisted as case evidence"
        print(f"      on-demand finding persisted: {novel_findings[0].to_dict()}", flush=True)
    else:
        print("      no on-demand hit -- correctly reported as 'not_found', not fabricated absence", flush=True)

    print("[8/9] Submitting an expert review action...", flush=True)
    submit_review(case_dir, reviewer_name="E2E Check", action="confirm", target_label=existing_label,
                  note="automated end-to-end check")
    review = load_review(case_dir)
    assert review["status"] == "submitted"
    judges_after = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
    assert judges_after["module_availability"]["clinician_review"] == "submitted"
    assert judges_after["module_availability"]["detection"] == "available"
    print(f"      review status={review['status']!r}, "
          f"module_availability={judges_after['module_availability']}", flush=True)

    print("[9/9] Reloading the case fresh from disk (simulated app reopen)...", flush=True)
    reloaded_detections = load_detections(case_dir)
    reloaded_review = load_review(case_dir)
    assert len(reloaded_detections) >= len(findings)
    assert reloaded_review["status"] == "submitted"
    print(f"      {len(reloaded_detections)} detection(s), review status "
          f"{reloaded_review['status']!r} -- all persisted correctly.", flush=True)

    print(f"\nDOAR MVP end-to-end check PASSED. Case: {case_dir}", flush=True)


if __name__ == "__main__":
    main()
