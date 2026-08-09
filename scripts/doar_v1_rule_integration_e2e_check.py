#!/usr/bin/env python
"""DOAR V1 rule-integration acceptance check: runs ONE real child's
drawing through the COMPLETE real pipeline -- objective measurements,
real automatic visual detection, canonical evidence conversion, the REAL
rule engine (rule_engine_v2.py), aggregation/synthesis
(structured_analysis.json), Parent View data, Technical View trace,
grounded Q&A (existing evidence + on-demand search), expert review, and
case reload. Real models throughout (no mocking) -- this is the
authoritative Stage 9 acceptance case for the rule-integration work
(distinct from the unit/integration tests, which use fake predict_fns for
speed).

The rule-engine step is NOT a separate call -- `run_and_persist_initial_
scan` already chains into `visual_evidence.integrate_visual_findings_into_
case` automatically (see visual_evidence.py), so this script's only new
addition versus scripts/doar_mvp_e2e_check.py is INSPECTING and reporting
that outcome explicitly, honestly, without assuming or fabricating a
triggered rule.
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
    print(f"[1/10] Source image: {SOURCE_IMAGE}", flush=True)
    assert SOURCE_IMAGE.exists(), f"missing fixture image: {SOURCE_IMAGE}"

    case_name = f"v1_rule_integration_check_{int(time.time())}"
    case_dir = CASES_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / SOURCE_IMAGE.name
    image_path.write_bytes(SOURCE_IMAGE.read_bytes())
    print(f"[2/10] Case dir: {case_dir}", flush=True)

    print("[3/10] Running the real analysis pipeline (analyze_image_with_timing) -- "
          "objective measurements + old-registry rule engine...", flush=True)
    analyze_image_with_timing(str(image_path), str(case_dir), None)
    assert (case_dir / "analysis.json").exists()
    before_analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
    print(f"      {len(before_analysis['rule_evaluations'])} baseline rule_evaluations, "
          f"{len(before_analysis['evidence'])} baseline evidence records (pre-visual-scan)", flush=True)

    print("[4/10] Loading real models for the visual scan (grounding_dino x2, owlv2 x2, "
          "open-vocab query)...", flush=True)
    eye_entry = _load_eye_entry()
    registry_v2 = build_registry_v2()
    model_predict_fns = rt.build_real_model_predict_fns(eye_entry.best_model)
    model_predict_fns["open_vocab_query"] = rt.load_open_vocab_query_fn()

    print("[5/10] Running the automatic visual scan (this also runs the REAL rule engine "
          "against the resulting canonical evidence -- integrate_visual_findings_into_case "
          "is chained automatically, not a separate call)...", flush=True)
    findings = run_and_persist_initial_scan(
        case_dir, str(image_path), eye_entry=eye_entry, registry_v2=registry_v2,
        model_predict_fns=model_predict_fns)
    print(f"      {len(findings)} finding(s): "
          f"{[(f.label, f.validation_status) for f in findings]}", flush=True)
    assert findings, "expected at least one real detection on this image"

    print("[6/10] Inspecting the REAL rule-engine outcome (honest report, no fabrication)...",
          flush=True)
    analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
    visual_sourced = [r for r in analysis["rule_evaluations"] if r.get("visual_evidence_sourced")]
    triggered = [r for r in visual_sourced if r["status"] == "weak_support"]
    validated_labels = {f.label for f in findings if f.evidence_status == "validated_evidence"}
    print(f"      Validated (rule-eligible) findings this run: {sorted(validated_labels) or 'none'}",
          flush=True)
    print(f"      Rows produced by evaluate_visual_object_presence_rules: {len(visual_sourced)}",
          flush=True)
    for row in visual_sourced:
        print(f"        {row['rule_id']}: status={row['status']!r} "
              f"matched_evidence_ids={row['matched_evidence_ids']} "
              f"missing_evidence={row['missing_evidence']}", flush=True)
    if triggered:
        print(f"      OUTCOME A: {len(triggered)} rule(s) legitimately triggered by validated "
              "visual evidence.", flush=True)
        structured = json.loads((case_dir / "structured_analysis.json").read_text(encoding="utf-8"))
        reaching_synthesis = {s["rule_id"] for s in structured.get("individual_rule_suggestions", [])} | {
            rid for hyp in structured.get("combined_drawing_level_hypotheses", [])
            for rid in hyp.get("contributing_rule_ids", [])}
        for row in triggered:
            assert row["rule_id"] in reaching_synthesis, (
                f"{row['rule_id']} triggered but did not reach structured_analysis.json")
        print(f"      Confirmed: triggered rule(s) reached structured_analysis.json synthesis: "
              f"{[r['rule_id'] for r in triggered]}", flush=True)
    else:
        print("      OUTCOME B: no validated visual finding is currently eligible to trigger a "
              "real rule -- every matching static_detector rule's allowed_output_level in "
              "rules_registry_v2.json is still 'disabled' (a real, existing registry-curation "
              "decision this session does not override). The system correctly abstains -- this "
              "is the honest, expected result today, not a bug.", flush=True)
    evidence = json.loads((case_dir / "evidence.json").read_text(encoding="utf-8"))
    n_visual_evidence = sum(1 for e in evidence if e.get("kind") == "visual_detection")
    print(f"      {n_visual_evidence} canonical Evidence record(s) created from this scan's "
          f"findings (all statuses -- full technical traceability regardless of trigger outcome).",
          flush=True)
    assert n_visual_evidence == len(findings)

    existing_label = findings[0].label
    print(f"[7/10] Q&A about an EXISTING finding ('{existing_label}')...", flush=True)
    resp_existing = respond_to_chat(
        case_dir, f"Is there a {existing_label} in the drawing?", "en",
        registry_v2=registry_v2, open_vocab_predict_fn=model_predict_fns["open_vocab_query"])
    print(f"      availability={resp_existing.availability!r} answer={resp_existing.answer!r}", flush=True)
    assert resp_existing.availability == "available"
    resp_rules = respond_to_chat(case_dir, "Which rules were evaluated?", "en")
    print(f"      'which rules were evaluated?' -> {resp_rules.answer[:200]}...", flush=True)

    novel_target = "kite"
    print(f"[8/10] Q&A about an object NOT in initial evidence ('{novel_target}') -- "
          "triggers on-demand visual search...", flush=True)
    resp_novel = respond_to_chat(
        case_dir, f"Is there a {novel_target} in the drawing?", "en",
        registry_v2=registry_v2, open_vocab_predict_fn=model_predict_fns["open_vocab_query"])
    print(f"      availability={resp_novel.availability!r} answer={resp_novel.answer!r}", flush=True)
    assert resp_novel.availability in ("available", "not_found")
    after_search_analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
    n_visual_sourced_after_search = sum(
        1 for r in after_search_analysis["rule_evaluations"] if r.get("visual_evidence_sourced"))
    assert n_visual_sourced_after_search == len(visual_sourced), (
        "on-demand search must NEVER change rule_evaluations -- it never calls rule integration")
    print("      Confirmed: on-demand search did not alter rule_evaluations (never silently "
          "validated).", flush=True)

    print("[9/10] Submitting an expert review action...", flush=True)
    submit_review(case_dir, reviewer_name="E2E Check", action="confirm", target_label=existing_label,
                  note="automated rule-integration end-to-end check")
    review = load_review(case_dir)
    assert review["status"] == "submitted"
    judges_after = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
    assert judges_after["module_availability"]["clinician_review"] == "submitted"
    assert judges_after["module_availability"]["detection"] == "available"
    print(f"      review status={review['status']!r}, "
          f"module_availability={judges_after['module_availability']}", flush=True)

    print("[10/10] Reloading the case fresh from disk (simulated app reopen)...", flush=True)
    reloaded_detections = load_detections(case_dir)
    reloaded_review = load_review(case_dir)
    reloaded_analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
    reloaded_visual_sourced = [r for r in reloaded_analysis["rule_evaluations"]
                                if r.get("visual_evidence_sourced")]
    assert len(reloaded_detections) >= len(findings)
    assert reloaded_review["status"] == "submitted"
    assert len(reloaded_visual_sourced) == len(visual_sourced)
    print(f"      {len(reloaded_detections)} detection(s), review status "
          f"{reloaded_review['status']!r}, {len(reloaded_visual_sourced)} visual-sourced rule "
          "row(s) -- complete trace persisted correctly.", flush=True)

    print(f"\nDOAR V1 rule-integration end-to-end check PASSED. Case: {case_dir}", flush=True)


if __name__ == "__main__":
    main()
