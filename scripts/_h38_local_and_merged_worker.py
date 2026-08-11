#!/usr/bin/env python
"""Worker process for doar_visual_observer_h38_check.py: runs the real
local detector pipeline (A) and the local+observer merged pipeline (C).
Isolated in its OWN process, invoked via `subprocess.run`, because real-
model CPU inference (grounding_dino x2 + owlv2 x2 + the eye combo) has
been observed to crash this specific sandboxed environment with a native
segfault (SIGSEGV, exit 139) -- reproduced twice, always right after
model loading completes and inference begins. Neither
`phase2c7/visual_detector.py` nor `phase2c7/runtime.py` (the code that
actually loads/runs these models) was touched by the DOAR Visual Observer
phase; this crash is a pre-existing environment/infrastructure limitation
of this sandbox, not something introduced here, and out of this phase's
scope to fix (DO-NOT list: no detector changes, no new detector
benchmark). Isolating it in a subprocess lets the orchestrator (the
calling script) catch that crash by exit code and still report the
outcome honestly, instead of losing the whole comparison run to it.

Writes its JSON result to the path given as argv[1]. On success, exits 0.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c7 import detector_policy as pol  # noqa: E402
from doar.phase2c7 import runtime as rt  # noqa: E402
from doar.registry_v2_build import build_registry_v2  # noqa: E402
from doar.visual_evidence import load_entities as load_entities_from_case  # noqa: E402
from doar.visual_evidence import run_and_persist_initial_scan  # noqa: E402
from doar.visual_observer import OpenAIVisualObserver  # noqa: E402


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
    out_path = Path(sys.argv[1])
    local_case_dir = Path(sys.argv[2])
    local_image_path = Path(sys.argv[3])
    merged_case_dir = Path(sys.argv[4])
    merged_image_path = Path(sys.argv[5])

    eye_entry = _load_eye_entry()
    registry_v2 = build_registry_v2()
    t_load = time.monotonic()
    model_predict_fns = rt.build_real_model_predict_fns(eye_entry.best_model)
    load_seconds = time.monotonic() - t_load

    t_local = time.monotonic()
    local_findings = run_and_persist_initial_scan(
        local_case_dir, str(local_image_path), eye_entry=eye_entry, registry_v2=registry_v2,
        model_predict_fns=model_predict_fns)
    local_runtime = time.monotonic() - t_local

    t_merged = time.monotonic()
    run_and_persist_initial_scan(
        merged_case_dir, str(merged_image_path), eye_entry=eye_entry, registry_v2=registry_v2,
        model_predict_fns=model_predict_fns, observer=OpenAIVisualObserver())
    merged_runtime = time.monotonic() - t_merged
    merged_entities = load_entities_from_case(merged_case_dir)

    result = {
        "model_load_seconds": round(load_seconds, 2),
        "local": {
            "runtime_seconds": round(local_runtime, 2),
            "findings": [{"label": f.label, "confidence": f.confidence,
                          "validation_status": f.validation_status, "evidence_status": f.evidence_status}
                         for f in local_findings],
        },
        "merged": {
            "runtime_seconds": round(merged_runtime, 2),
            "entities": [{"label": e.canonical_label, "entity_type": e.entity_type,
                          "source": e.source, "model_validation_status": e.model_validation_status,
                          "case_verification_status": e.case_verification_status,
                          "evidence_status": e.evidence_status}
                         for e in merged_entities],
        },
    }
    out_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
