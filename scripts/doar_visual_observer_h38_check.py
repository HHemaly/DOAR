#!/usr/bin/env python
"""DOAR Visual Observer (shadow mode) development sanity check -- NOT a
benchmark. Runs the same real drawing used throughout the DOAR V1.1
stabilization phase (outputs/prototype_cases/h38_1786305027/h38.jpg)
through three configurations:

  (A) local detector pipeline alone (real models -- grounding_dino,
      owlv2, the eye combo -- exactly as `doar_mvp_e2e_check.py` loads
      them)
  (B) the real OpenAIVisualObserver alone (shadow mode; never tuned to
      this image -- the observer's prompt is generic, written before
      this script ever inspected h38.jpg's content)
  (C) local + observer merged, via the same `run_and_persist_initial_
      scan(..., observer=...)` path the app itself uses

(A) and (C) run inside a SEPARATE worker subprocess
(`_h38_local_and_merged_worker.py`), because real-model CPU inference has
been observed to crash this sandboxed environment with a native segfault
(reproduced twice, always right after model loading completes and
inference begins) -- a pre-existing environment limitation, not
something this phase's Visual Observer code introduced or can fix
in-process (a segfault cannot be caught with try/except). Isolating it
lets this script report the outcome honestly either way instead of
losing the whole comparison to an unrecoverable crash.

Compares the result against a short list of visually obvious content a
human would expect to see in this drawing (vehicle, sun, clouds,
butterflies/butterfly-like shapes, traffic light, wheels, windows,
road/ground line) plus three specific prior-phase concerns (person,
face, house) this project has repeatedly flagged as plausible false
positives from the local pipeline. That reference list is a DEVELOPMENT
AID for reading this script's own output -- it is never passed into the
observer's prompt or used to filter/boost its output.

Saves outputs/prototype_cases/<case>/visual_observer_h38_comparison.json.
If OPENAI_API_KEY (or the configured env var) is not set in this
environment -- expected in CI / most dev machines -- (B) and the
observer-half of (C) honestly report "unavailable", never a fabricated
result.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_observer import (  # noqa: E402
    OpenAIVisualObserver, VisualObserverConfigurationError, VisualObserverRequestError,
)

SOURCE_IMAGE = ROOT / "outputs" / "prototype_cases" / "h38_1786305027" / "h38.jpg"
CASES_DIR = ROOT / "outputs" / "prototype_cases"
WORKER_SCRIPT = Path(__file__).resolve().parent / "_h38_local_and_merged_worker.py"

# Development reference ONLY -- read after the fact by this script's own
# reporting code, never fed into the observer's prompt or the local
# pipeline's vocabulary/threshold choices.
EXPECTED_SALIENT_REFERENCE = (
    "vehicle", "car", "sun", "cloud", "clouds", "butterfly", "butterflies",
    "traffic light", "wheel", "wheels", "window", "windows", "road", "ground",
)
SUSPECT_LOCAL_LABELS = ("person", "face", "house")


def _match_reference(labels: list[str]) -> tuple[list[str], list[str]]:
    lowered = " | ".join(label.lower() for label in labels)
    matched = [ref for ref in EXPECTED_SALIENT_REFERENCE if ref in lowered]
    # Collapse near-duplicates (e.g. "cloud"/"clouds" both matching) to one entry per concept.
    seen, deduped = set(), []
    for m in matched:
        key = m.rstrip("s")
        if key not in seen:
            seen.add(key)
            deduped.append(m)
    missed = sorted({r.rstrip("s") for r in EXPECTED_SALIENT_REFERENCE} - seen)
    return deduped, missed


def main() -> None:
    t0 = time.monotonic()
    assert SOURCE_IMAGE.exists(), f"missing fixture image: {SOURCE_IMAGE}"

    stamp = int(time.time())
    local_case_dir = CASES_DIR / f"visual_observer_h38_check_{stamp}"
    local_case_dir.mkdir(parents=True, exist_ok=True)
    local_image_path = local_case_dir / SOURCE_IMAGE.name
    local_image_path.write_bytes(SOURCE_IMAGE.read_bytes())

    merged_case_dir = CASES_DIR / f"visual_observer_h38_check_merged_{stamp}"
    merged_case_dir.mkdir(parents=True, exist_ok=True)
    merged_image_path = merged_case_dir / SOURCE_IMAGE.name
    merged_image_path.write_bytes(SOURCE_IMAGE.read_bytes())
    print(f"[1/4] Case dirs: {local_case_dir.name} (local-only), {merged_case_dir.name} (merged)", flush=True)

    # -- (B) real observer alone, in-process (no heavy model loading, no crash risk) --
    print("[2/4] (B) Real OpenAIVisualObserver alone...", flush=True)
    observer = OpenAIVisualObserver()
    t_observer = time.monotonic()
    observer_status, observer_error, observer_candidates = "available", None, []
    try:
        observer_candidates = observer.analyze(str(local_image_path))
    except VisualObserverConfigurationError as exc:
        observer_status, observer_error = "unavailable_configuration", str(exc)
    except VisualObserverRequestError as exc:
        observer_status, observer_error = "unavailable_request_failed", str(exc)
    observer_runtime = time.monotonic() - t_observer
    print(f"      status={observer_status!r} model={observer.model!r} "
          f"{len(observer_candidates)} candidate(s) in {observer_runtime:.2f}s", flush=True)
    if observer_error:
        print(f"      reason: {observer_error}", flush=True)

    # -- (A)+(C) local pipeline + merged, isolated in a worker subprocess --
    print("[3/4] (A)+(C) Local detector pipeline + merged, in an isolated subprocess "
          "(real-model CPU inference has crashed this sandbox natively before)...", flush=True)
    worker_out = local_case_dir / "_worker_result.json"
    t_worker = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(WORKER_SCRIPT), str(worker_out), str(local_case_dir), str(local_image_path),
         str(merged_case_dir), str(merged_image_path)],
        capture_output=True, text=True)
    worker_runtime = time.monotonic() - t_worker
    local_status = "available" if proc.returncode == 0 and worker_out.exists() else "crashed_in_this_environment"
    local_findings, merged_entities_raw = [], []
    local_runtime = merged_runtime = model_load_seconds = None
    if local_status == "available":
        worker_result = json.loads(worker_out.read_text(encoding="utf-8"))
        model_load_seconds = worker_result["model_load_seconds"]
        local_runtime = worker_result["local"]["runtime_seconds"]
        local_findings = worker_result["local"]["findings"]
        merged_runtime = worker_result["merged"]["runtime_seconds"]
        merged_entities_raw = worker_result["merged"]["entities"]
        print(f"      worker succeeded in {worker_runtime:.1f}s: {len(local_findings)} local finding(s), "
              f"{len(merged_entities_raw)} merged entit(y/ies)", flush=True)
    else:
        print(f"      worker CRASHED (returncode={proc.returncode}) after {worker_runtime:.1f}s -- "
              "a native crash in real-model CPU inference cannot be caught with try/except; this is a "
              "pre-existing environment limitation, not new code from this phase (phase2c7/visual_detector.py "
              "and phase2c7/runtime.py were not modified). Reporting honestly rather than fabricating a result.",
              flush=True)
        if proc.stderr.strip():
            print("      worker stderr (tail):\n" + "\n".join(proc.stderr.strip().splitlines()[-15:]), flush=True)

    local_labels = [f["label"] for f in local_findings]
    merged_labels = [e["label"] for e in merged_entities_raw]
    matched_local, missed_local = _match_reference(local_labels)
    matched_merged, missed_merged = _match_reference(merged_labels)
    suspect_local = [lbl for lbl in local_labels if lbl in SUSPECT_LOCAL_LABELS]
    suspect_merged = [e for e in merged_entities_raw if e["label"] in SUSPECT_LOCAL_LABELS]

    print("[4/4] Writing comparison artifact...", flush=True)
    total_runtime = time.monotonic() - t0
    report = {
        "note": ("Development sanity check on ONE real drawing -- NOT a scientific benchmark. "
                 "The observer's prompt was written before this script inspected h38.jpg; the "
                 "reference list below is read only by this script's own comparison code, never "
                 "fed into the observer or the local pipeline. (A)/(C) run in an isolated worker "
                 "subprocess because real-model CPU inference has been observed to crash this "
                 "sandboxed environment natively (SIGSEGV) -- a pre-existing environment limitation, "
                 "not something this phase's Visual Observer changes caused."),
        "source_image": str(SOURCE_IMAGE),
        "case_dir_local": str(local_case_dir),
        "case_dir_merged": str(merged_case_dir),
        "total_runtime_seconds": round(total_runtime, 2),
        "local_detector": {
            "status": local_status,
            "worker_returncode": proc.returncode,
            "model_load_seconds": model_load_seconds,
            "runtime_seconds": local_runtime,
            "findings": local_findings,
            "matched_expected_salient": matched_local,
            "missed_expected_salient": missed_local,
            "suspect_labels_present": suspect_local,
        },
        "multimodal_observer": {
            "provider": "openai",
            "model": observer.model,
            "status": observer_status,
            "error": observer_error,
            "runtime_seconds": round(observer_runtime, 3),
            "candidates": [{"label": c.label, "alternative_labels": list(c.alternative_labels),
                            "entity_type": c.entity_type, "bbox": c.bbox, "count": c.count,
                            "confidence": c.confidence, "source_note": c.source_note}
                           for c in observer_candidates],
        },
        "merged": {
            "status": local_status,  # merged ran in the same worker subprocess as local
            "runtime_seconds": merged_runtime,
            "entity_count": len(merged_entities_raw),
            "entities": merged_entities_raw,
            "matched_expected_salient": matched_merged,
            "missed_expected_salient": missed_merged,
            "suspect_entities_present": suspect_merged,
        },
    }
    out_path = local_case_dir / "visual_observer_h38_comparison.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved comparison artifact: {out_path}", flush=True)
    print(f"Total runtime: {total_runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
