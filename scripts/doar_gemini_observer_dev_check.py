#!/usr/bin/env python
"""DOAR Gemini Observer development check -- NOT a benchmark. Runs the
SAME frozen, provider-agnostic open-world prompt (no per-image tuning)
through the real `GeminiVisualObserver` on:

  1. outputs/prototype_cases/h38_1786305027/h38.jpg -- permanently
     DEVELOPMENT ONLY, must never enter a future held-out benchmark.
  2. up to 4 additional real development drawings from the existing
     public dataset (outputs/phase2c1/private_images/p2b_0001..0004),
     selected by simple sequential order (not cherry-picked for a
     favorable result).

The raw structured response for EACH image is saved to disk immediately
after the call, before any manual/eyeballed comparison happens --
comparison against a short "visually obvious content" reference list
(read only by a human/reviewer afterward, never fed into the prompt) is
a separate, later step. If the API/schema fails technically, this script
is fixed and rerun; the prompt itself is never tuned based on what a
specific image's content turns out to be.

This is exploratory development only, not the DOAR thesis benchmark.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_observer import (  # noqa: E402
    GeminiVisualObserver, VisualObserverConfigurationError, VisualObserverRequestError,
)

H38_IMAGE = ROOT / "outputs" / "prototype_cases" / "h38_1786305027" / "h38.jpg"
ADDITIONAL_DEV_IMAGES = [
    ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0001.jpg",
    ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0002.jpg",
    ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0003.jpeg",
    ROOT / "outputs" / "phase2c1" / "private_images" / "p2b_0004.jpg",
]
OUT_DIR = ROOT / "outputs" / "prototype_cases" / f"gemini_observer_dev_check_{int(time.time())}"


def _run_one(observer: GeminiVisualObserver, image_path: Path) -> dict:
    t0 = time.monotonic()
    status, error, candidates = "available", None, []
    try:
        candidates = observer.analyze(str(image_path))
    except VisualObserverConfigurationError as exc:
        status, error = "unavailable_configuration", str(exc)
    except VisualObserverRequestError as exc:
        status, error = "unavailable_request_failed", str(exc)
    runtime = time.monotonic() - t0
    return {
        "image": image_path.name,
        "status": status,
        "error": error,
        "runtime_seconds": round(runtime, 2),
        "model": observer.model,
        "candidate_count": len(candidates),
        "candidates": [asdict(c) for c in candidates],
    }


def main() -> None:
    assert H38_IMAGE.exists(), f"missing fixture image: {H38_IMAGE}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    observer = GeminiVisualObserver()
    print(f"model: {observer.model}", flush=True)

    print(f"[1/{1 + len(ADDITIONAL_DEV_IMAGES)}] h38.jpg (development-only, frozen prompt)...", flush=True)
    h38_result = _run_one(observer, H38_IMAGE)
    (OUT_DIR / "h38_raw.json").write_text(json.dumps(h38_result, indent=2), encoding="utf-8")
    print(f"      status={h38_result['status']!r} {h38_result['candidate_count']} candidate(s) "
          f"in {h38_result['runtime_seconds']}s -- raw response saved", flush=True)
    if h38_result["error"]:
        print(f"      reason: {h38_result['error']}", flush=True)

    dev_results = []
    for i, image_path in enumerate(ADDITIONAL_DEV_IMAGES, start=2):
        print(f"[{i}/{1 + len(ADDITIONAL_DEV_IMAGES)}] {image_path.name} (same frozen prompt)...", flush=True)
        result = _run_one(observer, image_path)
        (OUT_DIR / f"{image_path.stem}_raw.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        dev_results.append(result)
        print(f"      status={result['status']!r} {result['candidate_count']} candidate(s) "
              f"in {result['runtime_seconds']}s -- raw response saved", flush=True)
        if result["error"]:
            print(f"      reason: {result['error']}", flush=True)

    summary = {
        "note": ("Exploratory development check, NOT the DOAR thesis benchmark. h38.jpg is "
                 "permanently development-only. The SAME frozen prompt/schema ran on every image "
                 "below -- no per-image tuning. Raw responses were saved before any comparison."),
        "model": observer.model,
        "h38": {"image": h38_result["image"], "status": h38_result["status"],
                "candidate_count": h38_result["candidate_count"],
                "runtime_seconds": h38_result["runtime_seconds"]},
        "additional_dev_images": [
            {"image": r["image"], "status": r["status"], "candidate_count": r["candidate_count"],
             "runtime_seconds": r["runtime_seconds"]}
            for r in dev_results
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved all raw responses + summary to: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
