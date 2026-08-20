#!/usr/bin/env python
"""ONE-TIME E6-EXPANDED perception-generation script.

Runs the FROZEN, unmodified GeminiVisualObserver + GeminiVisualVerifier
(same classes, same prompts, same models -- gemini-3.6-flash / gemini-
3.5-flash-lite -- production `run_development_benchmark.run_live_
observer_and_verifier`, imported unchanged, never reimplemented) against
each new candidate image, and saves results to the SAME
`development_live_cache/` directory + format the existing 15-case
development set already uses, via the SAME `live_cache_path()` /
`_atomic_write_json()` functions -- so every downstream E6 tool
(`clinician_review_app.load_case_bundle`, `reasoning_chain`,
`drawing_synthesis`) reads this new evidence identically to the original
15 cases, no special-casing anywhere.

Checkpointed: writes each image's result to cache immediately after that
image completes, and appends a status line to `progress.jsonl` -- a
partial/interrupted run loses nothing already completed, and the exact
achieved N is always recoverable by reading progress.jsonl, never
silently reported as "done" if it is not.

NO scientific code changed: does not touch RULE_EVIDENCE_MATRIX.csv,
RULE_RELATIONSHIP_GRAPH.json, CONCERN_DOMAIN_MAP.json, GeminiVisualObserver
or GeminiVisualVerifier's prompts/models, or any threshold.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_development_benchmark as rdb  # noqa: E402
from doar.visual_observer import GeminiVisualObserver, GeminiVisualVerifier  # noqa: E402

CANDIDATES_PATH = ROOT / "outputs" / "human_interaction_v1" / "e6_expand_candidates.json"
PROGRESS_PATH = ROOT / "outputs" / "human_interaction_v1" / "e6_expand_progress.jsonl"


def already_done(image_id: str) -> bool:
    rows, _ = rdb.find_saved_verification_rows(image_id)
    return rows is not None


def main() -> None:
    data = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    candidates = data["candidates_new"]
    print(f"{len(candidates)} candidate images to process.")

    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    observer = GeminiVisualObserver()
    verifier = GeminiVisualVerifier()
    print(f"observer model: {observer.model}  verifier model: {verifier.model}")

    n_done = n_skipped = n_failed = 0
    consecutive_failures = 0
    t_start = time.time()
    with open(PROGRESS_PATH, "a", encoding="utf-8") as progress_f:
        for i, (image_id, rel_path) in enumerate(candidates, start=1):
            if already_done(image_id):
                n_skipped += 1
                progress_f.write(json.dumps({"image_id": image_id, "status": "skipped_already_cached"}) + "\n")
                progress_f.flush()
                continue
            image_path = ROOT / rel_path
            t0 = time.time()
            try:
                rows = rdb.run_live_observer_and_verifier(image_id, image_path, observer=observer, verifier=verifier)
                live_path = rdb.live_cache_path(image_id)
                rdb._atomic_write_json(live_path, rows)
                n_done += 1
                consecutive_failures = 0
                elapsed = time.time() - t0
                total_elapsed = time.time() - t_start
                print(f"[{i}/{len(candidates)}] {image_id}: OK, {len(rows)} rows, {elapsed:.1f}s "
                      f"(total {total_elapsed:.0f}s, done={n_done} failed={n_failed})", flush=True)
                progress_f.write(json.dumps({"image_id": image_id, "status": "ok", "n_rows": len(rows),
                                              "elapsed_seconds": elapsed}) + "\n")
                progress_f.flush()
            except Exception as exc:
                n_failed += 1
                consecutive_failures += 1
                print(f"[{i}/{len(candidates)}] {image_id}: FAILED -- {type(exc).__name__}: {exc}", flush=True)
                progress_f.write(json.dumps({"image_id": image_id, "status": "failed",
                                              "error_type": type(exc).__name__, "error": str(exc)}) + "\n")
                progress_f.flush()
                # 5 CONSECUTIVE failures (regardless of prior successes) means
                # further attempts are almost certainly futile (quota just got
                # exhausted mid-run) -- stop cleanly rather than burn through
                # the remaining candidates against a now-dead key/quota.
                if consecutive_failures >= 5:
                    print(f"5 consecutive failures -- stopping early (likely quota/auth issue partway "
                          f"through). {n_done} images successfully processed before this point. "
                          f"See progress.jsonl for detail.", flush=True)
                    break

    print(f"\nDONE. {n_done} newly processed, {n_skipped} already cached, {n_failed} failed. "
          f"Total time: {time.time()-t_start:.0f}s.")


if __name__ == "__main__":
    main()
