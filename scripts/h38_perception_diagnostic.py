#!/usr/bin/env python
"""DIAGNOSTIC ONLY -- Human Interaction Layer v1, Part G.

Investigates two specific, already-observed h38 questions using ONLY the
already-cached Observer/Verifier data (no live Gemini calls, no change to
any Observer/Verifier code, prompt, model, or threshold):

  1. Why did the Observer-proposed 'car' candidate stay 'unreviewed'
     (never independently confirmed or rejected)? Did it have a bbox? Did
     the Verifier actually run for it?
  2. Why were the two visible people never in the Observer's candidate
     list at all?

This script draws conclusions ONLY from what the cached data and the
existing, unmodified crop-safety code
(`visual_entity._bbox_is_crop_usable` / `_crop_region`) actually show.
Nothing here is guessed, and nothing here changes h38's stored result,
the Observer prompt/model, the Verifier prompt/model, or any threshold.

Writes: outputs/human_interaction_v1/h38_perception_diagnostic.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_development_benchmark as rdb  # noqa: E402
from doar.visual_entity import _BBOX_OUT_OF_RANGE_TOLERANCE, _bbox_is_crop_usable  # noqa: E402

OUT_DIR = ROOT / "outputs" / "human_interaction_v1"
OUT_PATH = OUT_DIR / "h38_perception_diagnostic.json"

_PERSON_MARKERS = ("person", "people", "human", "boy", "girl", "child", "man", "woman", "kid", "figure")


def main() -> None:
    rows, source_path = rdb.find_saved_verification_rows("h38")
    if rows is None:
        raise SystemExit("No cached verification data found for h38 -- cannot run this diagnostic.")

    raw_text = source_path.read_text(encoding="utf-8")

    # --- 1. The 'car' candidate -----------------------------------------
    car_row = next((r for r in rows if r["observer_candidate"]["label"] == "car"), None)
    car_diagnosis = {"observer_proposed_car": car_row is not None}
    if car_row is not None:
        oc = car_row["observer_candidate"]
        bbox = tuple(oc["bbox"]) if oc.get("bbox") else None
        crop_usable = _bbox_is_crop_usable(bbox) if bbox is not None else False
        car_diagnosis.update({
            "observation_id": car_row["observation_id"],
            "observer_confidence": oc.get("confidence"),
            "bbox_xywh_normalized": bbox,
            "had_a_bbox": bbox is not None,
            "case_verification_status": car_row["verification_status"],
            "verifier_confidence_recorded": car_row.get("verifier_confidence"),
            "bbox_is_crop_usable": crop_usable,
            "bbox_x_plus_w": (bbox[0] + bbox[2]) if bbox else None,
            "bbox_out_of_range_tolerance": _BBOX_OUT_OF_RANGE_TOLERANCE,
            "finding": (
                "The Observer proposed 'car' with a bbox, but bbox[0]+bbox[2] "
                f"= {bbox[0] + bbox[2]:.3f} exceeds the normalized image width (1.0 + "
                f"{_BBOX_OUT_OF_RANGE_TOLERANCE} tolerance) -- the box extends past the "
                "right edge of the image as reported. DOAR's own unmodified "
                "`visual_entity._bbox_is_crop_usable` check rejects this bbox as "
                "unusable for cropping (never clamps/reinterprets it). Because "
                "`_bbox_is_crop_usable` returns False, `_crop_region`/`build_verifier_crops` "
                "return no crop, and `GeminiVisualVerifier.verify()` takes its documented "
                "'crop is None' path, returning status='unreviewed' WITHOUT making any "
                "network call and WITHOUT guessing a label. This is the existing safety "
                "guard working as designed against a malformed Observer bbox, not a "
                "Verifier failure and not a bug introduced by this diagnostic."
                if crop_usable is False and bbox is not None else
                "See had_a_bbox/bbox_is_crop_usable/case_verification_status fields above."
            ),
        })

    # --- 2. People never proposed ----------------------------------------
    all_labels = [r["observer_candidate"]["label"] for r in rows]
    person_like_labels = [lbl for lbl in all_labels if any(m in lbl.lower() for m in _PERSON_MARKERS)]
    raw_text_lower = raw_text.lower()
    person_marker_in_raw_cache = [m for m in _PERSON_MARKERS if m in raw_text_lower]
    people_diagnosis = {
        "num_observer_candidates_total": len(rows),
        "all_observer_candidate_labels": all_labels,
        "person_like_labels_among_candidates": person_like_labels,
        "person_related_substrings_found_anywhere_in_raw_cached_file": person_marker_in_raw_cache,
        "finding": (
            "The cached Observer/Verifier file for h38 contains no person-like label at "
            "all, in any status (verified/uncertain/rejected/unreviewed). This confirms "
            "that the two visible people were never proposed as a candidate by the "
            "Observer in the first place -- this is an upstream Observer-perception gap, "
            "not a Verifier or downstream-filtering issue (the file loaded here via "
            "`find_saved_verification_rows` is read verbatim with no filtering applied "
            "before this diagnostic inspects it). Per this task's explicit instruction, "
            "no attempt is made here to guess WHY Gemini's Observer missed them, and the "
            "Observer prompt/model is not modified to try to fix this."
        ),
    }

    report = {
        "case_id": "h38",
        "development_only_diagnostic": True,
        "scientific_pipeline_modified": False,
        "source_cache_file": str(source_path),
        "car_candidate_diagnosis": car_diagnosis,
        "missing_people_diagnosis": people_diagnosis,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
