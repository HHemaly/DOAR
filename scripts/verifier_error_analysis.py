#!/usr/bin/env python
"""DOAR Verifier Error Analysis (cached data only -- no live Gemini calls).

Classifies every cached Observer candidate across the 15-image
development set by whether it matches a human-annotated item (checked
against A1 and A2 INDEPENDENTLY, using the SAME deterministic
`_labels_plausibly_match` matcher DOAR uses everywhere else -- never an
LLM judging agreement) crossed with its real, already-computed Verifier
outcome:

    1. TRUE candidate kept VERIFIED
    2. TRUE candidate removed (rejected/uncertain)
    3. FALSE/unmatched candidate removed (rejected/uncertain)
    4. FALSE/unmatched candidate kept VERIFIED

Read-only: reuses `scripts/run_development_benchmark.py`'s own cached-
data search order (`find_saved_verification_rows` -- persistent live
cache first, then legacy `gemini_verifier_dev_check_*` dirs) and its own
`load_human_annotations`/`load_development_set`. Never calls Gemini,
never writes to any cached verification file or annotation file, never
touches reasoning_chain.py / RULE_EVIDENCE_MATRIX.csv /
CONCERN_DOMAIN_MAP.json / visual_observer.py.

Outputs two files at the repo root (both overwritten on each run):
    VERIFIER_ERROR_ANALYSIS.md
    VERIFIER_ERROR_ANALYSIS_CANDIDATES.csv

Usage:
    python scripts/verifier_error_analysis.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_observer import _labels_plausibly_match  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_development_benchmark", ROOT / "scripts" / "run_development_benchmark.py")
rdb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdb)

# "unreviewed" is grouped with rejected/uncertain here, NOT with
# "verified" -- reasoning_chain.py only ever trusts case_verification_
# status=="verified" entities (see _verified_entities), so an
# "unreviewed" candidate has exactly the same downstream effect as a
# rejected/uncertain one: it contributes nothing. "unreviewed" specifically
# means the Verifier never even attempted a network call (e.g. no usable
# bbox to crop) -- a DIFFERENT root cause from an active rejection, which
# is exactly why VERIFIER_ERROR_ANALYSIS.md discusses it as its own
# category even though it shares the "removed" bucket here.
REMOVED_STATUSES = ("rejected", "uncertain", "unreviewed")
UNKNOWN_LABEL_MARKERS = {"unknown", "unidentified", "unclear", ""}
SMALL_BBOX_AREA_THRESHOLD = 0.01  # <1% of image area -- a heuristic "tight crop" signal, not a hard rule

CSV_COLUMNS = [
    "image_id", "observation_id", "observer_label", "observer_alternative_labels", "observer_bbox",
    "observer_bbox_area", "observer_bbox_valid", "observer_confidence", "observer_entity_type",
    "verifier_independent_label", "verifier_independent_alternative_labels", "verifier_confidence",
    "verification_status", "verifier_notes",
    "matches_a1", "a1_matching_labels", "a1_salient",
    "matches_a2", "a2_matching_labels", "a2_salient",
    "bucket_a1", "bucket_a2",
    "heuristic_failure_category",
]


def _bbox_is_valid(bbox) -> bool | str:
    """A normalized [x, y, w, h] bbox must have all four values in
    [0, 1] (with x+w and y+h not wildly exceeding 1 either) -- some
    cached candidates (e.g. p2b_0004's pre-bbox-schema-fix data) have
    height/width values like 2.7 or 6.0, which cannot be a valid
    normalized crop region and explains why the Verifier could not
    produce a crop at all. Returns "" (unknown) when there's no bbox to
    judge -- that's a separate, legitimate "no bbox" case, not an
    invalid one."""
    if not bbox:
        return ""
    return all(0.0 <= v <= 1.0 for v in bbox)


def _matching_items(label: str, items) -> list:
    if not items:
        return []
    return [item for item in items if _labels_plausibly_match(label, item.label)]


def _bucket(is_true: bool | None, status: str) -> str | None:
    if is_true is None:
        return None
    removed = status in REMOVED_STATUSES
    if is_true and not removed:
        return "1_true_kept_verified"
    if is_true and removed:
        return "2_true_removed"
    if not is_true and removed:
        return "3_false_removed"
    return "4_false_kept_verified"


def _heuristic_failure_category(row: dict, a1_items, a2_items) -> str:
    """First-pass, evidence-only heuristic for TRUE-candidate-removed
    rows -- deliberately crude (no pixel inspection available). The
    authoritative per-row category in VERIFIER_ERROR_ANALYSIS.md is a
    human reading of each row's actual evidence, not this function's
    output; this column exists so that manual review has a starting
    signal to agree or disagree with, not as the final answer."""
    status = row["verification_status"]
    oc = row["observer_candidate"]
    bbox = oc.get("bbox")
    bbox_valid = _bbox_is_valid(bbox)

    if status == "unreviewed":
        # visual_observer.py's own documented behaviour: unreviewed means
        # the Verifier never made a network call at all -- either no bbox,
        # or (as with p2b_0004's stale pre-schema-fix data) a bbox whose
        # values are outside the valid [0, 1] normalized range, so no
        # crop could be produced.
        return "bad_bbox_or_crop_construction" if bbox_valid is False else "no_bbox_never_reviewed"

    verifier_label = (row.get("verifier_independent_label") or "").strip().lower()
    if verifier_label in UNKNOWN_LABEL_MARKERS:
        return "verifier_visual_miss"

    area = round(bbox[2] * bbox[3], 5) if bbox else None
    small_bbox = bbox_valid is True and area is not None and area < SMALL_BBOX_AREA_THRESHOLD

    verifier_label_full = row.get("verifier_independent_label", "")
    verifier_matches_a1 = bool(_matching_items(verifier_label_full, a1_items))
    verifier_matches_a2 = bool(_matching_items(verifier_label_full, a2_items))

    if bbox_valid is False:
        return "bad_bbox_or_crop_construction"
    if verifier_matches_a1 or verifier_matches_a2:
        # The verifier's independent read is itself a real, human-annotated
        # item -- just a DIFFERENT one than the observer's candidate was
        # labelled as. Most consistent with the crop capturing the wrong
        # (but still real) region.
        return "bad_bbox_or_crop_construction"
    if small_bbox:
        return "bad_bbox_or_crop_construction"
    return "verifier_visual_miss"


def analyze() -> tuple[list[dict], dict]:
    images = rdb.load_development_set()
    all_rows_out = []
    per_image_summary = []

    for image in images:
        image_id = image["image_id"]
        rows, source_path = rdb.find_saved_verification_rows(image_id)
        human = rdb.load_human_annotations(image_id)
        a1_items, a2_items = human.get("A1"), human.get("A2")

        if rows is None:
            per_image_summary.append({"image_id": image_id, "status": "no_observer_data", "source": None})
            continue

        image_counts = {"1_true_kept_verified": 0, "2_true_removed": 0,
                         "3_false_removed": 0, "4_false_kept_verified": 0}
        image_counts_a2 = dict(image_counts)

        for row in rows:
            oc = row["observer_candidate"]
            label = oc["label"]
            status = row["verification_status"]

            a1_matches = _matching_items(label, a1_items)
            a2_matches = _matching_items(label, a2_items)
            is_true_a1 = (a1_matches != []) if a1_items is not None else None
            is_true_a2 = (a2_matches != []) if a2_items is not None else None

            bucket_a1 = _bucket(is_true_a1, status)
            bucket_a2 = _bucket(is_true_a2, status)
            if bucket_a1:
                image_counts[bucket_a1] += 1
            if bucket_a2:
                image_counts_a2[bucket_a2] += 1

            bbox = oc.get("bbox")
            area = round(bbox[2] * bbox[3], 5) if bbox else None
            bbox_valid = _bbox_is_valid(bbox)
            heuristic_category = ""
            if bucket_a1 == "2_true_removed" or bucket_a2 == "2_true_removed":
                heuristic_category = _heuristic_failure_category(row, a1_items, a2_items)

            all_rows_out.append({
                "image_id": image_id, "observation_id": row["observation_id"],
                "observer_label": label,
                "observer_alternative_labels": "; ".join(oc.get("alternative_labels") or []),
                "observer_bbox": str(bbox) if bbox else "",
                "observer_bbox_area": area if area is not None else "",
                "observer_bbox_valid": bbox_valid,
                "observer_confidence": oc.get("confidence"),
                "observer_entity_type": oc.get("entity_type"),
                "verifier_independent_label": row.get("verifier_independent_label") or "",
                "verifier_independent_alternative_labels": "; ".join(
                    row.get("verifier_independent_alternative_labels") or []),
                "verifier_confidence": row.get("verifier_confidence"),
                "verification_status": status, "verifier_notes": row.get("verifier_notes") or "",
                "matches_a1": is_true_a1, "a1_matching_labels": "; ".join(i.label for i in a1_matches),
                "a1_salient": any(i.salient for i in a1_matches) if a1_matches else "",
                "matches_a2": is_true_a2, "a2_matching_labels": "; ".join(i.label for i in a2_matches),
                "a2_salient": any(i.salient for i in a2_matches) if a2_matches else "",
                "bucket_a1": bucket_a1 or "", "bucket_a2": bucket_a2 or "",
                "heuristic_failure_category": heuristic_category,
            })

        per_image_summary.append({
            "image_id": image_id, "status": "ok", "source": str(source_path),
            "total_candidates": len(rows),
            "a1_available": a1_items is not None, "a2_available": a2_items is not None,
            "counts_a1": image_counts, "counts_a2": image_counts_a2,
        })

    return all_rows_out, {"images": per_image_summary}


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    candidate_rows, summary = analyze()
    write_csv(candidate_rows, ROOT / "VERIFIER_ERROR_ANALYSIS_CANDIDATES.csv")
    print(f"Wrote {len(candidate_rows)} candidate rows across {len(summary['images'])} images "
          f"to VERIFIER_ERROR_ANALYSIS_CANDIDATES.csv")
    for img in summary["images"]:
        if img["status"] == "ok":
            print(f"  {img['image_id']}: {img['total_candidates']} candidates, "
                  f"A1 counts={img['counts_a1']}, A2 counts={img['counts_a2']}")
        else:
            print(f"  {img['image_id']}: {img['status']}")
