#!/usr/bin/env python
"""Builds raw/cohort_manifest.csv -- the frozen, complete record of which
images are IN the E6-EXPANDED cohort, which are the reserved locked
benchmark, and which have usable cached DOAR evidence at freeze time.

Frozen BEFORE any strategy comparison is run (Section A/D requirement).
Never edited by hand after being written -- if the cohort needs to
change, re-run this script and note the change in PROTOCOL.md's
changelog, never silently overwrite without a record.

Does not modify DEVELOPMENT_SET_15.json (frozen, permanent) or the
private_pilot_mapping.csv (frozen provenance record) -- read-only.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation_expanded"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_development_benchmark as rdb  # noqa: E402


def main() -> None:
    mapping = {}
    with open(ROOT / "outputs" / "phase2c1" / "private_pilot_mapping.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mapping[r["pilot_id"]] = r

    dev15 = json.loads((ROOT / "DEVELOPMENT_SET_15.json").read_text(encoding="utf-8"))
    dev15_by_id = {im["image_id"]: im for im in dev15["images"]}

    candidates_path = ROOT / "outputs" / "human_interaction_v1" / "e6_expand_candidates.json"
    candidates_data = json.loads(candidates_path.read_text(encoding="utf-8"))
    locked_benchmark_ids = set(candidates_data["locked_benchmark"])
    new_candidate_pairs = {pid: rel for pid, rel in candidates_data["candidates_new"]}

    rows = []

    # 1. E6-PILOT's original 15 (including h38, which is not a pool image).
    for image_id, im in dev15_by_id.items():
        rows.append({
            "image_id": image_id, "relative_path": im["relative_path"],
            "cohort_role": "e6_pilot_original_15",
            "original_split": mapping.get(image_id, {}).get("original_split", "not_applicable_h38_anchor"),
            "already_exposed_to_observer_verifier_before_e6": im.get("already_exposed_to_observer_verifier", False),
        })

    # 2. New expansion candidates (may or may not have completed live perception yet).
    for image_id, rel_path in sorted(new_candidate_pairs.items()):
        rows.append({
            "image_id": image_id, "relative_path": rel_path,
            "cohort_role": "e6_expanded_new",
            "original_split": mapping.get(image_id, {}).get("original_split", "unknown"),
            "already_exposed_to_observer_verifier_before_e6": False,
        })

    # 3. Locked benchmark (reserved, NEVER processed or inspected by E6).
    for image_id in sorted(locked_benchmark_ids):
        r = mapping[image_id]
        rel_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
            p = ROOT / "outputs" / "phase2c1" / "private_images" / f"{image_id}{ext}"
            if p.exists():
                rel_path = str(p.relative_to(ROOT)).replace("\\", "/")
                break
        rows.append({
            "image_id": image_id, "relative_path": rel_path,
            "cohort_role": "locked_benchmark_reserved_do_not_process",
            "original_split": r["original_split"],
            "already_exposed_to_observer_verifier_before_e6": False,
        })

    # Determine, for every row, whether cached evidence currently exists --
    # WITHOUT triggering any live call (find_saved_verification_rows is a
    # pure cache read).
    for row in rows:
        if row["cohort_role"] == "locked_benchmark_reserved_do_not_process":
            row["has_cached_evidence"] = False  # deliberately never checked/processed
            row["excluded_reason"] = "reserved as locked independent benchmark (original_split=='test' in " \
                                      "private_pilot_mapping.csv, never used in E6-PILOT) -- not processed, " \
                                      "not inspected, per explicit instruction to preserve a genuinely blind set"
            continue
        cached_rows, _ = rdb.find_saved_verification_rows(row["image_id"])
        row["has_cached_evidence"] = cached_rows is not None
        row["excluded_reason"] = "" if cached_rows is not None else \
            "no cached Observer/Verifier evidence yet (live perception run pending/failed for this image)"

    out_path = EXP_DIR / "raw" / "cohort_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "relative_path", "cohort_role", "original_split",
                  "already_exposed_to_observer_verifier_before_e6", "has_cached_evidence", "excluded_reason"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n_usable = sum(1 for r in rows if r["cohort_role"] != "locked_benchmark_reserved_do_not_process"
                   and r["has_cached_evidence"])
    n_pending = sum(1 for r in rows if r["cohort_role"] == "e6_expanded_new" and not r["has_cached_evidence"])
    print(f"Wrote {out_path}: {len(rows)} total rows.")
    print(f"  e6_pilot_original_15: {sum(1 for r in rows if r['cohort_role']=='e6_pilot_original_15')}")
    print(f"  e6_expanded_new: {sum(1 for r in rows if r['cohort_role']=='e6_expanded_new')} "
          f"({sum(1 for r in rows if r['cohort_role']=='e6_expanded_new' and r['has_cached_evidence'])} cached, "
          f"{n_pending} pending live perception)")
    print(f"  locked_benchmark_reserved_do_not_process: "
          f"{sum(1 for r in rows if r['cohort_role']=='locked_benchmark_reserved_do_not_process')}")
    print(f"  TOTAL usable for E6-EXPANDED right now: {n_usable}")


if __name__ == "__main__":
    main()
