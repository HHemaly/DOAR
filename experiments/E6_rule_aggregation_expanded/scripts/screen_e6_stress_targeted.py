#!/usr/bin/env python
"""E6-STRESS-TARGETED screen -- predeclared selection criterion (Part 1's
own feasibility finding), checked against ALREADY-CACHED deterministic
features only. NO new API calls, NO new perception. Selection criterion,
declared BEFORE examining any strategy outcome: a case where
segmentation.bounding_box_coverage <= 0.20 (PSY_AR_SIZE_SMALL_016's own
threshold) AND stroke.intensity_proxy <= INTENSITY_PROXY_LIGHT_THRESHOLD
(EN_COMPILED_LINE_LIGHT_PRESSURE_031's own threshold) BOTH hold -- the
only currently-enabled two-independent-family same-domain combination in
the registry (anxiety_or_stress_related; see REGISTRY_FEASIBILITY_REPORT.md).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar import drawing_synthesis as ds  # noqa: E402
from doar.rule_engine_v2 import INTENSITY_PROXY_LIGHT_THRESHOLD  # noqa: E402

SIZE_SMALL_MAX = 0.20


def main():
    with open(EXP_DIR / "raw" / "cohort_manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    out_rows = []
    hits = []
    for row in manifest:
        if row["cohort_role"] == "locked_benchmark_reserved_do_not_process":
            continue
        image_id = row["image_id"]
        cache_path = ds.deterministic_cache_path(image_id)
        if not cache_path.exists():
            continue
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        of = data["objective_features"]
        bbox = of.get("segmentation.bounding_box_coverage", {})
        intensity = of.get("stroke.intensity_proxy", {})
        bbox_val = bbox.get("value") if not bbox.get("missing") else None
        intensity_val = intensity.get("value") if not intensity.get("missing") else None
        small_ok = bbox_val is not None and 0 < bbox_val <= SIZE_SMALL_MAX
        light_ok = intensity_val is not None and intensity_val <= INTENSITY_PROXY_LIGHT_THRESHOLD
        both = small_ok and light_ok
        if both:
            hits.append(image_id)
        out_rows.append({
            "case_id": image_id, "bounding_box_coverage": bbox_val, "size_small_satisfied": small_ok,
            "stroke_intensity_proxy": intensity_val, "line_light_satisfied": light_ok,
            "meets_both_r4_reachable_criteria": both,
        })

    out_path = EXP_DIR / "eligibility_fix" / "E6_STRESS_TARGETED_screen.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {out_path} ({len(out_rows)} cases screened)")
    print(f"Cases meeting BOTH criteria: {hits or 'NONE'}")
    n_small = sum(1 for r in out_rows if r["size_small_satisfied"])
    n_light = sum(1 for r in out_rows if r["line_light_satisfied"])
    print(f"size_small_satisfied alone: {n_small}/{len(out_rows)}")
    print(f"line_light_satisfied alone: {n_light}/{len(out_rows)}")
    return out_rows, hits


if __name__ == "__main__":
    main()
