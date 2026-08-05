"""Threshold provenance and sensitivity audit (DOAR-TRACE Phase 2A,
Section 6). Sweeps each currently-executable rule's threshold across a
documented sensitivity range and measures the real trigger rate on a
real, non-test sample -- **never** tuned to agree with the
Angry/Fear/Happy/Sad folder labels (this module never reads the `class`
column for anything other than building a class-balanced sample).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .analysis import _composition, _segment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs" / "phase5" / "manifest.csv"
CSV_PATH = ROOT / "artifacts" / "phase2a" / "threshold_sensitivity.csv"

CSV_FIELDS = [
    "rule_id", "feature_id", "current_threshold", "source_status", "rationale", "sweep_value",
    "n_triggered", "n_total", "pct_triggered", "impact_on_individual_suggestions", "impact_on_combined_hypotheses",
]

_COMBINED_IMPACT_NOTE = (
    "Negligible under the current Level A/B/C policy: no two of the 6 original executable rules ever share a "
    "target_construct, so none can combine with another registry rule; the only demonstrated Level-C convergence "
    "path requires the expressive-content model's independent contribution (see docs/AGGREGATION_POLICY.md)."
)

RULES: list[dict[str, Any]] = [
    {"rule_id": "PSY_AR_SIZE_SMALL_016", "feature_id": "composition.bounding_box_coverage",
     "current_threshold": "<= 0.20", "source_status": "directly_sourced",
     "rationale": "The Arabic source states this percentage exactly ('لا تتجاوز 20%').",
     "sweep": [0.10, 0.15, 0.20, 0.25, 0.30], "kind": "le"},
    {"rule_id": "PSY_AR_SIZE_HALF_014", "feature_id": "composition.bounding_box_coverage",
     "current_threshold": "0.40 <= x <= 0.60", "source_status": "source_centre_with_invented_band",
     "rationale": "Source states 'about 50%'; the +/-10 percentage-point band around that center is an implementation choice.",
     "sweep": [(0.45, 0.55), (0.425, 0.575), (0.40, 0.60), (0.375, 0.625), (0.35, 0.65)], "kind": "band"},
    {"rule_id": "PSY_AR_SIZE_FULL_015", "feature_id": "composition.bounding_box_coverage",
     "current_threshold": ">= 0.90", "source_status": "invented_operational_standin",
     "rationale": "Source says only 'covers the whole page' (no percentage); 0.90 is an invented numeric stand-in.",
     "sweep": [0.80, 0.85, 0.90, 0.925, 0.95], "kind": "ge"},
    {"rule_id": "PSY_AR_PLACE_TOP_017", "feature_id": "composition.centroid_normalized[1]",
     "current_threshold": "cy < 0.40", "source_status": "invented_operational_standin",
     "rationale": "Source gives no numeric boundary for 'top' at all.",
     "sweep": [0.30, 0.35, 0.40, 0.45], "kind": "lt_y"},
    {"rule_id": "PSY_AR_PLACE_LEFT_018", "feature_id": "composition.centroid_normalized[0]",
     "current_threshold": "cx < 0.40", "source_status": "invented_operational_standin",
     "rationale": "Source gives no numeric boundary for 'left' at all.",
     "sweep": [0.30, 0.35, 0.40, 0.45], "kind": "lt_x"},
    {"rule_id": "PSY_AR_PLACE_RIGHT_019", "feature_id": "composition.centroid_normalized[0]",
     "current_threshold": "cx > 0.60", "source_status": "invented_operational_standin",
     "rationale": "Source gives no numeric boundary for 'right' at all.",
     "sweep": [0.55, 0.60, 0.65, 0.70], "kind": "gt_x"},
]


def _sample(manifest_path: Path, n_per_class: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] == "test":
            continue
        by_class.setdefault(row["class"], []).append(row)
    sample = []
    for cls, items in sorted(by_class.items()):
        sample.extend(items if len(items) <= n_per_class else rng.sample(items, n_per_class))
    return sample


def _measure(rows: list[dict]) -> list[tuple[float, float]]:
    """Returns [(bbox_coverage, centroid_x_or_nan, centroid_y_or_nan), ...] as (cov, cx, cy) tuples."""
    out = []
    for row in rows:
        try:
            image = Image.open(row["path"]).convert("RGB")
            rgb = np.asarray(image)
            mask, _bg, _conf, _cands, _diag = _segment(rgb)
            comp = _composition(mask)
            centroid = comp["centroid_normalized"]
            out.append((comp["bounding_box_coverage"], centroid[0] if centroid else float("nan"),
                        centroid[1] if centroid else float("nan")))
        except Exception:
            continue
    return out


def run_threshold_sensitivity(manifest_path: Path = DEFAULT_MANIFEST, n_per_class: int = 75, seed: int = 42) -> dict[str, Any]:
    sample_rows = _sample(manifest_path, n_per_class, seed)
    measurements = _measure(sample_rows)
    n_total = len(measurements)
    coverages = [m[0] for m in measurements]
    cxs = [m[1] for m in measurements if m[1] == m[1]]  # drop NaN
    cys = [m[2] for m in measurements if m[2] == m[2]]

    out_rows = []
    for rule in RULES:
        kind = rule["kind"]
        for sweep_value in rule["sweep"]:
            if kind == "le":
                n_trig = sum(1 for c in coverages if 0 < c <= sweep_value)
            elif kind == "ge":
                n_trig = sum(1 for c in coverages if c >= sweep_value)
            elif kind == "band":
                lo, hi = sweep_value
                n_trig = sum(1 for c in coverages if lo <= c <= hi)
            elif kind == "lt_y":
                n_trig = sum(1 for c in cys if c < sweep_value)
            elif kind == "lt_x":
                n_trig = sum(1 for c in cxs if c < sweep_value)
            elif kind == "gt_x":
                n_trig = sum(1 for c in cxs if c > sweep_value)
            else:
                n_trig = 0
            denom = n_total if kind in ("le", "ge", "band") else len(cxs if "x" in kind else cys)
            out_rows.append({
                "rule_id": rule["rule_id"], "feature_id": rule["feature_id"],
                "current_threshold": rule["current_threshold"], "source_status": rule["source_status"],
                "rationale": rule["rationale"], "sweep_value": str(sweep_value),
                "n_triggered": n_trig, "n_total": denom,
                "pct_triggered": round(100 * n_trig / max(1, denom), 2),
                "impact_on_individual_suggestions": f"{n_trig} of {denom} sampled images would produce a Level-B suggestion at this value.",
                "impact_on_combined_hypotheses": _COMBINED_IMPACT_NOTE,
            })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    return {"n_total_sampled": n_total, "n_rows": len(out_rows), "output_path": str(CSV_PATH)}


if __name__ == "__main__":
    print(run_threshold_sensitivity())
