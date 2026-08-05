"""Runs the page-frame assessability audit over a real, non-test sample
of the dataset (DOAR-TRACE Phase 2A, Section 3). Writes
`artifacts/phase2a/dataset_page_frame_audit.csv`. Never accesses
`split == "test"`.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .analysis import _composition, _segment
from .page_frame import ASSESSABLE_STATUSES, assess_page_frame

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs" / "phase5" / "manifest.csv"
OUTPUT_PATH = ROOT / "artifacts" / "phase2a" / "dataset_page_frame_audit.csv"

FIELDS = [
    "image_id", "source_split", "source_folder_label", "page_frame_status", "confidence",
    "current_bounding_box_coverage", "page_size_rules_assessable", "placement_rules_assessable", "notes",
]


def _sample_balanced(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] == "test":
            continue
        by_class.setdefault(row["class"], []).append(row)
    sample = []
    for cls, items in sorted(by_class.items()):
        take = items if len(items) <= per_class else rng.sample(items, per_class)
        sample.extend(take)
    return sample


def audit_one_image(row: dict) -> dict[str, Any]:
    try:
        image = Image.open(row["path"]).convert("RGB")
        rgb = np.asarray(image)
        mask, _background, _conf, _candidates, diagnostics = _segment(rgb)
        composition = _composition(mask)
        assessment = assess_page_frame(rgb, mask, diagnostics.get("background_stability", 0.0))
        return {
            "image_id": row["image_id"], "source_split": row["split"], "source_folder_label": row["class"],
            "page_frame_status": assessment.page_frame_status, "confidence": assessment.confidence,
            "current_bounding_box_coverage": round(composition["bounding_box_coverage"], 4),
            "page_size_rules_assessable": assessment.assessable,
            "placement_rules_assessable": assessment.assessable,
            "notes": "",
        }
    except Exception as exc:  # a real per-image failure, honestly recorded, not silently skipped
        return {
            "image_id": row["image_id"], "source_split": row["split"], "source_folder_label": row["class"],
            "page_frame_status": "failed", "confidence": 0.0, "current_bounding_box_coverage": None,
            "page_size_rules_assessable": False, "placement_rules_assessable": False,
            "notes": f"error: {exc}",
        }


def run_audit(
    manifest_path: Path = DEFAULT_MANIFEST, per_class: int = 30, seed: int = 42, output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    """`output_path` defaults to the canonical committed artifact --
    override it (e.g. in tests using a small `per_class`) so a reduced
    sample never clobbers the real, full-sample audit result."""
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    sample = _sample_balanced(rows, per_class=per_class, seed=seed)
    results = [audit_one_image(row) for row in sample]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r["page_frame_status"]] = status_counts.get(r["page_frame_status"], 0) + 1
    n_assessable = sum(1 for r in results if r["page_frame_status"] in ASSESSABLE_STATUSES)
    return {
        "n_images": len(results), "status_counts": status_counts,
        "n_assessable": n_assessable, "assessable_fraction": round(n_assessable / max(1, len(results)), 4),
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    summary = run_audit()
    print(summary)
