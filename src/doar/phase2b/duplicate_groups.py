"""Phase 2B leakage control -- reuses Phase 7A/7B's existing union-find
duplicate-group computation (`partition.py::compute_duplicate_groups`)
rather than re-deriving it. Phase 2B does not lock or approve a dataset
partition (that is Phase 7B's own, separate, still-open task); it only
needs to know which images belong to the same near-duplicate group so its
own small annotation sample never treats two images from one group as
independent evidence.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The most recent real duplicate-group computation on disk. Not the
# "approved" partition (none is approved yet -- see PHASE2B_AUDIT_AND_PLAN.md
# Section 5) -- just the most complete existing grouping, reused as-is.
_PRECOMPUTED_MANIFEST = REPO_ROOT / "outputs" / "phase7b" / "final_partition" / "partition_manifest.csv"


def load_group_lookup(manifest_path: Path | None = None) -> dict[str, dict]:
    """Returns {image_id: {"group_id": ..., "group_size": ..., "path": ...,
    "conflict_status": ...}} read from the existing Phase 7B partition
    manifest. Raises FileNotFoundError with a clear message if that
    manifest (private, gitignored, local-only) is not present -- callers
    must treat that as a local-only-integration-test precondition, not
    something to fall back on silently."""
    manifest_path = manifest_path or _PRECOMPUTED_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- Phase 2B's leakage control reuses "
            "the existing Phase 7A/7B duplicate-group computation and does "
            "not recompute it from raw images. This file is local-only "
            "(gitignored, derived from the private dataset).")
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    return {
        r["image_id"]: {
            "group_id": r["group_id"],
            "group_size": int(r["group_size"]),
            "path": r["path"],
            "conflict_status": r["conflict_status"],
        }
        for r in rows
    }


def groups_available() -> bool:
    return _PRECOMPUTED_MANIFEST.exists()


def write_pilot_duplicate_groups_csv(selected_image_ids: list[str], output_path: Path,
                                      manifest_path: Path | None = None) -> Path:
    """Writes a small, committable CSV scoped ONLY to the images actually
    selected for the Phase 2B pilot sample (artifacts/phase2b/duplicate_groups.csv)
    -- never the full private manifest, which stays under gitignored outputs/."""
    lookup = load_group_lookup(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "source_image_group", "group_size_in_full_dataset"])
        for image_id in selected_image_ids:
            info = lookup.get(image_id)
            if info is None:
                raise KeyError(f"{image_id!r} not found in the Phase 7B partition manifest")
            writer.writerow([image_id, info["group_id"], info["group_size"]])
    return output_path
