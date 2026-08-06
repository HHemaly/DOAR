#!/usr/bin/env python
"""Phase 2B: select and blind-copy a pilot annotation sample, and write
the duplicate-group leakage-control artifact for the selected images.

Usage:
    python scripts/phase2b_audit_dataset.py --n 20 --output-dir <dir>

Requires outputs/phase7b/final_partition/partition_manifest.csv (local-
only, gitignored, derived from the private dataset) -- this is a real,
local-integration script, not something the automated test suite runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.dataset import select_pilot_sample, blind_copy_sample, write_pilot_mapping_csv  # noqa: E402
from doar.phase2b.duplicate_groups import write_pilot_duplicate_groups_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80,
                        help="number of images to select and blind-copy. NOTE: "
                             "blind_copy_sample's re-ID shuffle depends on the full "
                             "length of the selected list, so a smaller --n does NOT "
                             "reproduce a prefix of a larger run's pilot_id assignment "
                             "-- always pass the same --n used for the run you want to "
                             "reproduce. The real Phase 2B pilot used --n 80 and "
                             "hand-annotated only the first 20 by pilot_id (p2b_0000-"
                             "p2b_0019); see docs/PHASE2B_ANNOTATION_PROTOCOL.md.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="where to write the blinded image copies and mapping CSV "
                             "(should NOT be inside the git repo -- these are real, private "
                             "dataset images)")
    args = parser.parse_args()

    selected = select_pilot_sample(args.n, seed=args.seed)
    print(f"selected {len(selected)} images, one per duplicate group")

    records = blind_copy_sample(selected, args.output_dir / "images", seed=args.seed)
    mapping_path = write_pilot_mapping_csv(records, args.output_dir / "pilot_mapping.csv")
    print(f"wrote {len(records)} blinded images to {args.output_dir / 'images'}")
    print(f"wrote re-identification mapping to {mapping_path}")

    image_ids = [r["image_id"] for r in records]
    groups_path = write_pilot_duplicate_groups_csv(
        image_ids, ROOT / "artifacts" / "phase2b" / "duplicate_groups.csv")
    print(f"wrote leakage-control artifact to {groups_path}")


if __name__ == "__main__":
    main()
