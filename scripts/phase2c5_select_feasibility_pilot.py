#!/usr/bin/env python
"""Phase 2C.5 Stage 4: deterministic selection of a small (~15 image)
development-eligible feasibility-pilot subset. Locked-test images are
structurally excluded (drawn only from DEV_ELIGIBLE) and the selection is
asserted against the locked-test guard before being used anywhere.
Selection is a fixed even-stride formula over the SORTED dev-eligible
pilot_id list -- no image was hand-picked, no emotion label was consulted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import workspace as workspace_mod  # noqa: E402
from doar.phase2c2.cohorts import DEV_ELIGIBLE, split_pilot_ids_by_cohort  # noqa: E402


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    """k indices into range(n), spread as evenly as possible including
    both ends -- deterministic, no randomness, no image-content lookahead."""
    if k >= n:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def select_feasibility_pilot(mapping_path: Path, k: int = 15) -> list[str]:
    mapping_rows = workspace_mod.load_pilot_mapping(mapping_path)
    cohorts = split_pilot_ids_by_cohort(mapping_rows)
    dev_ids = cohorts[DEV_ELIGIBLE]  # already sorted by split_pilot_ids_by_cohort
    selected = [dev_ids[i] for i in evenly_spaced_indices(len(dev_ids), k)]
    workspace_mod.assert_pilot_ids_exclude_locked_test(selected, mapping_path)
    return selected


def main() -> None:
    mapping_path = ROOT / "outputs/phase2c1/private_pilot_mapping.csv"
    out_path = ROOT / "outputs/phase2c5/feasibility_pilot_ids_private.json"
    selected = select_feasibility_pilot(mapping_path, k=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"pilot_ids": selected, "n": len(selected),
                                     "cohort": DEV_ELIGIBLE, "method": "evenly_spaced_indices"},
                                    indent=2), encoding="utf-8")
    print(f"selected {len(selected)} pilot_ids -> {out_path}")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
