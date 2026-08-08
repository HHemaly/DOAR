"""Splits the 80 pilot_ids into the three reporting cohorts Phase 2C.2
requires: the full descriptive cohort, the development-eligible subset
(original test split excluded), and the locked-test subset (descriptive
only, never for model selection/threshold choice/calibration/training).
"""
from __future__ import annotations

FULL = "full_80"
DEV_ELIGIBLE = "dev_eligible_excl_test"
LOCKED_TEST = "locked_test_descriptive_only"

LOCKED_TEST_SPLIT_VALUE = "test"


def split_pilot_ids_by_cohort(mapping_rows: list[dict]) -> dict[str, list[str]]:
    all_ids = sorted(r["pilot_id"] for r in mapping_rows)
    locked_test = sorted(r["pilot_id"] for r in mapping_rows
                          if r.get("original_split") == LOCKED_TEST_SPLIT_VALUE)
    dev_eligible = sorted(set(all_ids) - set(locked_test))
    return {FULL: all_ids, DEV_ELIGIBLE: dev_eligible, LOCKED_TEST: locked_test}
