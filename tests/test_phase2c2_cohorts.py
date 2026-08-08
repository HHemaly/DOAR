"""Cohort-splitting tests -- synthetic mapping rows only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c2.cohorts import DEV_ELIGIBLE, FULL, LOCKED_TEST, split_pilot_ids_by_cohort


class SplitPilotIdsByCohortTests(unittest.TestCase):
    def _rows(self):
        return [
            {"pilot_id": "p2b_0000", "original_split": "train"},
            {"pilot_id": "p2b_0001", "original_split": "valid"},
            {"pilot_id": "p2b_0002", "original_split": "test"},
            {"pilot_id": "p2b_0003", "original_split": "test"},
        ]

    def test_full_includes_everyone(self):
        result = split_pilot_ids_by_cohort(self._rows())
        self.assertEqual(result[FULL], ["p2b_0000", "p2b_0001", "p2b_0002", "p2b_0003"])

    def test_locked_test_only_test_split(self):
        result = split_pilot_ids_by_cohort(self._rows())
        self.assertEqual(result[LOCKED_TEST], ["p2b_0002", "p2b_0003"])

    def test_dev_eligible_excludes_test(self):
        result = split_pilot_ids_by_cohort(self._rows())
        self.assertEqual(result[DEV_ELIGIBLE], ["p2b_0000", "p2b_0001"])

    def test_dev_eligible_and_locked_test_partition_full(self):
        result = split_pilot_ids_by_cohort(self._rows())
        self.assertEqual(sorted(result[DEV_ELIGIBLE] + result[LOCKED_TEST]), result[FULL])

    def test_no_test_split_present(self):
        rows = [{"pilot_id": "p2b_0000", "original_split": "train"}]
        result = split_pilot_ids_by_cohort(rows)
        self.assertEqual(result[LOCKED_TEST], [])
        self.assertEqual(result[DEV_ELIGIBLE], ["p2b_0000"])


if __name__ == "__main__":
    unittest.main()
