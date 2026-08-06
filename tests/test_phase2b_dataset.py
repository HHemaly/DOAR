"""Requires the local, private Phase 7B partition manifest -- see
test_phase2b_duplicate_groups.py's module docstring for why this is a
skip-guarded local-only integration test, not a CI test."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.dataset import blind_copy_sample, select_pilot_sample, write_pilot_mapping_csv
from doar.phase2b.duplicate_groups import groups_available

_AVAILABLE = groups_available()
_REASON = "requires the local, private Phase 7B partition manifest (gitignored, not available in CI)"


@unittest.skipUnless(_AVAILABLE, _REASON)
class SelectPilotSampleTests(unittest.TestCase):
    def test_selects_the_requested_count(self):
        selected = select_pilot_sample(5)
        self.assertEqual(len(selected), 5)

    def test_every_selected_image_is_in_a_distinct_duplicate_group(self):
        selected = select_pilot_sample(10)
        groups = [r["group_id"] for r in selected]
        self.assertEqual(len(groups), len(set(groups)))  # no group leakage

    def test_deterministic_given_the_same_seed(self):
        a = select_pilot_sample(10, seed=123)
        b = select_pilot_sample(10, seed=123)
        self.assertEqual(a, b)

    def test_different_seeds_can_select_different_samples(self):
        a = select_pilot_sample(10, seed=1)
        b = select_pilot_sample(10, seed=2)
        self.assertNotEqual([r["image_id"] for r in a], [r["image_id"] for r in b])

    def test_requesting_more_than_available_groups_raises(self):
        with self.assertRaises(ValueError):
            select_pilot_sample(10_000_000)


@unittest.skipUnless(_AVAILABLE, _REASON)
class BlindCopySampleTests(unittest.TestCase):
    def test_blinded_filenames_carry_no_class_or_split_information(self):
        selected = select_pilot_sample(5)
        with tempfile.TemporaryDirectory() as d:
            records = blind_copy_sample(selected, Path(d) / "images")
            for r in records:
                self.assertTrue(r["pilot_id"].startswith("p2b_"))
                # The real path (with class-folder info) must never leak
                # into the blinded filename itself.
                blind_name = Path(r["blind_path"]).name
                self.assertNotIn("Angry", blind_name)
                self.assertNotIn("Happy", blind_name)
                self.assertNotIn("Sad", blind_name)
                self.assertNotIn("Fear", blind_name)

    def test_same_n_and_seed_reproduces_the_same_pilot_id_assignment(self):
        selected = select_pilot_sample(8)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = blind_copy_sample(selected, Path(d1))
            b = blind_copy_sample(selected, Path(d2))
            self.assertEqual([(r["pilot_id"], r["image_id"]) for r in a],
                             [(r["pilot_id"], r["image_id"]) for r in b])

    def test_mapping_csv_has_the_required_columns(self):
        selected = select_pilot_sample(3)
        with tempfile.TemporaryDirectory() as d:
            records = blind_copy_sample(selected, Path(d) / "images")
            mapping_path = write_pilot_mapping_csv(records, Path(d) / "mapping.csv")
            header = mapping_path.read_text(encoding="utf-8").splitlines()[0]
            for col in ("pilot_id", "image_id", "source_image_group", "original_path"):
                self.assertIn(col, header)


if __name__ == "__main__":
    unittest.main()
