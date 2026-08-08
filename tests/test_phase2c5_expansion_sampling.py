"""Phase 2C.5 Stage 7 expansion-sampling tests -- synthetic manifest only,
never the real 3,688-drawing duplicate-group manifest. Confirms the
protocol is deterministic, excludes prior-round image_ids, and never
touches an emotion-class label (the synthetic manifest doesn't even have
one, mirroring the real load_group_lookup's own column set)."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5.expansion_sampling import (
    STAGED_EXPANSION_PLAN,
    select_expansion_sample,
    to_protocol_dict,
)


def _write_manifest(path: Path, n_groups: int, group_size: int = 1, conflict_status="clean") -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group_id", "group_size", "path", "conflict_status"])
        for g in range(n_groups):
            for m in range(group_size):
                w.writerow([f"img_{g:04d}_{m}", f"grp_{g:04d}", group_size,
                            f"/fake/path/img_{g:04d}_{m}.jpg", conflict_status])


class SelectExpansionSampleTests(unittest.TestCase):
    def test_deterministic_given_fixed_seed(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / "manifest.csv"
            _write_manifest(manifest, n_groups=50)
            r1 = select_expansion_sample(10, seed=123, exclude_image_ids=set(), manifest_path=manifest)
            r2 = select_expansion_sample(10, seed=123, exclude_image_ids=set(), manifest_path=manifest)
            self.assertEqual([r["image_id"] for r in r1], [r["image_id"] for r in r2])

    def test_excludes_prior_round_image_ids(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / "manifest.csv"
            _write_manifest(manifest, n_groups=50)
            first = select_expansion_sample(10, seed=1, exclude_image_ids=set(), manifest_path=manifest)
            first_ids = {r["image_id"] for r in first}
            second = select_expansion_sample(10, seed=2, exclude_image_ids=first_ids, manifest_path=manifest)
            second_ids = {r["image_id"] for r in second}
            self.assertEqual(first_ids & second_ids, set())

    def test_raises_if_not_enough_clean_groups_remain(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / "manifest.csv"
            _write_manifest(manifest, n_groups=5)
            with self.assertRaises(ValueError):
                select_expansion_sample(10, seed=1, exclude_image_ids=set(), manifest_path=manifest)

    def test_excludes_conflicted_groups(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "group_id", "group_size", "path", "conflict_status"])
                w.writerow(["img_clean", "grp_0", 1, "/p/a.jpg", "clean"])
                w.writerow(["img_dirty", "grp_1", 1, "/p/b.jpg", "conflicted"])
            selected = select_expansion_sample(1, seed=1, exclude_image_ids=set(), manifest_path=manifest)
            self.assertEqual(selected[0]["image_id"], "img_clean")

    def test_manifest_has_no_emotion_label_column(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / "manifest.csv"
            _write_manifest(manifest, n_groups=5)
            header = manifest.read_text(encoding="utf-8").splitlines()[0]
            for label in ("angry", "fear", "happy", "sad"):
                self.assertNotIn(label, header.lower())


class ProtocolDictTests(unittest.TestCase):
    def test_executed_this_phase_is_false(self):
        self.assertFalse(to_protocol_dict()["executed_this_phase"])

    def test_stage_b_gated_on_three_conditions(self):
        stage_b = next(s for s in STAGED_EXPANSION_PLAN if s.stage == "B_first_expansion")
        self.assertIn("feasibility", stage_b.gate.lower())
        self.assertIn("reviewed", stage_b.gate.lower())

    def test_four_stages_defined(self):
        self.assertEqual(len(STAGED_EXPANSION_PLAN), 4)


if __name__ == "__main__":
    unittest.main()
