"""Phase 2C.7 Stage 3 dev/holdout split tests -- synthetic pilot_ids only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c7 import dev_holdout_split as split_mod


class SplitDevHoldoutTests(unittest.TestCase):
    def test_deterministic_given_fixed_seed(self):
        ids = [f"p2c6_{i:04d}" for i in range(213)]
        s1 = split_mod.split_dev_holdout(ids, seed=42)
        s2 = split_mod.split_dev_holdout(ids, seed=42)
        self.assertEqual(s1.dev_pilot_ids, s2.dev_pilot_ids)
        self.assertEqual(s1.holdout_pilot_ids, s2.holdout_pilot_ids)

    def test_order_independent(self):
        ids = [f"p2c6_{i:04d}" for i in range(213)]
        import random
        shuffled = list(ids)
        random.Random(1).shuffle(shuffled)
        s1 = split_mod.split_dev_holdout(ids, seed=7)
        s2 = split_mod.split_dev_holdout(shuffled, seed=7)
        self.assertEqual(s1.dev_pilot_ids, s2.dev_pilot_ids)

    def test_no_overlap(self):
        ids = [f"p{i}" for i in range(100)]
        s = split_mod.split_dev_holdout(ids, seed=1)
        self.assertEqual(set(s.dev_pilot_ids) & set(s.holdout_pilot_ids), set())

    def test_covers_all_ids(self):
        ids = [f"p{i}" for i in range(100)]
        s = split_mod.split_dev_holdout(ids, seed=1)
        self.assertEqual(set(s.dev_pilot_ids) | set(s.holdout_pilot_ids), set(ids))

    def test_approximately_80_20_split(self):
        ids = [f"p{i}" for i in range(200)]
        s = split_mod.split_dev_holdout(ids, seed=1, dev_fraction=0.8)
        self.assertEqual(len(s.dev_pilot_ids), 160)
        self.assertEqual(len(s.holdout_pilot_ids), 40)

    def test_different_seed_gives_different_split(self):
        ids = [f"p{i}" for i in range(200)]
        s1 = split_mod.split_dev_holdout(ids, seed=1)
        s2 = split_mod.split_dev_holdout(ids, seed=2)
        self.assertNotEqual(s1.dev_pilot_ids, s2.dev_pilot_ids)

    def test_invalid_fraction_rejected(self):
        with self.assertRaises(ValueError):
            split_mod.split_dev_holdout(["p1"], dev_fraction=1.5)
        with self.assertRaises(ValueError):
            split_mod.split_dev_holdout(["p1"], dev_fraction=0.0)


class AssertNoOverlapTests(unittest.TestCase):
    def test_passes_for_real_split(self):
        ids = [f"p{i}" for i in range(50)]
        s = split_mod.split_dev_holdout(ids, seed=1)
        split_mod.assert_no_overlap(s)  # must not raise

    def test_raises_for_constructed_overlap(self):
        bad = split_mod.DevHoldoutSplit(dev_pilot_ids=("a", "b"), holdout_pilot_ids=("b", "c"),
                                         seed=1, dev_fraction=0.8)
        with self.assertRaises(ValueError):
            split_mod.assert_no_overlap(bad)


class AssertHoldoutUntouchedByTests(unittest.TestCase):
    def test_passes_when_only_dev_used(self):
        ids = [f"p{i}" for i in range(50)]
        s = split_mod.split_dev_holdout(ids, seed=1)
        split_mod.assert_holdout_untouched_by(set(s.dev_pilot_ids), s)  # must not raise

    def test_raises_when_holdout_id_used(self):
        ids = [f"p{i}" for i in range(50)]
        s = split_mod.split_dev_holdout(ids, seed=1)
        leaked_use = set(s.dev_pilot_ids) | {s.holdout_pilot_ids[0]}
        with self.assertRaises(ValueError):
            split_mod.assert_holdout_untouched_by(leaked_use, s)


if __name__ == "__main__":
    unittest.main()
