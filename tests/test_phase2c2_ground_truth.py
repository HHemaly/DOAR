"""Ground-truth extraction tests -- synthetic data only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c1 import store as store_mod
from doar.phase2c1.schema import AnnotationRecord
from doar.phase2c2.ground_truth import genuine_human_ground_truth


def _rec(**overrides):
    defaults = dict(
        pilot_id="p2b_0000", image_id="abc", source_image_group="grp_1",
        class_name="person", status="present", annotator_id="ann1",
        annotation_timestamp="t", instance_count=1, annotator_type="human",
    )
    defaults.update(overrides)
    return AnnotationRecord(**defaults)


class GenuineHumanGroundTruthTests(unittest.TestCase):
    def test_only_human_type_included(self):
        st = {}
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", annotator_type="human", status="present"))
        store_mod.upsert(st, _rec(pilot_id="p2b_0001", annotator_type="legacy_provisional_human",
                                   annotator_id="legacy", status="absent", instance_count=0))
        gt = genuine_human_ground_truth(st, "person")
        self.assertEqual(gt, {"p2b_0000": "present"})

    def test_only_matching_class(self):
        st = {}
        store_mod.upsert(st, _rec(class_name="person", status="present"))
        store_mod.upsert(st, _rec(class_name="face", status="absent", instance_count=0))
        gt = genuine_human_ground_truth(st, "person")
        self.assertEqual(gt, {"p2b_0000": "present"})

    def test_empty_store_returns_empty(self):
        self.assertEqual(genuine_human_ground_truth({}, "person"), {})

    def test_disagreeing_duplicate_human_judgments_raise(self):
        st = {}
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", annotator_id="ann1", status="present"))
        # A second, distinct human annotator disagreeing on the same (pilot_id, class).
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", annotator_id="ann2", status="absent",
                                   instance_count=0))
        with self.assertRaises(ValueError):
            genuine_human_ground_truth(st, "person")

    def test_agreeing_duplicate_human_judgments_do_not_raise(self):
        st = {}
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", annotator_id="ann1", status="present"))
        store_mod.upsert(st, _rec(pilot_id="p2b_0000", annotator_id="ann2", status="present"))
        gt = genuine_human_ground_truth(st, "person")
        self.assertEqual(gt, {"p2b_0000": "present"})


if __name__ == "__main__":
    unittest.main()
