"""Phase 2C.7 Stage 1 export-summary tests -- synthetic data only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5.schema import PartAnnotationRecord, PartInstance
from doar.phase2c7 import eye_export_summary as summ


def _rec(pilot_id, status="present", n_instances=1, bbox_source="human_drawn",
         attributes=None, annotator_id="1"):
    instances = tuple(
        PartInstance(instance_index=i, bbox=(0.1 * i, 0.1, 0.05, 0.05), bbox_source=bbox_source)
        for i in range(n_instances)
    ) if status == "present" else ()
    return PartAnnotationRecord(pilot_id=pilot_id, target_name="eye", status=status,
                                 annotator_id=annotator_id, annotation_timestamp="t",
                                 instances=instances, attributes=attributes or {})


class SummarizeTargetExportTests(unittest.TestCase):
    def test_basic_counts(self):
        store = {}
        for r in [_rec("p1", "present", 1), _rec("p2", "present", 2), _rec("p3", "absent"),
                  _rec("p4", "uncertain")]:
            store[r.annotation_id] = r
        result = summ.summarize_target_export(store, "eye")
        self.assertEqual(result["n_rows"], 4)
        self.assertEqual(result["n_unique_pilot_ids"], 4)
        self.assertEqual(result["status_counts"], {"present": 2, "absent": 1, "uncertain": 1})

    def test_instance_distribution(self):
        store = {}
        for pid, n in [("p1", 1), ("p2", 1), ("p3", 2), ("p4", 3)]:
            r = _rec(pid, "present", n)
            store[r.annotation_id] = r
        result = summ.summarize_target_export(store, "eye")
        self.assertEqual(result["n_images_with_one_instance"], 2)
        self.assertEqual(result["n_images_with_two_instances"], 1)
        self.assertEqual(result["n_images_with_more_than_two_instances"], 1)

    def test_bbox_source_counts(self):
        store = {}
        r1 = _rec("p1", "present", 2, bbox_source="human_drawn")
        r2 = _rec("p2", "present", 1, bbox_source="human_accepted")
        for r in (r1, r2):
            store[r.annotation_id] = r
        result = summ.summarize_target_export(store, "eye")
        self.assertEqual(result["bbox_source_counts"], {"human_drawn": 2, "human_accepted": 1})
        self.assertEqual(result["n_instances_total"], 3)

    def test_attribute_distributions(self):
        store = {}
        r1 = _rec("p1", "present", 1, attributes={"eye_state": "open"})
        r2 = _rec("p2", "present", 1, attributes={"eye_state": "closed"})
        for r in (r1, r2):
            store[r.annotation_id] = r
        result = summ.summarize_target_export(store, "eye")
        self.assertEqual(result["attribute_value_distributions"]["eye_state"],
                          {"open": 1, "closed": 1})

    def test_only_requested_target_included(self):
        store = {}
        eye_rec = _rec("p1", "present", 1)
        mouth_rec = PartAnnotationRecord(pilot_id="p1", target_name="mouth", status="present",
                                          annotator_id="1", annotation_timestamp="t",
                                          instances=(PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1),
                                                                   bbox_source="human_drawn"),))
        for r in (eye_rec, mouth_rec):
            store[r.annotation_id] = r
        result = summ.summarize_target_export(store, "eye")
        self.assertEqual(result["n_rows"], 1)


class CompletionAgainstManifestTests(unittest.TestCase):
    def test_complete_when_no_missing(self):
        result = summ.completion_against_manifest({"p1", "p2"}, {"p1", "p2"})
        self.assertTrue(result["complete"])
        self.assertEqual(result["n_missing"], 0)

    def test_incomplete_reports_missing_ids(self):
        result = summ.completion_against_manifest({"p1"}, {"p1", "p2", "p3"})
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_pilot_ids"], ["p2", "p3"])
        self.assertEqual(result["n_reviewed"], 1)
        self.assertEqual(result["n_manifest"], 3)

    def test_unexpected_ids_outside_manifest_flagged(self):
        result = summ.completion_against_manifest({"p1", "p99"}, {"p1"})
        self.assertEqual(result["unexpected_pilot_ids"], ["p99"])


if __name__ == "__main__":
    unittest.main()
