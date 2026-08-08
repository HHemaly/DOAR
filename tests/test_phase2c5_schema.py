"""Phase 2C.5 schema tests -- synthetic data only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5.schema import PartAnnotationRecord, PartInstance, record_from_row


def _instance(**overrides):
    defaults = dict(instance_index=0, bbox=(0.1, 0.1, 0.2, 0.2), bbox_source="human_drawn")
    defaults.update(overrides)
    return PartInstance(**defaults)


class PartInstanceValidationTests(unittest.TestCase):
    def test_valid_human_drawn_instance(self):
        inst = _instance()
        self.assertEqual(inst.bbox_source, "human_drawn")

    def test_invalid_bbox_source_rejected(self):
        with self.assertRaises(ValueError):
            _instance(bbox_source="not_a_real_source")

    def test_bbox_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            _instance(bbox=(1.5, 0.1, 0.2, 0.2))

    def test_zero_width_bbox_rejected(self):
        with self.assertRaises(ValueError):
            _instance(bbox=(0.1, 0.1, 0.0, 0.2))

    def test_model_proposed_requires_proposal_model(self):
        with self.assertRaises(ValueError):
            _instance(bbox_source="model_proposed", proposal_model="")

    def test_model_proposed_with_model_is_valid(self):
        inst = _instance(bbox_source="model_proposed", proposal_model="grounding_dino:tiny")
        self.assertEqual(inst.proposal_model, "grounding_dino:tiny")

    def test_human_drawn_cannot_carry_proposal_model(self):
        with self.assertRaises(ValueError):
            _instance(bbox_source="human_drawn", proposal_model="grounding_dino:tiny")

    def test_human_accepted_can_carry_proposal_model(self):
        inst = _instance(bbox_source="human_accepted", proposal_model="owlv2:base")
        self.assertEqual(inst.bbox_source, "human_accepted")


class PartAnnotationRecordValidationTests(unittest.TestCase):
    def _rec(self, **overrides):
        defaults = dict(pilot_id="p2b_0000", target_name="eye", status="present",
                         annotator_id="ann1", annotation_timestamp="t",
                         instances=(_instance(),))
        defaults.update(overrides)
        return PartAnnotationRecord(**defaults)

    def test_present_requires_at_least_one_instance(self):
        with self.assertRaises(ValueError):
            self._rec(status="present", instances=())

    def test_absent_forbids_instances(self):
        with self.assertRaises(ValueError):
            self._rec(status="absent", instances=(_instance(),))

    def test_absent_with_no_instances_is_valid(self):
        rec = self._rec(status="absent", instances=())
        self.assertEqual(rec.status, "absent")

    def test_unknown_target_rejected(self):
        with self.assertRaises(ValueError):
            self._rec(target_name="not_a_real_target", instances=(_instance(),))

    def test_duplicate_instance_index_rejected(self):
        with self.assertRaises(ValueError):
            self._rec(instances=(_instance(instance_index=0), _instance(instance_index=0)))

    def test_non_contiguous_instance_index_rejected(self):
        with self.assertRaises(ValueError):
            self._rec(instances=(_instance(instance_index=0), _instance(instance_index=2)))

    def test_multi_instance_supported(self):
        rec = self._rec(instances=(_instance(instance_index=0), _instance(instance_index=1)))
        self.assertEqual(len(rec.instances), 2)

    def test_eye_state_attribute_allowed(self):
        rec = self._rec(target_name="eye", attributes={"eye_state": "closed"})
        self.assertEqual(rec.attributes["eye_state"], "closed")

    def test_eye_attribute_not_allowed_on_hand(self):
        with self.assertRaises(ValueError):
            self._rec(target_name="hand", attributes={"eye_state": "closed"})

    def test_invalid_attribute_value_rejected(self):
        with self.assertRaises(ValueError):
            self._rec(target_name="eye", attributes={"eye_state": "sideways"})

    def test_empty_annotator_id_rejected(self):
        with self.assertRaises(ValueError):
            self._rec(annotator_id="  ")

    def test_annotation_id_deterministic(self):
        rec = self._rec()
        self.assertEqual(rec.annotation_id, "p2b_0000__eye__ann1")


class RoundTripTests(unittest.TestCase):
    def test_to_row_and_from_row_round_trip_multi_instance(self):
        rec = PartAnnotationRecord(
            pilot_id="p2b_0001", target_name="hand", status="present",
            annotator_id="ann2", annotation_timestamp="t2",
            instances=(_instance(instance_index=0, bbox_source="model_proposed",
                                  proposal_model="grounding_dino:tiny", proposal_threshold=0.25),
                       _instance(instance_index=1, bbox_source="human_drawn")),
            attributes={}, notes="two hands",
        )
        row = rec.to_row()
        restored = record_from_row(row)
        self.assertEqual(restored.pilot_id, rec.pilot_id)
        self.assertEqual(len(restored.instances), 2)
        self.assertEqual(restored.instances[0].bbox_source, "model_proposed")
        self.assertEqual(restored.instances[0].proposal_model, "grounding_dino:tiny")
        self.assertEqual(restored.instances[1].bbox_source, "human_drawn")
        self.assertEqual(restored.notes, "two hands")

    def test_round_trip_preserves_absent_status_with_no_instances(self):
        rec = PartAnnotationRecord(pilot_id="p2b_0002", target_name="mouth", status="absent",
                                    annotator_id="ann1", annotation_timestamp="t")
        restored = record_from_row(rec.to_row())
        self.assertEqual(restored.status, "absent")
        self.assertEqual(restored.instances, ())


if __name__ == "__main__":
    unittest.main()
