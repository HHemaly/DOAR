from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.schemas import DetectorEvidenceRecord, from_zero_shot_prediction


class DetectorEvidenceRecordInvariantTests(unittest.TestCase):
    def test_valid_detected_record_constructs(self):
        rec = DetectorEvidenceRecord(
            evidence_id="ev1", entity_id="ev1_0", class_name="person",
            status="detected", confidence=0.7)
        self.assertEqual(rec.status, "detected")

    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                   class_name="person", status="maybe", confidence=0.5)

    def test_uncertain_status_with_confidence_raises(self):
        # A non-conclusive status must never carry a confidence value --
        # that would look like a real, trustworthy measurement.
        with self.assertRaises(ValueError):
            DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                   class_name="person", status="uncertain", confidence=0.5)

    def test_not_assessable_status_with_confidence_raises(self):
        with self.assertRaises(ValueError):
            DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                   class_name="person", status="not_assessable", confidence=0.1)

    def test_failed_status_with_confidence_raises(self):
        with self.assertRaises(ValueError):
            DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                   class_name="person", status="failed", confidence=0.1)

    def test_uncertain_status_with_no_confidence_is_valid(self):
        rec = DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                     class_name="person", status="uncertain", confidence=None)
        self.assertIsNone(rec.confidence)

    def test_confidence_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                   class_name="person", status="detected", confidence=1.5)

    def test_default_validation_and_review_status_are_honest_defaults(self):
        rec = DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                     class_name="person", status="not_detected", confidence=0.1)
        self.assertEqual(rec.validation_status, "pilot_unvalidated")
        self.assertEqual(rec.human_review_status, "not_reviewed")

    def test_invalid_bounding_box_is_just_none_not_a_malformed_tuple(self):
        rec = DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                     class_name="person", status="not_detected", confidence=0.1)
        self.assertIsNone(rec.bounding_box)

    def test_to_dict_serializes_all_fields(self):
        rec = DetectorEvidenceRecord(evidence_id="ev1", entity_id="ev1_0",
                                     class_name="person", status="detected", confidence=0.6)
        d = rec.to_dict()
        self.assertEqual(d["class_name"], "person")
        self.assertIn("checkpoint_hash", d)


class FromZeroShotPredictionTests(unittest.TestCase):
    def test_detected_prediction_carries_confidence(self):
        rec = from_zero_shot_prediction("p2b_0000", "person", 0.4, "detected",
                                        model_name="ViT-B-32", model_version="openai",
                                        preprocessing_version="v1")
        self.assertEqual(rec.status, "detected")
        self.assertIsNotNone(rec.confidence)

    def test_uncertain_prediction_never_carries_confidence(self):
        rec = from_zero_shot_prediction("p2b_0000", "person", 0.28, "uncertain",
                                        model_name="ViT-B-32", model_version="openai",
                                        preprocessing_version="v1")
        self.assertIsNone(rec.confidence)

    def test_limitations_are_non_empty(self):
        rec = from_zero_shot_prediction("p2b_0000", "person", 0.4, "detected",
                                        model_name="ViT-B-32", model_version="openai",
                                        preprocessing_version="v1")
        self.assertGreater(len(rec.limitations), 0)


if __name__ == "__main__":
    unittest.main()
