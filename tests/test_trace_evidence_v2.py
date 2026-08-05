from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.features import FeatureValue
from doar.schemas import Evidence
from doar.trace_evidence import EvidenceRecordV2, EvidenceSetV2, make_provenance
from doar.trace_evidence_adapters import feature_value_to_v2, legacy_evidence_to_v2


def _record(**overrides) -> EvidenceRecordV2:
    base = dict(
        evidence_id="e1", evidence_type="objective_feature", producer="test",
        producer_version="1", value=1.0, unit=None, confidence=0.9, method="test",
        status="PASS", limitations=[], provenance=make_provenance("test", "v1"),
    )
    base.update(overrides)
    return EvidenceRecordV2(**base)


class EvidenceRecordV2InvariantTests(unittest.TestCase):
    def test_valid_pass_record_constructs(self):
        record = _record()
        self.assertEqual(record.status, "PASS")

    def test_invalid_status_raises(self):
        with self.assertRaises(ValueError):
            _record(status="NOT_A_STATUS")

    def test_abstain_with_value_raises(self):
        with self.assertRaises(ValueError):
            _record(status="ABSTAIN", value=1.0)

    def test_abstain_with_none_value_is_valid(self):
        record = _record(status="ABSTAIN", value=None)
        self.assertIsNone(record.value)

    def test_confidence_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            _record(confidence=1.5)

    def test_to_dict_serializes_provenance(self):
        record = _record()
        data = record.to_dict()
        self.assertIn("pipeline_stage", data["provenance"])
        self.assertIn("generated_at", data["provenance"])


class EvidenceSetV2Tests(unittest.TestCase):
    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            EvidenceSetV2(records=[_record(evidence_id="dup"), _record(evidence_id="dup")])

    def test_rejects_unknown_source_evidence_id(self):
        with self.assertRaises(ValueError):
            EvidenceSetV2(records=[_record(evidence_id="e1", source_evidence_ids=["does_not_exist"])])

    def test_accepts_valid_source_evidence_id(self):
        parent = _record(evidence_id="parent")
        child = _record(evidence_id="child", source_evidence_ids=["parent"])
        evidence_set = EvidenceSetV2(records=[parent, child])
        self.assertEqual(evidence_set.by_id("child").source_evidence_ids, ["parent"])

    def test_by_type_filters(self):
        evidence_set = EvidenceSetV2(records=[
            _record(evidence_id="a", evidence_type="objective_feature"),
            _record(evidence_id="b", evidence_type="rule_evaluation"),
        ])
        self.assertEqual(len(evidence_set.by_type("objective_feature")), 1)


class LegacyAdapterTests(unittest.TestCase):
    def test_feature_value_missing_becomes_abstain(self):
        fv = FeatureValue(
            value=float("nan"), valid_min=None, valid_max=None, confidence=0.0,
            method="not_evaluated_no_detector", evidence_id="ev_feature_shape_repetition_score",
            missing=True,
        )
        record = feature_value_to_v2("shape.repetition_score", fv)
        self.assertEqual(record.status, "ABSTAIN")
        self.assertIsNone(record.value)
        self.assertTrue(record.limitations)

    def test_feature_value_present_high_confidence_becomes_pass(self):
        fv = FeatureValue(
            value=0.5, valid_min=None, valid_max=None, confidence=0.9,
            method="objective_features_v3_1", evidence_id="ev_bbox_coverage", missing=False,
        )
        record = feature_value_to_v2("segmentation.bounding_box_coverage", fv)
        self.assertEqual(record.status, "PASS")
        self.assertEqual(record.value, 0.5)

    def test_feature_value_present_low_confidence_becomes_warn(self):
        fv = FeatureValue(
            value=0.5, valid_min=None, valid_max=None, confidence=0.2,
            method="objective_features_v3_1", evidence_id="ev_x", missing=False,
        )
        record = feature_value_to_v2("x", fv)
        self.assertEqual(record.status, "WARN")

    def test_legacy_evidence_becomes_pass(self):
        legacy = Evidence("ev_bbox_coverage", "objective_feature", 0.5, "method_x", 0.8)
        record = legacy_evidence_to_v2(legacy)
        self.assertEqual(record.status, "PASS")
        self.assertEqual(record.evidence_type, "objective_feature")
        self.assertEqual(record.value, 0.5)

    def test_adapters_produce_a_mutually_consistent_evidence_set(self):
        fv_present = FeatureValue(
            value=0.9, valid_min=None, valid_max=None, confidence=0.9,
            method="objective_features_v3_1", evidence_id="ev_bbox_coverage", missing=False,
        )
        fv_missing = FeatureValue(
            value=float("nan"), valid_min=None, valid_max=None, confidence=0.0,
            method="not_evaluated_no_detector", evidence_id="ev_feature_shape_repetition_score",
            missing=True,
        )
        records = [
            feature_value_to_v2("segmentation.bounding_box_coverage", fv_present),
            feature_value_to_v2("shape.repetition_score", fv_missing),
        ]
        evidence_set = EvidenceSetV2(records=records)
        self.assertEqual(len(evidence_set.records), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
