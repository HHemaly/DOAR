"""Phase 2C.6 Stage 8 fine-tuning readiness criteria tests -- synthetic
data only. No fine-tuning is performed anywhere in this test file or the
module it tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c6 import finetuning_readiness as ftr
from doar.phase2c5.schema import PartAnnotationRecord, PartInstance

_TARGETS = ("eye", "mouth", "hand", "face", "person")


def _present_rec(pilot_id, target, n_instances=1, annotator_id="ann1"):
    instances = tuple(PartInstance(instance_index=i, bbox=(0.1 * i, 0.1, 0.1, 0.1),
                                    bbox_source="human_drawn") for i in range(n_instances))
    return PartAnnotationRecord(pilot_id=pilot_id, target_name=target, status="present",
                                 annotator_id=annotator_id, annotation_timestamp="t", instances=instances)


class NoFineTuningHappensTests(unittest.TestCase):
    def test_module_contains_no_training_calls_or_imports(self):
        # "peft"/"lora" legitimately appear once in the module's own
        # docstring disclaiming their use -- checked for calls/imports
        # instead of raw text presence (same pattern as other phases'
        # self-referential-disclaimer test fixes).
        text = Path(ROOT / "src/doar/phase2c6/finetuning_readiness.py").read_text(encoding="utf-8").lower()
        for forbidden in (".fit(", ".train(", "optimizer", "backward()", "import peft",
                           "import torch", "from peft", "from torch"):
            self.assertNotIn(forbidden, text)


class EmptyStoreTests(unittest.TestCase):
    def test_empty_store_is_not_ready_with_clear_reasons(self):
        report = ftr.evaluate_finetuning_readiness({}, set(), targets=_TARGETS)
        self.assertFalse(report.overall_ready)
        self.assertFalse(report.meets_image_floor)
        self.assertGreater(len(report.blocking_reasons), 0)


class WellSupportedStoreTests(unittest.TestCase):
    def test_report_becomes_ready_once_thresholds_are_cleared(self):
        store = {}
        reviewed = set()
        n_images = ftr.MIN_REVIEWED_IMAGES + ftr.MIN_VALIDATION_SPLIT_IMAGES + 5
        per_target_positive = ftr.MIN_POSITIVE_INSTANCES_PER_TARGET + 5
        for i in range(n_images):
            pid = f"p{i:04d}"
            reviewed.add(pid)
            for target in _TARGETS:
                n_inst = 1 if i < per_target_positive else 0
                status = "present" if n_inst else "absent"
                rec = PartAnnotationRecord(
                    pilot_id=pid, target_name=target, status=status, annotator_id="ann1",
                    annotation_timestamp="t",
                    instances=(PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1),
                                             bbox_source="human_drawn"),) if n_inst else ())
                store[rec.annotation_id] = rec
        report = ftr.evaluate_finetuning_readiness(store, reviewed, targets=_TARGETS)
        self.assertTrue(report.overall_ready, report.blocking_reasons)

    def test_severe_imbalance_blocks_readiness(self):
        store = {}
        reviewed = set()
        n_images = ftr.MIN_REVIEWED_IMAGES + ftr.MIN_VALIDATION_SPLIT_IMAGES + 5
        for i in range(n_images):
            pid = f"p{i:04d}"
            reviewed.add(pid)
            for target in _TARGETS:
                # 'eye' gets abundant support, everything else gets almost none
                n_inst = 1 if (target == "eye" or i < 2) else 0
                status = "present" if n_inst else "absent"
                rec = PartAnnotationRecord(
                    pilot_id=pid, target_name=target, status=status, annotator_id="ann1",
                    annotation_timestamp="t",
                    instances=(PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1),
                                             bbox_source="human_drawn"),) if n_inst else ())
                store[rec.annotation_id] = rec
        report = ftr.evaluate_finetuning_readiness(store, reviewed, targets=_TARGETS)
        self.assertFalse(report.overall_ready)
        self.assertTrue(any("imbalance" in r or "positive instances" in r for r in report.blocking_reasons))


class ToDictTests(unittest.TestCase):
    def test_to_dict_is_json_serializable(self):
        import json
        report = ftr.evaluate_finetuning_readiness({}, set(), targets=_TARGETS)
        json.dumps(ftr.to_dict(report))  # raises if not serializable


if __name__ == "__main__":
    unittest.main()
