"""Phase 2C.7 visual-detector API tests -- synthetic fake predict_fns
only, no real model weights, no image files needed (fake predict_fns
ignore their image_path argument's actual content)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c7 import detector_policy as pol
from doar.phase2c7 import visual_detector as vd


def _eye_entry(status=pol.VALIDATED_AUTOMATIC, best_model="owlv2_parts"):
    return pol.build_eye_policy_entry(
        status=status, best_model=best_model, best_model_checkpoint="ckpt", prompt="a eye",
        threshold=0.1, precision=0.8, recall=0.6, balanced_accuracy=0.75,
        n_ground_truth_present=170, localization_validated=(status == pol.VALIDATED_AUTOMATIC),
        rationale="test")


class AnalyzeImageTests(unittest.TestCase):
    def setUp(self):
        self.policy = pol.full_policy(_eye_entry())
        self.gd_calls = []
        self.owlv2_calls = []

    def _grounding_dino_object_predict_fn(self, image_path):
        self.gd_calls.append(image_path)
        return {
            "person": (True, 0.9, (0.1, 0.1, 0.5, 0.5)),
            "face": (False, 0.0, None),
            "circle": (True, 0.99, (0.0, 0.0, 0.1, 0.1)),  # DISABLED target -- must never appear in output
        }

    def _owlv2_object_predict_fn(self, image_path):
        self.owlv2_calls.append(image_path)
        return {"heart": (True, 0.55, (0.3, 0.3, 0.1, 0.1))}

    def _owlv2_parts_predict_fn(self, image_path):
        self.owlv2_calls.append(image_path)
        return {"eye": (True, 0.4, (0.2, 0.2, 0.05, 0.05))}

    def _predict_fns(self):
        return {
            "grounding_dino_object_classes": self._grounding_dino_object_predict_fn,
            "owlv2_object_classes": self._owlv2_object_predict_fn,
            "owlv2_parts": self._owlv2_parts_predict_fn,
        }

    def test_image_path_only_no_other_inputs_required(self):
        records = vd.analyze_image("some/image.jpg", self.policy, model_predict_fns=self._predict_fns())
        self.assertTrue(records)
        self.assertEqual(self.gd_calls, ["some/image.jpg"])
        self.assertEqual(self.owlv2_calls, ["some/image.jpg", "some/image.jpg"])

    def test_disabled_target_never_appears_in_output(self):
        records = vd.analyze_image("img.jpg", self.policy, model_predict_fns=self._predict_fns())
        targets = {r.target for r in records}
        self.assertNotIn("circle", targets)  # circle is DISABLED in the real policy
        self.assertNotIn("vehicle", targets)

    def test_disabled_only_policy_never_calls_any_model(self):
        all_disabled = {t: pol.TargetPolicyEntry(
            target=t, status=pol.DISABLED, best_model=None, best_model_checkpoint=None, prompt=None,
            threshold=None, precision=None, recall=None, balanced_accuracy=0.5,
            n_ground_truth_present=0, localization_validated=False, rationale="r",
            related_rule_ids=(), allowed_downstream_usage="not used")
            for t in ("circle", "vehicle")}
        records = vd.analyze_image("img.jpg", all_disabled, model_predict_fns=self._predict_fns())
        self.assertEqual(records, [])
        self.assertEqual(self.gd_calls, [])
        self.assertEqual(self.owlv2_calls, [])

    def test_validated_target_gets_validated_evidence_status(self):
        records = vd.analyze_image("img.jpg", self.policy, model_predict_fns=self._predict_fns())
        person_record = next(r for r in records if r.target == "person")
        self.assertEqual(person_record.evidence_status, "validated_evidence")
        self.assertTrue(person_record.present)
        self.assertEqual(person_record.bbox, (0.1, 0.1, 0.5, 0.5))

    def test_experimental_target_gets_experimental_evidence_status(self):
        records = vd.analyze_image("img.jpg", self.policy, model_predict_fns=self._predict_fns())
        heart_record = next(r for r in records if r.target == "heart")
        self.assertEqual(heart_record.evidence_status, "experimental_evidence_technical_view_only")

    def test_heart_routes_to_owlv2_not_grounding_dino(self):
        """heart's frozen best_model is OWLv2 (detector_policy.py), unlike
        every other object class which freezes to Grounding DINO -- the
        routing must respect that per-target choice, not a hardcoded
        object/part split."""
        records = vd.analyze_image("img.jpg", self.policy, model_predict_fns=self._predict_fns())
        heart_record = next(r for r in records if r.target == "heart")
        self.assertEqual(heart_record.model, "owlv2_object_classes")
        self.assertTrue(heart_record.present)
        self.assertAlmostEqual(heart_record.confidence, 0.55)

    def test_non_dispatchable_reference_target_produces_no_record(self):
        records = vd.analyze_image("img.jpg", self.policy, model_predict_fns=self._predict_fns())
        targets = {r.target for r in records}
        self.assertNotIn("person_part_reference", targets)

    def test_missing_target_in_predictions_defaults_to_absent(self):
        records = vd.analyze_image("img.jpg", self.policy, model_predict_fns=self._predict_fns())
        face_record = next(r for r in records if r.target == "face")
        self.assertFalse(face_record.present)
        self.assertIsNone(face_record.bbox)

    def test_each_distinct_model_callable_called_at_most_once(self):
        """'owlv2_object_classes' and 'owlv2_parts' are two genuinely
        different callables (different prompts/vocabularies) sharing the
        same underlying model family -- each is called once; the family
        total is legitimately 2, not 1."""
        vd.analyze_image("img.jpg", self.policy, model_predict_fns=self._predict_fns())
        self.assertEqual(len(self.gd_calls), 1)
        self.assertEqual(len(self.owlv2_calls), 2)

    def test_missing_model_in_mapping_treated_as_no_detections_not_a_crash(self):
        records = vd.analyze_image("img.jpg", self.policy, model_predict_fns={})
        person_record = next(r for r in records if r.target == "person")
        self.assertFalse(person_record.present)


class ValidatedRecordsOnlyTests(unittest.TestCase):
    def test_filters_out_experimental_and_disabled(self):
        records = [
            vd.VisualEvidenceRecord("a", True, None, 0.9, "m", "c", 0.1, pol.VALIDATED_AUTOMATIC,
                                     "validated_evidence"),
            vd.VisualEvidenceRecord("b", True, None, 0.9, "m", "c", 0.1, pol.EXPERIMENTAL_AUTOMATIC,
                                     "experimental_evidence_technical_view_only"),
        ]
        result = vd.validated_records_only(records)
        self.assertEqual([r.target for r in result], ["a"])

    def test_empty_input_gives_empty_output(self):
        self.assertEqual(vd.validated_records_only([]), [])


class ToDictTests(unittest.TestCase):
    def test_to_dict_is_json_serializable(self):
        import json
        rec = vd.VisualEvidenceRecord("eye", True, (0.1, 0.1, 0.1, 0.1), 0.5, "m", "c", 0.1,
                                       pol.VALIDATED_AUTOMATIC, "validated_evidence")
        json.dumps(rec.to_dict())


if __name__ == "__main__":
    unittest.main()
