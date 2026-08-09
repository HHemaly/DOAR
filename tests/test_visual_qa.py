"""DOAR MVP: visual_qa.py -- grounded Q&A extended with on-demand visual
search. Any question not recognized as visual must fall straight through
to the existing, already-tested qa.py logic unchanged."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_evidence import save_detections, VisualFinding  # noqa: E402
from doar.visual_qa import answer_with_visual_grounding, extract_visual_target  # noqa: E402

FAKE_REGISTRY_V2 = {"rules": []}


def _analysis():
    return {
        "quality": {"quality_status": "supported", "supported": True},
        "composition": {}, "colour": {"meaningful_colours": []}, "emotion": {"status": "unavailable"},
        "rule_evaluations": [], "evidence": [], "concerns": [],
    }


def _finding(label, *, validation_status="VALIDATED", confidence=0.8):
    return VisualFinding(
        label=label, finding_id=f"vf_test_{label}", free_form_label=None, bbox=None,
        confidence=confidence, detector="m", checkpoint="c", prompt="p",
        validation_status=validation_status,
        evidence_status=("validated_evidence" if validation_status == "VALIDATED"
                          else "experimental_evidence_technical_view_only"),
        rule_mapping_status="UNMAPPED", related_rule_ids=(), source="initial_scan", query=None,
        timestamp="t")


class ExtractVisualTargetTests(unittest.TestCase):
    def test_known_synonym_maps_to_canonical_label(self):
        self.assertEqual(extract_visual_target("Is there a dog in the drawing?"), "dog")
        self.assertEqual(extract_visual_target("How many people are there?"), "person")
        self.assertEqual(extract_visual_target("do you see any hands"), "hand")

    def test_arabic_synonym_recognized(self):
        self.assertEqual(extract_visual_target("هل هناك قطة؟"), "cat")

    def test_unknown_noun_falls_back_to_pattern_extraction(self):
        self.assertEqual(extract_visual_target("is there a bicycle?"), "bicycle")

    def test_non_visual_question_returns_none(self):
        self.assertIsNone(extract_visual_target("What is the image quality?"))


class AnswerWithVisualGroundingTests(unittest.TestCase):
    def test_non_visual_question_falls_through_to_qa_answer_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            resp = answer_with_visual_grounding(tmp, "what colours?", _analysis())
            self.assertEqual(resp["source_module"], "colour")

    def test_existing_evidence_answers_without_on_demand_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_detections(tmp, [_finding("dog")])
            calls = []

            def open_vocab_predict_fn(image_path, target):
                calls.append(target)
                return []

            resp = answer_with_visual_grounding(
                tmp, "is there a dog?", _analysis(), registry_v2=FAKE_REGISTRY_V2,
                open_vocab_predict_fn=open_vocab_predict_fn)
            self.assertEqual(resp["source_module"], "visual_evidence")
            self.assertEqual(resp["availability"], "available")
            self.assertEqual(calls, [])  # existing evidence found -- no live query issued

    def test_no_evidence_and_no_predict_fn_is_honest_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            resp = answer_with_visual_grounding(tmp, "is there a dog?", _analysis())
            self.assertEqual(resp["availability"], "missing_detector")
            self.assertEqual(resp["source_module"], "visual_evidence")

    def test_on_demand_search_hit_is_reported_as_experimental_not_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "image.png").write_bytes(b"\x89PNG\r\n")

            def open_vocab_predict_fn(image_path, target):
                return [(True, 0.6, None, "fake_model", "ckpt", target)]

            resp = answer_with_visual_grounding(
                tmp, "is there a dog?", _analysis(), registry_v2=FAKE_REGISTRY_V2,
                open_vocab_predict_fn=open_vocab_predict_fn)
            self.assertEqual(resp["source_module"], "visual_evidence_on_demand")
            self.assertEqual(resp["availability"], "available")
            self.assertIsNotNone(resp["non_diagnostic_warning"])
            self.assertIn("experimental", resp["answer"].lower())

    def test_on_demand_search_miss_never_claims_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "image.png").write_bytes(b"\x89PNG\r\n")

            def open_vocab_predict_fn(image_path, target):
                return []

            resp = answer_with_visual_grounding(
                tmp, "is there a dog?", _analysis(), registry_v2=FAKE_REGISTRY_V2,
                open_vocab_predict_fn=open_vocab_predict_fn)
            self.assertEqual(resp["availability"], "not_found")
            self.assertIn("does not prove", resp["answer"].lower())


if __name__ == "__main__":
    unittest.main()
