"""Item 13 — expanded, evidence-grounded Q&A (EN + AR, standard envelope)."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _analysis():
    return {
        "quality": {"quality_status": "supported", "supported": True,
                    "unsupported_reasons": []},
        "segmentation": {"status": "verified", "confidence": 0.9},
        "composition": {"placement": "middle_center", "bounding_box_coverage": 0.2,
                        "empty_space_ratio": 0.8, "foreground_coverage": 0.2},
        "colour": {"meaningful_colours": ["blue"]},
        "emotion": {"status": "unavailable"},
        "evidence": [{"evidence_id": "ev_bbox_coverage"}, {"evidence_id": "ev_centroid"}],
        "rule_evaluations": [
            {"rule_id": "PSY_AR_ANIMAL_FOX_005", "status": "missing_detector",
             "matched_evidence_ids": [], "missing_evidence": ["detector_absent:fox"]},
            {"rule_id": "PSY_AR_SIZE_SMALL_016", "status": "weak_support",
             "matched_evidence_ids": ["ev_bbox_coverage"], "missing_evidence": []},
        ],
        "concerns": [],
    }


ENVELOPE_KEYS = {"answer", "evidence_ids", "source_module", "availability",
                 "limitations", "non_diagnostic_warning"}


class QAEnvelopeTests(unittest.TestCase):
    def test_all_topics_return_standard_envelope(self):
        from doar.qa import answer
        a = _analysis()
        topics = ["quality", "segmentation", "composition", "colour",
                  "emotion probabilities", "confidence and uncertainty",
                  "which rules", "evidence ids", "missing detector", "concerns",
                  "review history", "limitations", "is there a person"]
        for t in topics:
            resp = answer(t, a, {})
            self.assertTrue(ENVELOPE_KEYS.issubset(resp), f"{t}: {set(resp)}")

    def test_grounded_and_arabic(self):
        from doar.qa import answer
        a = _analysis()
        en = answer("what colours?", a)
        ar = answer("ما هي الألوان؟", a, language="ar")
        self.assertIn("blue", en["answer"])
        self.assertIn("ev_dominant_colour", en["evidence_ids"])
        self.assertIn("الألوان", ar["answer"])

    def test_missing_detector_not_negative_evidence(self):
        from doar.qa import answer
        resp = answer("which detectors are missing?", _analysis())
        self.assertEqual(resp["availability"], "missing_detector")
        self.assertIn("not treated as negative", resp["answer"])

    def test_emotion_unavailable_is_honest(self):
        from doar.qa import answer
        resp = answer("what emotion is predicted?", _analysis())
        self.assertEqual(resp["availability"], "unavailable")
        self.assertIsNotNone(resp["non_diagnostic_warning"])

    def test_unknown_question_is_safe_refusal(self):
        from doar.qa import answer
        resp = answer("what is the child's diagnosis?", _analysis())
        self.assertEqual(resp["availability"], "unavailable")
        self.assertEqual(resp["evidence_ids"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
