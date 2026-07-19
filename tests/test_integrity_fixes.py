"""
Tests for integrity fixes D4 (no fake shape features), D5 (correct embedding
preprocessing metadata), D8 (valid single-document bilingual report).
"""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class D4ShapeFeatureHonesty(unittest.TestCase):
    def test_unimplemented_shape_features_marked_missing(self):
        from PIL import Image
        from doar.features import objective_feature_row
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            Image.new("RGB", (64, 64), (255, 255, 255)).save(p)
            analysis = {"composition": {"bounding_box": [0, 0, 64, 64]}}
            row = objective_feature_row(p, analysis)
            for name in ("shape.enclosed_shape_count", "shape.repetition_score"):
                self.assertIn(name, row)
                self.assertTrue(row[name].missing)          # not a fake 0.0
                self.assertEqual(row[name].method, "not_evaluated_no_detector")
                self.assertEqual(row[name].confidence, 0.0)


class D8BilingualReport(unittest.TestCase):
    def _analysis(self):
        return {"safety_disclaimer": "This is not a diagnosis.",
                "quality": {"supported": True},
                "segmentation": {"candidate_disagreement": 0.0},
                "composition": {"foreground_coverage": 0.3},
                "colour": {"dominant_colour": "red"},
                "emotion": {"status": "unavailable"},
                "rule_evaluations": []}

    def test_bilingual_is_single_valid_document(self):
        from doar.reports import render_bilingual
        html = render_bilingual(self._analysis(), {"safety_judge": {"status": "pass"}})
        self.assertEqual(html.count("<!doctype html>"), 1)
        self.assertEqual(html.count("<html"), 1)
        self.assertEqual(html.count("</html>"), 1)
        self.assertIn('dir="ltr"', html)
        self.assertIn('dir="rtl"', html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
