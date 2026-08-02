"""Item 14 — Arabic localization of dynamic report content (not only headings)."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class LocalizationTests(unittest.TestCase):
    def test_status_and_emotion_values_translated(self):
        from doar.localization import localize_value
        self.assertEqual(localize_value("supported", "ar"), "مدعوم")
        self.assertEqual(localize_value("unsupported", "ar"), "غير مدعوم")
        self.assertEqual(localize_value("missing_detector", "ar"), "لا يوجد كاشف")
        self.assertEqual(localize_value("Happy", "ar"), "سعيد")
        self.assertEqual(localize_value("high", "ar"), "عالٍ")

    def test_technical_ids_and_numbers_pass_through(self):
        from doar.localization import localize_value
        self.assertEqual(localize_value("ev_bbox_coverage", "ar"), "ev_bbox_coverage")
        self.assertEqual(localize_value(0.42, "ar"), "0.42")

    def test_quality_reason_freetext_translated(self):
        from doar.localization import localize_value
        out = localize_value("resolution below 100px (min_dimension=40)", "ar")
        self.assertIn("الدقة أقل من 100", out)

    def test_english_is_untouched(self):
        from doar.localization import localize_value, localize_key
        self.assertEqual(localize_value("supported", "en"), "supported")
        self.assertEqual(localize_key("quality_status", "en"), "quality_status")

    def test_arabic_report_translates_values_not_only_headings(self):
        from doar.reports import render_report
        analysis = {
            "safety_disclaimer": "This is not a diagnosis.",
            "quality": {"quality_status": "unsupported", "supported": False,
                        "unsupported_reasons": ["resolution below 100px (min_dimension=40)"]},
            "segmentation": {"status": "verified", "candidate_disagreement": 0.1},
            "composition": {"placement": "middle_center", "foreground_coverage": 0.2},
            "colour": {"dominant_colour": "blue", "meaningful_colours": ["blue"]},
            "emotion": {"status": "unavailable", "top_class": None, "uncertainty": "high"},
            "rule_evaluations": [],
        }
        judges = {"quality_judge": {"status": "fail"}, "safety_judge": {"status": "pass"}}
        html_ar = render_report(analysis, judges, "ar")
        # Value translations present (not just Arabic headings):
        self.assertIn("غير مدعوم", html_ar)     # unsupported
        self.assertIn("مُتحقق", html_ar)         # verified
        self.assertIn("الدقة أقل من 100", html_ar)  # translated quality reason
        self.assertIn('dir="rtl"', html_ar)


if __name__ == "__main__":
    unittest.main(verbosity=2)
