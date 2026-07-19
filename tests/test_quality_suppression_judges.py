"""Items 9, 10, 12 — quality suppression, rule statuses, repaired judges."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _supported_image(path):
    from PIL import Image
    import numpy as np
    arr = np.indices((160, 160)).sum(0) % 255           # sharp, high-contrast
    Image.fromarray(arr.astype("uint8")).convert("RGB").save(path)


def _unsupported_image(path):
    from PIL import Image
    Image.new("RGB", (48, 48), "white").save(path)      # too small + flat


class QualitySuppressionTests(unittest.TestCase):
    def test_supported_runs_all_modules(self):
        from doar.analysis import analyze_image
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.png"; _supported_image(p)
            a = analyze_image(p, Path(d) / "o")
            self.assertEqual(a.quality["quality_status"], "supported")
            self.assertIn("psychologist_rules", a.module_execution["executed"])
            self.assertEqual(a.module_execution["suppressed"], [])

    def test_unsupported_suppresses_clinical_modules(self):
        from doar.analysis import analyze_image
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "u.png"; _unsupported_image(p)
            a = analyze_image(p, Path(d) / "o")
            self.assertEqual(a.quality["quality_status"], "unsupported")
            self.assertEqual(a.emotion["status"], "suppressed")
            self.assertEqual(a.rule_evaluations, [])
            self.assertEqual(a.concerns, [])
            for m in ("emotion_model", "psychologist_rules", "concern_profiles"):
                self.assertIn(m, a.module_execution["suppressed"])

    def test_thresholds_not_claimed_validated(self):
        from doar.analysis import analyze_image
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.png"; _supported_image(p)
            a = analyze_image(p, Path(d) / "o")
            self.assertFalse(a.quality["thresholds_validated_on_real_dataset"])


class RuleStatusTests(unittest.TestCase):
    def test_missing_detector_distinct_from_not_matched(self):
        from doar.rules import evaluate_rules
        composition = {"bounding_box_coverage": 0.5, "placement": "top-left",
                       "foreground_coverage": 0.5}
        evals, _ = evaluate_rules(composition, {}, [])
        statuses = {e["status"] for e in evals}
        self.assertIn("missing_detector", statuses)      # symbol rules
        # missing evidence is not negative evidence -> never not_matched for symbols
        fox = next(e for e in evals if e["rule_id"] == "PSY_AR_ANIMAL_FOX_005")
        self.assertEqual(fox["status"], "missing_detector")
        self.assertNotEqual(fox["status"], "not_matched")


class JudgeRepairTests(unittest.TestCase):
    def _analysis(self, p, d):
        from doar.analysis import analyze_image
        return analyze_image(p, Path(d) / "o").to_dict()

    def test_feature_judge_not_always_pass(self):
        from doar.judges import run_judges
        # Craft an analysis missing bbox evidence -> feature_judge requires_review.
        analysis = {
            "quality": {"quality_status": "supported", "supported": True,
                        "resolution_ok": True, "blur_ok": True, "contrast_ok": True,
                        "unsupported_reasons": []},
            "composition": {"foreground_coverage": 0.3, "empty_space_ratio": 0.7,
                            "centroid_normalized": [0.5, 0.5]},
            "segmentation": {"candidate_disagreement": 0.0},
            "evidence": [],                     # <- no ev_bbox_coverage
            "emotion": {"status": "unavailable"},
            "rule_evaluations": [], "concerns": [],
            "module_execution": {"suppressed": []},
            "safety_disclaimer": "This is not a diagnosis.",
        }
        judges = run_judges(analysis)
        self.assertEqual(judges["feature_judge"]["status"], "requires_review")

    def test_overall_status_and_emotion_judge(self):
        from doar.judges import run_judges
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.png"; _supported_image(p)
            judges = run_judges(self._analysis(p, d))
            self.assertIn(judges["overall_status"], ("pass", "requires_review", "fail"))
            self.assertFalse(judges["emotion_judge"]["emotion_model_ran"])  # no checkpoint

    def test_segmentation_judge_reflects_all_checks(self):
        from doar.judges import run_judges
        analysis = {
            "quality": {"quality_status": "supported", "supported": True,
                        "resolution_ok": True, "blur_ok": True, "contrast_ok": True,
                        "unsupported_reasons": []},
            "composition": {"foreground_coverage": 0.95, "empty_space_ratio": 0.05,
                            "centroid_normalized": [0.5, 0.5]},
            "segmentation": {"candidate_disagreement": 0.0},
            "evidence": [{"evidence_id": "ev_bbox_coverage"}],
            "emotion": {"status": "unavailable"},
            "rule_evaluations": [], "concerns": [],
            "module_execution": {"suppressed": []},
            "safety_disclaimer": "This is not a diagnosis.",
        }
        judges = run_judges(analysis)
        # coverage 0.95 -> implausibly full -> segmentation fails -> overall fail
        self.assertEqual(judges["segmentation_judge"]["status"], "fail")
        self.assertEqual(judges["overall_status"], "fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
