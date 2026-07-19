"""
Tests for the D3 (cross-split leakage block) and D7 (bilingual safety judge) fixes.

Pure-stdlib + PIL/numpy only; no dataset, torch, or network required.
"""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _make_img(path: Path, colour, size=(64, 64), textured=True):
    """Write an image. Textured (non-flat) so perceptual hashing is informative;
    a per-image seed varies the texture so distinct images get distinct hashes."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, colour)
    if textured:
        px = img.load()
        seed = sum(colour)
        for y in range(size[1]):
            for x in range(size[0]):
                if (x * 7 + y * 13 + seed) % 5 == 0:
                    px[x, y] = (0, 0, 0)
    img.save(path)


class LeakageTests(unittest.TestCase):
    def test_clean_split_has_no_leakage(self):
        from doar.dataset import build_manifest
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # distinct colours per split -> no exact or near duplicates
            _make_img(root / "train" / "Happy" / "a.png", (255, 0, 0))
            _make_img(root / "valid" / "Sad" / "b.png", (0, 255, 0))
            _make_img(root / "test" / "Angry" / "c.png", (0, 0, 255))
            summary = build_manifest(root, root / "manifest.csv")
            self.assertTrue(summary["leakage_ok"])
            self.assertEqual(summary["leakage_status"], "PASS")

    def test_exact_cross_split_duplicate_is_flagged(self):
        from doar.dataset import build_manifest
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # identical bytes in train and test -> exact cross-split leakage
            _make_img(root / "train" / "Happy" / "same.png", (123, 200, 50))
            _make_img(root / "test" / "Happy" / "same.png", (123, 200, 50))
            summary = build_manifest(root, root / "manifest.csv")
            self.assertFalse(summary["leakage_ok"])
            self.assertEqual(summary["leakage_status"], "FAIL_LEAKAGE_DETECTED")
            self.assertTrue(summary["cross_split_exact_leakage"])
            splits = summary["cross_split_exact_leakage"][0]["splits"]
            self.assertIn("train", splits)
            self.assertIn("test", splits)


class SafetyJudgeTests(unittest.TestCase):
    def _analysis(self, rule):
        return {
            "composition": {"foreground_coverage": 0.3, "empty_space_ratio": 0.7,
                            "centroid_normalized": [0.5, 0.5]},
            "quality": {"supported": True},
            "segmentation": {"candidate_disagreement": 0.0},
            "evidence": [{"evidence_id": "ev_bbox_coverage"}],
            "rule_evaluations": [rule],
            "safety_disclaimer": "This is not a diagnosis. غير تشخيصي.",
        }

    def test_english_broadened_pattern_caught(self):
        from doar.judges import run_judges
        rule = {"rule_id": "r1", "matched_evidence_ids": [],
                "professional_reasoning": "The child shows signs of trauma."}
        out = run_judges(self._analysis(rule))
        self.assertTrue(out["safety_judge"]["diagnostic_language_found"])
        self.assertEqual(out["safety_judge"]["status"], "fail")

    def test_arabic_diagnostic_language_caught(self):
        from doar.judges import run_judges
        rule = {"rule_id": "r2", "matched_evidence_ids": [],
                "arabic": "الطفل يعاني من اكتئاب."}  # "the child suffers from depression"
        out = run_judges(self._analysis(rule))
        self.assertTrue(out["safety_judge"]["diagnostic_language_found"])
        self.assertEqual(out["safety_judge"]["status"], "fail")

    def test_parent_wording_is_scanned(self):
        from doar.judges import run_judges
        rule = {"rule_id": "r3", "matched_evidence_ids": [],
                "parent_safe_wording": "This proves depression in the child."}
        out = run_judges(self._analysis(rule))
        self.assertTrue(out["safety_judge"]["diagnostic_language_found"])

    def test_safe_wording_passes(self):
        from doar.judges import run_judges
        rule = {"rule_id": "r4", "matched_evidence_ids": [],
                "professional_reasoning": "Wide eyes were detected; meaning is unknown without asking the child.",
                "parent_safe_wording": "Ask the child what the expression means."}
        out = run_judges(self._analysis(rule))
        self.assertFalse(out["safety_judge"]["diagnostic_language_found"])
        self.assertEqual(out["safety_judge"]["status"], "pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
