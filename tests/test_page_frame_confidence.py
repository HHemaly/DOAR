"""DOAR V1.1 Stage 6: page-frame confidence honesty.

`page_frame.py`'s own `limitations` text admits the heuristic "cannot
distinguish a genuinely full page with a very thin margin from a
tightly cropped photo" -- yet its raw formulas could reach confidence
1.0 exactly. `MAX_AUTOMATIC_CONFIDENCE` caps every AUTOMATIC estimate
below full certainty, reserving 1.0 exclusively for a genuine human
confirmation (page_reference.py's user_confirmed_full_frame/
user_defined_page_corners paths, untouched by this cap).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.page_frame import MAX_AUTOMATIC_CONFIDENCE, assess_page_frame  # noqa: E402
from doar.page_reference import resolve_page_reference  # noqa: E402
from doar.rule_engine_v2 import apply_page_frame_gating  # noqa: E402


def _uniform_white_image(size=200):
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _empty_mask(size=200):
    return np.zeros((size, size), dtype=bool)


def _mask_touching_all_edges(size=200):
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask[size // 2, size // 2] = True
    return mask


class AutomaticConfidenceCeilingTests(unittest.TestCase):
    def test_clearly_assessable_full_page_never_exceeds_the_ceiling(self):
        rgb, mask = _uniform_white_image(), _empty_mask()
        result = assess_page_frame(rgb, mask, background_stability=1.0)
        self.assertEqual(result.page_frame_status, "full_page_detected")
        self.assertLessEqual(result.confidence, MAX_AUTOMATIC_CONFIDENCE)
        self.assertGreater(result.confidence, 0.0)

    def test_confidence_never_reaches_1_0_for_any_automatic_branch(self):
        # Sweep a range of uniformity/edge-touch scenarios -- none may
        # ever report full (1.0) certainty; only a human confirmation may.
        scenarios = [
            (_uniform_white_image(), _empty_mask(), 1.0),
            (_uniform_white_image(), _mask_touching_all_edges(), 1.0),
        ]
        for rgb, mask, bg_stability in scenarios:
            result = assess_page_frame(rgb, mask, background_stability=bg_stability)
            self.assertLess(result.confidence, 1.0, result.page_frame_status)

    def test_uncertain_crop_reports_low_confidence_and_no_polygon(self):
        # A mask touching exactly 2 edges lightly falls into the genuinely
        # ambiguous middle ground -- must be reported as uncertain, never
        # guessed toward full-page or cropped.
        size = 200
        mask = np.zeros((size, size), dtype=bool)
        mask[0, :5] = True
        mask[-1, :5] = True
        rgb = np.full((size, size, 3), 128, dtype=np.uint8)  # noisy/non-uniform border
        result = assess_page_frame(rgb, mask, background_stability=0.3)
        self.assertIn(result.page_frame_status, ("uncertain", "cropped_or_content_only"))
        if result.page_frame_status == "uncertain":
            self.assertFalse(result.assessable)


class UserConfirmationUntouchedByCapTests(unittest.TestCase):
    def test_user_confirmed_full_frame_still_reports_full_confidence(self):
        # The cap applies ONLY to the automatic heuristic -- a genuine
        # recorded human decision legitimately reports confidence=1.0
        # (it reflects what the human asserted, not a machine estimate).
        automatic = assess_page_frame(_uniform_white_image(), _empty_mask(), background_stability=1.0)
        reference = resolve_page_reference(
            automatic.to_dict(), user_page_declaration={"mode": "user_confirmed_full_frame"})
        self.assertEqual(reference.confidence, 1.0)
        self.assertEqual(reference.obtained_via, "explicit_user_assertion_v1")

    def test_automatic_page_reference_confidence_is_capped(self):
        automatic = assess_page_frame(_uniform_white_image(), _empty_mask(), background_stability=1.0)
        reference = resolve_page_reference(automatic.to_dict(), user_page_declaration=None)
        self.assertEqual(reference.obtained_via, "classical_cv_border_uniformity_v1")
        self.assertLessEqual(reference.confidence, MAX_AUTOMATIC_CONFIDENCE)


class PageRelativeRuleAbstentionTests(unittest.TestCase):
    def test_page_relative_rules_gated_not_assessable_when_page_uncertain(self):
        rule_evaluations = [
            {"rule_id": "PSY_AR_SIZE_FULL_015", "status": "weak_support",
             "matched_evidence_ids": ["ev_bbox_coverage"], "missing_evidence": []},
        ]
        page_reference = {"page_relative_features_assessable": False}
        gated = apply_page_frame_gating(rule_evaluations, page_reference)
        self.assertEqual(gated[0]["status"], "not_assessable")
        self.assertNotEqual(gated[0]["status"], "weak_support")

    def test_page_relative_rules_stay_evaluated_when_page_confirmed_assessable(self):
        rule_evaluations = [
            {"rule_id": "PSY_AR_SIZE_FULL_015", "status": "weak_support",
             "matched_evidence_ids": ["ev_bbox_coverage"], "missing_evidence": []},
        ]
        page_reference = {"page_relative_features_assessable": True}
        gated = apply_page_frame_gating(rule_evaluations, page_reference)
        self.assertEqual(gated[0]["status"], "weak_support")


if __name__ == "__main__":
    unittest.main()
