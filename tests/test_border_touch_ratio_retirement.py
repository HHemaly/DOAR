"""Tests for DOAR-TRACE Phase 2A.1, Section 5: retirement of
segmentation.border_touch_ratio from downstream use, and a regression
test reproducing the exact root-cause mechanism (a `_segment`
candidate-selection bias, not primarily morphological cleanup as
Phase 2A's original comment claimed) -- see
docs/BORDER_TOUCH_RATIO_DECISION.md."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import _segment, analyze_image
from doar.features import objective_feature_row


def _top_edge_strip(w: int = 200, h: int = 200) -> Image.Image:
    image = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(image).rectangle((0, 0, w - 1, 10), fill="black")
    return image


class RootCauseReproductionTests(unittest.TestCase):
    """Reproduces the exact mechanism documented in
    docs/BORDER_TOUCH_RATIO_DECISION.md, so a future change to `_segment`
    that accidentally "fixes" this the wrong way (e.g. only patching
    cleanup) is caught by a test that actually checks WHERE the failure
    originates, not just the final aggregate number."""

    def test_adaptive_candidate_fails_at_the_border_before_cleanup_runs(self):
        rgb = np.asarray(_top_edge_strip())
        _mask, _bg, _conf, candidates, _diag = _segment(rgb)
        adaptive = candidates["adaptive_grayscale"]
        # The primary bug: the adaptive-threshold candidate itself misses
        # the border-touching strip, BEFORE any cleanup pass runs.
        self.assertEqual(int(adaptive[0].sum()), 0, "adaptive_grayscale candidate should miss row 0 pre-cleanup")

    def test_other_two_candidates_correctly_detect_the_border_touching_strip(self):
        rgb = np.asarray(_top_edge_strip())
        _mask, _bg, _conf, candidates, _diag = _segment(rgb)
        self.assertEqual(int(candidates["colour_distance"][0].sum()), 200)
        self.assertEqual(int(candidates["global_grayscale"][0].sum()), 200)

    def test_candidate_scoring_rewards_the_flawed_candidate(self):
        # The compounding bug: _candidate_score's (1 - border_ratio) term
        # scores the border-blind candidate HIGHER than the two correct ones.
        rgb = np.asarray(_top_edge_strip())
        _mask, _bg, _conf, _candidates, diag = _segment(rgb)
        scores = diag["candidate_scores"]
        self.assertGreater(scores["adaptive_grayscale"], scores["colour_distance"])
        self.assertGreater(scores["adaptive_grayscale"], scores["global_grayscale"])
        self.assertEqual(diag["selected_strategy"], "adaptive_grayscale")

    def test_cleanup_is_a_minor_not_primary_contributor(self):
        # Cleanup, applied AFTER the flawed candidate is selected, actually
        # partially RECOVERS the under-detected band (row 1) rather than
        # being the source of the erosion -- row 0 has zero cleanup-time
        # support either way, since it starts at False in the selected
        # candidate with nothing to inherit from.
        rgb = np.asarray(_top_edge_strip())
        mask, _bg, _conf, candidates, _diag = _segment(rgb)
        adaptive = candidates["adaptive_grayscale"]
        self.assertEqual(int(adaptive[1].sum()), 0, "row 1 should also be missed pre-cleanup")
        self.assertGreater(int(mask[1].sum()), 0, "cleanup should partially recover row 1")
        self.assertEqual(int(mask[0].sum()), 0, "row 0 has no support to recover from even after cleanup")


class FeatureRetirementTests(unittest.TestCase):
    def _features_for(self, image: Image.Image) -> dict:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        result = analyze_image(path, Path(temp.name) / "out")
        return objective_feature_row(path, result.to_dict())

    def test_border_touch_ratio_is_marked_unreliable_on_the_failure_case(self):
        fv = self._features_for(_top_edge_strip())["segmentation.border_touch_ratio"]
        self.assertEqual(fv.confidence, 0.0)
        self.assertTrue(fv.missing)
        self.assertIn("retired", fv.method)

    def test_border_touch_ratio_is_marked_unreliable_even_when_numerically_correct(self):
        # Retirement is unconditional -- the feature is never selectively
        # trusted just because a particular case happens to look right.
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 149, 149), fill="black")  # isolated, no edge touch
        fv = self._features_for(image)["segmentation.border_touch_ratio"]
        self.assertEqual(fv.value, 0.0)  # numerically "correct" for this case
        self.assertEqual(fv.confidence, 0.0)  # but still marked unreliable
        self.assertTrue(fv.missing)

    def test_historical_value_is_preserved_not_deleted(self):
        # "Retired from downstream use", not removed -- the feature must
        # still exist and carry a real (if untrusted) numeric value.
        fv = self._features_for(_top_edge_strip())["segmentation.border_touch_ratio"]
        self.assertIsInstance(fv.value, float)
        self.assertTrue(np.isfinite(fv.value))

    def test_other_features_are_unaffected_by_the_retirement(self):
        features = self._features_for(_top_edge_strip())
        fv = features["segmentation.foreground_coverage"]
        self.assertGreater(fv.confidence, 0.0)
        self.assertFalse(fv.missing)


class NoDownstreamConsumerTests(unittest.TestCase):
    """Confirms border_touch_ratio was, and remains, unconsumed by any
    rule -- the precondition that made retirement (rather than a
    disruptive fix to shared `_segment` logic) the safe choice."""

    def test_no_production_rule_references_it(self):
        registry = json.loads((ROOT / "resources/psychology_sources/rules_registry.json").read_text(encoding="utf-8"))
        for rule in registry["rules"]:
            self.assertNotIn("border_touch", json.dumps(rule), rule["rule_id"])

    def test_no_registry_v2_rule_references_it(self):
        registry_v2 = json.loads((ROOT / "resources/psychology_sources/rules_registry_v2.json").read_text(encoding="utf-8"))
        for rule in registry_v2["rules"]:
            self.assertNotIn("border_touch", json.dumps(rule), rule["rule_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
