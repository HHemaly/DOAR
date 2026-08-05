"""Regression test for DOAR-TRACE 4B: objective features must be computed
and persisted for EVERY normal analyze_image run, including plain
deep-model/no-checkpoint inference -- not only inside emotion.py's
fusion-checkpoint branch (the gap CURRENT_TO_TARGET_GAP_V2.md documents).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image
from doar.features import objective_feature_row


class ObjectiveFeaturePersistenceTests(unittest.TestCase):
    def _analyze(self, image: Image.Image, checkpoint=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        out = Path(temp.name) / "out"
        result = analyze_image(path, out, emotion_checkpoint=checkpoint)
        return result, out

    def test_no_checkpoint_run_still_computes_all_features(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((50, 50, 150, 150), fill="black")
        result, out = self._analyze(image, checkpoint=None)
        # 60, not 59: Phase 2A.1 Section 3 adds
        # segmentation.page_relative_bounding_box_coverage.
        self.assertEqual(len(result.objective_features), 60)

    def test_no_checkpoint_run_writes_objective_features_json(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((50, 50, 150, 150), fill="black")
        _, out = self._analyze(image, checkpoint=None)
        doc_path = out / "objective_features.json"
        self.assertTrue(doc_path.exists(), "objective_features.json was not written")
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["feature_count"], 60)
        # 3, not 2: Phase 2A.1 Section 5 retires segmentation.border_touch_ratio
        # from downstream use (confidence=0.0/missing=True) -- see
        # docs/BORDER_TOUCH_RATIO_DECISION.md. The other 2 missing features
        # (shape.enclosed_shape_count/repetition_score) are unchanged.
        self.assertEqual(doc["missing_count"], 3)

    def test_analysis_json_also_contains_objective_features(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((50, 50, 150, 150), fill="black")
        _, out = self._analyze(image, checkpoint=None)
        analysis_doc = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
        self.assertIn("objective_features", analysis_doc)
        self.assertEqual(len(analysis_doc["objective_features"]), 60)

    def test_every_feature_envelope_has_the_required_fields(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((50, 50, 150, 150), fill="black")
        _, out = self._analyze(image, checkpoint=None)
        doc = json.loads((out / "objective_features.json").read_text(encoding="utf-8"))
        required = {"feature_id", "value", "unit", "method", "missing", "evidence_id", "judge_status", "limitations"}
        for entry in doc["features"]:
            self.assertEqual(set(entry.keys()), required, entry)

    def test_missing_features_keep_honest_nan_not_fabricated_zero(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((50, 50, 150, 150), fill="black")
        _, out = self._analyze(image, checkpoint=None)
        doc = json.loads((out / "objective_features.json").read_text(encoding="utf-8"))
        by_id = {f["feature_id"]: f for f in doc["features"]}
        for name in ("shape.enclosed_shape_count", "shape.repetition_score"):
            self.assertTrue(by_id[name]["missing"])
            # json.dumps(..., allow_nan=True) round-trips NaN as the literal
            # token NaN, which json.loads parses back to the Python float nan.
            self.assertNotEqual(by_id[name]["value"], 0.0)
            self.assertTrue(by_id[name]["value"] != by_id[name]["value"])  # NaN != NaN

    def test_low_quality_image_still_computes_features_only_emotion_is_suppressed(self):
        # A tiny, blank image fails the quality gate -- emotion/rules/concerns
        # are suppressed, but segmentation/composition/colour (and therefore
        # objective features) are computed before the quality branch runs.
        image = Image.new("RGB", (20, 20), "white")
        result, out = self._analyze(image, checkpoint=None)
        self.assertEqual(result.quality["quality_status"], "unsupported")
        # 60, not 59: Phase 2A.1 Section 3 adds
        # segmentation.page_relative_bounding_box_coverage.
        self.assertEqual(len(result.objective_features), 60)
        doc = json.loads((out / "objective_features.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["feature_count"], 60)

    def test_result_matches_direct_objective_feature_row_call_in_shape(self):
        """Sanity check: analyze_image's persisted features have the same
        keys objective_feature_row itself produces -- no silent subset/rename."""
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((50, 50, 150, 150), fill="black")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        out = Path(temp.name) / "out"
        result = analyze_image(path, out)
        direct = objective_feature_row(path, result.to_dict())
        self.assertEqual(set(result.objective_features.keys()), set(direct.keys()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
