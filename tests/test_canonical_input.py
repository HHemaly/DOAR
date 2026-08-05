"""Tests for canonical_input.py (DOAR-TRACE Phase 2A.1, Section 6):
resolution-only normalization for the specific features Phase 2A found
resize-sensitive, kept strictly separate from objective_features."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import _segment, analyze_image
from doar.canonical_input import (
    CANONICAL_GEOMETRY_FEATURE_IDS,
    CANONICAL_MAX_DIMENSION,
    canonicalize_image,
    compute_canonical_geometry_features,
)


class CanonicalizeImageTests(unittest.TestCase):
    def test_small_image_is_left_unchanged(self):
        image = Image.new("RGB", (200, 200), "white")
        resized, info = canonicalize_image(image)
        self.assertFalse(info["resized"])
        self.assertEqual(resized.size, (200, 200))
        self.assertEqual(info["scale_factor"], 1.0)

    def test_image_exactly_at_the_limit_is_unchanged(self):
        image = Image.new("RGB", (CANONICAL_MAX_DIMENSION, 800), "white")
        resized, info = canonicalize_image(image)
        self.assertFalse(info["resized"])
        self.assertEqual(resized.size, (CANONICAL_MAX_DIMENSION, 800))

    def test_oversized_image_is_downscaled_preserving_aspect_ratio(self):
        image = Image.new("RGB", (2000, 2500), "white")
        resized, info = canonicalize_image(image)
        self.assertTrue(info["resized"])
        self.assertEqual(max(resized.size), CANONICAL_MAX_DIMENSION)
        original_ratio = 2000 / 2500
        new_ratio = resized.width / resized.height
        self.assertAlmostEqual(original_ratio, new_ratio, places=3)

    def test_never_upscales(self):
        image = Image.new("RGB", (50, 60), "white")
        resized, info = canonicalize_image(image)
        self.assertFalse(info["resized"])
        self.assertEqual(resized.size, (50, 60))

    def test_deterministic(self):
        image = Image.new("RGB", (3000, 1000), "white")
        r1, info1 = canonicalize_image(image)
        r2, info2 = canonicalize_image(image)
        self.assertEqual(info1, info2)
        self.assertEqual(np.asarray(r1).tolist(), np.asarray(r2).tolist())


class ComputeCanonicalGeometryFeaturesTests(unittest.TestCase):
    def test_returns_exactly_the_documented_feature_set(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 149, 149), fill="black")
        rgb = np.asarray(image)
        mask, *_ = _segment(rgb)
        result = compute_canonical_geometry_features(rgb, mask)
        self.assertEqual(set(result.keys()), CANONICAL_GEOMETRY_FEATURE_IDS)

    def test_method_field_marks_canonical_provenance(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 149, 149), fill="black")
        rgb = np.asarray(image)
        mask, *_ = _segment(rgb)
        result = compute_canonical_geometry_features(rgb, mask)
        for fv in result.values():
            self.assertEqual(fv["method"], "canonical_resized_v1")

    def test_full_confidence_when_finite(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 149, 149), fill="black")
        rgb = np.asarray(image)
        mask, *_ = _segment(rgb)
        result = compute_canonical_geometry_features(rgb, mask)
        for fv in result.values():
            self.assertEqual(fv["confidence"], 1.0)
            self.assertFalse(fv["missing"])


class EndToEndWiringTests(unittest.TestCase):
    def _analyze(self, image: Image.Image):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "image.png"
        image.save(path)
        result = analyze_image(path, Path(temp.name) / "out")
        return result.to_dict()

    def test_canonical_features_present_and_separate_from_objective_features(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 149, 149), fill="black")
        d = self._analyze(image)
        self.assertIn("canonical_features", d)
        self.assertIn("features", d["canonical_features"])
        self.assertEqual(set(d["canonical_features"]["features"].keys()), CANONICAL_GEOMETRY_FEATURE_IDS)
        # objective_features is untouched -- still exactly the 60 original-image features.
        self.assertEqual(len(d["objective_features"]), 60)
        for fid in CANONICAL_GEOMETRY_FEATURE_IDS:
            self.assertEqual(d["objective_features"][fid]["method"], "objective_features_v3_1")

    def test_small_image_canonical_values_match_original_exactly(self):
        # Below the canonical size threshold, no resize happens, so the
        # canonical and original values must be numerically identical --
        # not merely "close" -- since they come from the same pixels.
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 149, 149), fill="black")
        d = self._analyze(image)
        self.assertFalse(d["canonical_features"]["canonicalization"]["resized"])
        for fid in CANONICAL_GEOMETRY_FEATURE_IDS:
            self.assertEqual(
                d["canonical_features"]["features"][fid]["value"],
                d["objective_features"][fid]["value"],
            )

    def test_oversized_image_produces_a_real_resize_record(self):
        image = Image.new("RGB", (2000, 2500), "white")
        ImageDraw.Draw(image).rectangle((500, 500, 1500, 2000), fill="black")
        d = self._analyze(image)
        info = d["canonical_features"]["canonicalization"]
        self.assertTrue(info["resized"])
        self.assertEqual(info["original_size"], [2000, 2500])
        self.assertEqual(max(info["canonical_size"]), CANONICAL_MAX_DIMENSION)

    def test_brightness_and_colour_are_never_altered_by_canonicalization(self):
        # The canonical image's pixel VALUES (not just its size) must be
        # unchanged where no resampling occurred -- proves no brightness/
        # contrast/colour adjustment was silently applied.
        image = Image.new("RGB", (200, 200), (240, 100, 50))
        ImageDraw.Draw(image).rectangle((50, 50, 149, 149), fill=(10, 200, 30))
        canonical_image, info = canonicalize_image(image)
        self.assertFalse(info["resized"])
        self.assertEqual(np.asarray(canonical_image).tolist(), np.asarray(image).tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
