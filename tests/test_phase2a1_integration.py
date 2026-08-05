"""End-to-end integration tests for DOAR-TRACE Phase 2A.1, Section 9's
required guarantees that aren't already covered by a more targeted unit
test elsewhere: page-relative features genuinely use a confirmed page
region (not silently the whole image) all the way through the real
`analyze_image` pipeline, and canonical (resolution-normalized) evidence
is never confused with the original evidence line-proxy rules actually
evaluate against."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image


def _analyze(image: Image.Image, user_page_declaration=None) -> dict:
    temp = tempfile.TemporaryDirectory()
    try:
        path = Path(temp.name) / "image.png"
        image.save(path)
        result = analyze_image(path, Path(temp.name) / "out", user_page_declaration=user_page_declaration)
        return result.to_dict()
    finally:
        temp.cleanup()


class PageRelativeFeaturesUseARealPageRegionTests(unittest.TestCase):
    def test_user_defined_smaller_page_polygon_changes_the_coverage_value(self):
        # A real end-to-end proof that page-relative coverage is NOT just
        # a relabeled copy of image-relative coverage: a genuinely smaller
        # user-supplied page polygon must produce a different number.
        image = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
        declaration = {"mode": "user_defined_page_corners", "corners": [[20, 20], [280, 20], [280, 280], [20, 280]]}
        d = _analyze(image, user_page_declaration=declaration)
        self.assertEqual(d["page_reference"]["page_reference_mode"], "user_defined_page_corners")
        image_relative = d["objective_features"]["segmentation.bounding_box_coverage"]["value"]
        page_relative = d["objective_features"]["segmentation.page_relative_bounding_box_coverage"]["value"]
        self.assertNotAlmostEqual(image_relative, page_relative, places=3)
        self.assertGreater(page_relative, image_relative)  # smaller page area -> higher relative coverage

    def test_cropped_image_never_gets_a_page_relative_value(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        d = _analyze(image)
        self.assertNotIn(d["page_reference"]["page_reference_mode"], ("auto_detected_page", "user_confirmed_full_frame", "user_defined_page_corners"))
        feature = d["objective_features"]["segmentation.page_relative_bounding_box_coverage"]
        self.assertTrue(feature["missing"])

    def test_user_confirmed_full_frame_overrides_automatic_cropped_classification(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
        d = _analyze(image, user_page_declaration={"mode": "user_confirmed_full_frame"})
        self.assertEqual(d["page_reference"]["page_reference_mode"], "user_confirmed_full_frame")
        self.assertTrue(d["page_reference"]["page_relative_features_assessable"])
        by_rule = {r["rule_id"]: r["status"] for r in d["rule_evaluations"]}
        # A page-gated rule must now be genuinely evaluated (not forced
        # not_assessable), even though the automatic page_frame status for
        # this same thin-margin image is not full/likely-full.
        self.assertIn(by_rule["PSY_AR_SIZE_FULL_015"], ("weak_support", "not_matched"))


class CanonicalEvidenceNeverConfusedWithLineProxyEvidenceTests(unittest.TestCase):
    def test_line_proxy_rules_are_evaluated_from_original_not_canonical_features(self):
        # canonical_features only ever covers 4 geometry/colour features
        # (never stroke.intensity_proxy/fragmentation) -- so line-proxy
        # rule evaluation structurally cannot read a canonicalized value,
        # even for an oversized image that DOES get resized.
        image = Image.new("RGB", (2000, 2500), "white")
        ImageDraw.Draw(image).rectangle((500, 500, 1500, 2000), fill="black")
        d = _analyze(image)
        self.assertTrue(d["canonical_features"]["canonicalization"]["resized"])
        self.assertNotIn("stroke.intensity_proxy", d["canonical_features"]["features"])
        self.assertNotIn("stroke.fragmentation", d["canonical_features"]["features"])
        # The rule evaluations exist and are grounded in the real,
        # original-image objective_features value (not silently swapped).
        original_intensity = d["objective_features"]["stroke.intensity_proxy"]["value"]
        by_rule = {r["rule_id"]: r for r in d["rule_evaluations"]}
        heavy = by_rule["EN_COMPILED_LINE_HEAVY_PRESSURE_030"]
        if heavy["status"] == "weak_support":
            self.assertIn("ev_feature_stroke_intensity_proxy", heavy["matched_evidence_ids"])
        # Sanity: the original value is a real, finite proxy, not the
        # canonical geometry set's placeholder absence.
        self.assertTrue(0.0 <= original_intensity <= 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
