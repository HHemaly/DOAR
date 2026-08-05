"""Tests for page_reference.py (DOAR-TRACE Phase 2A.1, Section 3)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.page_reference import (
    ASSESSABLE_PAGE_REFERENCE_MODES,
    PARENT_PAGE_DECLARATION_CHOICES,
    InvalidPagePolygonError,
    PageReference,
    page_relative_bounding_box_coverage,
    resolve_page_reference,
    user_page_declaration_from_choice,
)


def _pf(status: str, confidence: float = 0.9, limitations=None) -> dict:
    return {"page_frame_status": status, "confidence": confidence, "limitations": limitations or []}


class AutoDetectedModeTests(unittest.TestCase):
    def test_full_page_detected_becomes_auto_detected_page(self):
        ref = resolve_page_reference(_pf("full_page_detected", 0.95))
        self.assertEqual(ref.page_reference_mode, "auto_detected_page")
        self.assertTrue(ref.page_relative_features_assessable)
        self.assertEqual(ref.page_polygon, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.assertEqual(ref.confidence, 0.95)
        self.assertEqual(ref.page_area_fraction, 1.0)

    def test_likely_full_page_becomes_auto_detected_page(self):
        ref = resolve_page_reference(_pf("likely_full_page", 0.6))
        self.assertEqual(ref.page_reference_mode, "auto_detected_page")
        self.assertTrue(ref.page_relative_features_assessable)

    def test_cropped_is_not_assessable_and_has_no_polygon(self):
        ref = resolve_page_reference(_pf("cropped_or_content_only", 0.8))
        self.assertEqual(ref.page_reference_mode, "cropped_or_content_only")
        self.assertFalse(ref.page_relative_features_assessable)
        self.assertIsNone(ref.page_polygon)
        self.assertIsNone(ref.page_area_fraction)

    def test_uncertain_is_not_assessable(self):
        ref = resolve_page_reference(_pf("uncertain", 0.3))
        self.assertEqual(ref.page_reference_mode, "uncertain")
        self.assertFalse(ref.page_relative_features_assessable)
        self.assertIsNone(ref.page_polygon)

    def test_failed_is_not_assessable_zero_confidence(self):
        ref = resolve_page_reference(_pf("failed", 0.0))
        self.assertEqual(ref.page_reference_mode, "failed")
        self.assertFalse(ref.page_relative_features_assessable)

    def test_never_infers_a_smaller_polygon_for_a_cropped_image(self):
        # Explicit requirement: never manufacture a page boundary.
        ref = resolve_page_reference(_pf("cropped_or_content_only"))
        self.assertIsNone(ref.page_polygon)


class UserConfirmedFullFrameTests(unittest.TestCase):
    def test_overrides_automatic_cropped_classification(self):
        # A human assertion takes precedence over the automatic heuristic --
        # even if page_frame.py itself said cropped.
        ref = resolve_page_reference(_pf("cropped_or_content_only"), user_page_declaration={"mode": "user_confirmed_full_frame"})
        self.assertEqual(ref.page_reference_mode, "user_confirmed_full_frame")
        self.assertTrue(ref.page_relative_features_assessable)
        self.assertEqual(ref.confidence, 1.0)
        self.assertEqual(ref.page_area_fraction, 1.0)

    def test_traceable_provenance(self):
        ref = resolve_page_reference(_pf("uncertain"), user_page_declaration={"mode": "user_confirmed_full_frame"})
        self.assertEqual(ref.obtained_via, "explicit_user_assertion_v1")
        self.assertTrue(any("user asserted" in lim.lower() for lim in ref.limitations))


class UserDefinedPageCornersTests(unittest.TestCase):
    def test_deterministic_normalized_coordinates(self):
        corners = [[10, 10], [190, 10], [190, 190], [10, 190]]
        ref = resolve_page_reference(
            _pf("cropped_or_content_only"),
            user_page_declaration={"mode": "user_defined_page_corners", "corners": corners},
            image_width=200, image_height=200,
        )
        self.assertEqual(ref.page_reference_mode, "user_defined_page_corners")
        self.assertTrue(ref.page_relative_features_assessable)
        expected = [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]
        for got, want in zip(ref.page_polygon, expected):
            self.assertAlmostEqual(got[0], want[0], places=6)
            self.assertAlmostEqual(got[1], want[1], places=6)
        self.assertAlmostEqual(ref.page_area_fraction, 0.81, places=6)

    def test_same_corners_always_produce_identical_polygon(self):
        corners = [[10, 10], [190, 10], [190, 190], [10, 190]]
        ref1 = resolve_page_reference(
            _pf("uncertain"), user_page_declaration={"mode": "user_defined_page_corners", "corners": corners},
            image_width=200, image_height=200,
        )
        ref2 = resolve_page_reference(
            _pf("uncertain"), user_page_declaration={"mode": "user_defined_page_corners", "corners": corners},
            image_width=200, image_height=200,
        )
        self.assertEqual(ref1.page_polygon, ref2.page_polygon)

    def test_rejects_wrong_corner_count(self):
        with self.assertRaises(InvalidPagePolygonError):
            resolve_page_reference(
                _pf("uncertain"),
                user_page_declaration={"mode": "user_defined_page_corners", "corners": [[0, 0], [1, 1], [2, 2]]},
                image_width=200, image_height=200,
            )

    def test_rejects_out_of_bounds_corner(self):
        with self.assertRaises(InvalidPagePolygonError):
            resolve_page_reference(
                _pf("uncertain"),
                user_page_declaration={
                    "mode": "user_defined_page_corners",
                    "corners": [[10, 10], [190, 10], [190, 190], [-5, 190]],
                },
                image_width=200, image_height=200,
            )

    def test_rejects_degenerate_near_zero_area_polygon(self):
        with self.assertRaises(InvalidPagePolygonError):
            resolve_page_reference(
                _pf("uncertain"),
                user_page_declaration={
                    "mode": "user_defined_page_corners",
                    "corners": [[100, 100], [101, 100], [101, 101], [100, 101]],
                },
                image_width=200, image_height=200,
            )

    def test_requires_image_dimensions(self):
        with self.assertRaises(ValueError):
            resolve_page_reference(
                _pf("uncertain"),
                user_page_declaration={"mode": "user_defined_page_corners", "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            )

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            resolve_page_reference(_pf("uncertain"), user_page_declaration={"mode": "not_a_real_mode"})


class PageReferenceSchemaTests(unittest.TestCase):
    def test_assessable_modes_are_exactly_the_three_with_a_polygon(self):
        self.assertEqual(ASSESSABLE_PAGE_REFERENCE_MODES, {
            "auto_detected_page", "user_confirmed_full_frame", "user_defined_page_corners",
        })

    def test_construction_rejects_assessable_true_without_polygon(self):
        with self.assertRaises(ValueError):
            PageReference(
                page_reference_mode="auto_detected_page", page_polygon=None, confidence=0.9,
                obtained_via="x", page_relative_features_assessable=True, limitations=[], evidence_id="ev_page_reference",
            )

    def test_construction_rejects_non_assessable_mode_marked_assessable(self):
        with self.assertRaises(ValueError):
            PageReference(
                page_reference_mode="cropped_or_content_only",
                page_polygon=[[0, 0], [1, 0], [1, 1], [0, 1]], confidence=0.9,
                obtained_via="x", page_relative_features_assessable=True, limitations=[], evidence_id="ev_page_reference",
            )

    def test_unknown_mode_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            PageReference(
                page_reference_mode="not_a_real_mode", page_polygon=None, confidence=0.5,
                obtained_via="x", page_relative_features_assessable=False, limitations=[], evidence_id="ev_page_reference",
            )

    def test_confidence_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            PageReference(
                page_reference_mode="uncertain", page_polygon=None, confidence=1.5,
                obtained_via="x", page_relative_features_assessable=False, limitations=[], evidence_id="ev_page_reference",
            )


class ParentPageDeclarationChoiceTests(unittest.TestCase):
    """DOAR-TRACE Phase 2A.2, Section 6: the 4-option parent-facing
    control, mapped to the real page_reference API."""

    def test_auto_means_no_declaration(self):
        self.assertIsNone(user_page_declaration_from_choice("auto"))

    def test_yes_maps_to_user_confirmed_full_frame(self):
        self.assertEqual(user_page_declaration_from_choice("yes"), {"mode": "user_confirmed_full_frame"})

    def test_no_maps_to_a_declaration_that_resolves_to_cropped(self):
        decl = user_page_declaration_from_choice("no")
        ref = resolve_page_reference(_pf("full_page_detected"), user_page_declaration=decl)
        self.assertEqual(ref.page_reference_mode, "cropped_or_content_only")
        self.assertFalse(ref.page_relative_features_assessable)
        self.assertEqual(ref.obtained_via, "explicit_user_assertion_v1")

    def test_unsure_maps_to_a_declaration_that_resolves_to_uncertain(self):
        decl = user_page_declaration_from_choice("unsure")
        ref = resolve_page_reference(_pf("full_page_detected"), user_page_declaration=decl)
        self.assertEqual(ref.page_reference_mode, "uncertain")
        self.assertFalse(ref.page_relative_features_assessable)
        self.assertEqual(ref.obtained_via, "explicit_user_assertion_v1")

    def test_no_declaration_overrides_an_automatic_full_page_reading(self):
        # A user explicitly saying "no, this is cropped" must override even
        # an automatic full_page_detected reading -- an explicit human
        # decision always wins.
        decl = user_page_declaration_from_choice("no")
        ref = resolve_page_reference(_pf("full_page_detected", 0.99), user_page_declaration=decl)
        self.assertFalse(ref.page_relative_features_assessable)

    def test_unknown_choice_rejected(self):
        with self.assertRaises(ValueError):
            user_page_declaration_from_choice("maybe")

    def test_exactly_four_choices_exist(self):
        self.assertEqual(set(PARENT_PAGE_DECLARATION_CHOICES), {"auto", "yes", "no", "unsure"})

    def test_user_declared_cropped_is_traceable_and_distinct_from_automatic(self):
        # Same page_reference_mode as an automatic cropped reading, but a
        # different obtained_via -- traceable/distinguishable in the saved
        # case output.
        auto = resolve_page_reference(_pf("cropped_or_content_only"))
        declared = resolve_page_reference(_pf("full_page_detected"), user_page_declaration={"mode": "user_declared_cropped"})
        self.assertEqual(auto.page_reference_mode, declared.page_reference_mode)
        self.assertNotEqual(auto.obtained_via, declared.obtained_via)


class PageRelativeCoverageTests(unittest.TestCase):
    def test_whole_image_modes_reduce_to_image_relative_coverage(self):
        ref = resolve_page_reference(_pf("full_page_detected"))
        result = page_relative_bounding_box_coverage(0.55, ref)
        self.assertAlmostEqual(result, 0.55, places=9)

    def test_smaller_page_polygon_increases_relative_coverage(self):
        corners = [[10, 10], [190, 10], [190, 190], [10, 190]]  # area_fraction = 0.81
        ref = resolve_page_reference(
            _pf("uncertain"), user_page_declaration={"mode": "user_defined_page_corners", "corners": corners},
            image_width=200, image_height=200,
        )
        result = page_relative_bounding_box_coverage(0.5, ref)
        self.assertAlmostEqual(result, 0.5 / 0.81, places=6)

    def test_not_assessable_returns_none_never_zero(self):
        ref = resolve_page_reference(_pf("cropped_or_content_only"))
        result = page_relative_bounding_box_coverage(0.9, ref)
        self.assertIsNone(result)

    def test_missing_bounding_box_coverage_returns_none(self):
        ref = resolve_page_reference(_pf("full_page_detected"))
        self.assertIsNone(page_relative_bounding_box_coverage(None, ref))

    def test_result_is_capped_at_one(self):
        # A pathological/rounding case: bbox coverage very close to the
        # page area must never report > 1.0 (100%).
        corners = [[0, 0], [199, 0], [199, 199], [0, 199]]
        ref = resolve_page_reference(
            _pf("uncertain"), user_page_declaration={"mode": "user_defined_page_corners", "corners": corners},
            image_width=200, image_height=200,
        )
        result = page_relative_bounding_box_coverage(0.999, ref)
        self.assertLessEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
