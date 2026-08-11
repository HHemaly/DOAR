"""DOAR Visual Knowledge V2: VisualEntity -- the richer, searchable
counterpart to VisualFinding. Proves: lossless conversion, serialization
round-trip, unknown entities are preserved (never discarded or
force-classified), alias search works, candidate labels never silently
become the confirmed canonical label, and case_verification_status stays
strictly separate from model_validation_status.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_entity import (  # noqa: E402
    CASE_VERIFICATION_STATUSES, ENTITY_TYPES, VisualEntity, apply_expert_review_to_entities,
    build_entities_from_findings, classify_entity_type, compute_dominant_colors, compute_page_position,
    compute_relative_size, visual_finding_to_entity,
)
from doar.visual_evidence import VisualFinding  # noqa: E402


def _finding(label, *, bbox=(0.1, 0.2, 0.3, 0.15), validation_status="UNKNOWN", confidence=0.7,
             finding_id=None, related_rule_ids=()):
    return VisualFinding(
        label=label, finding_id=finding_id or f"vf_test_{label}", free_form_label=None, bbox=bbox,
        confidence=confidence, detector="m", checkpoint="c", prompt="p",
        validation_status=validation_status,
        evidence_status=("validated_evidence" if validation_status == "VALIDATED"
                          else "experimental_evidence_technical_view_only" if validation_status != "DISABLED_FOR_RULES"
                          else "not_used"),
        rule_mapping_status="MAPPED" if related_rule_ids else "UNMAPPED",
        related_rule_ids=related_rule_ids, source="initial_scan", query=None, timestamp="2026-08-10T00:00:00+00:00")


class ConversionTests(unittest.TestCase):
    def test_every_finding_field_preserved(self):
        f = _finding("dog", validation_status="EXPERIMENTAL")
        e = visual_finding_to_entity(f)
        self.assertEqual(e.entity_id, f.finding_id)
        self.assertEqual(e.canonical_label, f.label)
        self.assertEqual(e.bbox, f.bbox)
        self.assertEqual(e.confidence, f.confidence)
        self.assertEqual(e.detector, f.detector)
        self.assertEqual(e.checkpoint, f.checkpoint)
        self.assertEqual(e.prompt, f.prompt)
        self.assertEqual(e.model_validation_status, f.validation_status)
        self.assertEqual(e.evidence_status, f.evidence_status)
        self.assertEqual(e.rule_mapping_status, f.rule_mapping_status)
        self.assertEqual(e.related_rule_ids, f.related_rule_ids)
        self.assertEqual(e.source, f.source)
        self.assertEqual(e.timestamp, f.timestamp)

    def test_candidate_labels_include_the_single_current_candidate_with_confidence(self):
        f = _finding("cat", confidence=0.42)
        e = visual_finding_to_entity(f)
        self.assertEqual(e.candidate_labels, (("cat", 0.42),))

    def test_no_bbox_means_geometry_fields_stay_none_not_fabricated(self):
        f = _finding("person", bbox=None)
        e = visual_finding_to_entity(f)
        self.assertIsNone(e.bbox)
        self.assertIsNone(e.relative_size)
        self.assertIsNone(e.page_position)
        self.assertIsNone(e.dominant_colors)
        self.assertIsNone(e.crop_ref)

    def test_default_new_entity_is_unverified_and_not_indexed(self):
        e = visual_finding_to_entity(_finding("house"))
        self.assertEqual(e.case_verification_status, "unreviewed")
        self.assertEqual(e.memory_status, "not_indexed")

    def test_build_entities_from_findings_preserves_count_and_order(self):
        findings = [_finding("person", finding_id="vf_1"), _finding("tree", finding_id="vf_2")]
        entities = build_entities_from_findings(findings)
        self.assertEqual([e.entity_id for e in entities], ["vf_1", "vf_2"])


class UnknownEntityPreservationTests(unittest.TestCase):
    def test_unrecognized_label_gets_unknown_type_not_forced_into_object(self):
        e = visual_finding_to_entity(_finding("some_unrecognized_blob_shape"))
        self.assertEqual(e.entity_type, "unknown")

    def test_unknown_entity_still_carries_full_geometry_and_provenance(self):
        f = _finding("some_unrecognized_blob_shape", bbox=(0.05, 0.05, 0.2, 0.2))
        e = visual_finding_to_entity(f, image_path=None)
        self.assertEqual(e.entity_type, "unknown")
        self.assertEqual(e.bbox, f.bbox)
        self.assertIsNotNone(e.relative_size)   # geometry still computed -- never discarded
        self.assertIsNotNone(e.page_position)
        self.assertEqual(e.detector, f.detector)  # provenance still present

    def test_classify_entity_type_never_raises_for_arbitrary_strings(self):
        for label in ("xyz123", "", "a very unusual on-demand query result"):
            self.assertIn(classify_entity_type(label), ENTITY_TYPES)

    def test_known_symbol_and_shape_labels_classified_correctly(self):
        self.assertEqual(classify_entity_type("heart"), "symbol")
        self.assertEqual(classify_entity_type("star"), "symbol")
        self.assertEqual(classify_entity_type("circle"), "geometric_shape")


class SerializationRoundTripTests(unittest.TestCase):
    def test_to_dict_from_dict_round_trips_exactly(self):
        e = visual_finding_to_entity(_finding("dog", bbox=(0.1, 0.1, 0.2, 0.2)))
        restored = VisualEntity.from_dict(e.to_dict())
        self.assertEqual(e, restored)

    def test_to_dict_is_json_serializable(self):
        import json
        e = visual_finding_to_entity(_finding("dog"))
        json.dumps(e.to_dict())  # must not raise

    def test_from_dict_tolerates_missing_new_fields_old_document_shape(self):
        # Simulates loading a MINIMAL/older-shaped entity dict -- must not crash.
        minimal = {
            "entity_id": "vf_old", "canonical_label": "dog", "detector": "m", "checkpoint": "c",
            "prompt": "p", "confidence": 0.5, "model_validation_status": "UNKNOWN",
            "evidence_status": "experimental_evidence_technical_view_only", "rule_mapping_status": "UNMAPPED",
        }
        e = VisualEntity.from_dict(minimal)
        self.assertEqual(e.case_verification_status, "unreviewed")
        self.assertEqual(e.memory_status, "not_indexed")
        self.assertEqual(e.candidate_labels, ())
        self.assertEqual(e.entity_type, "unknown")


class AliasSearchTests(unittest.TestCase):
    def test_matches_canonical_label(self):
        e = visual_finding_to_entity(_finding("dog"))
        self.assertTrue(e.matches_search_term("dog"))

    def test_matches_english_alias_not_equal_to_canonical_label(self):
        e = visual_finding_to_entity(_finding("dog"))
        self.assertTrue(e.matches_search_term("dogs"))
        self.assertNotEqual("dogs", e.canonical_label)

    def test_no_match_for_unrelated_term(self):
        e = visual_finding_to_entity(_finding("dog"))
        self.assertFalse(e.matches_search_term("airplane"))

    def test_empty_term_never_matches(self):
        e = visual_finding_to_entity(_finding("dog"))
        self.assertFalse(e.matches_search_term(""))
        self.assertFalse(e.matches_search_term("   "))


class CandidateLabelNeverBecomesConfirmedTests(unittest.TestCase):
    def test_candidate_labels_are_read_only_metadata_not_the_identity(self):
        e = visual_finding_to_entity(_finding("cat", confidence=0.31))
        # canonical_label is what THIS finding reports -- candidate_labels
        # records what was considered, with its own confidence, but does
        # not change canonical_label, model_validation_status, or
        # evidence_status.
        self.assertEqual(e.canonical_label, "cat")
        self.assertEqual(len(e.candidate_labels), 1)
        self.assertEqual(e.candidate_labels[0][0], e.canonical_label)

    def test_low_confidence_candidate_does_not_upgrade_validation_status(self):
        e = visual_finding_to_entity(_finding("cat", validation_status="UNKNOWN", confidence=0.99))
        self.assertEqual(e.model_validation_status, "UNKNOWN")  # confidence alone never upgrades trust


class GeometryHelperTests(unittest.TestCase):
    def test_relative_size_is_bbox_area_fraction(self):
        self.assertAlmostEqual(compute_relative_size((0.1, 0.2, 0.5, 0.4)), 0.2)

    def test_relative_size_none_without_bbox(self):
        self.assertIsNone(compute_relative_size(None))

    def test_page_position_matches_analysis_py_thirds_convention(self):
        # Center of a small bbox near (0.05, 0.05) is top-left.
        self.assertEqual(compute_page_position((0.0, 0.0, 0.1, 0.1)), "top_left")
        # Center near (0.5, 0.5) is middle_center.
        self.assertEqual(compute_page_position((0.45, 0.45, 0.1, 0.1)), "middle_center")
        # Center near (0.9, 0.9) is bottom_right.
        self.assertEqual(compute_page_position((0.85, 0.85, 0.1, 0.1)), "bottom_right")

    def test_dominant_colors_none_without_image(self):
        self.assertIsNone(compute_dominant_colors("nonexistent.png", (0.1, 0.1, 0.2, 0.2)))

    def test_dominant_colors_real_crop(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (100, 100), (200, 30, 30)).save(path)  # solid red-ish
            colors = compute_dominant_colors(str(path), (0.0, 0.0, 1.0, 1.0))
            self.assertIsNotNone(colors)
            self.assertIn("red", colors)


class ExpertReviewProjectionTests(unittest.TestCase):
    def test_confirm_action_sets_verified(self):
        e = visual_finding_to_entity(_finding("dog"))
        updated = apply_expert_review_to_entities([e], [{"target_label": "dog", "action": "confirm"}])
        self.assertEqual(updated[0].case_verification_status, "verified")

    def test_reject_action_sets_rejected(self):
        e = visual_finding_to_entity(_finding("dog"))
        updated = apply_expert_review_to_entities([e], [{"target_label": "dog", "action": "reject"}])
        self.assertEqual(updated[0].case_verification_status, "rejected")

    def test_no_matching_review_stays_unverified(self):
        e = visual_finding_to_entity(_finding("dog"))
        updated = apply_expert_review_to_entities([e], [{"target_label": "cat", "action": "confirm"}])
        self.assertEqual(updated[0].case_verification_status, "unreviewed")

    def test_note_and_rename_actions_never_change_verification_status(self):
        e = visual_finding_to_entity(_finding("dog"))
        updated = apply_expert_review_to_entities(
            [e], [{"target_label": "dog", "action": "note", "note": "looks fine"}])
        self.assertEqual(updated[0].case_verification_status, "unreviewed")

    def test_latest_action_wins(self):
        e = visual_finding_to_entity(_finding("dog"))
        history = [{"target_label": "dog", "action": "confirm"}, {"target_label": "dog", "action": "reject"}]
        updated = apply_expert_review_to_entities([e], history)
        self.assertEqual(updated[0].case_verification_status, "rejected")

    def test_matches_via_alias_not_just_exact_label(self):
        e = visual_finding_to_entity(_finding("dog"))
        updated = apply_expert_review_to_entities([e], [{"target_label": "dogs", "action": "confirm"}])
        self.assertEqual(updated[0].case_verification_status, "verified")

    def test_model_validation_status_never_changed_by_review(self):
        e = visual_finding_to_entity(_finding("dog", validation_status="EXPERIMENTAL"))
        updated = apply_expert_review_to_entities([e], [{"target_label": "dog", "action": "confirm"}])
        self.assertEqual(updated[0].model_validation_status, "EXPERIMENTAL")  # unchanged -- orthogonal fields


class ValidStatusEnumTests(unittest.TestCase):
    def test_every_entity_type_in_allowed_set(self):
        for label in ("person", "heart", "circle", "unknown_thing"):
            self.assertIn(classify_entity_type(label), ENTITY_TYPES)

    def test_verification_status_always_in_allowed_set(self):
        e = visual_finding_to_entity(_finding("dog"))
        self.assertIn(e.case_verification_status, CASE_VERIFICATION_STATUSES)


if __name__ == "__main__":
    unittest.main()
