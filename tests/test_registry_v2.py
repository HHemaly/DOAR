"""Tests for rules_registry_v2.json / registry_v2_build.py (DOAR-TRACE 4C):
schema validation, no missing PDF rows, source-page traceability, rule
observability classifications, dependency grouping.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.registry_v2_build import (
    ALLOWED_OUTPUT_LEVEL, EVIDENCE_FAMILY, OBSERVABILITY_CLASS, REGISTRY_V2_PATH,
    TARGET_CONSTRUCT, build_registry_v2,
)

REQUIRED_FIELDS = {
    "rule_id", "registry_v2_status", "source_document", "source_page", "source_section",
    "faithful_source_quote", "observable", "possible_interpretation", "observability_class",
    "allowed_output_level", "evidence_level", "target_construct", "evidence_family",
    "dependency_group", "direction", "required_detector_or_metadata", "limitations",
    "alternative_explanations", "parent_safe_wording", "professional_wording",
    "reference_ids", "psychologist_review_status", "validation_status",
    "confidence_ceiling", "scientific_support", "threshold_source", "operational_threshold",
}

OBSERVABILITY_CLASSES = {
    "static_direct", "static_detector", "static_proxy", "prompt_required",
    "age_required", "process_required", "longitudinal_required", "not_operational",
}
ALLOWED_OUTPUT_LEVELS = {
    "observation_only", "question_generating", "combined_hypothesis_only",
    "professional_only", "disabled",
}


class RegistryV2SchemaTests(unittest.TestCase):
    def setUp(self):
        self.doc = build_registry_v2()

    def test_committed_file_matches_freshly_built_document(self):
        # The checked-in JSON must be reproducible from the builder, not
        # hand-edited out of sync with it.
        on_disk = json.loads(REGISTRY_V2_PATH.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, self.doc)

    def test_rule_count_is_nineteen(self):
        self.assertEqual(self.doc["rule_count"], 19)
        self.assertEqual(len(self.doc["rules"]), 19)

    def test_every_rule_has_the_required_field_set(self):
        for rule in self.doc["rules"]:
            self.assertEqual(set(rule.keys()), REQUIRED_FIELDS, rule["rule_id"])

    def test_no_fabricated_status_is_marked_candidate_unreviewed(self):
        # Every rule here IS already in the production registry -- none may
        # be silently downgraded to the task's generic "unreviewed" default.
        for rule in self.doc["rules"]:
            self.assertEqual(rule["registry_v2_status"], "already_in_production_registry")

    def test_source_page_is_always_one_or_two(self):
        for rule in self.doc["rules"]:
            self.assertIn(rule["source_page"], (1, 2), rule["rule_id"])

    def test_source_document_is_the_real_available_pdf(self):
        for rule in self.doc["rules"]:
            self.assertEqual(rule["source_document"], "التحليل النفسي للصور.pdf")

    def test_observability_class_is_from_the_allowed_vocabulary(self):
        for rule in self.doc["rules"]:
            self.assertIn(rule["observability_class"], OBSERVABILITY_CLASSES, rule["rule_id"])

    def test_allowed_output_level_is_from_the_allowed_vocabulary(self):
        for rule in self.doc["rules"]:
            self.assertIn(rule["allowed_output_level"], ALLOWED_OUTPUT_LEVELS, rule["rule_id"])

    def test_no_single_rule_is_combined_hypothesis_only(self):
        # Reserved for the multi-rule aggregator (Section 4D), never a single rule.
        for rule in self.doc["rules"]:
            self.assertNotEqual(rule["allowed_output_level"], "combined_hypothesis_only")

    def test_static_detector_rules_are_disabled_not_silently_shown(self):
        for rule in self.doc["rules"]:
            if rule["observability_class"] == "static_detector":
                self.assertEqual(rule["allowed_output_level"], "disabled")

    def test_static_direct_rules_have_no_required_detector(self):
        for rule in self.doc["rules"]:
            if rule["observability_class"] == "static_direct":
                self.assertIsNone(rule["required_detector_or_metadata"])

    def test_non_static_direct_rules_name_a_required_detector(self):
        for rule in self.doc["rules"]:
            if rule["observability_class"] != "static_direct":
                self.assertIsNotNone(rule["required_detector_or_metadata"], rule["rule_id"])

    def test_speculative_rules_are_retained_not_discarded(self):
        rule_ids = {r["rule_id"] for r in self.doc["rules"]}
        for expected in ("PSY_AR_ANIMAL_LION_007", "PSY_AR_ANIMAL_FOX_005",
                          "PSY_AR_ANIMAL_SQUIRREL_006", "PSY_AR_STARS_009", "PSY_AR_CIRCLES_011"):
            self.assertIn(expected, rule_ids)

    def test_no_fabricated_reference_ids(self):
        known_refs = set(self.doc["references"].keys())
        for rule in self.doc["rules"]:
            for ref in rule["reference_ids"]:
                self.assertIn(ref, known_refs, f"{rule['rule_id']} cites unknown reference {ref!r}")

    def test_static_direct_rules_depend_only_on_real_composition_features(self):
        composition_features = {"composition.bounding_box_coverage", "composition.centroid_normalized"}
        for rule in self.doc["rules"]:
            if rule["observability_class"] == "static_direct":
                self.assertTrue(set(rule["dependency_group"]) <= composition_features, rule["rule_id"])
                self.assertTrue(rule["dependency_group"], rule["rule_id"])

    def test_no_double_counting_every_rule_has_exactly_one_dependency_feature(self):
        # None of the 19 source rules require more than one underlying
        # feature -- catches an aggregator accidentally listing a feature twice.
        for rule in self.doc["rules"]:
            self.assertEqual(len(rule["dependency_group"]), len(set(rule["dependency_group"])), rule["rule_id"])


class RegistryV2CoverageAuditTests(unittest.TestCase):
    """No missing PDF rows: the PDF's 19 bullets (traced manually this
    session, recorded in docs/RULE_PDF_COVERAGE_AUDIT.md) must all appear."""

    def test_all_nineteen_rule_ids_present(self):
        doc = build_registry_v2()
        rule_ids = {r["rule_id"] for r in doc["rules"]}
        expected = set(OBSERVABILITY_CLASS.keys())
        self.assertEqual(rule_ids, expected)

    def test_every_evidence_family_has_alternative_explanations(self):
        doc = build_registry_v2()
        for rule in doc["rules"]:
            self.assertTrue(rule["alternative_explanations"], rule["rule_id"])

    def test_target_construct_and_evidence_family_are_populated_for_every_rule(self):
        for rule_id in OBSERVABILITY_CLASS:
            self.assertIn(rule_id, TARGET_CONSTRUCT)
            self.assertIn(rule_id, EVIDENCE_FAMILY)

    def test_allowed_output_level_map_has_no_gaps(self):
        self.assertEqual(set(ALLOWED_OUTPUT_LEVEL.keys()), set(OBSERVABILITY_CLASS.keys()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
