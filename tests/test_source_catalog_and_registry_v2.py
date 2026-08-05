"""Tests for source_catalog_build.py, construct_registry_build.py, and the
Phase 1.5 rewrite of registry_v2_build.py."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.construct_registry_build import CONSTRUCT_REGISTRY_PATH, build_construct_registry
from doar.registry_v2_build import REGISTRY_V2_PATH, build_registry_v2
from doar.source_catalog_build import AR_PDF, CATALOG_PATH, EN_PDF, build_source_catalog

try:
    import pypdf  # noqa: F401
    _PYPDF = True
except Exception:
    _PYPDF = False

AR_PDF_PATH = ROOT / AR_PDF
EN_PDF_PATH = ROOT / "resources" / "psychology_sources" / EN_PDF

ORIGINAL_19_RULE_IDS = {
    "PSY_AR_EYES_WIDE_001", "PSY_AR_EYES_STERN_002", "PSY_AR_EYES_CLOSED_003",
    "PSY_AR_ANIMAL_TIGER_WOLF_004", "PSY_AR_ANIMAL_FOX_005", "PSY_AR_ANIMAL_SQUIRREL_006",
    "PSY_AR_ANIMAL_LION_007", "PSY_AR_GEOMETRY_008", "PSY_AR_STARS_009",
    "PSY_AR_FLOWERS_CLOUDS_SUN_010", "PSY_AR_CIRCLES_011", "PSY_AR_TRANSPORT_012",
    "PSY_AR_HEARTS_013", "PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_FULL_015",
    "PSY_AR_SIZE_SMALL_016", "PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018",
    "PSY_AR_PLACE_RIGHT_019",
}


class SourcePDFPresenceTests(unittest.TestCase):
    """No silent PDF substitution: both real files must exist and be read."""

    def test_arabic_pdf_exists(self):
        self.assertTrue(AR_PDF_PATH.exists(), AR_PDF_PATH)

    def test_compiled_english_pdf_exists_at_the_required_path(self):
        self.assertTrue(EN_PDF_PATH.exists(), EN_PDF_PATH)

    @unittest.skipUnless(_PYPDF, "pypdf not installed")
    def test_compiled_pdf_has_three_pages(self):
        reader = pypdf.PdfReader(str(EN_PDF_PATH))
        self.assertEqual(len(reader.pages), 3)

    @unittest.skipUnless(_PYPDF, "pypdf not installed")
    def test_arabic_pdf_has_two_pages(self):
        reader = pypdf.PdfReader(str(AR_PDF_PATH))
        self.assertEqual(len(reader.pages), 2)


class SourceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = build_source_catalog()

    def test_committed_file_matches_freshly_built_document(self):
        on_disk = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, self.catalog)

    def test_both_documents_represented(self):
        docs = {e["source_document"] for e in self.catalog["entries"]}
        self.assertEqual(docs, {AR_PDF, EN_PDF})

    def test_nineteen_arabic_rows(self):
        ar_entries = [e for e in self.catalog["entries"] if e["source_document"] == AR_PDF]
        self.assertEqual(len(ar_entries), 19)

    def test_approximately_43_english_candidate_rule_rows(self):
        en_candidate = [
            e for e in self.catalog["entries"]
            if e["source_document"] == EN_PDF and e["source_type"] != "general_guidance"
        ]
        self.assertEqual(len(en_candidate), 43)

    def test_every_entry_has_stable_source_page_traceability(self):
        for entry in self.catalog["entries"]:
            self.assertIn(entry["source_page"], (1, 2, 3), entry["source_entry_id"])
            if entry["source_document"] == AR_PDF:
                self.assertIn(entry["source_page"], (1, 2))
            else:
                self.assertIn(entry["source_page"], (1, 2, 3))

    def test_every_entry_has_original_text(self):
        for entry in self.catalog["entries"]:
            self.assertTrue(entry["original_text"], entry["source_entry_id"])

    def test_duplicate_relationships_are_explicit_and_reference_real_entries(self):
        known_ids = {e["source_entry_id"] for e in self.catalog["entries"]}
        self.assertGreater(len(self.catalog["relationships"]), 0)
        for rel in self.catalog["relationships"]:
            self.assertIn(rel["type"], {"exact_duplicate", "near_duplicate", "expands", "contradicts", "related_but_distinct"})
            for sid in rel["source_entry_ids"]:
                self.assertIn(sid, known_ids, f"relationship references unknown source_entry_id {sid!r}")

    def test_internal_english_duplicates_are_flagged_not_silently_merged(self):
        # The English PDF itself restates "repeated monsters/danger" across
        # two sections -- both rows must exist as separate catalog entries
        # AND be linked by an explicit relationship, never merged into one row.
        ids = {e["source_entry_id"] for e in self.catalog["entries"]}
        self.assertIn("SRC_EN_019", ids)
        self.assertIn("SRC_EN_037", ids)
        pair_flagged = any(
            set(rel["source_entry_ids"]) == {"SRC_EN_019", "SRC_EN_037"} for rel in self.catalog["relationships"]
        )
        self.assertTrue(pair_flagged)


class ConstructRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_construct_registry()

    def test_committed_file_matches_freshly_built_document(self):
        on_disk = json.loads(CONSTRUCT_REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, self.registry)

    def test_twelve_constructs(self):
        self.assertEqual(self.registry["construct_count"], 12)
        self.assertEqual(len(self.registry["constructs"]), 12)

    def test_every_construct_has_the_required_fields(self):
        required = {
            "construct_id", "display_name_en", "display_name_ar", "description",
            "allowed_wording", "prohibited_wording", "minimum_independent_evidence_families",
            "minimum_evidence_grade_requirement", "possible_supporting_families",
            "possible_contradicting_families", "limitations", "contextual_questions",
            "professional_review_recommendation",
        }
        for construct in self.registry["constructs"]:
            self.assertEqual(set(construct.keys()), required, construct["construct_id"])

    def test_minimum_independent_evidence_families_is_at_least_two(self):
        for construct in self.registry["constructs"]:
            self.assertGreaterEqual(construct["minimum_independent_evidence_families"], 2)


class RegistryV2PhaseOneFiveTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry_v2()
        self.rules_by_id = {r["rule_id"]: r for r in self.registry["rules"]}

    def test_committed_file_matches_freshly_built_document(self):
        on_disk = json.loads(REGISTRY_V2_PATH.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, self.registry)

    def test_original_nineteen_ids_are_stable(self):
        self.assertTrue(ORIGINAL_19_RULE_IDS <= set(self.rules_by_id.keys()))

    def test_forty_one_total_rules(self):
        self.assertEqual(self.registry["rule_count"], 41)
        self.assertEqual(self.registry["rule_count_original_production"], 19)
        self.assertEqual(self.registry["rule_count_new_from_compiled_pdf"], 22)

    def test_new_rules_use_a_distinct_id_prefix(self):
        new_rules = [rid for rid in self.rules_by_id if rid not in ORIGINAL_19_RULE_IDS]
        self.assertEqual(len(new_rules), 22)
        for rid in new_rules:
            self.assertTrue(rid.startswith("EN_COMPILED_"), rid)

    def test_no_fabricated_references(self):
        known_refs = set(self.registry["references"].keys())
        for rule in self.registry["rules"]:
            for ref in rule["reference_ids"]:
                self.assertIn(ref, known_refs, f"{rule['rule_id']} cites unknown reference {ref!r}")

    def test_only_ten_rules_are_individually_executable(self):
        # Phase 1.5: the original 6 tier-1 composition/placement rules.
        # Phase 2A, Section 7: 4 more rules wired via rule_engine_v2.py
        # (see registry_v2_build.py module docstring and
        # docs/STATIC_PROXY_RULE_POLICY.md). Every other rule remains
        # `disabled` -- this test must be updated deliberately, never
        # silently, whenever a rule is newly activated.
        executable = {rid for rid, r in self.rules_by_id.items() if r["allowed_output_level"] == "individual_heuristic_only"}
        self.assertEqual(executable, {
            "PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_FULL_015", "PSY_AR_SIZE_SMALL_016",
            "PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019",
            "EN_COMPILED_PLACEMENT_CENTER_029", "EN_COMPILED_LINE_HEAVY_PRESSURE_030",
            "EN_COMPILED_LINE_LIGHT_PRESSURE_031", "EN_COMPILED_LINE_SHAKY_BROKEN_032",
        })

    def test_all_construct_mappings_point_to_valid_constructs(self):
        construct_ids = {c["construct_id"] for c in build_construct_registry()["constructs"]}
        for rule in self.registry["rules"]:
            if rule["target_construct"] is not None:
                self.assertIn(rule["target_construct"], construct_ids, rule["rule_id"])

    def test_longitudinal_required_rules_are_correctly_classified(self):
        for rid in ("EN_COMPILED_REPEATED_MONSTERS_DANGER_025", "EN_COMPILED_REPEATED_FAMILY_CONFLICT_027",
                    "EN_COMPILED_REPEATED_ISOLATION_THEMES_028"):
            self.assertEqual(self.rules_by_id[rid]["observability_class"], "longitudinal_required")

    def test_process_required_rules_are_correctly_classified(self):
        for rid in ("PSY_AR_FLOWERS_CLOUDS_SUN_010", "EN_COMPILED_LINE_OVER_ERASING_034", "EN_COMPILED_REFUSAL_TO_DRAW_039"):
            self.assertEqual(self.rules_by_id[rid]["observability_class"], "process_required")

    def test_static_proxy_rules_never_claim_actual_pencil_pressure(self):
        for rid in ("EN_COMPILED_LINE_HEAVY_PRESSURE_030", "EN_COMPILED_LINE_LIGHT_PRESSURE_031"):
            rule = self.rules_by_id[rid]
            self.assertEqual(rule["observability_class"], "static_proxy")
            self.assertNotIn("actual pencil pressure", rule["possible_interpretation"])
            self.assertIn("pressure_appearance", rule["observable"])

    def test_speculative_rules_are_retained_not_discarded(self):
        for rid in ("PSY_AR_ANIMAL_LION_007", "PSY_AR_ANIMAL_FOX_005", "PSY_AR_STARS_009", "PSY_AR_CIRCLES_011"):
            self.assertIn(rid, self.rules_by_id)

    def test_no_single_rule_is_combined_hypothesis_only(self):
        # combined_hypothesis_only is reserved for the multi-rule aggregator
        # (Section 6), never assigned directly to a single registry rule.
        for rule in self.registry["rules"]:
            self.assertNotEqual(rule["allowed_output_level"], "combined_hypothesis_only")

    def test_static_direct_rules_have_no_required_detector(self):
        for rule in self.registry["rules"]:
            if rule["observability_class"] == "static_direct" and rule["allowed_output_level"] == "individual_heuristic_only":
                self.assertIsNone(rule["required_detector_or_metadata"])

    def test_non_executable_rules_name_a_required_detector_or_context(self):
        for rule in self.registry["rules"]:
            if rule["allowed_output_level"] == "disabled":
                self.assertIsNotNone(rule["required_detector_or_metadata"], rule["rule_id"])

    def test_no_double_counting_every_rule_dependency_group_has_unique_features(self):
        for rule in self.registry["rules"]:
            self.assertEqual(len(rule["dependency_group"]), len(set(rule["dependency_group"])), rule["rule_id"])

    def test_every_rule_has_alternative_explanations(self):
        for rule in self.registry["rules"]:
            self.assertTrue(rule["alternative_explanations"], rule["rule_id"])

    def test_every_rule_required_field_set(self):
        required = {
            "rule_id", "registry_v2_status", "source_entry_ids", "source_document", "source_page",
            "source_section", "faithful_source_quote", "observable", "possible_interpretation",
            "evidence_level_as_written", "confidence_ceiling", "scientific_support", "observability_class",
            "required_feature_ids", "required_detector_or_metadata", "evidence_family", "dependency_group",
            "target_construct", "direction", "allowed_output_level", "alternative_explanations", "reference_ids",
            "limitations", "parent_safe_wording", "professional_wording", "question_template",
            "psychologist_review_status", "validation_status", "threshold_source",
            "page_reference_requirement", "feature_version", "known_robustness_limitations",
            "expert_review_status", "version",
        }
        for rule in self.registry["rules"]:
            self.assertEqual(set(rule.keys()), required, rule["rule_id"])

    def test_confidence_ceiling_and_scientific_support_present_for_executable_rules(self):
        # Phase 2A: confidence_ceiling/scientific_support are reused from
        # rules_registry.json for the 6 already-in-production rules and
        # explicitly, conservatively set for the 4 newly-activated
        # EN_COMPILED_* rules -- but are allowed to be None for every other
        # (still-disabled) rule, since only executable rules are ever run
        # through rule_engine_v2.py's _base_eval, which requires them.
        executable_ids = {rid for rid, r in self.rules_by_id.items() if r["allowed_output_level"] == "individual_heuristic_only"}
        for rule_id in executable_ids:
            rule = self.rules_by_id[rule_id]
            self.assertIsNotNone(rule["confidence_ceiling"], rule_id)
            self.assertIsNotNone(rule["scientific_support"], rule_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
