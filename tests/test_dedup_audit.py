"""Phase G0.2 -- tests for src/doar/dedup_audit_build.py and
MASTER_RULE_DEDUP_AUDIT.csv, plus Step 12's registry validation checks
(scope/task/readiness classification, object-specificity preservation,
formal-feature-id uniqueness across evidence families)."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.dedup_audit_build import MASTER_DEDUP_AUDIT_PATH, build_dedup_audit
from doar.master_registry_v3_build import (
    PRIMARY_SCOPE_VALUES, TASK_REQUIREMENT_VALUES, TECHNICAL_READINESS_VALUES,
    build_master_registry_v3,
)


class DedupAuditCoverageTests(unittest.TestCase):
    def test_every_source_entry_has_exactly_one_audit_row(self):
        doc = build_master_registry_v3()
        rows = build_dedup_audit()
        self.assertEqual(len(rows), doc["source_entry_count"])
        self.assertEqual({r["source_entry_id"] for r in rows}, {e["id"] for e in doc["entries"]})

    def test_every_row_has_a_valid_review_status(self):
        valid = {"MERGED", "KEPT_DISTINCT", "SOURCE_SPECIFIC", "FRAMEWORK_ONLY",
                 "PROCESS_ONLY", "LONGITUDINAL_ONLY", "ETHICS_REVIEW"}
        for row in build_dedup_audit():
            self.assertIn(row["review_status"], valid, row["source_entry_id"])

    def test_merged_rows_have_a_merge_reason_and_partner_ids(self):
        for row in build_dedup_audit():
            if row["review_status"] == "MERGED":
                self.assertTrue(row["merge_reason"].strip(), row["source_entry_id"])
                self.assertTrue(row["merged_with_source_ids"].strip(), row["source_entry_id"])

    def test_non_merged_rows_have_a_kept_distinct_reason(self):
        for row in build_dedup_audit():
            if row["review_status"] in ("KEPT_DISTINCT", "SOURCE_SPECIFIC", "FRAMEWORK_ONLY",
                                        "PROCESS_ONLY", "LONGITUDINAL_ONLY", "ETHICS_REVIEW"):
                self.assertTrue(row["kept_distinct_reason"].strip(), row["source_entry_id"])

    def test_file_on_disk_matches_build_output_when_present(self):
        if not MASTER_DEDUP_AUDIT_PATH.exists():
            self.skipTest("MASTER_RULE_DEDUP_AUDIT.csv not yet generated in this checkout")
        with open(MASTER_DEDUP_AUDIT_PATH, encoding="utf-8") as f:
            on_disk = list(csv.DictReader(f))
        self.assertEqual(on_disk, build_dedup_audit())


class Step2WorkedExamplesTests(unittest.TestCase):
    """The task's own explicit worked examples -- both the positive merge
    case and the negative (must-stay-distinct) cases."""

    def _by_id(self):
        return {e["id"]: e for e in build_master_registry_v3()["entries"]}

    def test_koppitz_htp_family_small_person_all_merge(self):
        by_id = self._by_id()
        ids = ("KOPPITZ_07_TINY_FIGURE", "HTP_42_PERSON_VERY_SMALL_PERSON",
               "FAMILY_MARKER_10_UNUSUALLY_SMALL_FIGURES")
        canonical_ids = {by_id[i]["canonical_observable_id"] for i in ids}
        self.assertEqual(len(canonical_ids), 1, canonical_ids)

    def test_house_tree_person_small_size_stay_three_distinct_observables(self):
        by_id = self._by_id()
        house = by_id["HTP_18_HOUSE_VERY_SMALL_HOUSE"]["canonical_observable_id"]
        tree = by_id["HTP_33_TREE_VERY_SMALL_TREE"]["canonical_observable_id"]
        person = by_id["HTP_42_PERSON_VERY_SMALL_PERSON"]["canonical_observable_id"]
        self.assertEqual(len({house, tree, person}), 3)

    def test_line_quality_terms_are_not_automatically_merged(self):
        """weak/intermittent lines, shaky/broken lines, and line jitter are
        NOT collapsed together just because they are all 'about lines'."""
        by_id = self._by_id()
        weak = by_id["HTP_06_WHOLE_WEAK_OR_INTERMITTENT_LINES"]["canonical_observable_id"]
        shaky = by_id["EN_COMPILED_LINE_SHAKY_BROKEN_032"]["canonical_observable_id"]
        jitter_rows = [e for e in build_master_registry_v3()["entries"] if e["canonical_observable"] == "Line jitter/shakiness"]
        self.assertEqual(len(jitter_rows), 1)
        jitter = jitter_rows[0]["canonical_observable_id"]
        self.assertEqual(len({weak, shaky, jitter}), 3, f"weak={weak} shaky={shaky} jitter={jitter}")


class ScopeTaskReadinessValidationTests(unittest.TestCase):
    """Step 12's specific checks."""

    def setUp(self):
        self.doc = build_master_registry_v3()
        self.entries = self.doc["entries"]
        self.by_id = {e["id"]: e for e in self.entries}

    def test_every_entry_has_a_valid_primary_scope(self):
        for e in self.entries:
            self.assertIn(e["primary_scope"], PRIMARY_SCOPE_VALUES, e["id"])

    def test_every_entry_has_a_valid_task_requirement(self):
        for e in self.entries:
            self.assertIn(e["task_requirement"], TASK_REQUIREMENT_VALUES, e["id"])

    def test_every_entry_has_a_valid_technical_readiness(self):
        for e in self.entries:
            self.assertIn(e["technical_readiness"], TECHNICAL_READINESS_VALUES, e["id"])

    def test_scope_is_not_overwhelmingly_whole_drawing(self):
        """Regression guard against the exact problem this phase fixes --
        most entries defaulting to WHOLE_DRAWING with no real
        classification."""
        whole_drawing_count = self.doc["primary_scope_distribution"].get("WHOLE_DRAWING", 0)
        self.assertLess(whole_drawing_count / len(self.entries), 0.5)

    def test_human_figure_omissions_stay_distinct_from_whole_drawing_omission(self):
        missing_eyes = self.by_id["KOPPITZ_23_NO_EYES"]
        whole_scene = self.by_id["HTP_09_WHOLE_OMITTED_HOUSE_TREE_PERSON"]
        self.assertNotEqual(missing_eyes["canonical_observable_id"], whole_scene["canonical_observable_id"])
        self.assertEqual(missing_eyes["primary_scope"], "FACE_REGION")
        self.assertNotEqual(whole_scene["primary_scope"], "FACE_REGION")

    def test_htp_specific_items_carry_the_htp_task_requirement(self):
        for e in self.entries:
            if e["source_family"] == "htp_2023_meta_analysis":
                self.assertEqual(e["task_requirement"], "HTP", e["id"])

    def test_family_drawing_markers_carry_the_family_drawing_task_requirement(self):
        for e in self.entries:
            if e["source_family"] in ("family_drawing_2022_marker", "family_drawing_2022_global_scale"):
                self.assertEqual(e["task_requirement"], "FAMILY_DRAWING", e["id"])

    def test_process_required_entries_are_never_static_image_detectable(self):
        for e in self.entries:
            if e["technical_readiness"] == "PROCESS_DATA_REQUIRED":
                self.assertTrue(e["requires_process_data"], e["id"])
                self.assertEqual(e["primary_scope"], "PROCESS_REQUIRED", e["id"])

    def test_longitudinal_required_entries_are_never_single_image_detectable(self):
        for e in self.entries:
            if e["technical_readiness"] == "LONGITUDINAL_DATA_REQUIRED":
                self.assertTrue(e["requires_longitudinal_data"], e["id"])
                self.assertEqual(e["primary_scope"], "LONGITUDINAL_REQUIRED", e["id"])

    def test_htp_house_specific_items_are_house_region_scope(self):
        for e in self.entries:
            if e["source_family"] == "htp_2023_meta_analysis" and e["id"].split("_")[2] == "HOUSE":
                self.assertEqual(e["primary_scope"], "HOUSE_REGION", e["id"])


class FormalFeatureIdUniquenessTests(unittest.TestCase):
    """Step 7: a formal_features.py measurement must back exactly one
    evidence family, never counted as independent evidence twice."""

    def test_no_implemented_feature_id_spans_multiple_evidence_families(self):
        doc = build_master_registry_v3()
        from collections import defaultdict
        families_by_feature = defaultdict(set)
        for e in doc["entries"]:
            if e["existing_feature_id_if_any"]:
                families_by_feature[e["existing_feature_id_if_any"]].add(e["evidence_family"])
        offenders = {fid: fams for fid, fams in families_by_feature.items() if len(fams) > 1}
        self.assertEqual(offenders, {})


class NoConcernScoreEffectFromNewMetadataTests(unittest.TestCase):
    """Step 12, item 9: the new scope/task/readiness/dedup-audit metadata
    is presentation/planning metadata only -- must never be importable
    from, or affect, the aggregation path."""

    def test_case_interpretation_never_imports_dedup_audit_or_new_metadata(self):
        import inspect
        from doar import case_interpretation as ci
        source = inspect.getsource(ci)
        for forbidden in ("dedup_audit_build", "primary_scope", "technical_readiness", "MASTER_RULE_DEDUP_AUDIT"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
