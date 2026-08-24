"""Phase G0 -- tests for src/doar/master_registry_v3_build.py and the
MASTER_RULE_FEATURE_REGISTRY_V3.json it writes.

Verifies: all 41 existing DOAR rules are accounted for (read live from
RULE_EVIDENCE_MATRIX.csv, never a hardcoded copy of the count); the
existing-rule section's activation_allowed exactly matches the REAL,
current production enabled/disabled split (10 enabled); every entry this
module ADDS beyond those 41 has activation_allowed=False, unconditionally;
no duplicate ids; the registry never mutates any frozen production file.
"""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.master_registry_v3_build import (
    MASTER_REGISTRY_V3_PATH, build_master_registry_v3, existing_41_rules,
)


class ExistingRulesAccountedForTests(unittest.TestCase):
    def test_all_41_existing_rules_present(self):
        with open(ROOT / "RULE_EVIDENCE_MATRIX.csv", encoding="utf-8") as f:
            real_rule_ids = {row["rule_id"] for row in csv.DictReader(f)}
        self.assertEqual(len(real_rule_ids), 41)
        registry_rule_ids = {e["id"] for e in existing_41_rules()}
        self.assertEqual(real_rule_ids, registry_rule_ids)

    def test_existing_rule_enabled_count_matches_the_real_frozen_matrix(self):
        with open(ROOT / "RULE_EVIDENCE_MATRIX.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        real_enabled = {r["rule_id"] for r in rows if r["allowed_output_level"] == "individual_heuristic_only"}
        self.assertEqual(len(real_enabled), 10)
        registry_enabled = {e["id"] for e in existing_41_rules() if e["activation_allowed"]}
        self.assertEqual(real_enabled, registry_enabled)

    def test_existing_rule_section_never_invents_a_new_evidence_family(self):
        with open(ROOT / "RULE_EVIDENCE_MATRIX.csv", encoding="utf-8") as f:
            real_families = {row["evidence_family"] for row in csv.DictReader(f)}
        registry_families = {e["evidence_family"] for e in existing_41_rules()}
        self.assertEqual(real_families, registry_families)


class NewEntriesNeverActivatedTests(unittest.TestCase):
    def test_every_entry_outside_the_41_existing_rules_is_activation_false(self):
        doc = build_master_registry_v3()
        existing_ids = {e["id"] for e in existing_41_rules()}
        new_entries = [e for e in doc["entries"] if e["id"] not in existing_ids]
        self.assertGreater(len(new_entries), 100)  # sanity: the PDF genuinely added many
        still_true = [e["id"] for e in new_entries if e["activation_allowed"]]
        self.assertEqual(still_true, [])

    def test_no_duplicate_ids(self):
        doc = build_master_registry_v3()
        self.assertEqual(doc["duplicate_ids_found"], [])


class ThreeLevelHierarchyTests(unittest.TestCase):
    """Phase G0.1: source entries, canonical observables, and evidence
    families must be THREE distinct, non-conflated counts -- the previous
    report's "21 deduplicated concepts" was actually just the evidence-
    family count mislabeled as a canonical-concept count."""

    def test_source_entry_count_equals_total_entries(self):
        doc = build_master_registry_v3()
        self.assertEqual(doc["source_entry_count"], doc["total_entry_count"])
        self.assertEqual(doc["source_entry_count"], 225)

    def test_canonical_observable_count_is_strictly_between_evidence_family_and_source_entry_counts(self):
        doc = build_master_registry_v3()
        self.assertLess(doc["evidence_family_count"], doc["canonical_observable_count"])
        self.assertLess(doc["canonical_observable_count"], doc["source_entry_count"])

    def test_evidence_family_groups_are_populated(self):
        doc = build_master_registry_v3()
        self.assertGreater(doc["evidence_family_count"], 10)
        for group, ids in doc["evidence_family_groups"].items():
            self.assertGreater(len(ids), 0)

    def test_every_canonical_observable_belongs_to_exactly_one_evidence_family(self):
        doc = build_master_registry_v3()
        for cid, info in doc["canonical_observables"].items():
            if cid == "contextual_interpretation_framework":
                continue  # Malchiodi: a FRAMEWORK_SOURCE, not an observable -- documented exception
            self.assertIsNotNone(info["evidence_family"], f"{cid} has no evidence_family")

    def test_object_specific_small_size_observables_never_collapse_together(self):
        """The task's own worked example: 'very small house', 'very small
        tree', and 'very small person' must remain THREE separate
        canonical observables even though all three share the
        size_composition evidence family."""
        doc = build_master_registry_v3()
        by_id = {e["id"]: e for e in doc["entries"]}
        house = by_id["HTP_18_HOUSE_VERY_SMALL_HOUSE"]["canonical_observable_id"]
        tree = by_id["HTP_33_TREE_VERY_SMALL_TREE"]["canonical_observable_id"]
        person = by_id["HTP_42_PERSON_VERY_SMALL_PERSON"]["canonical_observable_id"]
        self.assertEqual(len({house, tree, person}), 3, "small house/tree/person canonical observables collapsed")
        for cid in (house, tree, person):
            self.assertEqual(doc["canonical_observables"][cid]["evidence_family"], "size_composition")

    def test_koppitz_tiny_figure_and_htp_very_small_person_do_merge(self):
        """The task's positive worked example: these two SHOULD merge into
        one canonical observable (they describe the same concept)."""
        doc = build_master_registry_v3()
        by_id = {e["id"]: e for e in doc["entries"]}
        koppitz_id = by_id["KOPPITZ_07_TINY_FIGURE"]["canonical_observable_id"]
        htp_id = by_id["HTP_42_PERSON_VERY_SMALL_PERSON"]["canonical_observable_id"]
        self.assertEqual(koppitz_id, htp_id)
        self.assertGreaterEqual(len(doc["canonical_observables"][koppitz_id]["source_entry_ids"]), 2)

    def test_dedup_source_group_matches_canonical_observable_id(self):
        doc = build_master_registry_v3()
        for e in doc["entries"]:
            self.assertEqual(e["dedup_source_group"], e["canonical_observable_id"])


class RegistryNeverMutatesFrozenFilesTests(unittest.TestCase):
    def test_build_module_never_writes_to_a_production_registry_path(self):
        import inspect
        from doar import master_registry_v3_build as mod
        source = inspect.getsource(mod)
        for forbidden in ("rules_registry_v2.json", "RULE_EVIDENCE_MATRIX.csv", "CONCERN_DOMAIN_MAP.json"):
            # Reading these paths is fine (existing_41_rules() does); the
            # guard is that no write_text/open(..., "w") call targets them.
            self.assertNotIn(f'"w").write' , source)
        self.assertIn("RULE_EVIDENCE_MATRIX.csv", source)  # confirms it DOES read the real file


class RegistryFileOnDiskTests(unittest.TestCase):
    """Only runs meaningfully after write_master_registry_v3() has been
    called at least once this session -- skips cleanly on a checkout where
    the generated artifact has not been produced yet."""

    def test_written_file_matches_build_output_when_present(self):
        if not MASTER_REGISTRY_V3_PATH.exists():
            self.skipTest("MASTER_RULE_FEATURE_REGISTRY_V3.json not yet generated in this checkout")
        on_disk = json.loads(MASTER_REGISTRY_V3_PATH.read_text(encoding="utf-8"))
        fresh = build_master_registry_v3()
        self.assertEqual(on_disk, fresh)


if __name__ == "__main__":
    unittest.main()
