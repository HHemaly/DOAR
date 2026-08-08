"""Rule visual-coverage classification tests."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c4 import rule_coverage


class RuleCoverageCompletenessTests(unittest.TestCase):
    def test_covers_exactly_the_41_registry_rules(self):
        registry = json.loads(
            (ROOT / "resources/psychology_sources/rules_registry_v2.json").read_text(encoding="utf-8"))
        registry_ids = {r["rule_id"] for r in registry["rules"]}
        my_ids = {r["rule_id"] for r in rule_coverage.RULE_COVERAGE}
        self.assertEqual(registry_ids, my_ids)
        self.assertEqual(len(rule_coverage.RULE_COVERAGE), 41)

    def test_no_duplicate_rule_ids(self):
        ids = [r["rule_id"] for r in rule_coverage.RULE_COVERAGE]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_classification_is_valid(self):
        for row in rule_coverage.RULE_COVERAGE:
            self.assertIn(row["primary_classification"], rule_coverage.PRIMARY_CLASSIFICATIONS)

    def test_every_row_has_a_nonempty_rationale(self):
        for row in rule_coverage.RULE_COVERAGE:
            self.assertTrue(row["rationale"].strip(), row["rule_id"])


class ClassificationConsistencyTests(unittest.TestCase):
    """Spot-checks that specific, known rules land in the classification
    their observability_class/registry facts actually justify -- catches an
    accidental miscategorization more precisely than the completeness
    checks above."""

    def _by_id(self, rule_id):
        return next(r for r in rule_coverage.RULE_COVERAGE if r["rule_id"] == rule_id)

    def test_already_measurable_static_direct_rule(self):
        self.assertEqual(
            self._by_id("PSY_AR_SIZE_HALF_014")["primary_classification"],
            "already_measurable_by_objective_features")

    def test_longitudinal_rule(self):
        self.assertEqual(
            self._by_id("EN_COMPILED_REPEATED_MONSTERS_DANGER_025")["primary_classification"],
            "requires_longitudinal_information")

    def test_not_operational_rule(self):
        self.assertEqual(
            self._by_id("EN_COMPILED_VERY_SMALL_DRAWING_026")["primary_classification"],
            "remains_not_operational")

    def test_process_required_rule(self):
        self.assertEqual(
            self._by_id("PSY_AR_FLOWERS_CLOUDS_SUN_010")["primary_classification"],
            "requires_process_information")

    def test_directly_benchmarked_presence_rule(self):
        self.assertEqual(
            self._by_id("PSY_AR_HEARTS_013")["primary_classification"],
            "potentially_measurable_by_benchmarked_detector")

    def test_species_attribute_rule_never_marked_directly_measurable(self):
        row = self._by_id("PSY_AR_ANIMAL_TIGER_WOLF_004")
        self.assertNotEqual(row["primary_classification"], "potentially_measurable_by_benchmarked_detector")


class WriteCsvTests(unittest.TestCase):
    def test_writes_41_data_rows(self):
        with tempfile.TemporaryDirectory() as d:
            out = rule_coverage.write_csv(Path(d) / "coverage.csv")
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
            self.assertEqual(len(rows), 41)

    def test_deterministic_across_repeated_writes(self):
        with tempfile.TemporaryDirectory() as d:
            out1 = rule_coverage.write_csv(Path(d) / "a.csv")
            out2 = rule_coverage.write_csv(Path(d) / "b.csv")
            self.assertEqual(out1.read_text(encoding="utf-8"), out2.read_text(encoding="utf-8"))


class SummaryCountsTests(unittest.TestCase):
    def test_counts_sum_to_41(self):
        counts = rule_coverage.summary_counts()
        self.assertEqual(sum(counts.values()), 41)

    def test_every_bucket_key_is_a_valid_classification(self):
        counts = rule_coverage.summary_counts()
        self.assertEqual(set(counts.keys()), rule_coverage.PRIMARY_CLASSIFICATIONS)


if __name__ == "__main__":
    unittest.main()
