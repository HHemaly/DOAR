"""Phase 2C.5 Stage 1 traceability tests -- verifies every referenced
rule_id/observable actually exists in the live rules_registry_v2.json,
and that every ontology.PART_TARGETS entry traces to at least one rule."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5.ontology import PART_TARGETS
from doar.phase2c5.rule_traceability import RULE_ANNOTATION_TRACEABILITY, to_rows

REGISTRY_PATH = ROOT / "resources/psychology_sources/rules_registry_v2.json"


def _load_registry_rule_ids() -> set[str]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {r["rule_id"] for r in data["rules"]}


class RuleIdsExistInLiveRegistryTests(unittest.TestCase):
    def test_every_referenced_rule_id_exists_in_registry(self):
        registry_ids = _load_registry_rule_ids()
        for target in RULE_ANNOTATION_TRACEABILITY:
            for rule_id in target.supported_rule_ids:
                self.assertIn(rule_id, registry_ids,
                              f"{target.target_name} references unknown rule_id {rule_id!r}")

    def test_all_9_expected_rule_ids_covered_somewhere(self):
        expected = {
            "PSY_AR_EYES_WIDE_001", "PSY_AR_EYES_STERN_002", "PSY_AR_EYES_CLOSED_003",
            "EN_COMPILED_EYES_MISSING_DETAIL_020", "EN_COMPILED_FACE_EXPRESSION_021",
            "EN_COMPILED_MISSING_HANDS_035", "EN_COMPILED_MISSING_MOUTH_036",
            "EN_COMPILED_EXAGGERATED_BODY_PARTS_037", "EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038",
        }
        covered = {rid for t in RULE_ANNOTATION_TRACEABILITY for rid in t.supported_rule_ids}
        self.assertEqual(covered, expected)


class AllowedOutputLevelStillDisabledTests(unittest.TestCase):
    def test_every_referenced_rule_is_still_allowed_output_level_disabled(self):
        """This phase must not have coincided with (or been preceded by) any
        change that activated these rules -- if this ever fails, a rule the
        traceability table assumed was disabled has been enabled somewhere
        else, and the report's 'nothing was activated' claim would be stale."""
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        by_id = {r["rule_id"]: r for r in data["rules"]}
        referenced = {rid for t in RULE_ANNOTATION_TRACEABILITY for rid in t.supported_rule_ids}
        for rule_id in referenced:
            self.assertEqual(by_id[rule_id]["allowed_output_level"], "disabled",
                              f"{rule_id} is no longer disabled")


class PartTargetsTraceToARuleTests(unittest.TestCase):
    def test_every_part_target_appears_in_traceability_table(self):
        traced_targets = {t.target_name for t in RULE_ANNOTATION_TRACEABILITY}
        for target in PART_TARGETS:
            self.assertIn(target, traced_targets)


class ToRowsTests(unittest.TestCase):
    def test_to_rows_flattens_rule_ids_to_semicolon_string(self):
        rows = to_rows()
        eye_row = next(r for r in rows if r["target_name"] == "eye")
        self.assertIn("PSY_AR_EYES_CLOSED_003", eye_row["supported_rule_ids"])
        self.assertIsInstance(eye_row["supported_rule_ids"], str)


if __name__ == "__main__":
    unittest.main()
