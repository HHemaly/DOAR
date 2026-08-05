"""Tests for DOAR-TRACE Phase 2A.2: Parent-View clarity, capability
transparency, and minimal page-reference user controls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.registry_v2_build import build_registry_v2
from doar.rule_engine_v2 import V2_RULE_IDS


class NaturalQuestionWordingTests(unittest.TestCase):
    """Section 8: every executable rule's question_template must be a
    natural, open-ended question -- never the mechanically-generated
    "ask about the <raw observable name>" pattern."""

    _FORBIDDEN_PATTERNS = ("ask your child about the", "light line pressure appearance",
                            "heavy line pressure appearance", "coverage about half",
                            "coverage full", "coverage small", "placement top", "placement left",
                            "placement right", "placement center", "shaky or broken lines")

    def setUp(self):
        self.rules_by_id = {r["rule_id"]: r for r in build_registry_v2()["rules"]}
        self.executable_ids = {
            rid for rid, r in self.rules_by_id.items() if r["allowed_output_level"] == "individual_heuristic_only"
        }

    def test_ten_executable_rules_exist(self):
        self.assertEqual(len(self.executable_ids), 10)

    def test_no_executable_rule_uses_the_mechanical_ask_about_pattern(self):
        for rule_id in self.executable_ids:
            question = self.rules_by_id[rule_id]["question_template"].lower()
            for forbidden in self._FORBIDDEN_PATTERNS:
                self.assertNotIn(forbidden, question, f"{rule_id}: {question!r}")

    def test_every_executable_question_is_a_real_question(self):
        for rule_id in self.executable_ids:
            question = self.rules_by_id[rule_id]["question_template"]
            self.assertTrue(question.strip().endswith("?"), rule_id)
            self.assertGreater(len(question), 15, rule_id)

    def test_line_proxy_rules_never_say_pressure_without_a_physical_pressure_hedge(self):
        # Whenever the word "pressure" appears, the same text must also
        # explicitly disclaim that it is not an actual physical pressure
        # measurement -- never a bare, unhedged "pressure" claim.
        for rule_id in ("EN_COMPILED_LINE_HEAVY_PRESSURE_030", "EN_COMPILED_LINE_LIGHT_PRESSURE_031"):
            rule = self.rules_by_id[rule_id]
            for field in ("parent_safe_wording", "professional_wording"):
                text = rule[field].lower()
                if "pressure" in text:
                    self.assertIn("physical", text, f"{rule_id}.{field}: {text!r}")
                    self.assertTrue(
                        "does not measure" in text or "not a measurement" in text,
                        f"{rule_id}.{field}: {text!r}",
                    )
            # question_template must never mention "pressure" at all --
            # only the appearance-based phrasing ("light or thin-line
            # appearance"), per the task's explicit instruction.
            self.assertNotIn("pressure", rule["question_template"].lower(), rule_id)

    def test_disabled_rules_are_unaffected(self):
        # The mechanical fallback still exists for the 31 disabled rules
        # (they never reach the Parent view) -- this is a deliberate scope
        # boundary, not an oversight.
        disabled_sample = next(
            r for rid, r in self.rules_by_id.items()
            if r["allowed_output_level"] != "individual_heuristic_only" and rid not in V2_RULE_IDS
        )
        self.assertIn("Would you be willing to ask your child about", disabled_sample["question_template"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
