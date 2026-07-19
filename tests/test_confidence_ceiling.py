"""D10 — confidence_ceiling is now numerically enforced on rule outputs."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class ConfidenceCeilingTests(unittest.TestCase):
    def _eval(self, coverage, placement="unavailable"):
        from doar.rules import evaluate_rules
        composition = {
            "bounding_box_coverage": coverage,
            "placement": placement,
            "foreground_coverage": coverage,
        }
        evaluations, concerns = evaluate_rules(composition, {}, [])
        return evaluations, concerns

    def test_emitted_confidence_never_exceeds_ceiling(self):
        evaluations, _ = self._eval(0.5, placement="top-left")
        self.assertTrue(evaluations)
        for rule in evaluations:
            self.assertIn("rule_confidence", rule)
            self.assertTrue(rule["confidence_ceiling_enforced"])
            self.assertLessEqual(rule["rule_confidence"], rule["confidence_ceiling"])

    def test_non_matches_have_zero_confidence(self):
        evaluations, _ = self._eval(0.5, placement="top-left")
        for rule in evaluations:
            if rule["status"] != "weak_support":
                self.assertEqual(rule["rule_confidence"], 0.0)

    def test_still_no_concerns_from_single_symbol(self):
        _, concerns = self._eval(0.5, placement="top-left")
        self.assertEqual(concerns, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
