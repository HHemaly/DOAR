"""Phase 2C.4A safety/scope regression tests, mirroring
tests/test_phase2c4_safety.py's style: no rule-engine import, no locked-
test leakage into model selection, calibration cannot skip the
locked-test guard, Florence-2's status is never generalized to the whole
architecture, default and calibrated results stay in separate code paths.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c4 import model_status as status_mod

_FORBIDDEN_RULE_ENGINE_MODULES = {"rule_engine_v2", "evidence_rule_engine", "rules", "structured_report"}
_PHASE2C4_SRC_DIR = ROOT / "src" / "doar" / "phase2c4"
_NEW_MODULES = ("macro_metrics.py", "calibration.py", "uncertainty.py", "model_status.py")


class NoRuleEngineDependencyTests(unittest.TestCase):
    def test_no_new_phase2c4a_module_imports_the_rule_engine(self):
        for name in _NEW_MODULES:
            tree = ast.parse((_PHASE2C4_SRC_DIR / name).read_text(encoding="utf-8"), filename=name)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[-1])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.split(".")[-1])
            forbidden = imported & _FORBIDDEN_RULE_ENGINE_MODULES
            self.assertFalse(forbidden, f"{name} imports rule-engine module(s): {forbidden}")


class ModelSelectionCohortIsolationTests(unittest.TestCase):
    """The orchestration script must build its dev-only ranking without
    ever touching LOCKED_TEST rows -- checked structurally in the
    macro_metrics module (cohort filtering happens inside
    macro_summaries_by_model, called once per desired cohort) and directly
    against a synthetic mixed-cohort row set."""

    def test_macro_summaries_by_model_excludes_other_cohorts(self):
        from doar.phase2c4 import macro_metrics as mm
        rows = [
            {"model": "m", "cohort": "dev_eligible_excl_test", "class_name": "a",
             "n_present": 5, "n_absent": 5, "f1": 0.5, "balanced_accuracy": 0.5},
            {"model": "m", "cohort": "locked_test_descriptive_only", "class_name": "a",
             "n_present": 100, "n_absent": 100, "f1": 0.99, "balanced_accuracy": 0.99},
        ]
        summaries = mm.macro_summaries_by_model(rows, "dev_eligible_excl_test", class_names=("a",))
        self.assertEqual(len(summaries), 1)
        self.assertAlmostEqual(summaries[0].macro_f1, 0.5)  # not the locked-test 0.99 row


class CalibrationCallsLockedTestGuardTests(unittest.TestCase):
    def test_orchestration_script_calls_assert_pilot_ids_exclude_locked_test(self):
        text = (ROOT / "scripts/phase2c4a_run_corrections.py").read_text(encoding="utf-8")
        self.assertIn("assert_pilot_ids_exclude_locked_test", text)
        # the guard must run before freeze_operating_points is ever called
        guard_pos = text.index("assert_pilot_ids_exclude_locked_test(")
        freeze_pos = text.index("freeze_operating_points(")
        self.assertLess(guard_pos, freeze_pos)

    def test_calibration_module_never_reads_locked_test_directly(self):
        text = (_PHASE2C4_SRC_DIR / "calibration.py").read_text(encoding="utf-8")
        self.assertNotIn("LOCKED_TEST", text)
        self.assertNotIn("locked_test", text)


class DefaultVsCalibratedSeparationTests(unittest.TestCase):
    def test_calibration_module_does_no_file_io_of_its_own(self):
        """calibration.py is pure functions over already-loaded dicts --
        it never opens any file itself (the orchestration script owns all
        I/O), so it structurally cannot overwrite the frozen default
        results file no matter what path is passed around it."""
        tree = ast.parse((_PHASE2C4_SRC_DIR / "calibration.py").read_text(encoding="utf-8"))
        open_calls = [n for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"]
        self.assertEqual(open_calls, [])

    def test_orchestration_script_writes_calibration_outputs_to_separate_files(self):
        text = (ROOT / "scripts/phase2c4a_run_corrections.py").read_text(encoding="utf-8")
        self.assertIn("calibration_dev_threshold_sweep.csv", text)
        self.assertIn("calibration_frozen_operating_points.csv", text)
        self.assertIn("calibration_locked_test_descriptive.csv", text)
        # never opened in write mode
        self.assertNotIn('"artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv", "w"', text)


class FlorenceStatusNotGeneralizedTests(unittest.TestCase):
    def test_no_forbidden_generalizing_phrase_in_status_field_values(self):
        """Checks the actual data (status/rationale/future_work/configuration
        field VALUES), not the module source -- the source legitimately
        contains the forbidden phrases once each, inside the
        FORBIDDEN_GENERALIZING_PHRASES tuple that documents what to avoid."""
        for entry in status_mod.MODEL_CONFIGURATION_STATUS:
            combined = " ".join([entry.model, entry.configuration, entry.status,
                                  entry.rationale, entry.future_work]).lower()
            for phrase in status_mod.FORBIDDEN_GENERALIZING_PHRASES:
                self.assertNotIn(phrase, combined, f"{entry.model}: found forbidden phrase {phrase!r}")

    def test_no_forbidden_generalizing_phrase_in_correction_report(self):
        report = (ROOT / "PHASE2C4A_VALIDATION_CORRECTION_REPORT.md").read_text(encoding="utf-8").lower()
        for phrase in status_mod.FORBIDDEN_GENERALIZING_PHRASES:
            self.assertNotIn(phrase, report)

    def test_florence2_status_is_configuration_scoped_not_incapable(self):
        florence_status = next(s for s in status_mod.MODEL_CONFIGURATION_STATUS if s.model == "florence2")
        self.assertEqual(florence_status.status, "failed_inconclusive_for_current_use")
        self.assertIn("configuration", florence_status.rationale.lower())


class NoRuleActivationTests(unittest.TestCase):
    def test_no_new_module_writes_rules_registry(self):
        for name in _NEW_MODULES:
            text = (_PHASE2C4_SRC_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("rules_registry_v2.json", text)
            self.assertNotIn("allowed_output_level", text)


if __name__ == "__main__":
    unittest.main()
