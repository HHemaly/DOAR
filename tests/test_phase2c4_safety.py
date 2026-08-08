"""Safety/scope regression tests for Phase 2C.4, mirroring
tests/test_phase2c1_safety.py and tests/test_phase2c2_safety.py: no
diagnostic language, no rule-engine import, no emotion-label exposure, no
threshold-tuning code path, and ground truth always genuine-human only."""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC

_FORBIDDEN_RULE_ENGINE_MODULES = {"rule_engine_v2", "evidence_rule_engine", "rules", "structured_report"}
_EMOTION_LABELS = {"angry", "fear", "happy", "sad"}
_PHASE2C4_SRC_DIR = ROOT / "src" / "doar" / "phase2c4"


def _has_diagnostic_language(text: str) -> bool:
    if _ARABIC_DIAGNOSTIC.search(text):
        return True
    return any(p.search(text) for p in DIAGNOSTIC_PATTERNS)


class NoDiagnosticLanguageTests(unittest.TestCase):
    def test_no_module_source_contains_diagnostic_language(self):
        for py_file in sorted(_PHASE2C4_SRC_DIR.glob("*.py")):
            text = py_file.read_text(encoding="utf-8")
            self.assertFalse(_has_diagnostic_language(text), py_file.name)


class NoRuleEngineDependencyTests(unittest.TestCase):
    def test_no_phase2c4_module_imports_the_rule_engine(self):
        for py_file in sorted(_PHASE2C4_SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_names.add(node.module.split(".")[-1])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.name.split(".")[-1])
            forbidden = imported_names & _FORBIDDEN_RULE_ENGINE_MODULES
            self.assertFalse(forbidden, f"{py_file.name} imports rule-engine module(s): {forbidden}")


class NoEmotionLabelExposureTests(unittest.TestCase):
    def test_no_source_file_references_emotion_class_names(self):
        for py_file in list(_PHASE2C4_SRC_DIR.glob("*.py")) + [ROOT / "scripts/phase2c4_run_detector.py"]:
            text = py_file.read_text(encoding="utf-8").lower()
            for label in _EMOTION_LABELS:
                self.assertIsNone(
                    re.search(rf"\b{label}\b", text),
                    f"{py_file.name} must never reference the emotion label {label!r}")


class GroundTruthAlwaysGenuineHumanTests(unittest.TestCase):
    def test_evaluation_module_uses_genuine_human_ground_truth_only(self):
        text = (_PHASE2C4_SRC_DIR / "evaluation.py").read_text(encoding="utf-8")
        self.assertIn("genuine_human_ground_truth", text)
        self.assertNotIn("legacy_provisional_human", text)


class NoThresholdTuningTests(unittest.TestCase):
    """Structural guard: nothing in phase2c4 may search over candidate
    thresholds -- every detector's operating point is a fixed, documented
    constant, never derived from these results."""

    def test_no_threshold_sweep_in_evaluation_module(self):
        text = (_PHASE2C4_SRC_DIR / "evaluation.py").read_text(encoding="utf-8")
        for forbidden_token in ("for threshold in", "np.arange", "threshold_sensitivity", "argmax"):
            self.assertNotIn(forbidden_token, text)

    def test_detectors_module_thresholds_are_named_constants_in_signatures(self):
        """Every load_real_* function takes its threshold as a keyword
        argument with a literal default -- never computed from data."""
        tree = ast.parse((_PHASE2C4_SRC_DIR / "detectors.py").read_text(encoding="utf-8"))
        checked = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("load_real_"):
                checked += 1
                defaults = node.args.kw_defaults
                for d in defaults:
                    if d is not None:
                        self.assertIsInstance(d, (ast.Constant, ast.UnaryOp),
                                               f"{node.name}'s default {ast.dump(d)} is not a literal constant")
        self.assertEqual(checked, 4, "expected exactly 4 load_real_* loaders")


class CohortProtectionTests(unittest.TestCase):
    def test_locked_test_cohort_name_present_in_evaluation_module(self):
        text = (_PHASE2C4_SRC_DIR / "evaluation.py").read_text(encoding="utf-8")
        self.assertIn("LOCKED_TEST", text)


if __name__ == "__main__":
    unittest.main()
