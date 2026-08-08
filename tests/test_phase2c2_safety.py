"""Safety/scope regression tests for Phase 2C.2, mirroring
tests/test_phase2c1_safety.py: no diagnostic language, no import path into
the rule-evaluation engine, and no threshold-tuning code path (the module
must only ever read the fixed detect_threshold/uncertain_margin's already-
computed __status field, never search for a better threshold)."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC
from doar.phase2c2 import cohorts, evaluation, ground_truth

_FORBIDDEN_RULE_ENGINE_MODULES = {"rule_engine_v2", "evidence_rule_engine", "rules", "structured_report"}
_PHASE2C2_SRC_DIR = ROOT / "src" / "doar" / "phase2c2"
_PHASE2C2_MODULES = [cohorts, evaluation, ground_truth]


def _has_diagnostic_language(text: str) -> bool:
    if _ARABIC_DIAGNOSTIC.search(text):
        return True
    return any(p.search(text) for p in DIAGNOSTIC_PATTERNS)


class NoDiagnosticLanguageTests(unittest.TestCase):
    def test_no_module_source_contains_diagnostic_language(self):
        for py_file in sorted(_PHASE2C2_SRC_DIR.glob("*.py")):
            text = py_file.read_text(encoding="utf-8")
            self.assertFalse(_has_diagnostic_language(text), py_file.name)


class NoRuleEngineDependencyTests(unittest.TestCase):
    def test_no_phase2c2_module_imports_the_rule_engine(self):
        for py_file in sorted(_PHASE2C2_SRC_DIR.glob("*.py")):
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


class NoThresholdTuningTests(unittest.TestCase):
    """Structural guard: this module must never contain code that searches
    over candidate thresholds (that would be threshold tuning, explicitly
    out of scope for Phase 2C.2). It may only ever read the fixed
    detect_threshold/uncertain_margin's already-computed __status output."""

    def test_no_threshold_search_loop_in_evaluation_module(self):
        text = (_PHASE2C2_SRC_DIR / "evaluation.py").read_text(encoding="utf-8")
        for forbidden_token in ("threshold_sensitivity", "detect_threshold=", "np.arange", "for t in"):
            self.assertNotIn(forbidden_token, text)

    def test_predicted_positive_reads_existing_status_only(self):
        """predicted_positive_from_clip must derive purely from the
        '__status' field already produced upstream by the fixed
        threshold -- never recompute against a raw similarity value."""
        import inspect
        source = inspect.getsource(evaluation.predicted_positive_from_clip)
        self.assertIn("__status", source)
        self.assertNotIn(">=", source)
        self.assertNotIn("<=", source)


class GroundTruthNeverProvisionalTests(unittest.TestCase):
    def test_ground_truth_module_only_reads_human_type(self):
        text = (_PHASE2C2_SRC_DIR / "ground_truth.py").read_text(encoding="utf-8")
        self.assertIn('annotator_type != "human"', text)
        self.assertNotIn("legacy_provisional_human ==", text)


if __name__ == "__main__":
    unittest.main()
