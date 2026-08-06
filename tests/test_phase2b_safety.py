"""Safety/scope regression tests for docs/PHASE2B_OBJECT_EVIDENCE_POLICY.md's
hard constraints: no diagnostic language anywhere in Phase 2B's own
strings, and no import path from src/doar/phase2b into the rule-
evaluation engine (i.e. no detector-dependent rule can be activated by
this phase, structurally, not just by convention)."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC
from doar.phase2b import annotations, dataset, duplicate_groups, evaluation, inference, ontology, schemas

_FORBIDDEN_RULE_ENGINE_MODULES = {"rule_engine_v2", "evidence_rule_engine", "rules", "structured_report"}

_PHASE2B_MODULES = [annotations, dataset, duplicate_groups, evaluation, inference, ontology, schemas]
_PHASE2B_SRC_DIR = ROOT / "src" / "doar" / "phase2b"


def _has_diagnostic_language(text: str) -> bool:
    if _ARABIC_DIAGNOSTIC.search(text):
        return True
    return any(p.search(text) for p in DIAGNOSTIC_PATTERNS)


class NoDiagnosticLanguageTests(unittest.TestCase):
    def test_ontology_class_notes_are_clean(self):
        from doar.phase2b.ontology import CLASSES
        for c in CLASSES:
            self.assertFalse(_has_diagnostic_language(c.notes), c.name)
            self.assertFalse(_has_diagnostic_language(c.minimum_visible_evidence), c.name)

    def test_zero_shot_evidence_limitations_are_clean(self):
        from doar.phase2b.schemas import from_zero_shot_prediction
        rec = from_zero_shot_prediction("p2b_0000", "person", 0.4, "detected",
                                        model_name="ViT-B-32", model_version="openai",
                                        preprocessing_version="v1")
        for line in rec.limitations:
            self.assertFalse(_has_diagnostic_language(line))


class NoRuleEngineDependencyTests(unittest.TestCase):
    """Structural check: no module under src/doar/phase2b/ may import any
    rule-evaluation module, so no detector-dependent rule can be
    activated by this phase even by accident."""

    def test_no_phase2b_module_imports_the_rule_engine(self):
        for py_file in sorted(_PHASE2B_SRC_DIR.glob("*.py")):
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

    def test_technical_view_module_also_avoids_the_rule_engine(self):
        tv_path = _PHASE2B_SRC_DIR / "technical_view.py"
        tree = ast.parse(tv_path.read_text(encoding="utf-8"), filename=str(tv_path))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module.split(".")[-1])
        self.assertFalse(imported_names & _FORBIDDEN_RULE_ENGINE_MODULES)


class DetectorStatusVocabularyTests(unittest.TestCase):
    def test_a_missing_detection_is_never_a_confirmed_negative(self):
        # not_detected is a real, distinct status from "confirmed absent" --
        # this test documents/pins that distinction at the schema level.
        from doar.phase2b.schemas import DETECTOR_STATUSES, NON_CONCLUSIVE_DETECTOR_STATUSES
        self.assertIn("not_detected", DETECTOR_STATUSES)
        self.assertNotIn("confirmed_absent", DETECTOR_STATUSES)
        self.assertIn("uncertain", NON_CONCLUSIVE_DETECTOR_STATUSES)
        self.assertIn("not_assessable", NON_CONCLUSIVE_DETECTOR_STATUSES)


if __name__ == "__main__":
    unittest.main()
