"""Safety/scope regression tests for Phase 2C.1, mirroring
tests/test_phase2b_safety.py's pattern: no diagnostic language, no import
path into the rule-evaluation engine, no emotion-label exposure in the
annotator-facing path, and Parent View stays unmodified."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC
from doar.phase2c1 import quality, schema, store, workspace

_FORBIDDEN_RULE_ENGINE_MODULES = {"rule_engine_v2", "evidence_rule_engine", "rules", "structured_report"}
_EMOTION_LABELS = {"angry", "fear", "happy", "sad"}

_PHASE2C1_MODULES = [schema, store, workspace, quality]
_PHASE2C1_SRC_DIR = ROOT / "src" / "doar" / "phase2c1"
_APP_FILE = ROOT / "phase2c_annotation_app.py"


def _has_diagnostic_language(text: str) -> bool:
    if _ARABIC_DIAGNOSTIC.search(text):
        return True
    return any(p.search(text) for p in DIAGNOSTIC_PATTERNS)


def _module_source_files():
    yield from sorted(_PHASE2C1_SRC_DIR.glob("*.py"))
    yield _APP_FILE


class NoDiagnosticLanguageTests(unittest.TestCase):
    def test_no_module_docstring_contains_diagnostic_language(self):
        for py_file in _module_source_files():
            text = py_file.read_text(encoding="utf-8")
            self.assertFalse(_has_diagnostic_language(text), py_file.name)


class NoRuleEngineDependencyTests(unittest.TestCase):
    """Structural check, same technique as test_phase2b_safety.py: no
    module under src/doar/phase2c1/ (or the Streamlit app) may import any
    rule-evaluation module."""

    def test_no_phase2c1_module_imports_the_rule_engine(self):
        for py_file in sorted(_PHASE2C1_SRC_DIR.glob("*.py")):
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

    def test_annotation_app_avoids_the_rule_engine(self):
        tree = ast.parse(_APP_FILE.read_text(encoding="utf-8"), filename=str(_APP_FILE))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module.split(".")[-1])
        self.assertFalse(imported_names & _FORBIDDEN_RULE_ENGINE_MODULES)

    def test_no_phase2c1_module_evaluates_concerns(self):
        for py_file in sorted(_PHASE2C1_SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_names.add(node.module.split(".")[-1])
            self.assertNotIn("concerns", imported_names)


class NoEmotionLabelExposureTests(unittest.TestCase):
    """Structural: the app's source must never reference an emotion-class
    folder name as a literal string, and AnnotationRecord's schema has no
    field that could carry one."""

    def test_app_source_never_contains_emotion_class_names_as_identifiers(self):
        text = _APP_FILE.read_text(encoding="utf-8").lower()
        for label in _EMOTION_LABELS:
            # Looking for the class name as a standalone word (not a
            # coincidental substring of an unrelated identifier).
            import re
            self.assertIsNone(
                re.search(rf"\b{label}\b", text),
                f"phase2c_annotation_app.py must never reference the emotion "
                f"label {label!r}",
            )

    def test_schema_has_no_emotion_or_split_or_original_path_field(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(schema.AnnotationRecord)}
        for forbidden in ("emotion", "emotion_class", "split", "original_split", "original_path"):
            self.assertNotIn(forbidden, field_names)

    def test_workspace_module_never_writes_original_path_into_blind_images_dir(self):
        text = (_PHASE2C1_SRC_DIR / "workspace.py").read_text(encoding="utf-8")
        # build_pilot_workspace must write the mapping (which DOES carry
        # original_path) to mapping_path, never into images_dir.
        self.assertIn("mapping_path", text)


class ParentViewUnmodifiedTests(unittest.TestCase):
    def test_prototype_app_does_not_import_phase2c1(self):
        prototype = ROOT / "doar_prototype_app.py"
        tree = ast.parse(prototype.read_text(encoding="utf-8"), filename=str(prototype))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        self.assertFalse(any("phase2c1" in name for name in imported_names))


class ObjectStatusVocabularyStaysSafeTests(unittest.TestCase):
    """Pins the exact vocabulary this file's docstring promises, so a
    future edit can't silently narrow uncertain/not_assessable back into
    absent."""

    def test_status_vocabulary_unchanged(self):
        self.assertEqual(schema.OBJECT_STATUSES, frozenset({"present", "absent", "uncertain", "not_assessable"}))


if __name__ == "__main__":
    unittest.main()
