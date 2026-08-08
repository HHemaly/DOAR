"""Phase 2C.6 safety/scope regression tests, mirroring
tests/test_phase2c5_safety.py's style: no rule-engine import, no emotion-
label exposure, locked-test exclusion, private-path leakage, no fine-
tuning, backward compatibility with the Phase 2C.5 store.
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5.schema import SCHEMA_VERSION as SCHEMA_V1
from doar.phase2c5.schema import PartAnnotationRecord, PartInstance, record_from_row
from doar.phase2c6.expansion_manifest import PILOT_ID_PREFIX

_FORBIDDEN_RULE_ENGINE_MODULES = {"rule_engine_v2", "evidence_rule_engine", "rules", "structured_report"}
_EMOTION_LABELS = {"angry", "fear", "happy", "sad"}
_PHASE2C6_SRC_DIR = ROOT / "src" / "doar" / "phase2c6"
_ALL_MODULES = sorted(_PHASE2C6_SRC_DIR.glob("*.py"))
_SCRIPTS = sorted((ROOT / "scripts").glob("phase2c6_*.py"))
_EXCLUDED_FROM_EMOTION_SCAN = {"__init__.py"}  # documents its own non-use, same carve-out as Phase 2C.5


class NoRuleEngineDependencyTests(unittest.TestCase):
    def test_no_phase2c6_module_imports_the_rule_engine(self):
        for py_file in _ALL_MODULES + _SCRIPTS:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[-1])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.split(".")[-1])
            forbidden = imported & _FORBIDDEN_RULE_ENGINE_MODULES
            self.assertFalse(forbidden, f"{py_file.name} imports rule-engine module(s): {forbidden}")

    def test_no_module_writes_allowed_output_level(self):
        for py_file in _ALL_MODULES + _SCRIPTS:
            text = py_file.read_text(encoding="utf-8")
            self.assertNotIn("allowed_output_level\"] =", text)


class NoEmotionLabelExposureTests(unittest.TestCase):
    def test_no_source_or_script_references_emotion_class_names(self):
        scanned = [f for f in _ALL_MODULES + _SCRIPTS if f.name not in _EXCLUDED_FROM_EMOTION_SCAN]
        for py_file in scanned:
            text = py_file.read_text(encoding="utf-8").lower()
            for label in _EMOTION_LABELS:
                self.assertIsNone(re.search(rf"\b{label}\b", text),
                                  f"{py_file.name} must never reference the emotion label {label!r}")

    def test_expansion_manifest_selection_path_has_no_emotion_conditional(self):
        """Structural guard: no If/Compare node in expansion_manifest.py's
        AST may mention an emotion label -- selection logic must be
        provably blind, not just documented as blind."""
        tree = ast.parse((_PHASE2C6_SRC_DIR / "expansion_manifest.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.Compare)):
                dumped = ast.dump(node).lower()
                for label in _EMOTION_LABELS:
                    self.assertNotIn(label, dumped)


class LockedTestExclusionTests(unittest.TestCase):
    def test_expansion_manifest_module_never_references_locked_test(self):
        text = (_PHASE2C6_SRC_DIR / "expansion_manifest.py").read_text(encoding="utf-8")
        self.assertNotIn("LOCKED_TEST", text)

    def test_build_manifest_script_excludes_prior_round_which_includes_locked_test(self):
        text = (ROOT / "scripts/phase2c6_build_expansion_manifest.py").read_text(encoding="utf-8")
        self.assertIn("load_used_image_ids", text)
        self.assertIn("exclude_image_ids", text)

    def test_verify_pilot_ui_script_asserts_no_locked_test_image(self):
        text = (ROOT / "scripts/phase2c6_verify_pilot_ui.py").read_text(encoding="utf-8")
        self.assertIn("LOCKED_TEST", text)
        self.assertIn("locked-test", text.lower())


class PrivatePathLeakageTests(unittest.TestCase):
    def test_write_expansion_summary_dict_keys_are_opaque_only(self):
        """AST-level check of the actual dict LITERAL keys assigned to
        `summary` in write_expansion_summary -- not the docstring (which
        legitimately names 'original_path'/'group_id' once, to disclaim
        including them). Covers the same ground as
        test_phase2c6_expansion_manifest.py's real-data test, structurally."""
        tree = ast.parse((_PHASE2C6_SRC_DIR / "expansion_manifest.py").read_text(encoding="utf-8"))
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "write_expansion_summary")
        dict_node = next(n for n in ast.walk(func) if isinstance(n, ast.Dict))
        keys = {k.value for k in dict_node.keys if isinstance(k, ast.Constant)}
        forbidden = {"original_path", "path", "group_id", "image_id"}
        self.assertEqual(keys & forbidden, set())

    def test_pilot_id_prefix_is_distinct_from_phase2c1_prefix(self):
        self.assertEqual(PILOT_ID_PREFIX, "p2c6")
        self.assertNotEqual(PILOT_ID_PREFIX, "p2b")


class NoFineTuningAnywhereInPackageTests(unittest.TestCase):
    def test_no_module_imports_a_training_framework(self):
        forbidden_imports = {"peft", "accelerate", "trl"}
        for py_file in _ALL_MODULES:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.split(".")[0])
            self.assertFalse(imported & forbidden_imports, f"{py_file.name} imports {imported & forbidden_imports}")


class BackwardCompatibilityWithPhase2C5StoreTests(unittest.TestCase):
    def test_phase2c6_produces_records_the_phase2c5_schema_accepts_unchanged(self):
        """A Phase 2C.6 canvas-produced instance list must be a completely
        ordinary Phase 2C.5 PartAnnotationRecord -- same schema_version,
        same validation, round-trips through record_from_row exactly like
        any Phase 2C.5-authored row."""
        from doar.phase2c6.canvas_helpers import reconcile_canvas_session

        seed = PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.2, 0.2), bbox_source="model_proposed",
                             proposal_model="grounding_dino:tiny", proposal_threshold=0.25)
        working = reconcile_canvas_session([(0.12, 0.12, 0.2, 0.2)], [seed])
        rec = PartAnnotationRecord(pilot_id="p2c6_0000", target_name="eye", status="present",
                                    annotator_id="ann1", annotation_timestamp="t", instances=working)
        self.assertEqual(rec.schema_version, SCHEMA_V1)
        row = rec.to_row()
        restored = record_from_row(row)
        self.assertEqual(restored.instances[0].bbox_source, "human_edited")

    def test_phase2c6_never_redefines_the_schema_version(self):
        text = (_PHASE2C6_SRC_DIR / "canvas_helpers.py").read_text(encoding="utf-8")
        self.assertNotIn("SCHEMA_VERSION", text)
        text2 = (_PHASE2C6_SRC_DIR / "proposal_batch.py").read_text(encoding="utf-8")
        self.assertNotIn("SCHEMA_VERSION", text2)

    def test_phase2c6_never_imports_phase2c1_store_module_at_all(self):
        """AST-level: no phase2c6 module or script imports
        `doar.phase2c1.store` (or `doar.phase2c1 import store`) under any
        alias -- stronger than a text scan, and immune to a docstring
        merely mentioning the module name in prose (e.g. the Stage 2
        verification script's own docstring, which disclaims writing to
        it)."""
        for py_file in _ALL_MODULES + _SCRIPTS:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotEqual(node.module, "doar.phase2c1.store", py_file.name)
                    if node.module in ("doar.phase2c1", "..phase2c1", ".phase2c1"):
                        imported_names = {alias.name for alias in node.names}
                        self.assertNotIn("store", imported_names, py_file.name)


class FrozenBenchmarkUntouchedTests(unittest.TestCase):
    def test_no_phase2c6_module_imports_phase2c4_or_phase2c4a(self):
        for py_file in _ALL_MODULES:
            text = py_file.read_text(encoding="utf-8")
            self.assertNotIn("phase2c4", text.replace("phase2c4_", "").replace("PHASE2C4", ""))

    def test_no_script_writes_into_artifacts_phase2c4_directories(self):
        for py_file in _SCRIPTS:
            text = py_file.read_text(encoding="utf-8")
            self.assertNotIn("artifacts/phase2c4/", text)
            self.assertNotIn("artifacts/phase2c4a/", text)


if __name__ == "__main__":
    unittest.main()
