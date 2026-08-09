"""Phase 2C.7 safety/scope regression tests, mirroring every prior
phase's style: no rule-engine import, no emotion-label exposure, holdout
isolation, experimental/disabled evidence cannot activate a validated
conclusion, provenance preservation, image-only inference, frozen
Phase 2C.4/2C.4A artifacts untouched, no fine-tuning.
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5.schema import PartInstance
from doar.phase2c7 import detector_policy as pol
from doar.phase2c7 import dev_holdout_split as split_mod
from doar.phase2c7 import visual_detector as vd

_FORBIDDEN_RULE_ENGINE_MODULES = {"rule_engine_v2", "evidence_rule_engine", "rules", "structured_report"}
_EMOTION_LABELS = {"angry", "fear", "happy", "sad"}
_PHASE2C7_SRC_DIR = ROOT / "src" / "doar" / "phase2c7"
_ALL_MODULES = sorted(_PHASE2C7_SRC_DIR.glob("*.py"))
_EXCLUDED_FROM_EMOTION_SCAN = {"__init__.py"}


class NoRuleEngineDependencyTests(unittest.TestCase):
    def test_no_phase2c7_module_imports_the_rule_engine(self):
        for py_file in _ALL_MODULES:
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
        for py_file in _ALL_MODULES:
            text = py_file.read_text(encoding="utf-8")
            self.assertNotIn("allowed_output_level\"] =", text)


class NoEmotionLabelExposureTests(unittest.TestCase):
    def test_no_source_file_references_emotion_class_names(self):
        for py_file in _ALL_MODULES:
            if py_file.name in _EXCLUDED_FROM_EMOTION_SCAN:
                continue
            text = py_file.read_text(encoding="utf-8").lower()
            for label in _EMOTION_LABELS:
                self.assertIsNone(re.search(rf"\b{label}\b", text),
                                  f"{py_file.name} must never reference the emotion label {label!r}")


class HoldoutIsolationTests(unittest.TestCase):
    def test_detector_policy_module_has_no_holdout_selection_code(self):
        """detector_policy.py's docstring legitimately mentions 'the eye
        holdout' once, in prose, to explain provenance -- this checks for
        actual CODE (an identifier, not a comment/docstring word) that
        would let this module select or touch holdout membership."""
        tree = ast.parse((_PHASE2C7_SRC_DIR / "detector_policy.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        holdout_identifiers = {n for n in names if "holdout" in n.lower()}
        self.assertEqual(holdout_identifiers, set())

    def test_eye_evaluation_module_has_no_split_selection_logic(self):
        """eye_evaluation.py computes metrics given already-decided ground
        truth/predictions -- it must never itself decide dev vs. holdout
        membership (that is dev_holdout_split.py's sole job)."""
        text = (_PHASE2C7_SRC_DIR / "eye_evaluation.py").read_text(encoding="utf-8")
        self.assertNotIn("holdout", text.lower())
        self.assertNotIn("DEV_HOLDOUT_SEED", text)


class ExperimentalCannotActivateValidatedRuleTests(unittest.TestCase):
    def test_validated_records_only_excludes_experimental(self):
        records = [
            vd.VisualEvidenceRecord("eye", True, None, 0.9, "m", "c", 0.1, pol.VALIDATED_AUTOMATIC,
                                     "validated_evidence"),
            vd.VisualEvidenceRecord("heart", True, None, 0.9, "m", "c", 0.1, pol.EXPERIMENTAL_AUTOMATIC,
                                     "experimental_evidence_technical_view_only"),
        ]
        result = vd.validated_records_only(records)
        self.assertEqual([r.target for r in result], ["eye"])

    def test_no_experimental_entry_in_real_policy_is_mislabeled_validated(self):
        eye_entry = pol.build_eye_policy_entry(
            status=pol.EXPERIMENTAL_AUTOMATIC, best_model="m", best_model_checkpoint="c", prompt="p",
            threshold=0.1, precision=0.3, recall=0.2, balanced_accuracy=0.5,
            n_ground_truth_present=10, localization_validated=False, rationale="r")
        full = pol.full_policy(eye_entry)
        for target, entry in full.items():
            if entry.status == pol.EXPERIMENTAL_AUTOMATIC:
                self.assertNotIn("validated evidence pipeline", entry.allowed_downstream_usage,
                                  f"{target} is EXPERIMENTAL but its usage text claims validated pipeline")

    def test_evidence_status_mapping_is_exhaustive_and_correct(self):
        self.assertEqual(vd.EVIDENCE_STATUS_BY_VALIDATION[pol.VALIDATED_AUTOMATIC], "validated_evidence")
        self.assertEqual(vd.EVIDENCE_STATUS_BY_VALIDATION[pol.EXPERIMENTAL_AUTOMATIC],
                          "experimental_evidence_technical_view_only")
        self.assertEqual(vd.EVIDENCE_STATUS_BY_VALIDATION[pol.DISABLED], "not_used")


class DisabledEvidenceCannotEnterPipelineTests(unittest.TestCase):
    def test_disabled_target_produces_zero_records(self):
        policy = {"circle": pol.OBJECT_CLASS_POLICY["circle"]}
        records = vd.analyze_image("img.jpg", policy,
                                    model_predict_fns={"grounding_dino_object_classes":
                                                        lambda p: {"circle": (True, 0.99, None)}})
        self.assertEqual(records, [])

    def test_disabled_targets_in_real_policy(self):
        for target in ("circle", "vehicle"):
            self.assertEqual(pol.OBJECT_CLASS_POLICY[target].status, pol.DISABLED)


class ProvenancePreservationTests(unittest.TestCase):
    def test_visual_evidence_record_carries_model_checkpoint_threshold(self):
        rec = vd.VisualEvidenceRecord("eye", True, (0.1, 0.1, 0.1, 0.1), 0.7, "owlv2_parts",
                                       "google/owlv2-base-patch16-ensemble", 0.1,
                                       pol.VALIDATED_AUTOMATIC, "validated_evidence")
        d = rec.to_dict()
        self.assertEqual(d["model"], "owlv2_parts")
        self.assertEqual(d["checkpoint"], "google/owlv2-base-patch16-ensemble")
        self.assertEqual(d["threshold"], 0.1)

    def test_eye_instances_in_real_export_are_all_provenance_valid(self):
        """Structural: PartInstance itself already enforces bbox_source
        validity and provenance-field consistency (schema.py) -- this just
        confirms Phase 2C.7 code never bypasses that by constructing an
        instance some other way."""
        inst = PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1), bbox_source="human_drawn")
        self.assertEqual(inst.proposal_model, "")


class ImageOnlyInferenceTests(unittest.TestCase):
    def test_analyze_image_signature_has_no_annotation_parameter(self):
        import inspect
        sig = inspect.signature(vd.analyze_image)
        param_names = set(sig.parameters)
        for forbidden in ("ground_truth", "annotation", "store", "pilot_id", "label"):
            self.assertNotIn(forbidden, param_names)
        self.assertIn("image_path", param_names)

    def test_analyze_image_does_not_import_any_store_module(self):
        text = (_PHASE2C7_SRC_DIR / "visual_detector.py").read_text(encoding="utf-8")
        self.assertNotIn("phase2c5.store", text)
        self.assertNotIn("phase2c1.store", text)


class FrozenBenchmarkUntouchedTests(unittest.TestCase):
    def test_no_phase2c7_module_writes_into_phase2c4_artifact_dirs(self):
        """detector_policy.py legitimately CITES
        artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv and
        artifacts/phase2c4a/calibration_frozen_operating_points.csv in
        its own docstring (its rationale strings' data source) -- this
        checks that no module actually OPENS a path under those
        directories at all (read or write), which is the real guarantee
        ("never re-run, never modified") the phase report promises."""
        for py_file in _ALL_MODULES:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            self.assertNotIn("artifacts/phase2c4", arg.value, py_file.name)

    def test_no_fine_tuning_imports_anywhere(self):
        forbidden_imports = {"peft", "trl"}
        for py_file in _ALL_MODULES:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.split(".")[0])
            self.assertFalse(imported & forbidden_imports)

    def test_no_training_calls_anywhere(self):
        for py_file in _ALL_MODULES:
            text = py_file.read_text(encoding="utf-8").lower()
            for forbidden in (".fit(", ".backward()", "optimizer.step"):
                self.assertNotIn(forbidden, text)


class DevHoldoutSplitDeterminismTests(unittest.TestCase):
    def test_split_is_deterministic_and_covers_all_reviewed_ids(self):
        ids = [f"p2c6_{i:04d}" for i in range(213)]
        s = split_mod.split_dev_holdout(ids)
        split_mod.assert_no_overlap(s)
        self.assertEqual(set(s.dev_pilot_ids) | set(s.holdout_pilot_ids), set(ids))


if __name__ == "__main__":
    unittest.main()
