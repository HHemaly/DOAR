"""Phase 2C.5 safety/scope regression tests, mirroring
tests/test_phase2c4_safety.py and tests/test_phase2c4a_safety.py's style:
emotion-label blindness, locked-test exclusion, proposal != ground truth,
provenance preservation, no rule activation, compatibility with Phase 2C.1.
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2c5 import app_helpers as helpers
from doar.phase2c5.schema import PartAnnotationRecord, PartInstance

_FORBIDDEN_RULE_ENGINE_MODULES = {"rule_engine_v2", "evidence_rule_engine", "rules", "structured_report"}
_EMOTION_LABELS = {"angry", "fear", "happy", "sad"}
_PHASE2C5_SRC_DIR = ROOT / "src" / "doar" / "phase2c5"
_ALL_MODULES = sorted(_PHASE2C5_SRC_DIR.glob("*.py"))
_TOP_LEVEL_APP = ROOT / "phase2c5_part_annotation_app.py"
_SCRIPTS = sorted((ROOT / "scripts").glob("phase2c5_*.py"))


class NoRuleEngineDependencyTests(unittest.TestCase):
    def test_no_phase2c5_module_imports_the_rule_engine(self):
        for py_file in _ALL_MODULES + [_TOP_LEVEL_APP] + _SCRIPTS:
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

    def test_rule_traceability_module_only_reads_registry_never_writes(self):
        text = (_PHASE2C5_SRC_DIR / "rule_traceability.py").read_text(encoding="utf-8")
        self.assertNotIn("allowed_output_level\"] =", text)
        self.assertNotIn("allowed_output_level'] =", text)


class NoEmotionLabelExposureTests(unittest.TestCase):
    def test_no_source_or_app_file_references_emotion_class_names(self):
        # __init__.py's own module docstring explicitly documents that no
        # emotion label is used ("No emotion label (Angry/Fear/Happy/Sad) is
        # referenced anywhere in it") -- that sentence necessarily contains
        # the words it is disclaiming; excluded from this scan for the same
        # reason PHASE2C4A's model_status.py test checks field values, not
        # raw source text containing its own forbidden-phrase list.
        # rule_traceability.py is documentation/rationale that quotes real
        # source-rule text explaining WHY EN_COMPILED_FACE_EXPRESSION_021 is
        # postponed (the source rule itself is written in emotion-adjacent
        # language, e.g. "sad, tense, or frightened faces") -- that is
        # meta-discussion of the registry's own wording, not the project's
        # actual forbidden pattern (using the dataset's Angry/Fear/Happy/Sad
        # FOLDER labels to drive sampling/annotation/interpretation).
        # Covered instead by RuleTraceabilityEmotionWordsAreInertTests below.
        excluded = {"__init__.py", "rule_traceability.py"}
        scanned = [f for f in _ALL_MODULES + [_TOP_LEVEL_APP] + _SCRIPTS if f.name not in excluded]
        for py_file in scanned:
            text = py_file.read_text(encoding="utf-8").lower()
            for label in _EMOTION_LABELS:
                self.assertIsNone(
                    re.search(rf"\b{label}\b", text),
                    f"{py_file.name} must never reference the emotion label {label!r}")

    def test_init_docstring_only_mentions_labels_in_its_own_disclaimer(self):
        text = (_PHASE2C5_SRC_DIR / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("No emotion label (Angry/Fear/Happy/Sad) is", text)
        # every emotion-label occurrence must be on that one disclaimer line
        offending_lines = [line for line in text.splitlines()
                            if re.search(r"\b(angry|fear|happy|sad)\b", line, re.IGNORECASE)
                            and "no emotion label" not in line.lower()]
        self.assertEqual(offending_lines, [])


class RuleTraceabilityEmotionWordsAreInertTests(unittest.TestCase):
    """rule_traceability.py quotes real source-rule text that uses
    emotion-adjacent English words (see the exclusion note in
    NoEmotionLabelExposureTests above). This confirms those words are
    inert prose -- never used in a conditional, comparison, filter, or any
    decision logic -- and the module has no sampling/annotation-decision
    capability of any kind (it defines a plain, static data table)."""

    def test_module_defines_no_functions_with_conditional_logic_on_emotion_words(self):
        tree = ast.parse((_PHASE2C5_SRC_DIR / "rule_traceability.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.Compare)):
                dumped = ast.dump(node).lower()
                for label in _EMOTION_LABELS:
                    self.assertNotIn(label, dumped,
                                      f"emotion label {label!r} used in a conditional/comparison")

    def test_module_has_no_image_or_pilot_id_handling(self):
        text = (_PHASE2C5_SRC_DIR / "rule_traceability.py").read_text(encoding="utf-8")
        self.assertNotIn("pilot_id", text)
        self.assertNotIn("image_id", text)


class LockedTestProtectionTests(unittest.TestCase):
    def test_feasibility_pilot_selection_calls_locked_test_guard(self):
        text = (ROOT / "scripts/phase2c5_select_feasibility_pilot.py").read_text(encoding="utf-8")
        self.assertIn("assert_pilot_ids_exclude_locked_test", text)
        self.assertIn("DEV_ELIGIBLE", text)
        self.assertNotIn("LOCKED_TEST", text)

    def test_feasibility_pilot_selection_actually_uses_dev_eligible_cohort(self):
        import sys as _sys
        sys_path_added = str(ROOT / "scripts") not in _sys.path
        if sys_path_added:
            _sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import importlib
            mod = importlib.import_module("phase2c5_select_feasibility_pilot")
            src = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
            called_names = set()
            for n in ast.walk(src):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                    called_names.add(n.func.attr)
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    called_names.add(n.func.id)
            self.assertIn("split_pilot_ids_by_cohort", called_names)
        finally:
            if sys_path_added:
                _sys.path.remove(str(ROOT / "scripts"))

    def test_expansion_sampling_module_never_references_locked_test(self):
        text = (_PHASE2C5_SRC_DIR / "expansion_sampling.py").read_text(encoding="utf-8")
        self.assertNotIn("LOCKED_TEST", text)
        self.assertNotIn("locked_test", text)


class ProposalNeverGroundTruthTests(unittest.TestCase):
    def test_model_proposed_instance_is_not_accepted_by_construction(self):
        proposal = PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1), bbox_source="model_proposed",
                                 proposal_model="grounding_dino:tiny")
        self.assertEqual(proposal.bbox_source, "model_proposed")
        self.assertNotEqual(proposal.bbox_source, "human_accepted")

    def test_seeding_from_proposals_never_produces_accepted_source(self):
        boxes = [{"instance_index": 0, "bbox": (0.1, 0.1, 0.1, 0.1)}]
        seeded = helpers.seed_instances_from_proposals(
            boxes, model_name="owlv2:base", checkpoint="c", prompt="a eye", threshold=0.1, timestamp="t")
        self.assertTrue(all(i.bbox_source == "model_proposed" for i in seeded))

    def test_app_module_never_auto_accepts_proposals(self):
        """The app must require an explicit user action (an Accept button
        click) to move a box out of 'model_proposed' -- the call site must
        be inside the `if ...button("Accept...")` block (within a few
        lines after it), not called unconditionally at module scope."""
        lines = (ROOT / "phase2c5_part_annotation_app.py").read_text(encoding="utf-8").splitlines()
        accept_idx = next(i for i, line in enumerate(lines) if "accept_proposal_instance(inst)" in line)
        preceding_window = "\n".join(lines[max(0, accept_idx - 5):accept_idx])
        self.assertIn('.button("Accept', preceding_window)


class ProvenancePreservationTests(unittest.TestCase):
    def test_accepted_instance_still_names_the_source_model(self):
        proposal = PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1), bbox_source="model_proposed",
                                 proposal_model="grounding_dino:tiny", proposal_checkpoint="ckpt",
                                 proposal_prompt="eye.", proposal_threshold=0.25, proposal_timestamp="t")
        accepted = helpers.accept_proposal_instance(proposal)
        self.assertEqual(accepted.proposal_model, "grounding_dino:tiny")
        self.assertEqual(accepted.proposal_checkpoint, "ckpt")
        self.assertEqual(accepted.proposal_threshold, 0.25)

    def test_human_drawn_instance_carries_no_false_model_provenance(self):
        inst = helpers.new_manual_instance(0, (0.1, 0.1, 0.1, 0.1))
        self.assertEqual(inst.proposal_model, "")
        self.assertEqual(inst.proposal_checkpoint, "")


class BboxSourceDistinctionTests(unittest.TestCase):
    def test_human_accepted_human_edited_human_drawn_are_distinct_and_all_reachable(self):
        proposal = PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1), bbox_source="model_proposed",
                                 proposal_model="m")
        accepted = helpers.accept_proposal_instance(proposal)
        edited = helpers.edit_proposal_instance(proposal, (0.2, 0.2, 0.1, 0.1))
        drawn = helpers.new_manual_instance(0, (0.3, 0.3, 0.1, 0.1))
        sources = {accepted.bbox_source, edited.bbox_source, drawn.bbox_source}
        self.assertEqual(sources, {"human_accepted", "human_edited", "human_drawn"})


class MultiInstanceSupportTests(unittest.TestCase):
    def test_record_supports_more_than_one_instance(self):
        rec = PartAnnotationRecord(
            pilot_id="p2b_0000", target_name="hand", status="present",
            annotator_id="ann1", annotation_timestamp="t",
            instances=(PartInstance(instance_index=0, bbox=(0.1, 0.1, 0.1, 0.1), bbox_source="human_drawn"),
                       PartInstance(instance_index=1, bbox=(0.5, 0.5, 0.1, 0.1), bbox_source="human_drawn")))
        self.assertEqual(len(rec.instances), 2)


class UncertainNotAssessableHandlingTests(unittest.TestCase):
    def test_uncertain_status_valid_with_no_instances(self):
        rec = PartAnnotationRecord(pilot_id="p1", target_name="eye", status="uncertain",
                                    annotator_id="a", annotation_timestamp="t",
                                    uncertainty_reason="occluded by hair")
        self.assertEqual(rec.status, "uncertain")

    def test_not_assessable_status_valid_with_no_instances(self):
        rec = PartAnnotationRecord(pilot_id="p1", target_name="mouth", status="not_assessable",
                                    annotator_id="a", annotation_timestamp="t")
        self.assertEqual(rec.status, "not_assessable")


class Phase2C1CompatibilityTests(unittest.TestCase):
    def test_phase2c5_never_imports_phase2c1_store_writers_for_writing(self):
        """phase2c5 modules may READ phase2c1 (e.g. for the 'prefill from
        existing presence' UI hint) but must never call its save/upsert
        writers -- Phase 2C.1's store stays untouched."""
        for py_file in _ALL_MODULES:
            text = py_file.read_text(encoding="utf-8")
            self.assertNotIn("phase2c1.store.upsert", text)
            self.assertNotIn("phase2c1.store.save_store", text)

    def test_app_reads_phase2c1_store_read_only(self):
        text = (ROOT / "phase2c5_part_annotation_app.py").read_text(encoding="utf-8")
        self.assertIn("store2c1_mod.load_store", text)
        self.assertNotIn("store2c1_mod.save_store", text)
        self.assertNotIn("store2c1_mod.upsert", text)

    def test_phase2c5_schema_version_distinct_from_phase2c1(self):
        from doar.phase2c1.schema import SCHEMA_VERSION as v1
        from doar.phase2c5.schema import SCHEMA_VERSION as v5
        self.assertNotEqual(v1, v5)


if __name__ == "__main__":
    unittest.main()
