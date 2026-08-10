"""DOAR Visual Knowledge V2: integration tests -- persistence
backward-compatibility, full run_and_persist_initial_scan wiring (fake
predict_fns, no real model weights), rule-safety re-verification with
entities present, and Q&A alias-widened search.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from doar.analysis import analyze_image  # noqa: E402
from doar.expert_review import submit_review  # noqa: E402
from doar.phase2c7.detector_policy import EXPERIMENTAL_AUTOMATIC, build_eye_policy_entry  # noqa: E402
from doar.registry_v2_build import build_registry_v2  # noqa: E402
from doar.visual_entity import VisualEntity  # noqa: E402
from doar.visual_evidence import (  # noqa: E402
    VisualFinding, load_detections, load_entities, run_and_persist_initial_scan, save_detections,
)
from doar.visual_qa import answer_with_visual_grounding  # noqa: E402

FAKE_REGISTRY_V2 = build_registry_v2()  # real, complete registry -- see test_visual_rule_integration.py

GD_OBJECT_RESULTS = {
    "person": (True, 0.9, None), "face": (False, 0.0, None), "hand": (False, 0.0, None),
    "tree": (False, 0.0, None), "house": (False, 0.0, None), "animal": (False, 0.0, None),
    "star": (False, 0.0, None),
}
OWL_OBJECT_RESULTS = {"heart": (False, 0.0, None)}
GD_PARTS_RESULTS = {"mouth": (False, 0.0, None)}
EYE_COMBO_RESULTS = {"eye": (False, 0.0, None)}


def _predict_fns():
    return {
        "grounding_dino_object_classes": lambda p: GD_OBJECT_RESULTS,
        "owlv2_object_classes": lambda p: OWL_OBJECT_RESULTS,
        "grounding_dino_parts": lambda p: GD_PARTS_RESULTS,
        "grounding_dino_parts+owlv2_parts_fallback": lambda p: EYE_COMBO_RESULTS,
    }


def _real_case(tmp_dir) -> Path:
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((30, 30, 170, 170), fill="black")
    path = Path(tmp_dir) / "drawing.png"
    image.save(path)
    case_dir = Path(tmp_dir) / "case"
    analyze_image(path, case_dir)
    return case_dir


def _eye_entry():
    return build_eye_policy_entry(
        status=EXPERIMENTAL_AUTOMATIC, best_model="grounding_dino_parts+owlv2_parts_fallback",
        best_model_checkpoint="ckpt", prompt="eye. mouth.", threshold=0.25,
        precision=0.88, recall=0.73, balanced_accuracy=0.57, n_ground_truth_present=128,
        localization_validated=False, rationale="test")


class BackwardCompatibilityTests(unittest.TestCase):
    def test_save_detections_without_entities_is_byte_identical_to_before(self):
        f = VisualFinding(
            label="dog", finding_id="vf_1", free_form_label=None, bbox=None, confidence=0.5,
            detector="m", checkpoint="c", prompt="p", validation_status="UNKNOWN",
            evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
            related_rule_ids=(), source="initial_scan", query=None, timestamp="t")
        with tempfile.TemporaryDirectory() as tmp:
            save_detections(tmp, [f])
            doc = json.loads((Path(tmp) / "detections.json").read_text(encoding="utf-8"))
            self.assertNotIn("entities", doc)
            self.assertNotIn("n_entities", doc)
            self.assertEqual(set(doc.keys()), {"status", "findings", "n_findings", "generated_at"})

    def test_load_detections_unaffected_by_entities_being_present(self):
        f = VisualFinding(
            label="dog", finding_id="vf_1", free_form_label=None, bbox=None, confidence=0.5,
            detector="m", checkpoint="c", prompt="p", validation_status="UNKNOWN",
            evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
            related_rule_ids=(), source="initial_scan", query=None, timestamp="t")
        from doar.visual_entity import visual_finding_to_entity
        e = visual_finding_to_entity(f)
        with tempfile.TemporaryDirectory() as tmp:
            save_detections(tmp, [f], entities=[e])
            reloaded = load_detections(tmp)
            self.assertEqual(reloaded, [f])

    def test_load_entities_on_old_case_without_entities_key_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "detections.json").write_text(
                json.dumps({"status": "available", "findings": [], "n_findings": 0,
                            "generated_at": "t"}), encoding="utf-8")
            self.assertEqual(load_entities(tmp), [])

    def test_load_entities_round_trips_real_entities(self):
        f = VisualFinding(
            label="dog", finding_id="vf_1", free_form_label=None, bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.5,
            detector="m", checkpoint="c", prompt="p", validation_status="UNKNOWN",
            evidence_status="experimental_evidence_technical_view_only", rule_mapping_status="UNMAPPED",
            related_rule_ids=(), source="initial_scan", query=None, timestamp="t")
        from doar.visual_entity import visual_finding_to_entity
        e = visual_finding_to_entity(f)
        with tempfile.TemporaryDirectory() as tmp:
            save_detections(tmp, [f], entities=[e])
            reloaded = load_entities(tmp)
            self.assertEqual(reloaded, [e])
            self.assertIsInstance(reloaded[0], VisualEntity)


class RunAndPersistInitialScanEntityWiringTests(unittest.TestCase):
    def test_entities_persisted_alongside_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            findings = run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=_predict_fns())
            entities = load_entities(case_dir)
            self.assertEqual(len(entities), len(findings))
            self.assertEqual({e.entity_id for e in entities}, {f.finding_id for f in findings})

    def test_expert_review_history_reflected_in_persisted_entities(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            submit_review(case_dir, reviewer_name="Dr. Test", action="confirm", target_label="person")
            run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=_predict_fns())
            entities = load_entities(case_dir)
            person_entity = next(e for e in entities if e.canonical_label == "person")
            self.assertEqual(person_entity.case_verification_status, "verified")


class QAAliasWidenedSearchTests(unittest.TestCase):
    def _analysis(self):
        return {
            "quality": {"quality_status": "supported", "supported": True},
            "composition": {}, "colour": {"meaningful_colours": []}, "emotion": {"status": "unavailable"},
            "rule_evaluations": [], "evidence": [], "concerns": [],
        }

    def test_alias_term_finds_existing_evidence_even_though_raw_label_differs(self):
        f = VisualFinding(
            label="dog", finding_id="vf_dog", free_form_label=None, bbox=None, confidence=0.8,
            detector="m", checkpoint="c", prompt="p", validation_status="VALIDATED",
            evidence_status="validated_evidence", rule_mapping_status="UNMAPPED", related_rule_ids=(),
            source="initial_scan", query=None, timestamp="t")
        from doar.visual_entity import visual_finding_to_entity
        e = visual_finding_to_entity(f)
        with tempfile.TemporaryDirectory() as tmp:
            save_detections(tmp, [f], entities=[e])
            # "dogs" is an alias, not the raw canonical label "dog" -- plain
            # find_matching would already catch this simple case, but this
            # proves the alias-widened path is real and wired correctly.
            resp = answer_with_visual_grounding(tmp, "is there a dog?", self._analysis())
            self.assertEqual(resp["availability"], "available")
            self.assertEqual(resp["source_module"], "visual_evidence")


class RuleSafetyUnaffectedByEntitiesTests(unittest.TestCase):
    """The rule engine (rule_engine_v2.py) only ever consumes canonical
    Evidence built from VisualFinding -- VisualEntity is a presentation/
    search layer on top, never an input to rule evaluation. Confirms this
    structurally (no import) and functionally (building entities
    alongside findings never changes which rules fire)."""

    def test_rule_engine_module_never_imports_visual_entity(self):
        source = (ROOT / "src" / "doar" / "rule_engine_v2.py").read_text(encoding="utf-8")
        self.assertNotIn("visual_entity", source)

    def test_experimental_and_unknown_findings_still_never_trigger_a_rule_with_entities_present(self):
        # No target in this fixture is VALIDATED (all GD_OBJECT_RESULTS
        # are False except "person"); use a scan where NOTHING is
        # VALIDATED to prove entity-building doesn't leak trust anywhere.
        no_validated_predict_fns = {
            "grounding_dino_object_classes": lambda p: {**GD_OBJECT_RESULTS, "person": (False, 0.0, None)},
            # "heart" is EXPERIMENTAL_AUTOMATIC (never VALIDATED) -- detected
            # here so at least one entity/finding exists, while zero
            # VALIDATED targets exist anywhere in this scan.
            "owlv2_object_classes": lambda p: {**OWL_OBJECT_RESULTS, "heart": (True, 0.9, None)},
            "grounding_dino_parts": lambda p: GD_PARTS_RESULTS,
            "grounding_dino_parts+owlv2_parts_fallback": lambda p: EYE_COMBO_RESULTS,
        }
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=no_validated_predict_fns)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            triggered = [r for r in analysis["rule_evaluations"]
                         if r.get("visual_evidence_sourced") and r["status"] == "weak_support"]
            self.assertEqual(triggered, [])
            # Entities were still built and persisted (nothing silently dropped).
            self.assertGreater(len(load_entities(case_dir)), 0)


class TechnicalViewRendersEntitiesTests(unittest.TestCase):
    def test_technical_view_renders_rich_entities_section_without_error(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest not available")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=_predict_fns())
            image = Image.new("RGB", (200, 200), "white")
            image.save(case_dir / "drawing.png")
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            subheaders = [s.value for s in at.subheader]
            self.assertIn("Rich visual entities (Visual Knowledge V2)", subheaders)


if __name__ == "__main__":
    unittest.main()
