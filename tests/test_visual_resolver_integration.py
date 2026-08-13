"""DOAR Visual Resolver: integration tests for the discovery + case
verification foundation -- merge, crop_ref, verifier application,
rule-safety with observer-sourced entities, and search.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from doar.analysis import analyze_image  # noqa: E402
from doar.phase2c7.detector_policy import EXPERIMENTAL_AUTOMATIC, build_eye_policy_entry  # noqa: E402
from doar.registry_v2_build import build_registry_v2  # noqa: E402
from doar.visual_entity import (  # noqa: E402
    VisualEntity, _bbox_is_crop_usable, _crop_region, apply_verifier_to_entities,
    merge_observer_candidates_into_entities, populate_crop_refs, save_entity_crop, visual_finding_to_entity,
)
from doar.visual_evidence import (  # noqa: E402
    VisualFinding, load_entities, run_and_persist_initial_scan, update_entities,
)
from doar.visual_observer import (  # noqa: E402
    CallableVisualObserver, CallableVisualVerifier, GeminiVisualObserver, GeminiVisualVerifier,
    OpenAIVisualObserver, VerificationResult, VisualObserverCandidate, VisualObserverConfigurationError,
)
from doar.visual_qa import answer_with_visual_grounding  # noqa: E402

FAKE_REGISTRY_V2 = build_registry_v2()

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


def _eye_entry():
    return build_eye_policy_entry(
        status=EXPERIMENTAL_AUTOMATIC, best_model="grounding_dino_parts+owlv2_parts_fallback",
        best_model_checkpoint="ckpt", prompt="eye. mouth.", threshold=0.25,
        precision=0.88, recall=0.73, balanced_accuracy=0.57, n_ground_truth_present=128,
        localization_validated=False, rationale="test")


def _real_case(tmp_dir) -> Path:
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((30, 30, 170, 170), fill="black")
    path = Path(tmp_dir) / "drawing.png"
    image.save(path)
    case_dir = Path(tmp_dir) / "case"
    analyze_image(path, case_dir)
    return case_dir


def _finding(label, *, bbox=None, validation_status="VALIDATED", confidence=0.8, finding_id=None):
    return VisualFinding(
        label=label, finding_id=finding_id or f"vf_{label}", free_form_label=None, bbox=bbox,
        confidence=confidence, detector="m", checkpoint="c", prompt="p",
        validation_status=validation_status,
        evidence_status="validated_evidence" if validation_status == "VALIDATED"
        else "experimental_evidence_technical_view_only",
        rule_mapping_status="UNMAPPED", related_rule_ids=(), source="initial_scan", query=None,
        timestamp="2026-08-11T00:00:00+00:00")


class MergeObserverCandidatesTests(unittest.TestCase):
    def test_candidate_matching_existing_entity_only_enriches_it(self):
        e = visual_finding_to_entity(_finding("dog"))
        candidate = VisualObserverCandidate(label="dogs", confidence=0.55, source_note="test")
        merged = merge_observer_candidates_into_entities([e], [candidate], registry_v2=FAKE_REGISTRY_V2)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].canonical_label, "dog")
        self.assertEqual(merged[0].model_validation_status, "VALIDATED")  # unchanged
        self.assertIn(("dogs", 0.55), merged[0].candidate_labels)

    def test_candidate_label_never_overwrites_canonical_label(self):
        e = visual_finding_to_entity(_finding("dog"))
        candidate = VisualObserverCandidate(label="dogs", confidence=0.99, source_note="test")
        merged = merge_observer_candidates_into_entities([e], [candidate], registry_v2=FAKE_REGISTRY_V2)
        self.assertEqual(merged[0].canonical_label, "dog")  # NOT "dogs", regardless of confidence

    def test_unmatched_candidate_creates_new_unknown_status_entity(self):
        e = visual_finding_to_entity(_finding("dog"))
        candidate = VisualObserverCandidate(label="kite", entity_type="object",
                                             bbox=(0.1, 0.1, 0.2, 0.2), confidence=0.5, source_note="test")
        merged = merge_observer_candidates_into_entities([e], [candidate], registry_v2=FAKE_REGISTRY_V2)
        self.assertEqual(len(merged), 2)
        new_entity = next(m for m in merged if m.canonical_label == "kite")
        self.assertEqual(new_entity.model_validation_status, "UNKNOWN")
        self.assertEqual(new_entity.source, "visual_observer")
        self.assertEqual(new_entity.case_verification_status, "unreviewed")
        self.assertEqual(new_entity.evidence_status, "experimental_evidence_technical_view_only")

    def test_new_entity_has_no_backing_finding_id(self):
        candidate = VisualObserverCandidate(label="kite", bbox=(0.1, 0.1, 0.2, 0.2), source_note="test")
        merged = merge_observer_candidates_into_entities([], [candidate], registry_v2=FAKE_REGISTRY_V2)
        self.assertFalse(merged[0].entity_id.startswith("vf_"))
        self.assertTrue(merged[0].entity_id.startswith("ve_"))

    def test_invalid_entity_type_from_candidate_clamped_to_unknown(self):
        candidate = VisualObserverCandidate(label="blob", entity_type="spaceship", source_note="test")
        merged = merge_observer_candidates_into_entities([], [candidate], registry_v2=FAKE_REGISTRY_V2)
        self.assertEqual(merged[0].entity_type, "unknown")

    def test_unknown_region_preserved_with_full_geometry(self):
        candidate = VisualObserverCandidate(label="odd_mark", entity_type="abstract_mark",
                                             bbox=(0.2, 0.3, 0.1, 0.1), confidence=0.4, source_note="test")
        merged = merge_observer_candidates_into_entities([], [candidate], registry_v2=FAKE_REGISTRY_V2)
        self.assertEqual(merged[0].entity_type, "abstract_mark")
        self.assertEqual(merged[0].bbox, (0.2, 0.3, 0.1, 0.1))
        self.assertIsNotNone(merged[0].relative_size)
        self.assertIsNotNone(merged[0].page_position)


class CropRefTests(unittest.TestCase):
    def test_save_entity_crop_writes_a_real_file_and_returns_case_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (100, 100), (10, 200, 10)).save(image_path)
            crop_ref = save_entity_crop(str(image_path), (0.1, 0.1, 0.3, 0.3),
                                         case_dir=case_dir, entity_id="vf_test")
            self.assertEqual(crop_ref, str(Path("artifacts") / "crops" / "vf_test.png"))
            self.assertTrue((case_dir / crop_ref).exists())

    def test_save_entity_crop_returns_none_without_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(save_entity_crop("fake.png", None, case_dir=tmp, entity_id="x"))

    def test_populate_crop_refs_fills_only_entities_with_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (100, 100), (10, 200, 10)).save(image_path)
            with_bbox = visual_finding_to_entity(_finding("dog", bbox=(0.1, 0.1, 0.3, 0.3), finding_id="vf_a"))
            without_bbox = visual_finding_to_entity(_finding("house", bbox=None, finding_id="vf_b"))
            updated = populate_crop_refs([with_bbox, without_bbox], str(image_path), case_dir)
            by_id = {e.entity_id: e for e in updated}
            self.assertIsNotNone(by_id["vf_a"].crop_ref)
            self.assertIsNone(by_id["vf_b"].crop_ref)

    def test_populate_crop_refs_never_overwrites_an_existing_crop_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (100, 100), (10, 200, 10)).save(image_path)
            e = visual_finding_to_entity(_finding("dog", bbox=(0.1, 0.1, 0.3, 0.3)))
            d = e.to_dict()
            d["crop_ref"] = "artifacts/crops/already_set.png"
            e = VisualEntity.from_dict(d)
            updated = populate_crop_refs([e], str(image_path), case_dir)
            self.assertEqual(updated[0].crop_ref, "artifacts/crops/already_set.png")


class BboxValidityTests(unittest.TestCase):
    """DOAR Visual Verification stabilization: Gemini's own documented,
    trained-in bbox convention ([ymin, xmin, ymax, xmax] scaled 0-1000,
    per ai.google.dev's object-detection guide) is NOT this project's
    normalized-xywh-in-[0,1] contract -- even with an explicit custom
    schema asking for the latter, a real response can still land
    (moderately) outside it. Proves the fix: an out-of-range bbox is
    REJECTED outright (never clamped into a differently-shaped, silently
    WRONG crop -- the h38 "yellow car" bug, whose bbox landed on the
    traffic light instead), while ordinary in-range boxes and tiny
    (rounding-noise-sized) overshoots still crop normally -- exactly
    once, with no double normalization anywhere in this path."""

    def test_valid_bbox_is_usable(self):
        self.assertTrue(_bbox_is_crop_usable((0.1, 0.1, 0.3, 0.3)))

    def test_edge_bbox_touching_bounds_is_usable(self):
        self.assertTrue(_bbox_is_crop_usable((0.0, 0.0, 1.0, 1.0)))
        self.assertTrue(_bbox_is_crop_usable((0.7, 0.7, 0.3, 0.3)))  # x+w == 1.0 exactly

    def test_tiny_rounding_overshoot_still_usable(self):
        # x + w = 1.01 -- ordinary floating-point noise, not a real error.
        self.assertTrue(_bbox_is_crop_usable((0.7, 0.7, 0.31, 0.31)))

    def test_the_h38_yellow_car_bbox_is_rejected_not_clamped(self):
        # The exact real bbox that produced the localization bug: x=0.81,
        # w=0.54 -> x+w=1.35, far beyond any rounding tolerance.
        self.assertFalse(_bbox_is_crop_usable((0.81, 0.343, 0.54, 0.596)))

    def test_grossly_out_of_range_bbox_is_rejected(self):
        # The exact shape of p2b_0004's observer bug: y and h values in a
        # completely different (non-[0,1]) scale.
        self.assertFalse(_bbox_is_crop_usable((0.08, 6.32, 0.44, 3.68)))

    def test_non_positive_area_bbox_is_rejected(self):
        self.assertFalse(_bbox_is_crop_usable((0.1, 0.1, 0.0, 0.3)))
        self.assertFalse(_bbox_is_crop_usable((0.1, 0.1, 0.3, 0.0)))
        self.assertFalse(_bbox_is_crop_usable((0.1, 0.1, -0.1, 0.3)))

    def test_negative_origin_beyond_tolerance_is_rejected(self):
        self.assertFalse(_bbox_is_crop_usable((-0.5, 0.1, 0.3, 0.3)))

    def test_none_bbox_is_not_usable(self):
        self.assertFalse(_bbox_is_crop_usable(None))

    def test_crop_region_returns_none_for_the_yellow_car_bbox_not_a_wrong_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "drawing.png"
            Image.new("RGB", (300, 300), "white").save(image_path)
            self.assertIsNone(_crop_region(str(image_path), (0.81, 0.343, 0.54, 0.596)))

    def test_crop_region_produces_the_exact_requested_pixels_no_double_normalization(self):
        # A valid, fully in-range bbox must crop EXACTLY the pixels its
        # own normalized coordinates describe -- no extra scaling step
        # applied on top of the provider's already-normalized numbers.
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "drawing.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            crop = _crop_region(str(image_path), (0.25, 0.5, 0.5, 0.5))
            self.assertIsNotNone(crop)
            # x=0.25*200=50, w=0.5*200=100 -> width 100; y=0.5*100=50, h=0.5*100=50 -> height 50.
            self.assertEqual(crop.size, (100, 50))

    def test_populate_crop_refs_never_guesses_a_crop_for_an_invalid_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            bad_bbox_entity = visual_finding_to_entity(
                _finding("yellow car", bbox=(0.81, 0.343, 0.54, 0.596), finding_id="vf_car"))
            updated = populate_crop_refs([bad_bbox_entity], str(image_path), case_dir)
            self.assertIsNone(updated[0].crop_ref)
            # The raw (invalid) bbox itself is still preserved on the entity --
            # never discarded, only refused as a crop source.
            self.assertEqual(updated[0].bbox, (0.81, 0.343, 0.54, 0.596))


class VerifierApplicationTests(unittest.TestCase):
    def test_verified_status_applied(self):
        e = visual_finding_to_entity(_finding("dog"))
        verifier = CallableVisualVerifier(fn=lambda p, ent: VerificationResult(status="verified"))
        updated = apply_verifier_to_entities([e], "fake.png", verifier)
        self.assertEqual(updated[0].case_verification_status, "verified")
        self.assertEqual(updated[0].model_validation_status, "VALIDATED")  # untouched

    def test_uncertain_status_applied(self):
        e = visual_finding_to_entity(_finding("dog"))
        verifier = CallableVisualVerifier(fn=lambda p, ent: VerificationResult(status="uncertain"))
        updated = apply_verifier_to_entities([e], "fake.png", verifier)
        self.assertEqual(updated[0].case_verification_status, "uncertain")

    def test_rejected_status_preserves_original_provenance(self):
        f = _finding("dog", validation_status="VALIDATED")
        e = visual_finding_to_entity(f)
        verifier = CallableVisualVerifier(fn=lambda p, ent: VerificationResult(status="rejected"))
        updated = apply_verifier_to_entities([e], "fake.png", verifier)
        rejected = updated[0]
        self.assertEqual(rejected.case_verification_status, "rejected")
        # Original detector provenance is fully intact -- nothing deleted.
        self.assertEqual(rejected.detector, f.detector)
        self.assertEqual(rejected.confidence, f.confidence)
        self.assertEqual(rejected.bbox, f.bbox)
        self.assertEqual(rejected.model_validation_status, "VALIDATED")

    def test_out_of_vocabulary_verifier_status_falls_back_to_uncertain(self):
        e = visual_finding_to_entity(_finding("dog"))
        verifier = CallableVisualVerifier(fn=lambda p, ent: VerificationResult(status="definitely_true"))
        updated = apply_verifier_to_entities([e], "fake.png", verifier)
        self.assertEqual(updated[0].case_verification_status, "uncertain")

    def test_verifier_exception_on_one_entity_leaves_it_unchanged_not_fatal(self):
        # A real verifier can raise (missing config, a transient request
        # error, ...) -- must never take down verification of the rest
        # of the batch, and must never fabricate a status for the entity
        # that failed.
        def flaky(p, ent):
            if ent.canonical_label == "cat":
                raise RuntimeError("simulated verifier failure")
            return VerificationResult(status="verified")

        dog = visual_finding_to_entity(_finding("dog"))
        cat = visual_finding_to_entity(_finding("cat", finding_id="vf_cat"))
        updated = apply_verifier_to_entities([dog, cat], "fake.png", CallableVisualVerifier(fn=flaky))
        by_label = {e.canonical_label: e for e in updated}
        self.assertEqual(by_label["dog"].case_verification_status, "verified")
        self.assertEqual(by_label["cat"].case_verification_status, "unreviewed")  # untouched, not guessed


class UpdateEntitiesPersistenceTests(unittest.TestCase):
    def test_update_entities_preserves_findings_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            from doar.visual_evidence import save_detections
            f = _finding("dog")
            e = visual_finding_to_entity(f)
            save_detections(tmp, [f], entities=[e])
            verified = VisualEntity.from_dict({**e.to_dict(), "case_verification_status": "verified"})
            update_entities(tmp, [verified])
            doc = json.loads((Path(tmp) / "detections.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["entities"][0]["case_verification_status"], "verified")
            self.assertEqual(len(doc["findings"]), 1)
            self.assertEqual(doc["status"], "available")

    def test_update_entities_is_a_no_op_when_detections_json_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            update_entities(tmp, [])  # must not raise
            self.assertFalse((Path(tmp) / "detections.json").exists())


class RuleSafetyWithObserverEntitiesTests(unittest.TestCase):
    def test_observer_entities_never_reach_the_rule_engine(self):
        source = (ROOT / "src" / "doar" / "rule_engine_v2.py").read_text(encoding="utf-8")
        self.assertNotIn("visual_observer", source)
        self.assertNotIn("visual_entity", source)

    def test_full_scan_with_observer_and_verifier_never_triggers_a_rule_from_unknown_entities(self):
        candidates = [VisualObserverCandidate(label="kite", entity_type="object",
                                               bbox=(0.2, 0.2, 0.1, 0.1), confidence=0.9, source_note="test")]
        observer = CallableVisualObserver(fn=lambda p: candidates)
        # Even a verifier that marks EVERYTHING "verified" must not cause
        # an observer-only (no backing VisualFinding) entity to trigger a
        # rule -- it structurally never reaches the rule engine.
        verifier = CallableVisualVerifier(fn=lambda p, ent: VerificationResult(status="verified"))
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=_predict_fns(), observer=observer, verifier=verifier)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            triggered = [r for r in analysis["rule_evaluations"]
                         if r.get("visual_evidence_sourced") and r["status"] == "weak_support"]
            self.assertEqual(triggered, [])
            entities = load_entities(case_dir)
            kite = next(e for e in entities if e.canonical_label == "kite")
            self.assertEqual(kite.case_verification_status, "verified")
            self.assertEqual(kite.model_validation_status, "UNKNOWN")

    def test_observer_and_verifier_omitted_by_default_zero_behavior_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            findings = run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=_predict_fns())
            entities = load_entities(case_dir)
            self.assertEqual(len(entities), len(findings))
            self.assertTrue(all(e.source == "initial_scan" for e in entities))


class SearchAcrossObserverEntitiesTests(unittest.TestCase):
    def _analysis(self):
        return {
            "quality": {"quality_status": "supported", "supported": True},
            "composition": {}, "colour": {"meaningful_colours": []}, "emotion": {"status": "unavailable"},
            "rule_evaluations": [], "evidence": [], "concerns": [],
        }

    def test_existing_visualentity_search_still_works_after_this_phase(self):
        f = _finding("dog")
        e = visual_finding_to_entity(f)
        with tempfile.TemporaryDirectory() as tmp:
            from doar.visual_evidence import save_detections
            save_detections(tmp, [f], entities=[e])
            resp = answer_with_visual_grounding(tmp, "is there a dog?", self._analysis())
            self.assertEqual(resp["availability"], "available")


class TechnicalViewRendersCropsTests(unittest.TestCase):
    def test_technical_view_renders_entity_crops_without_error(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("streamlit.testing.v1.AppTest not available")
        candidates = [VisualObserverCandidate(label="kite", entity_type="object",
                                               bbox=(0.2, 0.2, 0.1, 0.1), confidence=0.7, source_note="test")]
        observer = CallableVisualObserver(fn=lambda p: candidates)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            run_and_persist_initial_scan(
                case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                model_predict_fns=_predict_fns(), observer=observer)
            image = Image.new("RGB", (200, 200), "white")
            image.save(case_dir / "drawing.png")
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            subheaders = [s.value for s in at.subheader]
            self.assertIn("Rich visual entities (Visual Knowledge V2)", subheaders)


class RealObserverShadowModeIntegrationTests(unittest.TestCase):
    """Same guarantees as RuleSafetyWithObserverEntitiesTests, but wired
    through the REAL `OpenAIVisualObserver` class (with an injected
    `request_fn` test seam -- never a real network call) instead of
    `CallableVisualObserver`, to close the loop for the actual provider
    class this phase adds."""

    _KEY_VAR = "DOAR_TEST_OPENAI_API_KEY_UNUSED_INTEGRATION"

    def _fake_response(self, candidates):
        body = {"id": "resp_it_1", "choices": [{"message": {"content": json.dumps({"candidates": candidates})}}]}
        return json.dumps(body).encode("utf-8")

    def test_real_observer_candidate_never_reaches_the_rule_engine(self):
        candidates_payload = [{"label": "kite", "alternative_labels": [], "entity_type": "object",
                                "bbox": [0.2, 0.2, 0.1, 0.1], "count": 1, "confidence": 0.9}]
        observer = OpenAIVisualObserver(api_key_env_var=self._KEY_VAR,
                                         request_fn=lambda *a: self._fake_response(candidates_payload))
        verifier = CallableVisualVerifier(fn=lambda p, ent: VerificationResult(status="verified"))
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            with mock.patch.dict(os.environ, {self._KEY_VAR: "sk-fake"}):
                run_and_persist_initial_scan(
                    case_dir, str(image_path), eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                    model_predict_fns=_predict_fns(), observer=observer, verifier=verifier)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            triggered = [r for r in analysis["rule_evaluations"]
                         if r.get("visual_evidence_sourced") and r["status"] == "weak_support"]
            self.assertEqual(triggered, [])
            entities = load_entities(case_dir)
            kite = next(e for e in entities if e.canonical_label == "kite")
            # Even verified by the (fake, permissive) verifier, the entity
            # has no backing VisualFinding, so it stays structurally out
            # of the rule engine -- "verified" only ever changes
            # case_verification_status, never model_validation_status.
            self.assertEqual(kite.case_verification_status, "verified")
            self.assertEqual(kite.model_validation_status, "UNKNOWN")
            self.assertIn("openai", kite.detector)

    def test_ai_candidate_stays_candidate_unless_explicitly_verified(self):
        candidates_payload = [{"label": "kite", "alternative_labels": [], "entity_type": "object",
                                "bbox": [0.2, 0.2, 0.1, 0.1], "count": 1, "confidence": 0.9}]
        observer = OpenAIVisualObserver(api_key_env_var=self._KEY_VAR,
                                         request_fn=lambda *a: self._fake_response(candidates_payload))
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            with mock.patch.dict(os.environ, {self._KEY_VAR: "sk-fake"}):
                # No verifier supplied -- the candidate must stay "unreviewed", never auto-verified.
                run_and_persist_initial_scan(
                    case_dir, str(image_path), eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                    model_predict_fns=_predict_fns(), observer=observer)
            entities = load_entities(case_dir)
            kite = next(e for e in entities if e.canonical_label == "kite")
            self.assertEqual(kite.case_verification_status, "unreviewed")

    def test_observer_failure_does_not_break_base_doar_analysis(self):
        # No API key configured -- the observer raises VisualObserverConfigurationError.
        # The base analysis (local detector findings, analysis.json, entities) must still
        # be produced -- a real observer's unavailability is never a reason to lose the
        # rest of the scan.
        observer = OpenAIVisualObserver(api_key_env_var=self._KEY_VAR, request_fn=lambda *a: b"unreachable")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(self._KEY_VAR, None)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    findings = run_and_persist_initial_scan(
                        case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                        model_predict_fns=_predict_fns(), observer=observer)
            self.assertTrue(any("Visual observer failed" in str(w.message) for w in caught))
            self.assertTrue((case_dir / "analysis.json").exists())
            self.assertGreater(len(findings), 0)
            entities = load_entities(case_dir)
            self.assertTrue(all(e.source == "initial_scan" for e in entities))  # no observer candidates merged

    def test_observer_configuration_error_is_the_exact_raised_type(self):
        observer = OpenAIVisualObserver(api_key_env_var=self._KEY_VAR, request_fn=lambda *a: b"{}")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self._KEY_VAR, None)
            with self.assertRaises(VisualObserverConfigurationError):
                observer.analyze("fake.png")


class RealGeminiObserverShadowModeIntegrationTests(unittest.TestCase):
    """Same guarantees as RealObserverShadowModeIntegrationTests, but
    wired through the REAL `GeminiVisualObserver` class (injected
    `request_fn` test seam -- never a real network call), proving the
    DOAR Gemini Observer phase's own shadow-mode requirement: Gemini-only
    evidence stays psychologically rule-ineligible, exactly like every
    other observer-only entity, because it has no backing VisualFinding."""

    _KEY_VAR = "DOAR_TEST_GEMINI_API_KEY_UNUSED_INTEGRATION"

    def _fake_response(self, candidates):
        body = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({"candidates": candidates})}],
                                         "role": "model"}, "finishReason": "STOP", "index": 0}],
            "modelVersion": "gemini-3.6-flash", "responseId": "gemini_resp_it_1",
        }
        return json.dumps(body).encode("utf-8")

    def test_gemini_only_candidate_never_reaches_the_rule_engine(self):
        candidates_payload = [{"label": "kite", "alternative_labels": [], "entity_type": "object",
                                "bbox": [0.2, 0.2, 0.1, 0.1], "count": 1, "confidence": 0.9}]
        observer = GeminiVisualObserver(api_key_env_var=self._KEY_VAR,
                                         request_fn=lambda *a: self._fake_response(candidates_payload))
        verifier = CallableVisualVerifier(fn=lambda p, ent: VerificationResult(status="verified"))
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            with mock.patch.dict(os.environ, {self._KEY_VAR: "fake-gemini-key"}):
                run_and_persist_initial_scan(
                    case_dir, str(image_path), eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                    model_predict_fns=_predict_fns(), observer=observer, verifier=verifier)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            triggered = [r for r in analysis["rule_evaluations"]
                         if r.get("visual_evidence_sourced") and r["status"] == "weak_support"]
            self.assertEqual(triggered, [])
            entities = load_entities(case_dir)
            kite = next(e for e in entities if e.canonical_label == "kite")
            # Even verified by the (fake, permissive) verifier, the entity
            # has no backing VisualFinding, so it stays structurally out
            # of the rule engine -- "verified" only ever changes
            # case_verification_status, never model_validation_status.
            self.assertEqual(kite.case_verification_status, "verified")
            self.assertEqual(kite.model_validation_status, "UNKNOWN")
            self.assertIn("gemini", kite.detector)

    def test_gemini_candidate_stays_candidate_unless_explicitly_verified(self):
        candidates_payload = [{"label": "kite", "alternative_labels": [], "entity_type": "object",
                                "bbox": [0.2, 0.2, 0.1, 0.1], "count": 1, "confidence": 0.9}]
        observer = GeminiVisualObserver(api_key_env_var=self._KEY_VAR,
                                         request_fn=lambda *a: self._fake_response(candidates_payload))
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            with mock.patch.dict(os.environ, {self._KEY_VAR: "fake-gemini-key"}):
                # No verifier supplied -- the candidate must stay "unreviewed", never auto-verified.
                run_and_persist_initial_scan(
                    case_dir, str(image_path), eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                    model_predict_fns=_predict_fns(), observer=observer)
            entities = load_entities(case_dir)
            kite = next(e for e in entities if e.canonical_label == "kite")
            self.assertEqual(kite.case_verification_status, "unreviewed")

    def test_gemini_observer_failure_does_not_break_base_doar_analysis(self):
        # No API key configured -- the observer raises VisualObserverConfigurationError.
        # The base analysis (local detector findings, analysis.json, entities) must still
        # be produced -- a real observer's unavailability is never a reason to lose the
        # rest of the scan.
        observer = GeminiVisualObserver(api_key_env_var=self._KEY_VAR, request_fn=lambda *a: b"unreachable")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(self._KEY_VAR, None)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    findings = run_and_persist_initial_scan(
                        case_dir, "fake.png", eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                        model_predict_fns=_predict_fns(), observer=observer)
            self.assertTrue(any("Visual observer failed" in str(w.message) for w in caught))
            self.assertTrue((case_dir / "analysis.json").exists())
            self.assertGreater(len(findings), 0)
            entities = load_entities(case_dir)
            self.assertTrue(all(e.source == "initial_scan" for e in entities))  # no observer candidates merged

    def test_gemini_configuration_error_is_the_exact_raised_type(self):
        observer = GeminiVisualObserver(api_key_env_var=self._KEY_VAR, request_fn=lambda *a: b"{}")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self._KEY_VAR, None)
            with self.assertRaises(VisualObserverConfigurationError):
                observer.analyze("fake.png")


class RealGeminiVerifierShadowModeIntegrationTests(unittest.TestCase):
    """The Visual Verifier phase's own CRITICAL requirement: even a
    Gemini-only entity the REAL verifier marks "verified" must remain
    psychologically rule-ineligible -- it still has no backing
    VisualFinding, exactly like an unverified observer-only entity.
    Verification changes case correctness (case_verification_status),
    never rule eligibility (which is governed structurally, not by any
    status field)."""

    _OBSERVER_KEY_VAR = "DOAR_TEST_GEMINI_API_KEY_UNUSED_VERIFIER_IT"
    _VERIFIER_KEY_VAR = "DOAR_TEST_GEMINI_VERIFIER_API_KEY_UNUSED_VERIFIER_IT"

    def _fake_observer_response(self, candidates):
        body = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({"candidates": candidates})}],
                                         "role": "model"}, "finishReason": "STOP", "index": 0}],
            "modelVersion": "gemini-3.6-flash", "responseId": "gemini_obs_resp_it_1",
        }
        return json.dumps(body).encode("utf-8")

    def _fake_verifier_response(self, label, alternative_labels=()):
        inner = {"label": label, "alternative_labels": list(alternative_labels), "confidence": 0.95}
        body = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(inner)}], "role": "model"},
                             "finishReason": "STOP", "index": 0}],
            "modelVersion": "gemini-3.5-flash-lite", "responseId": "gemini_verifier_resp_it_1",
        }
        return json.dumps(body).encode("utf-8")

    def test_verified_gemini_only_entity_never_reaches_the_rule_engine(self):
        candidates_payload = [{"label": "kite", "alternative_labels": [], "entity_type": "object",
                                "bbox": [0.2, 0.2, 0.3, 0.3], "count": 1, "confidence": 0.9}]
        observer = GeminiVisualObserver(api_key_env_var=self._OBSERVER_KEY_VAR,
                                         request_fn=lambda *a: self._fake_observer_response(candidates_payload))
        # The REAL verifier, independently confirming the SAME region as "kite"
        # -- a genuine VERIFIED outcome, not a stub that always says verified.
        verifier = GeminiVisualVerifier(api_key_env_var=self._VERIFIER_KEY_VAR,
                                         request_fn=lambda *a: self._fake_verifier_response("kite"))
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            with mock.patch.dict(os.environ, {self._OBSERVER_KEY_VAR: "fake-observer-key",
                                               self._VERIFIER_KEY_VAR: "fake-verifier-key"}):
                run_and_persist_initial_scan(
                    case_dir, str(image_path), eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                    model_predict_fns=_predict_fns(), observer=observer, verifier=verifier)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            triggered = [r for r in analysis["rule_evaluations"]
                         if r.get("visual_evidence_sourced") and r["status"] == "weak_support"]
            self.assertEqual(triggered, [])
            entities = load_entities(case_dir)
            kite = next(e for e in entities if e.canonical_label == "kite")
            # Genuinely VERIFIED by the real, independent verifier --
            # and STILL structurally excluded from the rule engine.
            self.assertEqual(kite.case_verification_status, "verified")
            self.assertEqual(kite.model_validation_status, "UNKNOWN")

    def test_verifier_failure_does_not_break_base_doar_analysis(self):
        # No verifier API key configured -- GeminiVisualVerifier.verify()
        # raises VisualObserverConfigurationError for every entity. The
        # base analysis and the observer's own candidates must still be
        # produced -- apply_verifier_to_entities catches this per-entity.
        candidates_payload = [{"label": "kite", "alternative_labels": [], "entity_type": "object",
                                "bbox": [0.2, 0.2, 0.3, 0.3], "count": 1, "confidence": 0.9}]
        observer = GeminiVisualObserver(api_key_env_var=self._OBSERVER_KEY_VAR,
                                         request_fn=lambda *a: self._fake_observer_response(candidates_payload))
        verifier = GeminiVisualVerifier(api_key_env_var=self._VERIFIER_KEY_VAR, request_fn=lambda *a: b"unreachable")
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = _real_case(tmp)
            image_path = case_dir / "drawing.png"
            Image.new("RGB", (200, 200), "white").save(image_path)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ[self._OBSERVER_KEY_VAR] = "fake-observer-key"
                os.environ.pop(self._VERIFIER_KEY_VAR, None)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    run_and_persist_initial_scan(
                        case_dir, str(image_path), eye_entry=_eye_entry(), registry_v2=FAKE_REGISTRY_V2,
                        model_predict_fns=_predict_fns(), observer=observer, verifier=verifier)
            self.assertTrue(any("Visual verifier failed" in str(w.message) for w in caught))
            self.assertTrue((case_dir / "analysis.json").exists())
            entities = load_entities(case_dir)
            kite = next(e for e in entities if e.canonical_label == "kite")
            # Observer candidate still merged in -- verifier failure never
            # loses it -- but stays "unreviewed" since verification itself failed.
            self.assertEqual(kite.case_verification_status, "unreviewed")


if __name__ == "__main__":
    unittest.main()
