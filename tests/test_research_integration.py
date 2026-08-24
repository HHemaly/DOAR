"""Focused tests for the experiment-to-runtime integration layer:
research_runtime.py, experiment_manifest.py, providers.py, and
human_interaction.GeminiGlobalObserver/run_gemini_global_observation.

No live API/network access is required -- GeminiGlobalObserver is only
exercised with a stub request_fn."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

from doar import case_interpretation as ci
from doar import human_interaction as hi
from doar import providers
from doar import research_runtime as rr
from doar.experiment_manifest import ExperimentManifest, example_mobilenet_v3_small_development_manifest
from doar.live_case_bundle import build_live_case_bundle
from doar.timed_analysis import analyze_image_with_timing


def _build_case(tmp_dir: str) -> Path:
    case_dir = Path(tmp_dir) / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "drawing.png"
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
    image.save(path)
    analyze_image_with_timing(str(path), str(case_dir), None)
    (case_dir / "detections.json").write_text(json.dumps({"entities": []}), encoding="utf-8")
    return case_dir


class ResearchRuntimeConfigurationTests(unittest.TestCase):
    def test_resolves_all_five_roles_with_a_valid_status(self):
        cfg = rr.resolve_research_runtime_configuration()
        for component in cfg.components():
            self.assertIn(component.status, rr.COMPONENT_STATUSES)
            self.assertTrue(component.implementation.strip())

    def test_emotion_provider_is_development_not_final_when_e1_checkpoint_is_used(self):
        cfg = rr.resolve_research_runtime_configuration()
        if cfg.emotion_provider.checkpoint and "E1_visual_representation" in str(cfg.emotion_provider.checkpoint):
            self.assertEqual(cfg.emotion_provider.status, "DEVELOPMENT")
            self.assertIn("E1_DEVELOPMENT", cfg.emotion_provider.provenance.get("model_version", ""))

    def test_evidence_aggregator_is_independent_family_ratio_with_no_invented_weights(self):
        cfg = rr.resolve_research_runtime_configuration()
        self.assertEqual(cfg.evidence_aggregator.model_or_artifact, "INDEPENDENT_FAMILY_RATIO")
        self.assertIn("planned_future_candidates", cfg.evidence_aggregator.provenance)

    def test_to_dict_is_json_serializable(self):
        cfg = rr.resolve_research_runtime_configuration()
        json.dumps(cfg.to_dict())  # must not raise

    def test_invalid_status_is_rejected(self):
        with self.assertRaises(ValueError):
            rr.ResearchComponent(role="x", experiment=None, implementation="x", model_or_artifact=None,
                                 status="WINNER")


class ExperimentManifestTests(unittest.TestCase):
    def test_example_manifest_round_trips(self):
        manifest = example_mobilenet_v3_small_development_manifest()
        restored = ExperimentManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored, manifest)

    def test_example_manifest_is_development_not_final(self):
        manifest = example_mobilenet_v3_small_development_manifest()
        self.assertEqual(manifest.status, "DEVELOPMENT")
        self.assertEqual(manifest.class_mapping, ("Angry", "Fear", "Happy", "Sad"))

    def test_invalid_status_is_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentManifest(
                experiment_id="x", task="x", model_name="x", artifact_version="x", checkpoint_path="x",
                class_mapping=(), preprocessing={}, validation_metrics={}, split_manifest_sha256=None,
                training_seed=None, status="WINNER")

    def test_documented_manifest_file_matches_the_example(self):
        path = ROOT / "resources" / "experiment_manifests" / "e1_mobilenet_v3_small_development.json"
        self.assertTrue(path.exists())
        on_disk = ExperimentManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(on_disk, example_mobilenet_v3_small_development_manifest())


class ProviderInterfaceTests(unittest.TestCase):
    def test_default_semantic_concept_provider_matches_direct_call(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = build_live_case_bundle(_build_case(d))
            via_provider = providers.DefaultSemanticConceptProvider().build_concepts(bundle)
            direct = ci.build_visual_concepts(bundle)
            self.assertEqual(via_provider, direct)

    def test_production_emotion_provider_returns_the_standard_shape(self):
        result = providers.ProductionEmotionProvider(checkpoint=None).predict("fake.png")
        self.assertIn(result["status"], ("unavailable", "failed"))
        self.assertIn("probabilities", result)

    def test_reexported_pluggable_interfaces_are_the_same_objects(self):
        self.assertIs(providers.EvidenceAggregator, ci.EvidenceAggregator)
        self.assertIs(providers.GlobalImageRepresentationProvider, ci.GlobalImageRepresentationProvider)


class GeminiGlobalObserverTests(unittest.TestCase):
    def _stub_request_fn(self, parsed: dict):
        def _fn(api_key, model, payload, timeout):
            return json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(parsed)}]}}]}).encode()
        return _fn

    def test_observe_parses_structured_output(self):
        parsed = {
            "overall_scene": "A bright outdoor scene.", "overall_visual_tone": "cheerful",
            "expressive_descriptors": ["bright", "lively"], "salient_relationships": ["figures grouped together"],
            "important_visual_observations": ["sun", "flowers"],
            "candidate_concern_hypotheses": [
                {"domain": "positive_affect_or_social_engagement", "reason": "bright colors", "observations": ["sun"]},
            ],
            "uncertainty": "Hard to tell facial expression.",
        }
        with tempfile.TemporaryDirectory() as d:
            image_path = Path(d) / "sample.png"
            Image.new("RGB", (50, 50), "white").save(image_path)
            observer = hi.GeminiGlobalObserver(api_key="fake", request_fn=self._stub_request_fn(parsed))
            result = observer.observe(str(image_path))
        self.assertEqual(result["overall_scene"], parsed["overall_scene"])
        self.assertEqual(len(result["candidate_concern_hypotheses"]), 1)

    def test_run_gemini_global_observation_unavailable_with_no_observer(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = build_live_case_bundle(_build_case(d))
            result = hi.run_gemini_global_observation(bundle, observer=None)
            self.assertEqual(result["status"], "unavailable")

    def _bundle_with_resolvable_image(self, tmp_dir: str) -> dict:
        """build_live_case_bundle sets image.relative_path=None when the
        case lives outside the repo ROOT (the correct, documented
        degrade-safely behavior for a tempdir case -- see live_case_
        bundle.py) -- these tests need a truthy relative_path so run_
        gemini_global_observation actually reaches the observer; the
        observers here never open the file, so any real, existing path
        under ROOT (case_dir.png itself) is sufficient."""
        bundle = build_live_case_bundle(_build_case(tmp_dir))
        return {**bundle, "image": {**bundle["image"], "relative_path": "drawing.png"}}

    def test_run_gemini_global_observation_error_is_sanitized(self):
        class _RaisingObserver:
            def observe(self, image_path):
                raise RuntimeError("network failure api_key=AIzaSyFAKESECRETVALUE1234567890abcdef")

        with tempfile.TemporaryDirectory() as d:
            bundle = self._bundle_with_resolvable_image(d)
            result = hi.run_gemini_global_observation(bundle, observer=_RaisingObserver())
            self.assertEqual(result["status"], "error")
            self.assertNotIn("AIzaSyFAKESECRETVALUE1234567890abcdef", result["reason"])

    def test_end_to_end_gemini_candidates_never_inflate_governed_support(self):
        """The critical safety property, exercised through the real
        GeminiGlobalObserver parsing path (stub network only): a dramatic
        Gemini-suggested domain never becomes SUPPORTED unless DOAR's own
        governed evidence already supports it."""
        parsed = {
            "overall_scene": "x", "overall_visual_tone": "x", "expressive_descriptors": [],
            "salient_relationships": [], "important_visual_observations": [],
            "candidate_concern_hypotheses": [
                {"domain": "maltreatment_or_safety_concern", "reason": "alarming imagery", "observations": []},
                {"domain": "depressive_or_low_mood_related", "reason": "dark and heavy", "observations": []},
            ],
            "uncertainty": "",
        }
        class _StubObserver:
            """Bypasses image I/O entirely -- GeminiGlobalObserver's own
            JSON-parsing path is separately covered by
            test_observe_parses_structured_output."""
            def observe(self, image_path):
                return {**parsed, "candidate_concern_hypotheses": [
                    {"domain": h["domain"], "reason": h["reason"], "observations": h["observations"]}
                    for h in parsed["candidate_concern_hypotheses"]
                ]}

        with tempfile.TemporaryDirectory() as d:
            bundle = self._bundle_with_resolvable_image(d)
            gem = hi.run_gemini_global_observation(bundle, observer=_StubObserver())
            self.assertEqual(gem["status"], "ok")
            interp = ci.build_case_interpretation(
                bundle, gemini_global_observation=gem["observation"],
                gemini_concern_candidates=gem["observation"]["candidate_concern_hypotheses"])
            by_domain = {c.domain: c for c in interp.gemini_candidates}
            self.assertEqual(by_domain["maltreatment_or_safety_concern"].verification_status, "NOT_CURRENTLY_ASSESSED")
            self.assertNotEqual(by_domain["depressive_or_low_mood_related"].verification_status, "SUPPORTED")


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class FullAnalysisUiTests(unittest.TestCase):
    def test_run_full_analysis_button_and_categorized_feedback_render(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=90)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            psych = at.tabs[1]
            button_labels = [b.label for b in psych.button]
            self.assertIn("Run Full Analysis", button_labels)
            radio_labels = [r.label for r in psych.radio]
            self.assertIn("Overall interpretation", radio_labels)
            self.assertIn("Emotion interpretation", radio_labels)
            self.assertIn("Concern-domain interpretation", radio_labels)
            self.assertIn("Explanation usefulness", radio_labels)

    def test_technical_view_shows_research_runtime_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d)
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=90)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            tech = at.tabs[2]
            headers = " ".join(h.value for h in tech.header)
            self.assertIn("Research runtime configuration", headers)
            self.assertIn("Traceability", headers)


if __name__ == "__main__":
    unittest.main()
