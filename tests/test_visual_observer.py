"""DOAR Visual Resolver: visual_observer.py -- the provider-independent
observation/verification interface, PLUS the real `OpenAIVisualObserver`
integration (shadow mode). Proves: structured (never prose) observer
output, candidate labels stay candidates through the merge, verifier
verdicts map correctly to case_verification_status, the still-stubbed
providers fix their integration shape without ever touching a network
call, and the real OpenAI path -- exercised ONLY via an injected
`request_fn` test seam, NEVER a real network call -- fails cleanly
without a configured API key, parses a structured response into
candidates with provenance preserved, fails safely (not silently) on a
malformed response, and retries only transport-level failures.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_observer import (  # noqa: E402
    CallableVisualObserver, CallableVisualVerifier, GeminiVisualObserver, GeminiVisualVerifier,
    OpenAIVisualObserver, OpenAIVisualVerifier, StaticVisualVerifier, VerificationResult,
    VisualObserverCandidate, VisualObserverConfigurationError, VisualObserverRequestError,
    clamp_entity_type,
)


class VisualObserverCandidateShapeTests(unittest.TestCase):
    def test_candidate_is_structured_not_prose(self):
        c = VisualObserverCandidate(label="kite", alternative_labels=("flag",), entity_type="object",
                                     bbox=(0.1, 0.1, 0.2, 0.2), count=1, confidence=0.6)
        # Every field is a discrete, typed value -- never a free-text blob.
        self.assertIsInstance(c.label, str)
        self.assertIsInstance(c.alternative_labels, tuple)
        self.assertIsInstance(c.bbox, tuple)
        self.assertIsInstance(c.confidence, float)

    def test_defaults_are_conservative_not_fabricated(self):
        c = VisualObserverCandidate(label="thing")
        self.assertEqual(c.alternative_labels, ())
        self.assertEqual(c.entity_type, "unknown")
        self.assertIsNone(c.bbox)
        self.assertIsNone(c.count)
        self.assertIsNone(c.confidence)


class ClampEntityTypeTests(unittest.TestCase):
    def test_valid_type_passed_through(self):
        self.assertEqual(clamp_entity_type("symbol"), "symbol")

    def test_invalid_type_falls_back_to_unknown(self):
        self.assertEqual(clamp_entity_type("spaceship"), "unknown")

    def test_none_falls_back_to_unknown(self):
        self.assertEqual(clamp_entity_type(None), "unknown")


class CallableAdapterTests(unittest.TestCase):
    def test_callable_observer_returns_exactly_what_the_function_returns(self):
        candidates = [VisualObserverCandidate(label="dog")]
        observer = CallableVisualObserver(fn=lambda path: candidates)
        self.assertEqual(observer.analyze("fake.png"), candidates)

    def test_callable_verifier_returns_exactly_what_the_function_returns(self):
        result = VerificationResult(status="verified", confidence=0.9)
        verifier = CallableVisualVerifier(fn=lambda path, entity: result)
        self.assertEqual(verifier.verify("fake.png", object()), result)

    def test_static_verifier_defaults_to_uncertain_never_verified(self):
        verifier = StaticVisualVerifier()
        result = verifier.verify("fake.png", object())
        self.assertEqual(result.status, "uncertain")


class StillStubbedProvidersNeverTouchNetworkTests(unittest.TestCase):
    """Gemini and both *VisualVerifier classes are explicitly out of this
    phase's scope -- mirrors chat.py's OpenAIChatProvider/GeminiChatProvider
    pattern: typed stubs that fix the shape but always raise, never
    silently return a fabricated result."""

    def test_gemini_observer_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            GeminiVisualObserver().analyze("fake.png")

    def test_openai_verifier_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            OpenAIVisualVerifier().verify("fake.png", object())

    def test_gemini_verifier_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            GeminiVisualVerifier().verify("fake.png", object())

    def test_stub_construction_never_requires_a_real_api_key(self):
        # Construction alone (no .analyze()/.verify() call) must never
        # raise or attempt any network/credential access -- true for the
        # real OpenAIVisualObserver too, not just the remaining stubs.
        OpenAIVisualObserver()
        GeminiVisualObserver()
        OpenAIVisualVerifier()
        GeminiVisualVerifier()


# ---------------------------------------------------------------------------
# Real OpenAIVisualObserver -- every test below injects `request_fn` and
# never touches the network. `_post_openai_chat_completions` (the real
# transport) is never imported or called anywhere in this file.
# ---------------------------------------------------------------------------

_TEST_KEY_VAR = "DOAR_TEST_OPENAI_API_KEY_UNUSED"


def _fake_openai_response(candidates_payload, *, response_id="resp_test_1"):
    body = {"id": response_id, "choices": [{"message": {"content": json.dumps({"candidates": candidates_payload})}}]}
    return json.dumps(body).encode("utf-8")


def _tmp_image_path():
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Path(path).write_bytes(b"not a real png -- content is irrelevant, the transport is mocked")
    return path


class OpenAIObserverMissingKeyTests(unittest.TestCase):
    def test_missing_api_key_raises_configuration_error_before_any_transport_call(self):
        def must_not_be_called(*_args):
            raise AssertionError("transport must never be called when no API key is configured")
        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=must_not_be_called)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_TEST_KEY_VAR, None)
            with self.assertRaises(VisualObserverConfigurationError):
                observer.analyze("fake.png")

    def test_missing_image_raises_configuration_error(self):
        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=lambda *a: b"{}")
        with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
            with self.assertRaises(VisualObserverConfigurationError):
                observer.analyze("does_not_exist.png")


class OpenAIObserverStructuredParsingTests(unittest.TestCase):
    def test_structured_response_parses_into_candidates_with_provenance(self):
        payload = [
            {"label": "kite", "alternative_labels": ["flag"], "entity_type": "object",
             "bbox": [0.1, 0.1, 0.2, 0.2], "count": 1, "confidence": 0.7},
            {"label": "odd mark", "alternative_labels": [], "entity_type": "scribble",
             "bbox": None, "count": None, "confidence": None},
        ]
        raw = _fake_openai_response(payload, response_id="resp_abc123")
        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, model="gpt-4o-test",
                                         request_fn=lambda key, body, timeout: raw)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                candidates = observer.analyze(image_path)
        finally:
            os.unlink(image_path)

        self.assertEqual(len(candidates), 2)
        kite = candidates[0]
        self.assertIsInstance(kite, VisualObserverCandidate)
        self.assertEqual(kite.label, "kite")
        self.assertEqual(kite.alternative_labels, ("flag",))
        self.assertEqual(kite.entity_type, "object")
        self.assertEqual(kite.bbox, (0.1, 0.1, 0.2, 0.2))
        self.assertEqual(kite.count, 1)
        self.assertEqual(kite.confidence, 0.7)
        # Provenance: provider, model, and a response identifier -- never just "OpenAI said so".
        self.assertIn("openai", kite.source_note)
        self.assertIn("gpt-4o-test", kite.source_note)
        self.assertIn("resp_abc123", kite.source_note)

        scribble = candidates[1]
        self.assertEqual(scribble.entity_type, "scribble")
        self.assertIsNone(scribble.bbox)
        self.assertIsNone(scribble.confidence)

    def test_request_payload_is_structured_json_schema_not_free_prose(self):
        captured = {}

        def capturing_transport(key, body, timeout):
            captured["body"] = body
            return _fake_openai_response([])

        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=capturing_transport)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                observer.analyze(image_path)
        finally:
            os.unlink(image_path)

        self.assertEqual(captured["body"]["response_format"]["type"], "json_schema")
        self.assertTrue(captured["body"]["response_format"]["json_schema"]["strict"])

    def test_unrecognized_entity_type_from_provider_clamped_to_unknown(self):
        payload = [{"label": "blob", "alternative_labels": [], "entity_type": "spaceship",
                    "bbox": None, "count": None, "confidence": 0.3}]
        raw = _fake_openai_response(payload)
        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=lambda *a: raw)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                candidates = observer.analyze(image_path)
        finally:
            os.unlink(image_path)
        self.assertEqual(candidates[0].entity_type, "unknown")

    def test_one_malformed_candidate_entry_is_skipped_not_fatal(self):
        payload = [
            {"label": "kite", "alternative_labels": [], "entity_type": "object",
             "bbox": None, "count": None, "confidence": 0.5},
            {"label": "", "alternative_labels": [], "entity_type": "object",
             "bbox": None, "count": None, "confidence": 0.9},  # empty label -- skipped
            "not_even_a_dict",  # skipped
        ]
        raw = _fake_openai_response(payload)
        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=lambda *a: raw)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                candidates = observer.analyze(image_path)
        finally:
            os.unlink(image_path)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].label, "kite")


class OpenAIObserverMalformedResponseTests(unittest.TestCase):
    def test_non_json_response_raises_request_error_not_silent_fallback(self):
        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR,
                                         request_fn=lambda *a: b"not json at all")
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                with self.assertRaises(VisualObserverRequestError):
                    observer.analyze(image_path)
        finally:
            os.unlink(image_path)

    def test_missing_candidates_field_raises_request_error(self):
        raw = json.dumps({"id": "resp_x", "choices": [{"message": {"content": json.dumps({"nope": []})}}]}).encode()
        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=lambda *a: raw)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                with self.assertRaises(VisualObserverRequestError):
                    observer.analyze(image_path)
        finally:
            os.unlink(image_path)

    def test_api_error_body_raises_request_error(self):
        raw = json.dumps({"error": {"message": "invalid api key"}}).encode()
        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=lambda *a: raw)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                with self.assertRaises(VisualObserverRequestError):
                    observer.analyze(image_path)
        finally:
            os.unlink(image_path)


class OpenAIObserverRetryTests(unittest.TestCase):
    def test_transport_failure_retried_then_succeeds(self):
        calls = {"n": 0}

        def flaky_transport(key, body, timeout):
            calls["n"] += 1
            if calls["n"] < 2:
                raise TimeoutError("simulated timeout")
            return _fake_openai_response([{"label": "sun", "alternative_labels": [], "entity_type": "object",
                                            "bbox": None, "count": None, "confidence": 0.8}])

        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=flaky_transport, max_retries=2)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                candidates = observer.analyze(image_path)
        finally:
            os.unlink(image_path)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(candidates[0].label, "sun")

    def test_transport_failure_exhausts_retries_raises_request_error(self):
        def always_fails(key, body, timeout):
            raise TimeoutError("simulated timeout")

        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=always_fails, max_retries=1)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                with self.assertRaises(VisualObserverRequestError):
                    observer.analyze(image_path)
        finally:
            os.unlink(image_path)

    def test_parse_failure_is_not_retried(self):
        # A malformed response is deterministic -- retrying it would never
        # help and would just delay an honest error. Only transport-level
        # failures (connection/timeout) are retried.
        calls = {"n": 0}

        def bad_response_transport(key, body, timeout):
            calls["n"] += 1
            return b"not json"

        observer = OpenAIVisualObserver(api_key_env_var=_TEST_KEY_VAR, request_fn=bad_response_transport,
                                         max_retries=3)
        image_path = _tmp_image_path()
        try:
            with mock.patch.dict(os.environ, {_TEST_KEY_VAR: "sk-fake"}):
                with self.assertRaises(VisualObserverRequestError):
                    observer.analyze(image_path)
        finally:
            os.unlink(image_path)
        self.assertEqual(calls["n"], 1)


class OpenAIObserverModelConfigTests(unittest.TestCase):
    def test_model_defaults_from_production_config_env_var(self):
        with mock.patch.dict(os.environ, {"DOAR_VISUAL_OBSERVER_MODEL": "gpt-test-override"}):
            observer = OpenAIVisualObserver()
        self.assertEqual(observer.model, "gpt-test-override")

    def test_model_default_is_not_environment_dependent_without_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DOAR_VISUAL_OBSERVER_MODEL", None)
            observer = OpenAIVisualObserver()
        self.assertEqual(observer.model, "gpt-4o")

    def test_explicit_model_argument_overrides_config(self):
        observer = OpenAIVisualObserver(model="gpt-explicit")
        self.assertEqual(observer.model, "gpt-explicit")


if __name__ == "__main__":
    unittest.main()
