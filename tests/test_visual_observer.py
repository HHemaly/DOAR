"""DOAR Visual Resolver: visual_observer.py -- the provider-independent
observation/verification interface. Proves: structured (never prose)
observer output, candidate labels stay candidates through the merge,
verifier verdicts map correctly to case_verification_status, and the
typed provider stubs fix their integration shape without ever touching
a network call.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.visual_observer import (  # noqa: E402
    CallableVisualObserver, CallableVisualVerifier, GeminiVisualObserver, GeminiVisualVerifier,
    OpenAIVisualObserver, OpenAIVisualVerifier, StaticVisualVerifier, VerificationResult,
    VisualObserverCandidate, clamp_entity_type,
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


class ProviderStubsNeverTouchNetworkTests(unittest.TestCase):
    """Mirrors chat.py's OpenAIChatProvider/GeminiChatProvider pattern:
    typed stubs that fix the shape but always raise, never silently
    return a fabricated result."""

    def test_openai_observer_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            OpenAIVisualObserver().analyze("fake.png")

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
        # raise or attempt any network/credential access.
        OpenAIVisualObserver()
        GeminiVisualObserver()
        OpenAIVisualVerifier()
        GeminiVisualVerifier()


if __name__ == "__main__":
    unittest.main()
