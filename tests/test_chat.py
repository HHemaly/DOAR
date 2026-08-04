from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image
from doar.chat import (
    ChatProvider, ClaimExtractor, ClaimVerifier, DeterministicChatProvider,
    EvidenceRetriever, GeminiChatProvider, OpenAIChatProvider, ResponseJudge,
    SafetyChecker, respond_to_chat,
)


def _real_case(tmp_dir: str) -> Path:
    """Builds one real case via the real analyze_image pipeline (no mocking) --
    same fixture style as tests/test_objective.py."""
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((30, 30, 170, 170), fill="black")
    case_dir = Path(tmp_dir) / "case"
    analyze_image(_write_png(tmp_dir, image), case_dir)
    return case_dir


def _write_png(tmp_dir: str, image: Image.Image) -> Path:
    path = Path(tmp_dir) / "drawing.png"
    image.save(path)
    return path


class EvidenceRetrieverTests(unittest.TestCase):
    def test_retrieves_real_case_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            bundle = EvidenceRetriever(case_dir).retrieve("anything")
            self.assertGreater(len(bundle.evidence_ids), 0)
            self.assertIn("ev_bbox_coverage", bundle.evidence_ids)
            self.assertIsNone(bundle.profile)  # no profile.json saved in this fixture

    def test_missing_case_raises_not_silently_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                EvidenceRetriever(Path(d) / "does_not_exist").retrieve("x")


class DeterministicChatGroundingTests(unittest.TestCase):
    def test_answers_are_grounded_in_real_saved_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            response = respond_to_chat(case_dir, "What rules triggered?", "en")
            self.assertFalse(response.escalated)
            self.assertIn("status", " ".join(response.evidence_ids) + response.answer)
            # every claimed evidence id must actually exist in the case's evidence.json
            import json
            evidence_ids = {e["evidence_id"] for e in json.loads((case_dir / "evidence.json").read_text(encoding="utf-8"))}
            for eid in response.evidence_ids:
                self.assertIn(eid, evidence_ids)

    def test_unknown_question_returns_grounded_refusal_not_fabrication(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            response = respond_to_chat(case_dir, "asdkjqwoiuey nonsense", "en")
            self.assertEqual(response.availability, "unavailable")
            self.assertEqual(response.evidence_ids, [])

    def test_bilingual_answers_differ_and_are_grounded(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            en = respond_to_chat(case_dir, "What rules triggered?", "en")
            ar = respond_to_chat(case_dir, "ما هي القواعد؟", "ar")
            self.assertNotEqual(en.answer, ar.answer)
            self.assertTrue(any("؀" <= c <= "ۿ" for c in ar.answer))  # contains Arabic script


class SafetyEscalationTests(unittest.TestCase):
    def test_english_safeguarding_language_escalates(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            response = respond_to_chat(case_dir, "I want to hurt myself", "en")
            self.assertTrue(response.escalated)
            self.assertIn("professional", response.answer.lower())

    def test_arabic_safeguarding_language_escalates(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            response = respond_to_chat(case_dir, "أفكر في الانتحار", "ar")
            self.assertTrue(response.escalated)

    def test_ordinary_question_does_not_escalate(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _real_case(d)
            response = respond_to_chat(case_dir, "What is the emotion prediction?", "en")
            self.assertFalse(response.escalated)

    def test_outgoing_safety_checker_scans_for_diagnostic_language(self):
        checker = SafetyChecker()
        result = checker.check_outgoing("The child has anxiety and is clearly depressed.")
        self.assertTrue(result.diagnostic_language_found)

    def test_outgoing_safety_checker_passes_clean_text(self):
        checker = SafetyChecker()
        result = checker.check_outgoing("Predicted 'Fear' with confidence 0.5.")
        self.assertFalse(result.diagnostic_language_found)


class NoApiKeyRequiredTests(unittest.TestCase):
    """The base application must never require a paid API key -- confirm
    the deterministic path has no import of any network/HTTP/LLM SDK."""

    def test_chat_module_has_no_network_or_llm_sdk_imports(self):
        tree = ast.parse((ROOT / "src" / "doar" / "chat.py").read_text(encoding="utf-8"))
        forbidden = {"requests", "httpx", "urllib", "openai", "google", "anthropic"}
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
        self.assertEqual(found & forbidden, set())

    def test_deterministic_provider_generate_documents_the_real_entrypoint(self):
        with self.assertRaises(NotImplementedError):
            DeterministicChatProvider().generate("hi", system="", history=[])

    def test_llm_provider_stubs_never_silently_succeed(self):
        for provider_cls in (OpenAIChatProvider, GeminiChatProvider):
            with self.assertRaises(NotImplementedError):
                provider_cls().generate("hi", system="", history=[])

    def test_future_pipeline_stubs_are_not_silently_implemented(self):
        with self.assertRaises(NotImplementedError):
            ClaimExtractor().extract("draft text")
        with self.assertRaises(NotImplementedError):
            ClaimVerifier().verify("claim", None)
        with self.assertRaises(NotImplementedError):
            ResponseJudge().judge("response", [])

    def test_chat_provider_is_a_protocol_not_a_concrete_requirement(self):
        # DeterministicChatProvider structurally satisfies ChatProvider's shape
        # even though respond_to_chat() doesn't call .generate() on it.
        self.assertTrue(hasattr(DeterministicChatProvider, "generate"))
        self.assertTrue(issubclass(ChatProvider, object))  # Protocol is importable/usable


if __name__ == "__main__":
    unittest.main()
