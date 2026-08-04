"""
chat.py -- grounded follow-up chat, provider-neutral (LLM_GROUNDING_AND_SAFETY_DESIGN.md).

No paid API key is required to run this module: `DeterministicChatProvider`
answers strictly by delegating to `qa.py::answer()`, which is itself
deterministic and grounded in a saved case's evidence. Any real LLM
provider is designed behind the `ChatProvider` protocol below and is
intentionally NOT implemented here -- `OpenAIChatProvider`/`GeminiChatProvider`
are typed stubs that raise `NotImplementedError`, so no network call, no key
handling, and no "unsafe generation" is ever reachable from this module.

Pipeline (see design doc for the full graph):
  incoming message -> SafetyChecker (safeguarding escalation short-circuit)
  -> EvidenceRetriever.retrieve()
  -> ChatProvider.generate() [deterministic: delegates straight to qa.answer]
  -> SafetyChecker on the outgoing text
  -> ChatResponse with evidence references attached.

ClaimExtractor/ClaimVerifier/ResponseJudge exist as typed stubs: the
deterministic path never generates free text to extract claims from (the
answer IS the evidence, by construction, via qa.py), so there is nothing
for them to do yet. They must be implemented for real before any
free-text-generating ChatProvider is ever enabled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC
from .qa import answer as qa_answer

# ---------------------------------------------------------------------------
# Safeguarding escalation -- fixed, human-authored text. Never generated.
# ---------------------------------------------------------------------------
_SAFEGUARD_PATTERNS_EN = (
    "suicide", "kill myself", "self harm", "self-harm", "hurt myself",
    "abuse", "abused", "being hurt", "in danger", "emergency",
)
_SAFEGUARD_PATTERNS_AR = (
    "انتحار", "أؤذي نفسي", "إيذاء نفسي", "اعتداء", "إساءة", "خطر", "طوارئ",
)

_ESCALATION_MESSAGE = {
    "en": (
        "This sounds urgent or serious. This tool cannot help with safety, "
        "abuse, or emergency situations. Please contact a qualified "
        "professional, a local child-protection service, or emergency "
        "services right away if you or a child may be in danger."
    ),
    "ar": (
        "يبدو أن هذا أمر عاجل أو خطير. لا يمكن لهذه الأداة المساعدة في "
        "مسائل السلامة أو الإساءة أو الطوارئ. يُرجى التواصل فوراً مع مختص "
        "مؤهل أو جهة حماية الطفل المحلية أو خدمات الطوارئ إذا كنت أنت أو "
        "طفل في خطر."
    ),
}


@dataclass(frozen=True)
class SafetyResult:
    escalate: bool
    diagnostic_language_found: bool
    matched_terms: list[str] = field(default_factory=list)


class SafetyChecker:
    """Deterministic, regex/keyword-based. Reuses judges.py's diagnostic-
    language patterns (single source of truth) plus a small safeguarding
    keyword list for the incoming-message escalation check."""

    def check_incoming(self, text: str, language: str = "en") -> SafetyResult:
        text_cf = (text or "").casefold()
        patterns = _SAFEGUARD_PATTERNS_AR if language == "ar" else _SAFEGUARD_PATTERNS_EN
        matched = [p for p in patterns if p in text_cf or p in (text or "")]
        return SafetyResult(escalate=bool(matched), diagnostic_language_found=False,
                             matched_terms=matched)

    def check_outgoing(self, text: str) -> SafetyResult:
        found = bool(_ARABIC_DIAGNOSTIC.search(text or "")) or any(
            p.search(text or "") for p in DIAGNOSTIC_PATTERNS)
        return SafetyResult(escalate=False, diagnostic_language_found=found)


# ---------------------------------------------------------------------------
# Evidence retrieval -- read-only, never writes to the case.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceBundle:
    case_id: str
    analysis: dict
    judges: dict
    profile: dict | None
    evidence_ids: list[str]


class EvidenceRetriever:
    """Reads a saved case's analysis.json/judges.json/profile.json.
    Read-only -- never modifies the case directory."""

    def __init__(self, case_dir: str | Path):
        self.case_dir = Path(case_dir)

    def retrieve(self, query: str = "") -> EvidenceBundle:
        del query  # the deterministic path does not filter by query; qa.py does its own matching
        analysis = json.loads((self.case_dir / "analysis.json").read_text(encoding="utf-8"))
        judges_path = self.case_dir / "judges.json"
        judges = json.loads(judges_path.read_text(encoding="utf-8")) if judges_path.exists() else {}
        profile_path = self.case_dir / "profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else None
        evidence_ids = [e.get("evidence_id") for e in analysis.get("evidence", [])]
        return EvidenceBundle(case_id=self.case_dir.name, analysis=analysis, judges=judges,
                               profile=profile, evidence_ids=evidence_ids)


# ---------------------------------------------------------------------------
# Provider protocol + the one real, working implementation.
# ---------------------------------------------------------------------------
class Message(dict):
    """{'role': 'user'|'assistant', 'content': str} -- a plain dict subtype
    kept simple on purpose; no chat-history-dependent logic exists yet in
    the deterministic path (see module docstring)."""


class ChatProvider(Protocol):
    def generate(self, prompt: str, *, system: str, history: list[Message]) -> str:
        """Return raw draft text. Callers MUST run it through the
        verification pipeline before displaying it -- see
        LLM_GROUNDING_AND_SAFETY_DESIGN.md Section 4. Not called directly
        by respond_to_chat() for the deterministic provider (see below)."""
        ...


class DeterministicChatProvider:
    """No API key, no network call. Delegates directly to the existing,
    real, evidence-grounded src/doar/qa.py::answer(). The 'generated' text
    IS the grounded answer by construction -- there is nothing to verify
    that qa.py hasn't already grounded, so this bypasses the
    extract/verify claim steps (they exist for a future free-text
    provider, not for this one)."""

    def generate(self, prompt: str, *, system: str, history: list[Message]) -> str:
        raise NotImplementedError(
            "DeterministicChatProvider does not implement free-text generate(); "
            "use respond_to_chat(), which calls qa.answer() directly instead."
        )


class OpenAIChatProvider:
    """NOT IMPLEMENTED. A real integration would need: an API key read from
    an environment variable (never hardcoded or required for the base app),
    a request/response schema, timeout + retry policy, and -- critically --
    every output routed through ClaimExtractor/ClaimVerifier/SafetyChecker/
    ResponseJudge (Section 4 of LLM_GROUNDING_AND_SAFETY_DESIGN.md) before
    ever being shown to a user. None of that exists yet; this class exists
    only to fix the shape of the future integration point."""

    def generate(self, prompt: str, *, system: str, history: list[Message]) -> str:
        raise NotImplementedError("OpenAI integration is not implemented in this release.")


class GeminiChatProvider:
    """NOT IMPLEMENTED -- see OpenAIChatProvider's docstring; identical caveat."""

    def generate(self, prompt: str, *, system: str, history: list[Message]) -> str:
        raise NotImplementedError("Gemini integration is not implemented in this release.")


# ---------------------------------------------------------------------------
# Future free-text-generation pipeline stubs (Section 3/4 of the design doc).
# Never called by the deterministic path. Must be implemented for real
# before any free-text ChatProvider may be enabled.
# ---------------------------------------------------------------------------
class ClaimExtractor:
    def extract(self, draft: str) -> list[str]:
        raise NotImplementedError(
            "Not needed by DeterministicChatProvider (its answers are already "
            "grounded by construction). Required before enabling any "
            "free-text-generating ChatProvider."
        )


class ClaimVerifier:
    def verify(self, claim: str, evidence: EvidenceBundle) -> dict:
        raise NotImplementedError("See ClaimExtractor docstring.")


class ResponseJudge:
    """Optional, advisory-only even once implemented -- see design doc
    Section 5 for why an LLM judge alone is never a sufficient safety
    boundary."""

    def judge(self, response: str, verified_claims: list) -> dict:
        raise NotImplementedError("See ClaimExtractor docstring.")


# ---------------------------------------------------------------------------
# Orchestration -- the actual, working, default (deterministic) pipeline.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChatResponse:
    answer: str
    evidence_ids: list[str]
    escalated: bool
    availability: str
    limitations: list[str]
    non_diagnostic_warning: str | None


def respond_to_chat(case_dir: str | Path, message: str, language: str = "en") -> ChatResponse:
    """The real, working, no-API-key chat pipeline. Always uses
    DeterministicChatProvider (via qa.py) -- there is no code path in this
    function that calls an LLM."""
    checker = SafetyChecker()
    incoming = checker.check_incoming(message, language)
    if incoming.escalate:
        return ChatResponse(
            answer=_ESCALATION_MESSAGE["ar" if language == "ar" else "en"],
            evidence_ids=[], escalated=True, availability="escalated",
            limitations=[], non_diagnostic_warning=None,
        )

    bundle = EvidenceRetriever(case_dir).retrieve(message)
    envelope = qa_answer(message, bundle.analysis, bundle.judges, language)

    outgoing = checker.check_outgoing(envelope["answer"])
    if outgoing.diagnostic_language_found:
        # Should be unreachable -- qa.py's own vocabulary is pre-vetted and
        # never uses diagnostic language. Fail safe rather than display it.
        safe_text = {
            "en": "This information could not be safely displayed.",
            "ar": "تعذر عرض هذه المعلومة بأمان.",
        }["ar" if language == "ar" else "en"]
        return ChatResponse(answer=safe_text, evidence_ids=[], escalated=False,
                             availability="blocked", limitations=["safety_check_failed"],
                             non_diagnostic_warning=None)

    return ChatResponse(
        answer=envelope["answer"], evidence_ids=envelope["evidence_ids"],
        escalated=False, availability=envelope["availability"],
        limitations=envelope["limitations"],
        non_diagnostic_warning=envelope["non_diagnostic_warning"],
    )
