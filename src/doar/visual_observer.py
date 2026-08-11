"""DOAR Visual Resolver: provider-independent multimodal visual
observation/verification interface.

Mirrors `chat.py`'s own established pattern exactly (`ChatProvider` /
`DeterministicChatProvider` / `OpenAIChatProvider` / `GeminiChatProvider`):
a `Protocol` fixes the contract, small deterministic test adapters exist
for offline use (no network, no fabricated "vision"), and typed provider
stubs fix the shape of a future real integration without ever touching a
network call, an API key, or a real multimodal capability that doesn't
exist yet in this codebase.

    VisualObserver.analyze(image_path) -> list[VisualObserverCandidate]
        Broad, STRUCTURED candidate discovery -- never prose. Each
        candidate is a proposal, not a fact: it only reaches
        `VisualEntity` via `visual_entity.merge_observer_candidates_into_
        entities` (never trusted directly), and case-level trust only
        ever comes from verification (`VisualVerifier` and/or expert
        review), never from the observer alone.

    VisualVerifier.verify(image_path, entity) -> VerificationResult
        Independently assesses ONE existing `VisualEntity` (using its own
        `bbox`/`crop_ref`) -- verified / uncertain / rejected. Never
        touches `model_validation_status` (a detector-level property);
        only ever informs `case_verification_status`.

**Connecting a real provider next** (not done in this phase): a real
`OpenAIVisualObserver`/`GeminiVisualObserver` would need (1) an API key
read from an environment variable, never hardcoded and never required for
the base app to run; (2) an image-encoding step (base64 or a file
upload API, depending on provider); (3) a structured-output request
(JSON schema / function-calling / tool-call, not free text) mapped onto
`VisualObserverCandidate`; (4) timeout + retry policy; (5) the same
grounding/safety review `LLM_GROUNDING_AND_SAFETY_DESIGN.md` already
requires before any LLM-derived text reaches a user -- observer/verifier
output here never reaches Parent View text directly, only structured
`VisualEntity` fields, but a future free-text explanation layer built on
top of this would still need that review. None of that exists yet; the
stub classes below exist only to fix the integration point's shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .visual_entity import ENTITY_TYPES

# ---------------------------------------------------------------------------
# Structured candidate/result shapes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualObserverCandidate:
    """One proposed visual entity from an observer pass -- structured
    data, never prose. `entity_type` is validated/clamped to
    `visual_entity.ENTITY_TYPES` by `merge_observer_candidates_into_
    entities` (an invalid/unrecognized value from a real provider is
    never trusted verbatim -- falls back to "unknown"). `count`, when the
    observer reports multiple instances of the same label, is informational
    only in this phase -- it does not multiply into separate entities
    (that would mean guessing individual positions this codebase has no
    basis for); a future phase may use it once per-instance localization
    exists."""
    label: str
    alternative_labels: tuple[str, ...] = ()
    entity_type: str = "unknown"
    bbox: tuple[float, float, float, float] | None = None
    count: int | None = None
    confidence: float | None = None
    source_note: str = ""  # e.g. "callable_test_observer_v1" -- provenance, not a model name guess


@dataclass(frozen=True)
class VerificationResult:
    status: str  # "verified" | "uncertain" | "rejected" -- see visual_entity.CASE_VERIFICATION_STATUSES
    confidence: float | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Protocols -- structural typing, so any object with a matching method
# (a real provider, a test double, a future local model wrapper) works
# without inheriting from anything.
# ---------------------------------------------------------------------------


class VisualObserver(Protocol):
    def analyze(self, image_path: str) -> list[VisualObserverCandidate]: ...


class VisualVerifier(Protocol):
    def verify(self, image_path: str, entity) -> VerificationResult: ...


# ---------------------------------------------------------------------------
# Deterministic, offline, no-network test/foundation adapters -- the
# "mock/test provider" this phase requires. Wraps a plain callable so
# tests (and any future local heuristic) can inject exact, deterministic
# behavior -- mirrors the `model_predict_fns`-style injectable-backend
# pattern used everywhere else in this project. Never claims a real
# vision capability: a caller that constructs one of these is always
# supplying its own candidate/verdict logic explicitly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallableVisualObserver:
    fn: Callable[[str], list[VisualObserverCandidate]]

    def analyze(self, image_path: str) -> list[VisualObserverCandidate]:
        return self.fn(image_path)


@dataclass(frozen=True)
class CallableVisualVerifier:
    fn: Callable[[str, object], VerificationResult]

    def verify(self, image_path: str, entity) -> VerificationResult:
        return self.fn(image_path, entity)


@dataclass(frozen=True)
class StaticVisualVerifier:
    """The simplest possible verifier: a fixed status/confidence for
    every entity, regardless of image or entity content. Useful for
    tests and as an explicit "mark everything uncertain until a real
    verifier exists" default -- never silently promotes anything to
    "verified" on its own unless the caller explicitly configures that."""
    status: str = "uncertain"
    confidence: float | None = None
    notes: str = "No real visual verifier is configured in this environment."

    def verify(self, image_path: str, entity) -> VerificationResult:
        del image_path, entity
        return VerificationResult(status=self.status, confidence=self.confidence, notes=self.notes)


# ---------------------------------------------------------------------------
# Typed provider stubs -- fix the future integration shape, never touch a
# network call or an API key. Mirrors chat.py's OpenAIChatProvider/
# GeminiChatProvider exactly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAIVisualObserver:
    """NOT IMPLEMENTED. See module docstring, "Connecting a real provider
    next", for exactly what a real integration needs."""
    api_key_env_var: str = "OPENAI_API_KEY"
    model: str = "gpt-4o"

    def analyze(self, image_path: str) -> list[VisualObserverCandidate]:
        raise NotImplementedError(
            "OpenAI visual observation is not implemented in this release -- "
            "see visual_observer.py's module docstring for the integration shape needed."
        )


@dataclass(frozen=True)
class GeminiVisualObserver:
    """NOT IMPLEMENTED -- see OpenAIVisualObserver's docstring; identical caveat."""
    api_key_env_var: str = "GEMINI_API_KEY"
    model: str = "gemini-1.5-pro"

    def analyze(self, image_path: str) -> list[VisualObserverCandidate]:
        raise NotImplementedError(
            "Gemini visual observation is not implemented in this release -- "
            "see visual_observer.py's module docstring for the integration shape needed."
        )


@dataclass(frozen=True)
class OpenAIVisualVerifier:
    """NOT IMPLEMENTED -- see OpenAIVisualObserver's docstring; identical caveat."""
    api_key_env_var: str = "OPENAI_API_KEY"
    model: str = "gpt-4o"

    def verify(self, image_path: str, entity) -> VerificationResult:
        raise NotImplementedError(
            "OpenAI visual verification is not implemented in this release -- "
            "see visual_observer.py's module docstring for the integration shape needed."
        )


@dataclass(frozen=True)
class GeminiVisualVerifier:
    """NOT IMPLEMENTED -- see OpenAIVisualObserver's docstring; identical caveat."""
    api_key_env_var: str = "GEMINI_API_KEY"
    model: str = "gemini-1.5-pro"

    def verify(self, image_path: str, entity) -> VerificationResult:
        raise NotImplementedError(
            "Gemini visual verification is not implemented in this release -- "
            "see visual_observer.py's module docstring for the integration shape needed."
        )


def clamp_entity_type(entity_type: str | None) -> str:
    """A real provider (or a careless test) may propose an entity_type
    string outside `visual_entity.ENTITY_TYPES` -- never trusted
    verbatim; falls back to "unknown", the same safe default
    `visual_entity.classify_entity_type` uses for an unrecognized label."""
    return entity_type if entity_type in ENTITY_TYPES else "unknown"
