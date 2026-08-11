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

**Real provider connected: `OpenAIVisualObserver`** (shadow mode). Reads
its API key ONLY from an environment variable (never hardcoded, never
required for the base app to run -- construction never touches it, only
`.analyze()` does), base64-encodes the image, sends a structured-output
(JSON-schema) Chat Completions request over the stdlib's own
`urllib.request` (no new third-party dependency -- consistent with this
project's minimal `numpy`+`Pillow` dependency footprint), and parses the
response into `VisualObserverCandidate` instances -- never free prose.
Timeout + a small bounded retry apply only to transport-level failures
(connection/timeout), never to a malformed response (retrying a parse
failure would never help and would just delay an honest error). A
missing key or an unusable response raises a clear, typed exception --
`VisualObserverConfigurationError` / `VisualObserverRequestError` --
never a silently fabricated candidate list. `request_fn` exists purely
as a test seam (mirrors this project's `model_predict_fns` injectable-
backend pattern everywhere else): production code never sets it, so the
real network path (`_post_openai_chat_completions`) is never exercised
by the test suite.

SHADOW MODE (still applies unchanged): a real observer's candidates flow
through the exact same `merge_observer_candidates_into_entities` path as
the fake/test observers always did -- they can create/enrich
`VisualEntity` records, appear in Technical View, be searched, be
verified -- but they have no backing `VisualFinding`, so they remain
structurally invisible to the rule engine regardless of
`case_verification_status`. See `visual_entity.py` and
`test_visual_resolver_integration.py` for where that structural
exclusion is enforced and tested.

`GeminiVisualObserver` and both `*VisualVerifier` classes remain
unimplemented stubs -- out of this phase's explicit scope (see this
phase's own DO-NOT list); they still fix their shape exactly as before.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .production_config import resolve_visual_observer_model
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
# Real-provider errors -- always raised, never swallowed into a fabricated
# candidate list. Distinct from NotImplementedError (still used by the
# still-stubbed Gemini/*Verifier classes) so callers can tell "this
# provider was never built" apart from "this provider is built but
# unusable right now".
# ---------------------------------------------------------------------------


class VisualObserverConfigurationError(RuntimeError):
    """Raised before any network call when a real observer cannot run at
    all in this environment -- e.g. no API key in the configured
    environment variable, or an unreadable image path. Never a reason to
    fabricate a result; the caller sees a clean, explicit failure."""


class VisualObserverRequestError(RuntimeError):
    """Raised when a real observer's HTTP request fails after retries, or
    when the provider's response cannot be parsed into structured
    candidates. Never a reason to fabricate a result."""


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
# Provider integrations. `OpenAIVisualObserver` below is real (shadow mode);
# `GeminiVisualObserver` and both `*VisualVerifier` classes remain typed
# stubs that fix the future integration shape without ever touching a
# network call or an API key -- mirrors chat.py's OpenAIChatProvider/
# GeminiChatProvider split between its real and stub providers.
# ---------------------------------------------------------------------------


_OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

# Broad, vocabulary-unrestricted, uncertainty-preserving instructions --
# deliberately does NOT enumerate expected objects (no "look for a car, a
# sun, ..."); the observer must inspect the whole image and report
# whatever it can actually justify, including "I can't tell" via
# entity_type="unknown" and a low/absent confidence.
_OPENAI_SYSTEM_PROMPT = (
    "You are a careful visual observer inspecting a hand-drawn sketch for a "
    "structural analysis pipeline. Look broadly across the ENTIRE image -- do "
    "not limit yourself to any fixed list of expected objects. Identify every "
    "distinct visual element you can actually see: whole objects, parts of "
    "objects, symbols, geometric shapes, decorative marks, text-like regions, "
    "abstract marks, scribbles, or regions you cannot confidently classify. "
    "For each element report only what you can visually justify -- if you are "
    "unsure what something is, say so with a lower confidence and/or "
    "entity_type 'unknown' rather than guessing a specific object. Never "
    "invent an element that is not actually present in the image. Report each "
    "bounding box as [x, y, width, height] normalized to the 0..1 range of the "
    "image, or null if you cannot localize the element. Report count only "
    "when you can distinguish multiple separate instances of the same label."
)

_OPENAI_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "alternative_labels": {"type": "array", "items": {"type": "string"}},
                    "entity_type": {"type": "string", "enum": sorted(ENTITY_TYPES)},
                    "bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "count": {"type": ["integer", "null"]},
                    "confidence": {"type": ["number", "null"]},
                },
                "required": [
                    "label", "alternative_labels", "entity_type", "bbox", "count", "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

_MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _guess_image_mime(image_path: str) -> str:
    return _MIME_BY_SUFFIX.get(Path(image_path).suffix.lower(), "image/jpeg")


def _build_openai_payload(model: str, image_b64: str, mime: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _OPENAI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Analyze this drawing and report every distinct visual element you can identify."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "visual_candidates", "strict": True, "schema": _OPENAI_CANDIDATE_SCHEMA},
        },
    }


def _post_openai_chat_completions(api_key: str, payload: dict, timeout_seconds: float) -> bytes:
    """The one real network call this module makes -- never invoked by
    the test suite (tests always inject `request_fn`), mirrors chat.py's
    own real-vs-test transport split."""
    request = urllib.request.Request(
        _OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 -- fixed https URL
        return response.read()


def _parse_openai_candidates(raw: bytes, *, model: str) -> list[VisualObserverCandidate]:
    """Parses one OpenAI chat-completions response into structured
    candidates. Anything that doesn't fit the expected shape raises
    `VisualObserverRequestError` -- never silently drops into a fabricated
    or partially-guessed candidate list; individual malformed candidate
    entries are skipped (not fabricated-around) rather than failing the
    whole batch, since one bad entry among many good ones shouldn't
    discard everything else the model actually saw."""
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisualObserverRequestError(f"OpenAI response was not valid JSON: {exc}") from exc
    if not isinstance(response, dict):
        raise VisualObserverRequestError("OpenAI response was not a JSON object.")
    if "error" in response:
        raise VisualObserverRequestError(f"OpenAI API returned an error: {response['error']!r}")
    response_id = response.get("id", "unknown")
    source_note = f"openai:{model}:response={response_id}"
    try:
        content = response["choices"][0]["message"]["content"]
        raw_candidates = json.loads(content)["candidates"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise VisualObserverRequestError(
            f"OpenAI response (id={response_id}) could not be parsed into structured candidates: {exc}"
        ) from exc
    if not isinstance(raw_candidates, list):
        raise VisualObserverRequestError(f"OpenAI response (id={response_id}) 'candidates' field was not a list.")

    candidates = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        bbox = item.get("bbox")
        bbox_tuple = (
            tuple(float(v) for v in bbox)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4
            else None
        )
        alt_labels = item.get("alternative_labels")
        alt_tuple = tuple(str(a) for a in alt_labels) if isinstance(alt_labels, (list, tuple)) else ()
        count = item.get("count")
        count_int = int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else None
        confidence = item.get("confidence")
        confidence_float = (
            float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None
        )
        candidates.append(VisualObserverCandidate(
            label=label.strip(),
            alternative_labels=alt_tuple,
            entity_type=clamp_entity_type(item.get("entity_type")),
            bbox=bbox_tuple,
            count=count_int,
            confidence=confidence_float,
            source_note=source_note,
        ))
    return candidates


@dataclass(frozen=True)
class OpenAIVisualObserver:
    """Real OpenAI multimodal observer -- shadow mode only (see module
    docstring). `api_key_env_var` names the environment variable holding
    the API key (never read at construction time, only inside
    `.analyze()`, and never hardcoded anywhere in this codebase). `model`
    defaults to `production_config.resolve_visual_observer_model()` -- a
    research/production config surface (`DOAR_VISUAL_OBSERVER_MODEL`
    env var), never a normal-user UI choice. `request_fn`, when set,
    replaces the real HTTP transport entirely -- the test suite always
    sets it; production code never does."""
    api_key_env_var: str = "OPENAI_API_KEY"
    model: str = field(default_factory=resolve_visual_observer_model)
    timeout_seconds: float = 30.0
    max_retries: int = 2
    request_fn: Callable[[str, dict, float], bytes] | None = None

    def analyze(self, image_path: str) -> list[VisualObserverCandidate]:
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise VisualObserverConfigurationError(
                f"No OpenAI API key found in the '{self.api_key_env_var}' environment variable -- "
                "set it before using OpenAIVisualObserver. Refusing to fabricate a result."
            )
        if not image_path or not Path(image_path).exists():
            raise VisualObserverConfigurationError(f"Image not found: {image_path!r}")

        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        payload = _build_openai_payload(self.model, image_b64, _guess_image_mime(image_path))
        raw = self._send_with_retries(api_key, payload)
        return _parse_openai_candidates(raw, model=self.model)

    def _send_with_retries(self, api_key: str, payload: dict) -> bytes:
        transport = self.request_fn or _post_openai_chat_completions
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                return transport(api_key, payload, self.timeout_seconds)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                continue
        raise VisualObserverRequestError(
            f"OpenAI vision request failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error


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
