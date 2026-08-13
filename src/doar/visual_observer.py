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

**Real provider connected: `GeminiVisualObserver`** (shadow mode, same
contract). Reads `GEMINI_API_KEY` only inside `.analyze()`, calls the
stable `generateContent` REST API (model ID verified against
ai.google.dev, not assumed -- see `production_config.
resolve_gemini_visual_observer_model`), requests structured JSON output
via `generationConfig.responseSchema`, and parses the response with the
exact same defensive, never-fabricate discipline as
`_parse_openai_candidates`. Uses the SAME broad, provider-agnostic
`_BROAD_OPEN_WORLD_SYSTEM_PROMPT` as OpenAI -- no per-provider prompt
tuning, so any difference in what the two providers find reflects the
model, not a differently-worded ask.

**Real verifier connected: `GeminiVisualVerifier`** (shadow mode, same
contract). Independent from the observer in TWO ways, not just prompt
wording: a DIFFERENT model (`gemini-3.5-flash-lite`, vs. the observer's
`gemini-3.6-flash`) and a label-blind prompt that never contains the
observer's candidate label -- only the entity's own bbox crop (+ a small
padded context crop) and "what do you see here?". The independent label
it returns is compared to the entity's own labels/aliases in PLAIN CODE
(`_compare_verifier_to_entity`) -- never by asking a second LLM to judge
agreement -- producing verified/uncertain/rejected (or "unreviewed",
untouched, for an entity with no usable bbox to crop). `OpenAIVisualVerifier`
remains an unimplemented stub -- out of this phase's scope.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .production_config import (
    resolve_gemini_visual_observer_model, resolve_gemini_visual_verifier_model, resolve_visual_observer_model,
)
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
    """`status` is the only field `apply_verifier_to_entities` reads --
    everything below is additive, optional context a real verifier can
    fill in without breaking any existing caller (`StaticVisualVerifier`,
    `CallableVisualVerifier` test doubles, and every existing test all
    construct this with only `status`[, `confidence`[, `notes`]])."""
    status: str  # "verified" | "uncertain" | "rejected" | "unreviewed" -- see visual_entity.CASE_VERIFICATION_STATUSES
    confidence: float | None = None
    notes: str = ""
    # The verifier's OWN independent observation of the region -- what
    # `status` was deterministically computed FROM (see
    # `_compare_verifier_to_entity`), never the observer's label fed back.
    # Empty for verifiers that don't produce one.
    independent_label: str | None = None
    independent_alternative_labels: tuple[str, ...] = ()
    source_note: str = ""  # provider/model/provenance, same convention as VisualObserverCandidate.source_note


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
# entity_type="unknown" and a low/absent confidence. SHARED verbatim by
# every real provider (OpenAI, Gemini, ...) -- the prompt is
# provider-agnostic, and using the identical wording keeps any future
# cross-provider comparison meaningful (a difference in output reflects
# the model, not a differently-worded ask).
_BROAD_OPEN_WORLD_SYSTEM_PROMPT = (
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
            {"role": "system", "content": _BROAD_OPEN_WORLD_SYSTEM_PROMPT},
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


_GEMINI_GENERATE_CONTENT_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")

# generateContent (the stable, long-documented REST API) was chosen over
# the newer "Interactions API" Google now recommends for new development:
# generateContent remains fully supported with no deprecation timeline
# (confirmed via ai.google.dev's own migration guide, 2026-08-12), and its
# request/response shape is precisely documented and verifiable, unlike
# the Interactions API's still-unsettled-in-tooling docs at the time this
# was written. Revisit if/when generateContent is actually deprecated.
#
# `gemini-3.6-flash` was VERIFIED (not assumed) on 2026-08-12 against
# ai.google.dev: current stable/GA, Flash-family, accepts image input,
# supports structured JSON output, free-tier available. The Gemini 2.0
# Flash generation this repo might otherwise have defaulted to was shut
# down 2026-06-01.
#
# response_schema here uses Gemini's OpenAPI-3.0-subset dialect (uppercase
# type names, `nullable: true` rather than a `["type", "null"]` union) --
# the long-established, stable shape for this API. Response PARSING below
# is defensive regardless (mirrors `_parse_openai_candidates`): a schema
# quirk on either side degrades to skipped/malformed entries, never a
# fabricated candidate.
_GEMINI_CANDIDATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "alternative_labels": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "entity_type": {"type": "STRING", "enum": sorted(ENTITY_TYPES)},
                    "bbox": {"type": "ARRAY", "items": {"type": "NUMBER"}, "nullable": True},
                    "count": {"type": "INTEGER", "nullable": True},
                    "confidence": {"type": "NUMBER", "nullable": True},
                },
                "required": ["label", "alternative_labels", "entity_type"],
            },
        },
    },
    "required": ["candidates"],
}
# A version tag for this request shape -- bumped whenever the schema or
# prompt actually changes, so stored provenance can tell which version of
# the ask produced a given candidate. NOT a hash of the live schema
# object (dict key order isn't guaranteed stable across Python versions).
_GEMINI_SCHEMA_VERSION = "gemini_observer_schema_v1"


def _build_gemini_payload(image_b64: str, mime: str) -> dict:
    return {
        "systemInstruction": {"parts": [{"text": _BROAD_OPEN_WORLD_SYSTEM_PROMPT}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "Analyze this drawing and report every distinct visual element you can identify."},
                {"inlineData": {"mimeType": mime, "data": image_b64}},
            ],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _GEMINI_CANDIDATE_SCHEMA,
            # Minimal reasoning: this is broad visual discovery, not multi-step
            # reasoning -- keeps the call fast/cheap. `thinkingLevel: "minimal"`
            # is gemini-3.6-flash's actual accepted control (VERIFIED live
            # against the real API 2026-08-12 -- the older `thinkingBudget: 0`
            # numeric form this repo might otherwise have assumed returns
            # HTTP 400 INVALID_ARGUMENT on this model generation).
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }


def _post_gemini_generate_content(api_key: str, model: str, payload: dict, timeout_seconds: float) -> bytes:
    """The one real network call this function makes -- never invoked by
    the test suite (tests always inject `request_fn`)."""
    url = _GEMINI_GENERATE_CONTENT_URL_TEMPLATE.format(model=model)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 -- fixed https URL
        return response.read()


def _parse_gemini_candidates(raw: bytes, *, model: str) -> list[VisualObserverCandidate]:
    """Parses one Gemini generateContent response into structured
    candidates. Mirrors `_parse_openai_candidates` exactly: anything that
    doesn't fit the expected shape raises `VisualObserverRequestError`;
    individual malformed candidate entries are skipped, not fatal."""
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisualObserverRequestError(f"Gemini response was not valid JSON: {exc}") from exc
    if not isinstance(response, dict):
        raise VisualObserverRequestError("Gemini response was not a JSON object.")
    if "error" in response:
        raise VisualObserverRequestError(f"Gemini API returned an error: {response['error']!r}")
    response_id = response.get("responseId", "unknown")
    model_version = response.get("modelVersion", model)
    source_note = (f"gemini:{model}:model_version={model_version}:schema={_GEMINI_SCHEMA_VERSION}"
                    f":response={response_id}")
    try:
        content_text = response["candidates"][0]["content"]["parts"][0]["text"]
        raw_candidates = json.loads(content_text)["candidates"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise VisualObserverRequestError(
            f"Gemini response (id={response_id}) could not be parsed into structured candidates: {exc}"
        ) from exc
    if not isinstance(raw_candidates, list):
        raise VisualObserverRequestError(f"Gemini response (id={response_id}) 'candidates' field was not a list.")

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
class GeminiVisualObserver:
    """Real Gemini multimodal observer -- shadow mode only (see module
    docstring; identical safety contract to `OpenAIVisualObserver`).
    `api_key_env_var` names the environment variable holding the API key
    (never read at construction time, only inside `.analyze()`, never
    hardcoded). `model` defaults to `production_config.
    resolve_gemini_visual_observer_model()` -- a research/production
    config surface (`DOAR_GEMINI_VISUAL_OBSERVER_MODEL` env var), never a
    normal-user UI choice. `request_fn`, when set, replaces the real HTTP
    transport entirely -- the test suite always sets it; production code
    never does."""
    api_key_env_var: str = "GEMINI_API_KEY"
    model: str = field(default_factory=resolve_gemini_visual_observer_model)
    timeout_seconds: float = 30.0
    max_retries: int = 2
    request_fn: Callable[[str, str, dict, float], bytes] | None = None

    def analyze(self, image_path: str) -> list[VisualObserverCandidate]:
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise VisualObserverConfigurationError(
                f"No Gemini API key found in the '{self.api_key_env_var}' environment variable -- "
                "set it before using GeminiVisualObserver. Refusing to fabricate a result."
            )
        if not image_path or not Path(image_path).exists():
            raise VisualObserverConfigurationError(f"Image not found: {image_path!r}")

        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        payload = _build_gemini_payload(image_b64, _guess_image_mime(image_path))
        raw = self._send_with_retries(api_key, payload)
        return _parse_gemini_candidates(raw, model=self.model)

    def _send_with_retries(self, api_key: str, payload: dict) -> bytes:
        transport = self.request_fn or _post_gemini_generate_content
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                return transport(api_key, self.model, payload, self.timeout_seconds)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                continue
        raise VisualObserverRequestError(
            f"Gemini vision request failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error


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


# ---------------------------------------------------------------------------
# Real GeminiVisualVerifier -- independent, label-blind, crop/context-
# conditioned case-level verification. NEVER open-world: this is a
# single-region check, not another whole-image scan, and the prompt below
# never contains the observer's label -- only "what do you see here?".
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM_PROMPT = (
    "You are independently inspecting ONE small cropped region taken from a "
    "larger hand-drawn sketch. You have NOT been told what this region is "
    "supposed to be -- look only at what is actually visible in it and "
    "describe it narrowly and literally. The first image is the exact region "
    "in question. A second image, if provided, shows that same region with a "
    "small amount of surrounding context, included only to help you "
    "understand what you're looking at -- describe the first region, not the "
    "surrounding context. If the region is too small, too abstract, too "
    "ambiguous, or otherwise not reliably identifiable on its own, report "
    "your label as 'unknown' rather than guessing. Report one short primary "
    "label, any reasonable alternative labels, and how confident you are."
)

_VERIFIER_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING"},
        "alternative_labels": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence": {"type": "NUMBER", "nullable": True},
    },
    "required": ["label", "alternative_labels"],
}
_VERIFIER_SCHEMA_VERSION = "gemini_verifier_schema_v1"

# How far the "context" crop expands beyond the candidate's own bbox, as a
# fraction of the bbox's own width/height -- enough to see immediate
# surroundings without turning this back into whole-image analysis.
_VERIFIER_CONTEXT_PADDING_FRACTION = 0.5

# A verifier label this ambiguous/absent is never treated as a claim to
# compare against anything -- always "uncertain", never forced into a
# verified/rejected decision either way.
_UNKNOWN_LABEL_MARKERS = frozenset({
    "", "unknown", "unclear", "unsure", "uncertain", "ambiguous", "not identifiable",
    "cannot identify", "can't tell", "not sure", "n/a", "none",
})

# Below this verifier-reported confidence, a LABEL MISMATCH is treated as
# "uncertain" rather than "rejected" -- a low-confidence guess that happens
# to differ from the observer's label isn't a confident, actionable
# contradiction. A confident mismatch (or no confidence reported at all --
# a plain, unhedged claim) still rejects.
_VERIFIER_LOW_CONFIDENCE_THRESHOLD = 0.5


def _expand_bbox_for_context(
        bbox: tuple[float, float, float, float],
        padding_fraction: float = _VERIFIER_CONTEXT_PADDING_FRACTION) -> tuple[float, float, float, float]:
    """Expands a normalized xywh bbox by `padding_fraction` of its own
    width/height on each side, clamped to the [0, 1] image bounds -- same
    normalized-coordinate contract every bbox in this codebase already
    uses (see `visual_entity.compute_relative_size`/`compute_page_position`)."""
    x, y, w, h = bbox
    pad_x, pad_y = w * padding_fraction, h * padding_fraction
    left, top = max(0.0, x - pad_x), max(0.0, y - pad_y)
    right, bottom = min(1.0, x + w + pad_x), min(1.0, y + h + pad_y)
    return (left, top, max(0.0, right - left), max(0.0, bottom - top))


def build_verifier_crops(image_path: str, bbox: tuple[float, float, float, float] | None):
    """Returns `(crop, context_crop)` as real PIL Images, or `(None, None)`
    if the bbox/image is unusable -- never invents a crop. Reuses
    `visual_entity._crop_region`, the ONE place this project's crop math
    exists, for both the tight candidate crop and the padded context crop,
    so a verifier's idea of "this region" always agrees with the same
    crop Technical View/`save_entity_crop` would produce. Public (no
    leading underscore) so the development-check script can reuse it to
    persist context crops alongside the tight ones."""
    from .visual_entity import _crop_region
    if bbox is None:
        return None, None
    crop = _crop_region(image_path, bbox)
    if crop is None:
        return None, None
    context_crop = _crop_region(image_path, _expand_bbox_for_context(bbox)) or crop
    return crop, context_crop


def _pil_image_to_png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_gemini_verifier_payload(crop_b64: str, context_b64: str | None) -> dict:
    parts = [
        {"text": "Region to identify (exact crop):"},
        {"inlineData": {"mimeType": "image/png", "data": crop_b64}},
    ]
    if context_b64 is not None and context_b64 != crop_b64:
        parts.append({"text": "Same region with a small amount of surrounding context, for reference only:"})
        parts.append({"inlineData": {"mimeType": "image/png", "data": context_b64}})
    return {
        "systemInstruction": {"parts": [{"text": _VERIFIER_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _VERIFIER_RESPONSE_SCHEMA,
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }


def _parse_gemini_verifier_response(raw: bytes, *, model: str) -> dict:
    """Parses one Gemini generateContent response from the VERIFIER
    request into the independent (label, alternatives, confidence,
    provenance) observation -- mirrors `_parse_gemini_candidates`'
    defensive discipline, just for a single object instead of a list.
    Never compares against an entity here -- that's a separate,
    deterministic step (`_compare_verifier_to_entity`), never done by
    asking a second LLM call to judge agreement."""
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisualObserverRequestError(f"Gemini verifier response was not valid JSON: {exc}") from exc
    if not isinstance(response, dict):
        raise VisualObserverRequestError("Gemini verifier response was not a JSON object.")
    if "error" in response:
        raise VisualObserverRequestError(f"Gemini verifier API returned an error: {response['error']!r}")
    response_id = response.get("responseId", "unknown")
    model_version = response.get("modelVersion", model)
    try:
        content_text = response["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(content_text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise VisualObserverRequestError(
            f"Gemini verifier response (id={response_id}) could not be parsed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise VisualObserverRequestError(f"Gemini verifier response (id={response_id}) content was not an object.")

    label = parsed.get("label")
    label = label.strip() if isinstance(label, str) and label.strip() else "unknown"
    alt_labels = parsed.get("alternative_labels")
    alt_tuple = tuple(str(a) for a in alt_labels) if isinstance(alt_labels, (list, tuple)) else ()
    confidence = parsed.get("confidence")
    confidence_float = (
        float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None
    )
    source_note = (f"gemini:{model}:model_version={model_version}:schema={_VERIFIER_SCHEMA_VERSION}"
                    f":response={response_id}")
    return {"label": label, "alternative_labels": alt_tuple, "confidence": confidence_float,
            "source_note": source_note}


def _normalize_label_for_comparison(label: str) -> str:
    """Lowercases, strips, and drops a simple trailing-'s' plural -- the
    smallest normalization that handles case/plural differences without
    building a semantic-matching system this phase deliberately avoids."""
    normalized = label.strip().lower()
    if len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
        normalized = normalized[:-1]
    return normalized


# Small, fixed filler-word list -- NOT a synonym/ontology system, just the
# handful of words that add no identifying content to a label ("drawing
# of a person" vs "person inside car" should compare on "person", not on
# "of"/"a"/"drawing"). Deliberately short; if a real label ever needs a
# word here that isn't, that's a sign this list needs one more entry, not
# that this needs to grow into semantic matching.
_GENERIC_LABEL_WORDS = frozenset({
    "drawing", "drawings", "image", "picture", "sketch", "shape", "region", "mark", "marks",
    "of", "a", "an", "the", "with", "in", "on", "inside", "part", "figure", "like",
})


def _tokenize_label_for_comparison(label: str) -> frozenset:
    """Words only (punctuation stripped), each singular/case-normalized
    the same simple way as `_normalize_label_for_comparison`, with short
    and generic/filler words dropped -- the token set two labels are
    compared against as a FALLBACK when they don't already match as
    whole strings (see `_labels_plausibly_match`)."""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in label.lower())
    tokens = set()
    for word in cleaned.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if len(word) > 2 and word not in _GENERIC_LABEL_WORDS:
            tokens.add(word)
    return frozenset(tokens)


def _labels_plausibly_match(a: str, b: str) -> bool:
    """Two-tier check: (1) whole-string equality/containment after simple
    case/plural normalization -- the strong signal, unchanged from
    before; (2) a FALLBACK token-overlap check -- after dropping generic
    filler words, do the two labels share at least one real content word
    ("person" in both "person inside car" and "drawing of a person")?
    Still deterministic, plain-code comparison -- never a second LLM
    judging "do these agree?", and never full semantic/embedding
    matching -- just literal shared words."""
    na, nb = _normalize_label_for_comparison(a), _normalize_label_for_comparison(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    tokens_a, tokens_b = _tokenize_label_for_comparison(a), _tokenize_label_for_comparison(b)
    if not tokens_a or not tokens_b:
        return False
    return bool(tokens_a & tokens_b)


def _compare_verifier_to_entity(label: str, alternative_labels: tuple[str, ...], confidence: float | None,
                                 entity) -> str:
    """The ONE place observer and verifier labels are compared -- in
    plain code, never by asking another LLM whether they "agree". An
    honest 'unknown' from the verifier is always "uncertain", never
    forced toward verified or rejected. A match against the entity's
    canonical label, any candidate label, or any alias is "verified". A
    confident (or unhedged) mismatch is "rejected"; a LOW-confidence
    mismatch is "uncertain" -- a hedged wrong guess isn't a confident
    contradiction."""
    if _normalize_label_for_comparison(label) in _UNKNOWN_LABEL_MARKERS:
        return "uncertain"
    accepted_labels = {entity.canonical_label, *(lbl for lbl, _conf in entity.candidate_labels),
                        *entity.aliases_en, *entity.aliases_ar}
    verifier_labels = (label, *alternative_labels)
    for accepted in accepted_labels:
        for observed in verifier_labels:
            if _labels_plausibly_match(accepted, observed):
                return "verified"
    if confidence is not None and confidence < _VERIFIER_LOW_CONFIDENCE_THRESHOLD:
        return "uncertain"
    return "rejected"


@dataclass(frozen=True)
class GeminiVisualVerifier:
    """Real, independent Gemini case-level verifier -- shadow mode only
    (see module docstring; identical safety contract to the observers).
    Deliberately narrow and LABEL-BLIND: given one `VisualEntity`'s own
    bbox, crops the region (`build_verifier_crops`, reusing this
    project's one shared crop function), sends ONLY the crop (+ a padded
    context crop) to Gemini with a prompt that never mentions the
    observer's label, and independently asks "what is visibly shown
    here?". The resulting independent label is compared to the entity's
    own canonical/candidate labels/aliases IN CODE
    (`_compare_verifier_to_entity`) -- never by asking another LLM to
    judge agreement -- to produce verified/uncertain/rejected. Uses a
    DIFFERENT model than the observer (`gemini-3.5-flash-lite`, verified
    GA/stable and smaller than the observer's `gemini-3.6-flash`) -- an
    independently-chosen model, not just an independent prompt. An entity
    with no usable bbox is never guessed at -- returns "unreviewed"
    (the same safe status new entities already start in) without ever
    making a network call."""
    api_key_env_var: str = "GEMINI_API_KEY"
    model: str = field(default_factory=resolve_gemini_visual_verifier_model)
    timeout_seconds: float = 30.0
    max_retries: int = 2
    request_fn: Callable[[str, str, dict, float], bytes] | None = None

    def verify(self, image_path: str, entity) -> VerificationResult:
        if entity.bbox is None:
            return VerificationResult(
                status="unreviewed",
                notes="No bbox available for this entity -- nothing to crop and verify; not guessed.")

        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise VisualObserverConfigurationError(
                f"No Gemini API key found in the '{self.api_key_env_var}' environment variable -- "
                "set it before using GeminiVisualVerifier. Refusing to fabricate a result."
            )
        if not image_path or not Path(image_path).exists():
            raise VisualObserverConfigurationError(f"Image not found: {image_path!r}")

        crop, context_crop = build_verifier_crops(image_path, entity.bbox)
        if crop is None:
            return VerificationResult(
                status="unreviewed",
                notes="Bbox present but a crop could not be produced from this image -- not guessed.")

        crop_b64 = _pil_image_to_png_b64(crop)
        context_b64 = _pil_image_to_png_b64(context_crop) if context_crop is not None else None
        payload = _build_gemini_verifier_payload(crop_b64, context_b64)
        raw = self._send_with_retries(api_key, payload)
        observed = _parse_gemini_verifier_response(raw, model=self.model)
        status = _compare_verifier_to_entity(
            observed["label"], observed["alternative_labels"], observed["confidence"], entity)
        return VerificationResult(
            status=status, confidence=observed["confidence"],
            notes=f"Independent verifier observation: {observed['label']!r} "
                  f"(alternatives: {list(observed['alternative_labels'])})",
            independent_label=observed["label"],
            independent_alternative_labels=observed["alternative_labels"],
            source_note=observed["source_note"],
        )

    def _send_with_retries(self, api_key: str, payload: dict) -> bytes:
        transport = self.request_fn or _post_gemini_generate_content
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                return transport(api_key, self.model, payload, self.timeout_seconds)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                continue
        raise VisualObserverRequestError(
            f"Gemini verifier request failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error


def clamp_entity_type(entity_type: str | None) -> str:
    """A real provider (or a careless test) may propose an entity_type
    string outside `visual_entity.ENTITY_TYPES` -- never trusted
    verbatim; falls back to "unknown", the same safe default
    `visual_entity.classify_entity_type` uses for an unrecognized label."""
    return entity_type if entity_type in ENTITY_TYPES else "unknown"
