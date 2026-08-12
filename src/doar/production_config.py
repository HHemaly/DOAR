"""DOAR V1.1: centralized, frozen production analysis configuration.

Normal users make ZERO model/checkpoint/threshold choices (Stage 3/7 of
the V1.1 stabilization) -- this module is the SINGLE place that decides
which concrete model/checkpoint/policy the automatic pipeline actually
uses, resolved once per process and reused by both the real analysis
call and Technical View's "Production configuration / provenance"
display.

Never invents a new model choice: every identifier here is exactly the
one the repository's own prior phases already established as
scientifically frozen/recommended (Phase 5's calibrated efficientnet_b0
checkpoint -- CURRENT_CAPABILITY_AUDIT.md Section 3: "used in this
session's own trace", macro-F1 0.7364, temperature-scaled T=1.144; Phase
2C.7's frozen visual detector policy). This module only decides WHETHER
that asset is actually present on the current machine -- it never
substitutes a different, less-validated checkpoint just to make
something appear to work (DOAR V1.1 Problem B / Stage 4's explicit
instruction: no fabricated fallback, no hidden re-routing).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The one scientifically frozen/recommended expressive-model identifier
# and checkpoint path. Never a user choice; never silently swapped for a
# different, less-validated checkpoint if this one is absent on the
# current machine -- see resolve_production_config().
EXPRESSIVE_MODEL_IDENTIFIER = "efficientnet_b0_seed_42_calibrated"
EXPRESSIVE_MODEL_CHECKPOINT_PATH = (
    ROOT / "outputs" / "phase5" / "seed42_reference" / "efficientnet_b0_seed_42" / "best.pt")

VISUAL_DETECTOR_POLICY_PATH = ROOT / "artifacts" / "phase2c7" / "visual_detector_policy.json"

# DOAR Visual Observer (shadow mode): which OpenAI model the real
# multimodal observer uses. A research/production config surface, not a
# normal Streamlit UI choice -- set DOAR_VISUAL_OBSERVER_MODEL in the
# environment to override for A/B model comparisons; otherwise the
# repository's current default applies uniformly.
VISUAL_OBSERVER_MODEL_ENV_VAR = "DOAR_VISUAL_OBSERVER_MODEL"
DEFAULT_VISUAL_OBSERVER_MODEL = "gpt-4o"


def resolve_visual_observer_model() -> str:
    """The OpenAI vision model `OpenAIVisualObserver` uses when the
    caller doesn't inject a different one explicitly. Only configurable
    via `DOAR_VISUAL_OBSERVER_MODEL` -- never exposed as a normal-user UI
    choice, matching this module's own "zero model choices for normal
    users" principle."""
    return os.environ.get(VISUAL_OBSERVER_MODEL_ENV_VAR, DEFAULT_VISUAL_OBSERVER_MODEL)


# DOAR Gemini Observer: which Gemini model the real multimodal observer
# uses. `gemini-3.6-flash` was VERIFIED (not assumed) against
# ai.google.dev on 2026-08-12: current stable/GA, Flash-family, accepts
# image input, supports structured JSON output, free-tier available --
# the Gemini 2.0 Flash generation this repo might otherwise have assumed
# was shut down 2026-06-01. Same "research/production config surface,
# never a UI choice" contract as VISUAL_OBSERVER_MODEL_ENV_VAR above.
GEMINI_VISUAL_OBSERVER_MODEL_ENV_VAR = "DOAR_GEMINI_VISUAL_OBSERVER_MODEL"
DEFAULT_GEMINI_VISUAL_OBSERVER_MODEL = "gemini-3.6-flash"


def resolve_gemini_visual_observer_model() -> str:
    """The Gemini vision model `GeminiVisualObserver` uses when the
    caller doesn't inject a different one explicitly. Only configurable
    via `DOAR_GEMINI_VISUAL_OBSERVER_MODEL` -- never exposed as a
    normal-user UI choice."""
    return os.environ.get(GEMINI_VISUAL_OBSERVER_MODEL_ENV_VAR, DEFAULT_GEMINI_VISUAL_OBSERVER_MODEL)


# DOAR Visual Verifier: which Gemini model the independent, label-blind
# case-level verifier uses. Deliberately a DIFFERENT model than the
# observer (`gemini-3.6-flash`) -- an independent model, not just an
# independent prompt, for a narrower single-region task. `gemini-3.5-
# flash-lite` was VERIFIED (not assumed) against ai.google.dev on
# 2026-08-13: current stable/GA, a smaller Flash-Lite model (vs. the
# observer's full Flash model), accepts image input, supports structured
# JSON output. Same "research/production config surface, never a UI
# choice" contract as the observer model env vars above.
GEMINI_VISUAL_VERIFIER_MODEL_ENV_VAR = "DOAR_GEMINI_VISUAL_VERIFIER_MODEL"
DEFAULT_GEMINI_VISUAL_VERIFIER_MODEL = "gemini-3.5-flash-lite"


def resolve_gemini_visual_verifier_model() -> str:
    """The Gemini vision model `GeminiVisualVerifier` uses when the
    caller doesn't inject a different one explicitly. Only configurable
    via `DOAR_GEMINI_VISUAL_VERIFIER_MODEL` -- never exposed as a
    normal-user UI choice."""
    return os.environ.get(GEMINI_VISUAL_VERIFIER_MODEL_ENV_VAR, DEFAULT_GEMINI_VISUAL_VERIFIER_MODEL)


@dataclass(frozen=True)
class ProductionAnalysisConfig:
    """Resolved once per process, reused for every case. Immutable --
    nothing in the normal analysis pipeline mutates this after
    resolution; a different production configuration requires a code
    change and a new phase, never a runtime UI choice."""
    expressive_model_identifier: str
    expressive_model_checkpoint: str | None    # None if not available on this machine -- never a stand-in path
    expressive_model_available: bool
    expressive_model_unavailable_reason: str | None
    visual_detector_policy_path: str
    visual_detector_policy_version: str        # cites the frozen eye-policy status for provenance
    config_version: str = "production_config_v1"

    def to_dict(self) -> dict:
        return {
            "expressive_model_identifier": self.expressive_model_identifier,
            "expressive_model_checkpoint": self.expressive_model_checkpoint,
            "expressive_model_available": self.expressive_model_available,
            "expressive_model_unavailable_reason": self.expressive_model_unavailable_reason,
            "visual_detector_policy_path": self.visual_detector_policy_path,
            "visual_detector_policy_version": self.visual_detector_policy_version,
            "config_version": self.config_version,
        }


def resolve_production_config() -> ProductionAnalysisConfig:
    """The one function the app (or any other caller) should use to
    decide what to run -- never construct `ProductionAnalysisConfig`
    directly with a different checkpoint choice. Checking `Path.exists()`
    here, once, is exactly what lets `emotion.py::predict` receive a
    clean `checkpoint=None` (its own already-correct, already-tested
    "unavailable" path) instead of a path to a file that doesn't exist
    (which produces the less honest "failed" status -- see
    DOAR_V1_1_STABILIZATION_REPORT.md Problem B)."""
    checkpoint_available = EXPRESSIVE_MODEL_CHECKPOINT_PATH.exists()
    reason = None if checkpoint_available else (
        f"No trained checkpoint present at {EXPRESSIVE_MODEL_CHECKPOINT_PATH} in this environment "
        "(outputs/ is git-ignored and not populated on this machine) -- the same production "
        "identifier resolves automatically wherever this checkpoint file exists.")
    eye_status = "unknown"
    if VISUAL_DETECTOR_POLICY_PATH.exists():
        rows = json.loads(VISUAL_DETECTOR_POLICY_PATH.read_text(encoding="utf-8"))
        eye_row = next((r for r in rows if r["target"] == "eye"), None)
        eye_status = eye_row["status"] if eye_row else "unknown"
    return ProductionAnalysisConfig(
        expressive_model_identifier=EXPRESSIVE_MODEL_IDENTIFIER,
        expressive_model_checkpoint=(str(EXPRESSIVE_MODEL_CHECKPOINT_PATH) if checkpoint_available else None),
        expressive_model_available=checkpoint_available,
        expressive_model_unavailable_reason=reason,
        visual_detector_policy_path=str(VISUAL_DETECTOR_POLICY_PATH),
        visual_detector_policy_version=f"phase2c7_frozen (eye={eye_status})",
    )
