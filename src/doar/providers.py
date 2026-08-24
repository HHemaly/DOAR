"""Pluggable provider boundaries (additive, define-only).

Thin interfaces + default implementations wrapping ALREADY-EXISTING,
unmodified functions -- this module invents no new inference logic. Its
only purpose is to give each scientific role (emotion, object evidence,
semantic concepts) a stable seam so a future experimentally-selected
winner can be swapped in via `research_runtime.py` without rewriting
`case_interpretation.py`/`doar_prototype_app.py`.

`GlobalImageRepresentationProvider` and `EvidenceAggregator` already live
in `case_interpretation.py` (added in an earlier milestone) and are
re-exported here for a single, discoverable import point -- not
duplicated.
"""
from __future__ import annotations

from .case_interpretation import EvidenceAggregator, GlobalImageRepresentationProvider  # noqa: F401 (re-export)


class EmotionProvider:
    """Interface: predict the 4-class expressive distribution for one
    image. `predict()` returns the SAME dict shape doar.emotion.predict()
    already produces (status/probabilities/top_class/... -- see
    emotion.py's own `_summary`/`unavailable` for the canonical shape)."""

    def predict(self, image_path: str) -> dict:  # pragma: no cover -- interface only
        raise NotImplementedError


class ProductionEmotionProvider(EmotionProvider):
    """Default implementation -- wraps doar.emotion.predict() with
    whatever checkpoint doar.production_config.resolve_production_config()
    resolves (frozen production checkpoint if present, else the
    explicitly-labeled E1 DEVELOPMENT checkpoint, else unavailable).
    Zero new inference logic; this class only fixes the checkpoint choice
    once so callers don't each re-resolve it."""

    def __init__(self, checkpoint: str | None = None):
        self._explicit_checkpoint = checkpoint

    def predict(self, image_path: str) -> dict:
        from . import emotion
        checkpoint = self._explicit_checkpoint
        if checkpoint is None:
            from .production_config import resolve_production_config
            checkpoint = resolve_production_config().expressive_model_checkpoint
        return emotion.predict(image_path, checkpoint)


class ObjectEvidenceProvider:
    """Interface: run the deep, open-vocabulary object/entity scan for
    one case and persist detections.json -- the SAME contract
    doar.visual_evidence.run_and_persist_initial_scan() already has."""

    def scan(self, case_dir, image_path: str, *, eye_entry, registry_v2, model_predict_fns) -> None:
        raise NotImplementedError  # pragma: no cover -- interface only


class GroundingDinoOwlv2ObjectEvidenceProvider(ObjectEvidenceProvider):
    """Default implementation -- wraps doar.visual_evidence.
    run_and_persist_initial_scan() (Grounding-DINO primary / OWLv2
    fallback, the frozen phase2c7 eye policy) UNCHANGED. Deferred/
    on-demand only -- callers decide when to invoke it (the Psychologist
    view's "Run deep visual analysis" / "Run Full Analysis" actions)."""

    def scan(self, case_dir, image_path: str, *, eye_entry, registry_v2, model_predict_fns) -> None:
        from .visual_evidence import run_and_persist_initial_scan
        run_and_persist_initial_scan(case_dir, image_path, eye_entry=eye_entry, registry_v2=registry_v2,
                                     model_predict_fns=model_predict_fns)


class SemanticConceptProvider:
    """Interface: turn a governed bundle into human-readable visual
    concepts -- the SAME return shape case_interpretation.
    build_visual_concepts() already produces."""

    def build_concepts(self, bundle: dict) -> tuple:
        raise NotImplementedError  # pragma: no cover -- interface only


class DefaultSemanticConceptProvider(SemanticConceptProvider):
    """Default implementation -- wraps case_interpretation.
    build_visual_concepts() (entities + already-eligible rule matches)
    UNCHANGED. A future learned concept extractor would implement this
    same interface without CaseInterpretation needing to change."""

    def build_concepts(self, bundle: dict) -> tuple:
        from .case_interpretation import build_visual_concepts
        return build_visual_concepts(bundle)
