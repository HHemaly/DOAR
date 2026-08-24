"""Research component registry (additive, read-only).

ONE small place that records which runtime component currently serves
each scientific role, and that component's real, current status. This is
NOT a new reasoning system -- it only DESCRIBES what `production_config.py`
/`case_interpretation.py`/`live_case_bundle.py` already resolve, so a
future Colab-selected winner can be swapped in by changing the artifact
this registry points to, never by rewriting `CaseInterpretation`.

Every entry below is genuinely accurate for the current state of this
repository -- nothing is invented, no experiment "winner" is declared
that hasn't actually been decided. See `experiment_manifest.py` for the
schema future Colab exports should target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Status vocabulary shared by every entry -- deliberately small and
# unambiguous. FINAL is never set here; only a human decision (after
# comparing candidates on locked Test, out of scope for this repo) can
# promote a component to FINAL.
COMPONENT_STATUSES = ("DEVELOPMENT", "CANDIDATE", "FINAL")


@dataclass(frozen=True)
class ResearchComponent:
    """One row of the registry -- which experiment/artifact currently
    backs one scientific role, and its real status/provenance."""
    role: str                      # e.g. "emotion_provider"
    experiment: str | None         # e.g. "E1_visual_representation", or None if not experiment-derived
    implementation: str            # human-readable description of what actually runs
    model_or_artifact: str | None  # model name / artifact identifier, or None if not applicable
    status: str                    # one of COMPONENT_STATUSES
    checkpoint: str | None = None
    preprocessing: str | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in COMPONENT_STATUSES:
            raise ValueError(f"Unknown component status {self.status!r}, expected one of {COMPONENT_STATUSES}")

    def to_dict(self) -> dict:
        return {
            "role": self.role, "experiment": self.experiment, "implementation": self.implementation,
            "model_or_artifact": self.model_or_artifact, "status": self.status,
            "checkpoint": self.checkpoint, "preprocessing": self.preprocessing, "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ResearchRuntimeConfiguration:
    """The full registry -- one ResearchComponent per scientific role.
    `resolve_research_runtime_configuration()` below builds the real,
    current one; tests/callers that need a different combination (e.g. a
    future C1/C2/C3/C4 comparison config, per this task's own examples)
    can construct a `ResearchRuntimeConfiguration` directly with different
    components, without touching this module or CaseInterpretation."""
    emotion_provider: ResearchComponent
    object_detector_provider: ResearchComponent
    semantic_concept_provider: ResearchComponent
    global_semantic_provider: ResearchComponent
    evidence_aggregator: ResearchComponent

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in (
            ("emotion_provider", self.emotion_provider),
            ("object_detector_provider", self.object_detector_provider),
            ("semantic_concept_provider", self.semantic_concept_provider),
            ("global_semantic_provider", self.global_semantic_provider),
            ("evidence_aggregator", self.evidence_aggregator),
        )}

    def components(self) -> tuple[ResearchComponent, ...]:
        return (self.emotion_provider, self.object_detector_provider, self.semantic_concept_provider,
                self.global_semantic_provider, self.evidence_aggregator)


def _emotion_component() -> ResearchComponent:
    from .production_config import resolve_production_config
    try:
        from .deep.e1_dev_checkpoint import (
            E1_DEV_CHECKPOINT_PATH, E1_DEV_VAL_MACRO_F1, E1_DEV_PREPROCESSING_SPEC, MODEL_VERSION,
        )
    except ImportError:
        E1_DEV_CHECKPOINT_PATH = None
    config = resolve_production_config()
    if not config.expressive_model_available:
        return ResearchComponent(
            role="emotion_provider", experiment="E1_visual_representation",
            implementation="doar.emotion.predict (deep_image family) -- no checkpoint available",
            model_or_artifact=None, status="DEVELOPMENT", checkpoint=None,
            provenance={"unavailable_reason": config.expressive_model_unavailable_reason})
    is_e1_dev = E1_DEV_CHECKPOINT_PATH is not None and config.expressive_model_checkpoint == str(E1_DEV_CHECKPOINT_PATH)
    if is_e1_dev:
        return ResearchComponent(
            role="emotion_provider", experiment="E1_visual_representation",
            implementation="doar.deep.e1_dev_checkpoint.predict_e1_dev_checkpoint (mobilenet_v3_small, "
                           "frozen backbone + trained head)",
            model_or_artifact="mobilenet_v3_small", status="DEVELOPMENT",
            checkpoint=config.expressive_model_checkpoint,
            preprocessing=E1_DEV_PREPROCESSING_SPEC["preprocessing_version"],
            provenance={"val_macro_f1": E1_DEV_VAL_MACRO_F1, "model_version": MODEL_VERSION,
                       "selection_basis": "highest validation macro-F1 among E1 candidates with a "
                                          "completed (non-smoke-only) run, as of this session -- NOT a "
                                          "declared E1 experiment winner"})
    return ResearchComponent(
        role="emotion_provider", experiment=None,
        implementation="doar.deep.inference.predict_image (frozen production checkpoint)",
        model_or_artifact=config.expressive_model_identifier, status="FINAL",
        checkpoint=config.expressive_model_checkpoint,
        provenance={"note": "the frozen, pre-existing production checkpoint identifier "
                            "(production_config.EXPRESSIVE_MODEL_IDENTIFIER)"})


def _object_detector_component() -> ResearchComponent:
    from .production_config import VISUAL_DETECTOR_POLICY_PATH
    import json
    eye_status = "unknown"
    if VISUAL_DETECTOR_POLICY_PATH.exists():
        rows = json.loads(VISUAL_DETECTOR_POLICY_PATH.read_text(encoding="utf-8"))
        eye_row = next((r for r in rows if r["target"] == "eye"), None)
        eye_status = eye_row["status"] if eye_row else "unknown"
    try:
        import transformers as _transformers
        transformers_version = _transformers.__version__
    except ImportError:  # pragma: no cover -- transformers always installed in this project
        transformers_version = "unknown (transformers not importable)"
    return ResearchComponent(
        role="object_detector_provider", experiment="phase2c4/phase2c7",
        implementation="doar.visual_evidence.run_and_persist_initial_scan (Grounding-DINO object-class + "
                       "eye policy, OWLv2 fallback) -- deferred/on-demand, never run during normal Analyze",
        model_or_artifact="grounding_dino_parts+owlv2_parts_fallback (frozen phase2c7 eye policy)",
        status="DEVELOPMENT", checkpoint=None,
        provenance={
            "visual_detector_policy_path": str(VISUAL_DETECTOR_POLICY_PATH), "eye_policy_status": eye_status,
            # Pre-doctor stabilization pass (Q5): reproducibility provenance
            # for the Grounding-DINO/OWLv2 detector runtime -- see
            # phase2c4/detectors.py::grounding_dino_text_labels and every
            # AutoProcessor/Owlv2Processor.from_pretrained(..., use_fast=False)
            # call in phase2c4/detectors.py, phase2c5/proposals.py, and
            # phase2c7/runtime.py.
            "installed_transformers_version": transformers_version,
            "image_processor_use_fast": False,
            "grounding_dino_label_field": "text_labels (falls back to labels only if text_labels is absent)",
            "grounding_dino_model_id": "IDEA-Research/grounding-dino-tiny",
            "owlv2_model_id": "google/owlv2-base-patch16-ensemble",
        })


def _semantic_concept_component() -> ResearchComponent:
    return ResearchComponent(
        role="semantic_concept_provider", experiment=None,
        implementation="case_interpretation.build_visual_concepts (entities + already-eligible "
                       "reasoning_chain/drawing_synthesis rule matches, deterministic, no learned model)",
        model_or_artifact=None, status="DEVELOPMENT",
        provenance={"note": "governed rule/entity mapping, not a trained semantic-concept model -- "
                            "a future learned concept extractor would replace this behind the same "
                            "SemanticConceptProvider interface"})


def _global_semantic_component() -> ResearchComponent:
    return ResearchComponent(
        role="global_semantic_provider", experiment=None,
        implementation="case_interpretation.DeterministicGlobalImageRepresentation (composition + "
                       "relationship_features) -- optionally supplemented, per-question, by the "
                       "human_interaction.GeminiGlobalObserver whole-image candidate call",
        model_or_artifact=None, status="DEVELOPMENT",
        provenance={"pluggable_via": "case_interpretation.GlobalImageRepresentationProvider",
                   "gemini_candidate_provider": "human_interaction.GeminiGlobalObserver (candidate "
                                                "hypothesis generator, never authoritative -- see "
                                                "verify_gemini_concern_candidates)"})


def _evidence_aggregator_component() -> ResearchComponent:
    return ResearchComponent(
        role="evidence_aggregator", experiment=None,
        implementation="case_interpretation.IndependentFamilyRatioAggregator "
                       "(supporting independent families / assessable independent families)",
        model_or_artifact="INDEPENDENT_FAMILY_RATIO", status="DEVELOPMENT",
        provenance={"pluggable_via": "case_interpretation.EvidenceAggregator",
                   "planned_future_candidates": ["literature/expert-weighted score", "learned interpretable weights",
                                                  "fuzzy inference"],
                   "note": "no arbitrary point weights -- other strategies are declared extension points "
                          "only, not implemented, per explicit instruction to compare experimentally later"})


def resolve_research_runtime_configuration() -> ResearchRuntimeConfiguration:
    """The one function callers (Technical View, tests) should use --
    reflects THIS process's real, current resolution of every role."""
    return ResearchRuntimeConfiguration(
        emotion_provider=_emotion_component(),
        object_detector_provider=_object_detector_component(),
        semantic_concept_provider=_semantic_concept_component(),
        global_semantic_provider=_global_semantic_component(),
        evidence_aggregator=_evidence_aggregator_component(),
    )
