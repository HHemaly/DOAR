"""Adapters: existing evidence pipeline -> ontology layer (DOAR V2).

This is the seam between the already-working, already-tested V1.5
evidence pipeline (`evidence_schema.py`, `evidence_adapter.py`,
`rule_schema.py`, `evidence_rule_engine.py` -- all untouched by this
package) and the new general ontology primitives
(`observation.py`/`store.py`/`rule.py`/`engine.py`).

Nothing here re-implements extraction or re-derives the 19-rule
registry's source citations: `evidence_item_to_observation` reuses real
`EvidenceItem`s produced by `evidence_adapter.build_evidence_set`
(itself reusing `analysis.py`), and `legacy_rule_to_ontology_rule` reuses
the provenance/threshold data already assembled in `rule_schema.py`.

`run_ontology_demo_on_image` is the task's required minimal end-to-end
demonstration: image -> existing extractor(s) -> FeatureObservation(s)
-> feature store -> existing rules (re-expressed declaratively) -> rule
trace. No mocked data anywhere in this path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..analysis import analyze_image
from ..evidence_adapter import build_evidence_set
from ..evidence_schema import EvidenceItem, EvidenceSet
from ..rule_schema import RuleV2, load_rules_v2
from .context import EMPTY_CONTEXT
from .engine import RuleEvaluation, evaluate_all_rules
from .feature_schema import FeatureType
from .observation import BoundingBox, FeatureObservation
from .rule import Citation, ContextRequirements, EvidenceGrade, FeatureCondition, AllOf, Rule, RuleFramework
from .store import FeatureStore

# rule_schema.py's `scientific_support` free-text values -> this package's
# closed EvidenceGrade vocabulary. Every value actually present in
# resources/psychology_sources/rules_registry.json is mapped explicitly;
# an unmapped value raises rather than silently defaulting, so a new rule
# added to that registry cannot slip through unclassified.
_SCIENTIFIC_SUPPORT_TO_GRADE: dict[str, str] = {
    "not_found_for_specific_claim": EvidenceGrade.NOT_FOUND,
    "not_found_for_combined_claim": EvidenceGrade.NOT_FOUND,
    "not_found_for_personality_claim": EvidenceGrade.NOT_FOUND,
    "not_found_for_threshold_claim": EvidenceGrade.NOT_FOUND,
    "indirect_expression_research_only": EvidenceGrade.INDIRECT,
    "indirect_size_research_conflicting": EvidenceGrade.CONFLICTING,
    "indirect_size_research_does_not_validate_claim": EvidenceGrade.INDIRECT,
    "placement_effects_weak_not_specific": EvidenceGrade.INDIRECT,
}


def _infer_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return FeatureType.BOOLEAN
    if isinstance(value, (int, float)):
        return FeatureType.CONTINUOUS
    if isinstance(value, str):
        return FeatureType.CATEGORICAL
    if isinstance(value, (list, tuple)):
        return FeatureType.GEOMETRY
    return FeatureType.CATEGORICAL


def evidence_item_to_observation(item: EvidenceItem) -> FeatureObservation:
    """Direct, lossless conversion. Both schemas share the same core
    invariant (unassessable => value is None), so this never needs to
    invent a placeholder value."""
    bbox = None
    region = None
    if item.location is not None:
        region = item.location.region
        if item.location.bbox_xywh is not None:
            x, y, w, h = item.location.bbox_xywh
            bbox = BoundingBox(x=x, y=y, width=w, height=h, coordinate_space=item.location.coordinate_space)

    assessable = item.status in ("available", "experimental")
    return FeatureObservation(
        feature_id=item.feature_id,
        value=item.value,
        value_type=_infer_value_type(item.value) if item.value is not None else FeatureType.CATEGORICAL,
        assessable=assessable,
        source="existing_cv",
        not_assessable_reason=None if assessable else (item.reason or f"status={item.status}"),
        extractor_name=item.extractor,
        extractor_version=item.extractor_version,
        raw_confidence=item.confidence,
        bbox=bbox,
        evidence_region=region,
        notes=item.reason,
    )


def evidence_set_to_store(evidence: EvidenceSet) -> FeatureStore:
    """Populate a fresh `FeatureStore` from a real `EvidenceSet`. Also
    splits the combined `composition.centroid_normalized` geometry item
    into two derived scalar continuous features
    (`composition.centroid_x`/`composition.centroid_y`) -- the same
    scalar decomposition `features.py` already performs independently --
    so ontology `FeatureCondition`s can express simple placement
    thresholds without a "geometry" comparison operator."""
    store = FeatureStore()
    for item in evidence.items:
        obs = evidence_item_to_observation(item)
        store.add_observation(obs)
        if item.feature_id == "composition.centroid_normalized":
            for axis, idx in (("x", 0), ("y", 1)):
                derived_id = f"composition.centroid_{axis}"
                if obs.assessable:
                    derived = FeatureObservation(
                        feature_id=derived_id, value=float(obs.value[idx]), value_type=FeatureType.CONTINUOUS,
                        assessable=True, source="existing_cv", extractor_name=obs.extractor_name,
                        extractor_version=obs.extractor_version, raw_confidence=obs.raw_confidence,
                        notes=f"Derived from composition.centroid_normalized[{idx}].",
                    )
                else:
                    derived = FeatureObservation(
                        feature_id=derived_id, value=None, value_type=FeatureType.CONTINUOUS,
                        assessable=False, source="existing_cv",
                        not_assessable_reason=obs.not_assessable_reason,
                        extractor_name=obs.extractor_name, extractor_version=obs.extractor_version,
                    )
                store.add_observation(derived)
    return store


# `observable` (rule_schema.py's key into REQUIRED_FEATURE_IDS) -> a
# premise built over the ontology feature_ids it corresponds to. Built by
# hand once here because the arithmetic (which threshold, which
# direction) is real, already-reviewed domain knowledge copied verbatim
# from `evidence_rule_engine.py`'s `_COVERAGE_OBSERVABLES`/
# `_PLACEMENT_SPLIT` -- not re-derived or guessed.
def _premise_for_observable(observable: str) -> Any:
    coverage_feature = "composition.bounding_box_coverage"
    if observable == "coverage_about_half":
        return AllOf((FeatureCondition(coverage_feature, "gte", 0.40), FeatureCondition(coverage_feature, "lte", 0.60)))
    if observable == "coverage_full":
        return FeatureCondition(coverage_feature, "gte", 0.90)
    if observable == "coverage_small":
        return AllOf((FeatureCondition(coverage_feature, "gt", 0.0), FeatureCondition(coverage_feature, "lte", 0.20)))
    if observable == "placement_top":
        return FeatureCondition("composition.centroid_y", "lt", 0.4)
    if observable == "placement_left":
        return FeatureCondition("composition.centroid_x", "lt", 0.4)
    if observable == "placement_right":
        return FeatureCondition("composition.centroid_x", "gt", 0.6)
    # Every tier-2 (content-conditional) observable maps to exactly one
    # canonical "unavailable" feature_id in evidence_adapter.py's
    # REQUIRED_FEATURE_IDS-equivalent mapping (rule_schema.py). A single
    # presence check is the finest distinction the current evidence layer
    # supports for these -- see docs/DOAR_V2_RULE_AUDIT.md for the
    # multiple-rules-per-feature ambiguity this causes.
    from ..rule_schema import REQUIRED_FEATURE_IDS
    feature_id = REQUIRED_FEATURE_IDS[observable][0]
    return FeatureCondition(feature_id, "equals", True)


def legacy_rule_to_ontology_rule(rule: RuleV2) -> Rule:
    """Every rule in the existing 19-rule registry is sourced from one
    non-clinical, non-peer-reviewed Arabic notes document (see
    `resources/psychology_sources/rules_registry.json`'s own `source`
    block: `scientifically_validated: false`). None of them is an
    empirically-supported expressive-drawing rule -- they are folk/
    projective symbol interpretations -- so every one is tagged
    CLASSICAL_PROJECTIVE here. This is not a downgrade: it is the
    correct classification the source material itself documents."""
    grade = _SCIENTIFIC_SUPPORT_TO_GRADE.get(rule.scientific_support)
    if grade is None:
        raise ValueError(
            f"Rule {rule.rule_id!r} has scientific_support {rule.scientific_support!r} with no "
            "EvidenceGrade mapping in _SCIENTIFIC_SUPPORT_TO_GRADE -- add one rather than defaulting."
        )
    citation = Citation(
        document=rule.source_document, page=rule.source_page, section=rule.source_section,
        quote=rule.faithful_source_quote, status="documented",
    )
    return Rule(
        rule_id=rule.rule_id,
        name=rule.title_english,
        framework=RuleFramework.CLASSICAL_PROJECTIVE,
        premise=_premise_for_observable(rule.observable),
        domain=rule.observable,
        citation=citation,
        evidence_grade=grade,
        requirements=ContextRequirements(),  # all 19 current rules are image-only; see rule audit
        confidence_ceiling=rule.confidence_ceiling,
        professional_reasoning=rule.professional_reasoning,
        parent_safe_wording=rule.parent_safe_wording,
        limitations=tuple(rule.limitations),
        requires_clinician_review=rule.requires_clinician_review,
    )


def load_legacy_rules_as_ontology_rules() -> list[Rule]:
    return [legacy_rule_to_ontology_rule(r) for r in load_rules_v2()]


def run_ontology_demo_on_image(image_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Minimal end-to-end demonstration (task requirement):

        image -> analyze_image (existing) -> build_evidence_set (existing)
        -> FeatureObservation(s) -> FeatureStore -> ontology Rules
        (converted from the existing 19-rule registry) -> RuleEvaluation
        trace, for every rule, with an empty CaseContext (no age/task/
        narrative supplied -- proving the pipeline runs and degrades
        correctly without them, not that it needs them).

    Writes `ontology_observations.json` and `ontology_rule_trace.json` to
    `output_dir` and returns a summary dict.
    """
    output_dir = Path(output_dir)
    analysis = analyze_image(image_path, output_dir).to_dict()
    evidence = build_evidence_set(analysis, artifacts=analysis["artifacts"])
    store = evidence_set_to_store(evidence)
    rules = load_legacy_rules_as_ontology_rules()
    evaluations: list[RuleEvaluation] = evaluate_all_rules(rules, store, EMPTY_CONTEXT)

    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "ontology_observations.json"
    trace_path = output_dir / "ontology_rule_trace.json"

    all_observations = {
        fid: [obs.to_dict() for obs in store.history(fid)] for fid in store.feature_ids()
    }
    observations_path.write_text(json.dumps(all_observations, indent=2, default=str), encoding="utf-8")
    trace_path.write_text(json.dumps([e.to_dict() for e in evaluations], indent=2, default=str), encoding="utf-8")

    outcome_counts: dict[str, int] = {}
    for evaluation in evaluations:
        outcome_counts[evaluation.outcome] = outcome_counts.get(evaluation.outcome, 0) + 1

    return {
        "image_path": str(image_path),
        "feature_count": len(store.feature_ids()),
        "rule_count": len(rules),
        "outcome_counts": outcome_counts,
        "output_files": {"observations": str(observations_path), "rule_trace": str(trace_path)},
    }
