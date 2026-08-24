"""DOAR interpretable case-synthesis layer (additive extension).

Builds ONE governed `CaseInterpretation` from a case's EXISTING,
already-computed evidence (the same `bundle` shape
`live_case_bundle.py`/`human_interaction.py` already use) -- this module
invents no new detector, no new rule, and no new threshold. It only:

1. Groups already-eligible rule matches and already-verified entities
   into human-readable "visual concepts" (`build_visual_concepts`).
2. Reads the already-computed emotion prediction (`bundle["emotion"]`,
   raw grounding data added in Milestone 1) into an `ExpressiveProfile`,
   adding a SMALL, fully-declared table of richer expressive descriptors
   -- each one gated behind real, cited evidence (never invented).
3. Scores every concern domain in the frozen `CONCERN_DOMAIN_MAP.json`
   via a small, PLUGGABLE `EvidenceAggregator` interface (see "3. Concern
   domains" below) -- the only implementation provided is the transparent
   INDEPENDENT_FAMILY_RATIO baseline (supporting independent evidence
   families / assessable independent evidence families, no invented point
   weights). Domains with no assessable evidence, or no rule mapped to
   them at all (including the permanently-empty, explicitly-forbidden
   `maltreatment_or_safety_concern` domain -- see CONCERN_DOMAIN_MAP.
   json's own prohibition, grounded in SCIENTIFIC_LIMITATIONS.md Section
   3), are reported as INSUFFICIENT/NOT_CURRENTLY_ASSESSED, never
   guessed. The expressive Happy/Sad/Fear/Angry classifier is NEVER an
   input to this scoring -- it is used only for `ExpressiveProfile` and
   the consistency comparisons below, so it can never indiscriminately
   strengthen a domain it has no relevance to.
4. Builds a whole-drawing `GlobalImpression` (composition + relational
   entity layout + the expressive classifier -- never one isolated local
   feature) behind a pluggable `GlobalImageRepresentationProvider`.
5. Compares expressive-emotion valence against concern-domain support,
   and separately the global impression against concern-domain support,
   to produce two governed consistency verdicts (`consistency_status` and
   `global_local_consistency`) plus a deterministic, template-composed
   final synthesis paragraph -- never an LLM call, never per-case
   hard-coding.

This module does NOT modify drawing_synthesis.py/reasoning_chain.py/
concerns.py/rules.py/live_case_bundle.py, does NOT change E1 methodology,
and does NOT touch any locked scientific threshold or rule definition.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from . import drawing_synthesis as ds
from . import reasoning_chain as rc

ROOT = Path(__file__).resolve().parents[2]
RULE_SOURCE_REGISTER_PATH = ROOT / "RULE_SOURCE_REGISTER.csv"


# Small presentation helpers, deliberately duplicated (not imported) from
# human_interaction.py: human_interaction.py's own AnswerProvider grounding
# package will import THIS module (to ground Ask DOAR's answers in the new
# synthesis, per the "one focused synthesis module" requirement), so this
# module must not import human_interaction.py back -- that would be a
# circular import. Each function here is a pure, few-line wrapper over
# already-frozen registry data, identical in behavior to its
# human_interaction.py counterpart.
def rule_display_name(rule_id: str, matrix_row: dict) -> str:
    observable = (matrix_row.get("observable_feature") or rule_id).replace("_", " ").strip()
    return observable[:1].upper() + observable[1:] if observable else rule_id


def family_display(evidence_family: str) -> str:
    return (evidence_family or "").replace("_", " ").strip().capitalize()


def domain_display(concern_domain: str) -> dict:
    entry = rc.load_concern_domain_map()["domains"].get(concern_domain, {})
    return {"label": entry.get("example_clinician_hypothesis_wording", concern_domain.replace("_", " ")),
            "description": entry.get("description", "")}


def _load_rule_source_register() -> dict[str, dict]:
    if not RULE_SOURCE_REGISTER_PATH.exists():
        return {}
    with open(RULE_SOURCE_REGISTER_PATH, encoding="utf-8") as f:
        return {row["rule_id"]: row for row in csv.DictReader(f)}


def source_citation(rule_id: str) -> dict:
    register_row = _load_rule_source_register().get(rule_id)
    matrix_row = rc.load_rule_matrix().get(rule_id, {})
    pdf_section = matrix_row.get("source_pdf_page_section") or None
    if register_row is None:
        return {"available": bool(pdf_section), "citation_title": None, "source_pdf_page_section": pdf_section}
    return {"available": True, "citation_title": register_row.get("citation_title") or None,
            "source_pdf_page_section": pdf_section}


# ---------------------------------------------------------------------------
# Domains excluded from concern-domain classification entirely -- not
# "unsupported", genuinely NOT clinical concern constructs at all (per
# their own CONCERN_DOMAIN_MAP.json descriptions: composition-only facts,
# or a personality-trait source claim). Kept separate from
# `_FORBIDDEN_DOMAINS` below on purpose.
_NON_CLINICAL_DOMAINS = frozenset({
    "neutral_descriptive_only", "neutral_descriptive_only_no_construct_proposed",
})

# CONCERN_DOMAIN_MAP.json's own explicit, permanent prohibition: zero
# candidate rules exist for this domain today, and none may ever be added
# without a separate, deliberate registry change -- "NEVER a hypothesis
# DOAR proposes on its own" (SCIENTIFIC_LIMITATIONS.md Section 3). Always
# reported as NOT_CURRENTLY_ASSESSED, never WEAK/MODERATE/STRONG.
_FORBIDDEN_DOMAINS = frozenset({"maltreatment_or_safety_concern"})

# Domains sometimes asked about that have NO rule/literature mapping in
# DOAR's registry at all (distinct from `_FORBIDDEN_DOMAINS`, which IS in
# the registry but deliberately empty) -- exposed as NOT_CURRENTLY_ASSESSED
# for transparency rather than silently omitted.
_UNMAPPED_DOMAINS = {
    "trauma_related": "Trauma-related indicators",
    "bullying_related": "Bullying-related indicators",
    "autism_related": "Autism-related indicators",
}

_POSITIVE_EMOTION_CLASSES = frozenset({"Happy"})
_NEGATIVE_EMOTION_CLASSES = frozenset({"Sad", "Fear", "Angry"})

# Domains whose support counts as a "negative-valence concern" for the
# consistency/conflict check -- excludes the positive-affect domain itself
# and the two non-clinical/forbidden/unmapped groups above.
_NEGATIVE_VALENCE_DOMAINS = frozenset({
    "anxiety_or_stress_related", "depressive_or_low_mood_related",
    "aggression_or_threat_related", "social_withdrawal_related",
    "developmental_or_attention_related",
})

_SUPPORT_RANK = {"NOT_CURRENTLY_ASSESSED": -1, "NONE": 0, "INSUFFICIENT": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}


def _to_dict(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    return obj


@dataclass(frozen=True)
class VisualConcept:
    concept_name: str
    human_readable_label: str
    status: str                       # "verified"/"uncertain"/"rejected"/"unreviewed"/"observed"
    supporting_evidence: tuple[str, ...]
    value: str | float | None
    confidence: float | None
    source: str                       # entity_id or rule_id this concept traces to
    provenance: dict
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "concept_name": self.concept_name, "human_readable_label": self.human_readable_label,
            "status": self.status, "supporting_evidence": list(self.supporting_evidence),
            "value": self.value, "confidence": self.confidence, "source": self.source,
            "provenance": self.provenance, "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ExpressiveDescriptor:
    label: str
    supporting_evidence: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict:
        return {"label": self.label, "supporting_evidence": list(self.supporting_evidence), "reason": self.reason}


@dataclass(frozen=True)
class ExpressiveProfile:
    availability: str                  # "available" / "unavailable" / "suppressed"
    unavailable_reason: str | None
    base_emotion_probabilities: dict | None
    top_emotion: str | None
    calibration_status: str | None
    model_identity: dict
    richer_descriptors: tuple[ExpressiveDescriptor, ...]
    supporting_visual_evidence: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "availability": self.availability, "unavailable_reason": self.unavailable_reason,
            "base_emotion_probabilities": self.base_emotion_probabilities, "top_emotion": self.top_emotion,
            "calibration_status": self.calibration_status, "model_identity": self.model_identity,
            "richer_descriptors": _to_dict(self.richer_descriptors),
            "supporting_visual_evidence": list(self.supporting_visual_evidence),
        }


@dataclass(frozen=True)
class ConcernDomainResult:
    domain: str
    domain_label: str
    support_level: str                 # NONE/WEAK/MODERATE/STRONG/INSUFFICIENT/NOT_CURRENTLY_ASSESSED
    support_ratio: float | None        # supporting_family_count / assessable_family_count, or None (see AggregationResult)
    assessable_family_count: int
    supporting_observations: tuple[str, ...]
    supporting_rule_ids: tuple[str, ...]
    independent_evidence_families: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    missing_expected_evidence: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    sources: tuple[dict, ...]
    plain_language_reason: str

    def to_dict(self) -> dict:
        return {
            "domain": self.domain, "domain_label": self.domain_label, "support_level": self.support_level,
            "support_ratio": self.support_ratio, "assessable_family_count": self.assessable_family_count,
            "supporting_observations": list(self.supporting_observations),
            "supporting_rule_ids": list(self.supporting_rule_ids),
            "independent_evidence_families": list(self.independent_evidence_families),
            "contradictory_evidence": list(self.contradictory_evidence),
            "missing_expected_evidence": list(self.missing_expected_evidence),
            "alternative_explanations": list(self.alternative_explanations),
            "sources": list(self.sources), "plain_language_reason": self.plain_language_reason,
        }


@dataclass(frozen=True)
class GlobalImpression:
    """The WHOLE-DRAWING level -- built from evidence that already
    describes the complete image (composition/coverage/placement,
    relational entity layout, the expressive classifier), never from one
    isolated local feature. `availability`/`provenance` make explicit
    which representation produced it, so a future E1-selected whole-image
    model can be swapped in via `GlobalImageRepresentationProvider`
    without changing this shape."""
    overall_scene: str
    expressive_tone: str
    supporting_global_evidence: tuple[str, ...]
    supporting_local_concepts: tuple[str, ...]
    contradictions: tuple[str, ...]
    availability: str                  # "available" / "partial" / "unavailable"
    provenance: dict

    def to_dict(self) -> dict:
        return {
            "overall_scene": self.overall_scene, "expressive_tone": self.expressive_tone,
            "supporting_global_evidence": list(self.supporting_global_evidence),
            "supporting_local_concepts": list(self.supporting_local_concepts),
            "contradictions": list(self.contradictions), "availability": self.availability,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CaseInterpretation:
    visual_concepts: tuple[VisualConcept, ...]
    expressive_profile: ExpressiveProfile
    concern_domains: tuple[ConcernDomainResult, ...]
    global_impression: GlobalImpression
    contradictions: tuple[str, ...]
    missing_information: tuple[str, ...]
    consistency_status: str            # CONSISTENT/MIXED/CONFLICT/INSUFFICIENT_EVIDENCE (emotion vs concern domains)
    global_local_consistency: str      # GLOBAL_LOCAL_CONSISTENT/_MIXED/_CONFLICT/INSUFFICIENT_EVIDENCE
    final_synthesis: str
    # Both optional/on-demand -- populated only when "Run Full Analysis" (or
    # a direct caller) supplies a GeminiGlobalObserver result. Never
    # required, never authoritative -- see verify_gemini_concern_candidates.
    gemini_global_observation: dict | None = None
    gemini_candidates: "tuple[GeminiConcernCandidate, ...]" = ()

    def to_dict(self) -> dict:
        return {
            "visual_concepts": _to_dict(self.visual_concepts),
            "expressive_profile": self.expressive_profile.to_dict(),
            "concern_domains": _to_dict(self.concern_domains),
            "global_impression": self.global_impression.to_dict(),
            "contradictions": list(self.contradictions),
            "missing_information": list(self.missing_information),
            "consistency_status": self.consistency_status,
            "global_local_consistency": self.global_local_consistency,
            "final_synthesis": self.final_synthesis,
            "gemini_global_observation": self.gemini_global_observation,
            "gemini_candidates": _to_dict(self.gemini_candidates),
        }


# ---------------------------------------------------------------------------
# 1. Visual concepts -- entities (already governed/verified) + already-
#    ELIGIBLE rule matches (semantic + deterministic, same eligibility
#    gate reasoning_chain.py/drawing_synthesis.py already enforce).
# ---------------------------------------------------------------------------

def _all_eligible_matches(entities: list, deterministic_features: dict) -> list:
    semantic = rc.build_eligible_matches(entities)
    det = ds.build_deterministic_eligible_matches(
        deterministic_features["composition"], deterministic_features["objective_features"],
        deterministic_features.get("page_reference"))
    return semantic + det


def build_visual_concepts(bundle: dict) -> tuple[VisualConcept, ...]:
    entities = bundle.get("entities") or []
    concepts = [
        VisualConcept(
            concept_name=e.canonical_label, human_readable_label=e.canonical_label,
            status=e.case_verification_status, supporting_evidence=(e.entity_id,),
            value=None, confidence=e.confidence, source=e.entity_id,
            provenance={"entity_id": e.entity_id, "detector": e.detector, "source": e.source},
            limitations=(),
        )
        for e in entities
    ]
    matches = _all_eligible_matches(entities, bundle["deterministic_features"])
    matrix = rc.load_rule_matrix()
    for m in matches:
        row = matrix.get(m.rule_id, {})
        concepts.append(VisualConcept(
            concept_name=row.get("observable_feature", m.rule_id),
            human_readable_label=rule_display_name(m.rule_id, row),
            status="observed", supporting_evidence=m.matched_entity_ids,
            value=None, confidence=None, source=m.rule_id,
            provenance={"rule_id": m.rule_id, "evidence_family": m.evidence_family,
                        "concern_domain": m.concern_domain, "allowed_output_level": m.allowed_output_level},
            limitations=m.alternative_explanations,
        ))
    return tuple(concepts)


# ---------------------------------------------------------------------------
# 2. Expressive profile -- base classifier probabilities (already computed,
#    raw grounding data from live_case_bundle.py) + a small, declared table
#    of richer descriptors, each requiring real supporting evidence.
# ---------------------------------------------------------------------------

def _emotion_valence(top_emotion: str | None) -> str:
    if top_emotion in _POSITIVE_EMOTION_CLASSES:
        return "positive"
    if top_emotion in _NEGATIVE_EMOTION_CLASSES:
        return "negative"
    return "neutral"


def _derive_richer_descriptors(top_emotion: str | None, concept_names: set[str],
                                domain_support: dict[str, str]) -> tuple[ExpressiveDescriptor, ...]:
    descriptors = []
    if top_emotion in _POSITIVE_EMOTION_CLASSES:
        matched_concepts = {"sun", "flower", "heart"} & concept_names
        if domain_support.get("positive_affect_or_social_engagement") in ("WEAK", "MODERATE", "STRONG"):
            descriptors.append(ExpressiveDescriptor(
                "cheerful", tuple(sorted(matched_concepts)) + ("positive_affect_or_social_engagement",),
                "Happy expressive classification, together with positive-affect-related visual evidence."))
        else:
            descriptors.append(ExpressiveDescriptor(
                "positive", (), "The expressive classifier's top class is Happy."))
    elif top_emotion in _NEGATIVE_EMOTION_CLASSES:
        descriptors.append(ExpressiveDescriptor(
            "negative", (), f"The expressive classifier's top class is {top_emotion}."))
    tense_concepts = {"heavy_line_pressure_appearance", "light_line_pressure_appearance"} & concept_names
    if tense_concepts:
        descriptors.append(ExpressiveDescriptor(
            "tense", tuple(sorted(tense_concepts)),
            "Line-pressure appearance associated in the literature with tension/energy was observed."))
    if domain_support.get("social_withdrawal_related") in ("WEAK", "MODERATE", "STRONG"):
        descriptors.append(ExpressiveDescriptor(
            "withdrawn-looking", ("social_withdrawal_related",),
            "Placement/composition evidence associated with social withdrawal was observed."))
    return tuple(descriptors)


def build_expressive_profile(bundle: dict, concept_names: set[str],
                              domain_support: dict[str, str]) -> ExpressiveProfile:
    emotion = bundle.get("emotion") or {}
    status = emotion.get("status", "unavailable")
    probabilities = emotion.get("probabilities") if status == "available" else None
    top_emotion = emotion.get("top_class") if status == "available" else None
    model_identity = {
        "model_name": emotion.get("model_name"), "model_version": emotion.get("model_version"),
        "checkpoint_sha256": emotion.get("checkpoint_sha256"),
        "preprocessing_version": emotion.get("preprocessing_version"),
    }
    descriptors = _derive_richer_descriptors(top_emotion, concept_names, domain_support) if top_emotion else ()
    supporting = tuple(sorted({e for d in descriptors for e in d.supporting_evidence}))
    return ExpressiveProfile(
        availability=status, unavailable_reason=emotion.get("reason") if status != "available" else None,
        base_emotion_probabilities=probabilities, top_emotion=top_emotion,
        calibration_status=emotion.get("calibration_status") if status == "available" else None,
        model_identity=model_identity, richer_descriptors=descriptors, supporting_visual_evidence=supporting,
    )


# ---------------------------------------------------------------------------
# 3. Concern domains -- a small, PLUGGABLE evidence-aggregation interface.
#
# No arbitrary point weights ("flower = +2 happy") anywhere here. The only
# implementation provided now is INDEPENDENT_FAMILY_RATIO: for each concern
# domain, every candidate rule in CONCERN_DOMAIN_MAP.json/RULE_EVIDENCE_
# MATRIX.csv is checked (reusing reasoning_chain.check_visual_preconditions/
# drawing_synthesis.check_deterministic_preconditions UNCHANGED) and grouped
# by evidence_family. A family only counts as "assessable" if DOAR can
# currently evaluate it at all (allowed_output_level=individual_heuristic_
# only AND the precondition check itself returned satisfied/not_satisfied,
# never blocked/not_assessable/disabled) -- a family DOAR cannot currently
# check is excluded from the denominator entirely, never counted as
# negative evidence. "Supporting" families are the assessable ones whose
# precondition was actually satisfied this case.
#
#     support_ratio = supporting_family_count / assessable_family_count
#
# IMPORTANT: the expressive Happy/Sad/Fear/Angry classifier is NEVER an
# input here -- emotion's only roles are the ExpressiveProfile and the
# consistency comparisons (_consistency_status/_global_local_consistency).
# Concern-domain support comes ONLY from domain-relevant governed rule
# evidence, so the classifier can never indiscriminately strengthen an
# unrelated domain just because that domain happens to have any match.
#
# Other aggregation strategies for later thesis comparison (literature/
# expert-weighted scoring, learned interpretable weights, fuzzy inference)
# are intentionally NOT implemented yet -- only declared as the pluggable
# interface's future extension points.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregationResult:
    support_level: str                 # NONE/WEAK/MODERATE/INSUFFICIENT (see EvidenceAggregator)
    support_ratio: float | None        # supporting_family_count / assessable_family_count, or None if 0 assessable
    supporting_family_count: int
    assessable_family_count: int
    supporting_families: tuple[str, ...]
    assessable_families: tuple[str, ...]


class EvidenceAggregator:
    """Pluggable interface -- swap the implementation passed to
    `build_concern_domains(bundle, aggregator=...)` without changing any
    other part of this module. Planned future implementations (NOT built
    yet, to be compared experimentally rather than chosen arbitrarily):
    literature/expert-weighted scoring, learned interpretable weights,
    fuzzy inference."""

    def aggregate(self, *, assessable_families: set[str], supporting_families: set[str]) -> AggregationResult:
        raise NotImplementedError


class IndependentFamilyRatioAggregator(EvidenceAggregator):
    """The only strategy implemented today. `support_level` is a coarse,
    COUNT-based (never point-weighted) convenience label reusing this
    codebase's own pre-existing convention (structured_report.py/
    drawing_synthesis.py already escalate a combined hypothesis only at
    >=2 independent evidence families) -- 0 supporting -> NONE, exactly 1
    -> WEAK, >=2 -> MODERATE. 0 assessable families at all -> INSUFFICIENT
    (nothing could be checked for this domain in this case, distinct from
    NOT_CURRENTLY_ASSESSED, which means no rule maps to the domain at
    all). The precise ratio/counts are always preserved regardless of
    which coarse label they round to."""

    def aggregate(self, *, assessable_families: set[str], supporting_families: set[str]) -> AggregationResult:
        n_assessable = len(assessable_families)
        n_supporting = len(supporting_families)
        if n_assessable == 0:
            return AggregationResult("INSUFFICIENT", None, 0, 0, (), ())
        ratio = n_supporting / n_assessable
        level = "NONE" if n_supporting == 0 else "WEAK" if n_supporting == 1 else "MODERATE"
        return AggregationResult(level, ratio, n_supporting, n_assessable,
                                  tuple(sorted(supporting_families)), tuple(sorted(assessable_families)))


_ASSESSABLE_CHECK_STATUSES = frozenset({"satisfied", "not_satisfied"})


def _domain_rule_checks(entities: list, deterministic_features: dict) -> tuple[dict, dict]:
    """Per-rule {rule_id: VisualPreconditionCheck} for ALL 41 rules (not
    just matched ones) -- reuses check_visual_preconditions/check_
    deterministic_preconditions UNCHANGED; the deterministic checks
    override the semantic checker's own "not_applicable_to_this_module"
    placeholder for the 10 composition/line rules it doesn't cover."""
    matrix = rc.load_rule_matrix()
    semantic = {c.rule_id: c for c in rc.check_visual_preconditions(entities)}
    det = {c.rule_id: c for c in ds.check_deterministic_preconditions(
        deterministic_features["composition"], deterministic_features["objective_features"],
        deterministic_features.get("page_reference"))}
    return matrix, {**semantic, **det}


def _plain_language_reason(agg: AggregationResult, supporting_observations: tuple[str, ...]) -> str:
    if agg.assessable_family_count == 0:
        return "No DOAR rule currently mapped to this domain could be assessed for this drawing."
    if agg.supporting_family_count == 0:
        return (f"None of the {agg.assessable_family_count} assessable evidence famil"
                f"{'y' if agg.assessable_family_count == 1 else 'ies'} for this domain were satisfied "
                f"for this drawing.")
    families = ", ".join(family_display(f) for f in agg.supporting_families)
    feature_text = f" ({', '.join(supporting_observations)})" if supporting_observations else ""
    return (f"{agg.supporting_family_count} of {agg.assessable_family_count} assessable independent evidence "
            f"famil{'y' if agg.assessable_family_count == 1 else 'ies'} support this domain: {families}"
            f"{feature_text}.")


def build_concern_domains(bundle: dict, *, aggregator: EvidenceAggregator | None = None
                           ) -> tuple[ConcernDomainResult, ...]:
    aggregator = aggregator or IndependentFamilyRatioAggregator()
    entities = bundle.get("entities") or []
    matrix, checks = _domain_rule_checks(entities, bundle["deterministic_features"])
    relationship_graph = rc.load_relationship_graph()

    domain_map = rc.load_concern_domain_map()["domains"]
    results = []
    for domain, defn in domain_map.items():
        if domain in _NON_CLINICAL_DOMAINS:
            continue
        label = domain_display(domain)["label"]
        if domain in _FORBIDDEN_DOMAINS:
            results.append(ConcernDomainResult(
                domain=domain, domain_label=label, support_level="NOT_CURRENTLY_ASSESSED",
                support_ratio=None, assessable_family_count=0,
                supporting_observations=(), supporting_rule_ids=(), independent_evidence_families=(),
                contradictory_evidence=(), missing_expected_evidence=(), alternative_explanations=(),
                sources=(),
                plain_language_reason=("DOAR does not generate hypotheses for this domain -- no rule in the "
                                        "registry maps here, by explicit design (see CONCERN_DOMAIN_MAP.json)."),
            ))
            continue

        candidate_rule_ids = defn.get("candidate_atomic_rule_ids", [])
        if not candidate_rule_ids:
            results.append(ConcernDomainResult(
                domain=domain, domain_label=label, support_level="NOT_CURRENTLY_ASSESSED",
                support_ratio=None, assessable_family_count=0,
                supporting_observations=(), supporting_rule_ids=(), independent_evidence_families=(),
                contradictory_evidence=(), missing_expected_evidence=(), alternative_explanations=(), sources=(),
                plain_language_reason="No rule is currently mapped to this domain in DOAR's registry.",
            ))
            continue

        assessable_families, supporting_families = set(), set()
        supporting_rule_ids, missing_reasons, alt_explanations = [], [], []
        for rid in candidate_rule_ids:
            row = matrix.get(rid)
            check = checks.get(rid)
            if row is None or check is None:
                continue
            family = row["evidence_family"]
            eligible = row["allowed_output_level"] == rc.ELIGIBLE_OUTPUT_LEVEL
            assessable = eligible and check.status in _ASSESSABLE_CHECK_STATUSES
            if not assessable:
                missing_reasons.append(f"{rule_display_name(rid, row)}: {check.reason}"
                                       if eligible else f"{rule_display_name(rid, row)}: no validated detector yet")
                continue
            assessable_families.add(family)
            if check.status == "satisfied":
                supporting_families.add(family)
                supporting_rule_ids.append(rid)
                if row.get("alternative_explanations"):
                    alt_explanations.extend(row["alternative_explanations"].split(" | "))

        contradicting = [edge["reference_id"] for edge in relationship_graph.get("context_limited_by_edges", [])
                         if any(rid in edge["rule_ids"] for rid in supporting_rule_ids)]

        agg = aggregator.aggregate(assessable_families=assessable_families, supporting_families=supporting_families)
        supporting_observations = tuple(sorted({matrix[rid]["observable_feature"] for rid in supporting_rule_ids}))
        results.append(ConcernDomainResult(
            domain=domain, domain_label=label, support_level=agg.support_level,
            support_ratio=agg.support_ratio, assessable_family_count=agg.assessable_family_count,
            supporting_observations=supporting_observations,
            supporting_rule_ids=tuple(sorted(supporting_rule_ids)),
            independent_evidence_families=agg.supporting_families,
            contradictory_evidence=tuple(sorted(set(contradicting))),
            missing_expected_evidence=tuple(sorted(set(missing_reasons))),
            alternative_explanations=tuple(sorted(set(alt_explanations))),
            sources=tuple(source_citation(rid) for rid in supporting_rule_ids),
            plain_language_reason=_plain_language_reason(agg, supporting_observations),
        ))

    for domain, label in _UNMAPPED_DOMAINS.items():
        results.append(ConcernDomainResult(
            domain=domain, domain_label=label, support_level="NOT_CURRENTLY_ASSESSED",
            support_ratio=None, assessable_family_count=0,
            supporting_observations=(), supporting_rule_ids=(), independent_evidence_families=(),
            contradictory_evidence=(), missing_expected_evidence=(), alternative_explanations=(), sources=(),
            plain_language_reason="No DOAR rule or literature mapping currently exists for this domain.",
        ))
    return tuple(results)


# ---------------------------------------------------------------------------
# 3.5. Global (whole-drawing) impression -- built ONLY from evidence that
# already describes the COMPLETE image (composition/coverage/placement,
# relational entity layout, the expressive classifier), never from one
# isolated local feature. The representation itself is pluggable
# (`GlobalImageRepresentationProvider`) so a future E1-selected whole-image
# model can replace `DeterministicGlobalImageRepresentation` without any
# change to `CaseInterpretation`'s shape or downstream consumers.
# ---------------------------------------------------------------------------


class GlobalImageRepresentationProvider:
    """Interface a whole-image representation provider must implement.
    `describe(bundle)` returns {"overall_scene": str, "evidence": list[str]}
    built from data that characterizes the COMPLETE drawing (never a
    single local feature). Not a `typing.Protocol` (kept a plain duck-typed
    interface, matching this module's dependency-light style) -- any
    object with a compatible `describe` method works."""

    def describe(self, bundle: dict) -> dict:  # pragma: no cover -- interface only
        raise NotImplementedError


class DeterministicGlobalImageRepresentation(GlobalImageRepresentationProvider):
    """The current, always-available whole-image representation: real
    composition (coverage/placement) + real relational entity layout
    (`relationship_features.compute_relationship_features`, already used
    by drawing_synthesis.py's own unified_evidence) -- no new detector, no
    new measurement. This is the DEVELOPMENT default `build_global_
    impression` uses when no provider is injected; swap in a different
    `GlobalImageRepresentationProvider` (e.g. the eventual E1-selected
    whole-image model) without touching any other part of this module."""

    def describe(self, bundle: dict) -> dict:
        from . import relationship_features as rf

        det = bundle.get("deterministic_features") or {}
        comp = det.get("composition") or {}
        entities = bundle.get("entities") or []
        rel = rf.compute_relationship_features(entities)

        coverage = comp.get("bounding_box_coverage")
        placement = comp.get("placement")
        count = rel["entity_count"]
        scene_parts = []
        if coverage is not None:
            scene_parts.append(f"the drawing covers about {coverage:.0%} of the page"
                               if isinstance(coverage, (int, float)) else f"the drawing covers {coverage}")
        if placement:
            scene_parts.append(f"is positioned {placement.replace('_', ' ')} on the page")
        if count:
            figure_word = "verified figure/object" if count == 1 else "verified figures/objects"
            scene_parts.append(f"contains {count} {figure_word}")
        if rel["repeated_labels"]:
            repeated = ", ".join(sorted(rel["repeated_labels"]))
            scene_parts.append(f"repeats: {repeated}")
        if rel["spatially_separated_entity_ids"]:
            scene_parts.append(f"{len(rel['spatially_separated_entity_ids'])} figure(s) spatially separated "
                               f"from the rest")
        overall_scene = ("Overall, " + "; ".join(scene_parts) + "."
                         ) if scene_parts else "Not enough whole-image measurement is available for this case."

        evidence = []
        if coverage is not None:
            evidence.append("composition.bounding_box_coverage")
        if placement:
            evidence.append("composition.placement")
        if count:
            evidence.append("relationships.entity_count")
        if rel["repeated_labels"]:
            evidence.append("relationships.repeated_labels")
        if rel["spatially_separated_entity_ids"]:
            evidence.append("relationships.spatially_separated_entity_ids")
        return {"overall_scene": overall_scene, "evidence": evidence}


def build_global_impression(bundle: dict, concepts: tuple[VisualConcept, ...],
                             profile: ExpressiveProfile, concern_by_domain: dict[str, str], *,
                             provider: GlobalImageRepresentationProvider | None = None,
                             visual_judge_audit: dict | None = None) -> GlobalImpression:
    """`visual_judge_audit` (optional): an ALREADY-RUN, session-only
    Visual Consistency Judge result (see human_interaction.run_visual_
    consistency_audit) -- included here ONLY as a clearly-labeled,
    AUDIT_ONLY note in `supporting_global_evidence`/`contradictions`,
    exactly like every other caller of that function. It never changes
    `overall_scene`/`expressive_tone`/support levels, and is never
    required (defaults to None -- the audit is on-demand, not automatic)."""
    provider = provider or DeterministicGlobalImageRepresentation()
    described = provider.describe(bundle)
    overall_scene = described.get("overall_scene", "")
    global_evidence = list(described.get("evidence", []))

    expressive_tone = (", ".join(d.label for d in profile.richer_descriptors)
                       or profile.top_emotion or "unavailable")
    if profile.availability == "available":
        global_evidence.append("expressive_model")

    supporting_local = tuple(sorted({c.concept_name for c in concepts
                                     if c.status in ("verified", "observed")}))

    contradictions = []
    valence = _emotion_valence(profile.top_emotion)
    negative_ranks = [_SUPPORT_RANK.get(concern_by_domain.get(d, "NONE"), 0) for d in _NEGATIVE_VALENCE_DOMAINS]
    if valence == "positive" and any(r >= 2 for r in negative_ranks):
        contradictions.append("The overall positive expressive tone conflicts with independently "
                              "supported concern-domain evidence -- see concern_domains below.")
    if visual_judge_audit and visual_judge_audit.get("status") == "ok":
        audit = visual_judge_audit["audit"]
        global_evidence.append("visual_consistency_judge(AUDIT_ONLY)")
        if audit.get("disputed_observations"):
            contradictions.append("AUDIT_ONLY: the Visual Consistency Judge disputed "
                                  f"{len(audit['disputed_observations'])} observation(s) -- for human review, "
                                  "never automatically applied.")
        if audit.get("likely_missed_observations"):
            contradictions.append("AUDIT_ONLY: the Visual Consistency Judge flagged possible missed content "
                                  f"({len(audit['likely_missed_observations'])} item(s)) -- for human review only.")

    availability = "available" if (global_evidence and overall_scene) else "unavailable"
    return GlobalImpression(
        overall_scene=overall_scene, expressive_tone=expressive_tone,
        supporting_global_evidence=tuple(global_evidence), supporting_local_concepts=supporting_local,
        contradictions=tuple(contradictions), availability=availability,
        provenance={"provider": provider.__class__.__name__},
    )


# ---------------------------------------------------------------------------
# 4. Consistency / conflict + final synthesis.
# ---------------------------------------------------------------------------

def _valence_vs_concern_consistency(
        valence: str, concern_by_domain: dict[str, str], *,
        labels: tuple[str, str, str, str] = ("CONSISTENT", "MIXED", "CONFLICT", "INSUFFICIENT_EVIDENCE"),
) -> str:
    """Shared decision logic behind both `consistency_status` (expressive
    classifier vs concern domains) and `global_local_consistency` (global
    impression vs concern domains) -- same rank-based, multi-family-aware
    reasoning either way, only the returned label set differs. A tiny,
    single-family local hint can never alone reach the CONFLICT-equivalent
    label (that requires >=2 independent evidence families, i.e. MODERATE+
    per `concerns._aggregation_strength`); a strong classifier likewise
    never silently erases WEAK+ concern-domain evidence -- it downgrades
    to the MIXED-equivalent label instead of being hidden."""
    consistent, mixed, conflict, insufficient = labels
    negative_levels = [concern_by_domain.get(d, "NONE") for d in _NEGATIVE_VALENCE_DOMAINS]
    negative_ranks = [_SUPPORT_RANK.get(lv, 0) for lv in negative_levels]
    strong_negative_count = sum(1 for r in negative_ranks if r >= 2)   # MODERATE or STRONG
    any_negative = any(r >= 1 for r in negative_ranks)                 # WEAK or above
    positive_level = concern_by_domain.get("positive_affect_or_social_engagement", "NONE")

    if valence == "neutral" and not any_negative and positive_level in ("NONE", "NOT_CURRENTLY_ASSESSED"):
        return insufficient
    if valence == "positive" and strong_negative_count >= 1:
        return conflict
    if valence == "positive" and any_negative:
        return mixed
    if valence == "negative" and any_negative:
        return consistent
    if valence == "negative" and not any_negative:
        return mixed
    return consistent


def _consistency_status(top_emotion: str | None, concern_by_domain: dict[str, str]) -> str:
    return _valence_vs_concern_consistency(_emotion_valence(top_emotion), concern_by_domain)


def _global_local_consistency(global_impression: GlobalImpression, top_emotion: str | None,
                               concern_by_domain: dict[str, str]) -> str:
    if global_impression.availability != "available":
        return "INSUFFICIENT_EVIDENCE"
    # Reuses the SAME emotion valence the expressive classifier already
    # provides -- the global scene text itself carries no separate learned
    # valence signal today; a future whole-image provider can add one via
    # GlobalImageRepresentationProvider.describe()'s "evidence" without
    # changing this function's signature.
    valence = _emotion_valence(top_emotion)
    labels = ("GLOBAL_LOCAL_CONSISTENT", "GLOBAL_LOCAL_MIXED", "GLOBAL_LOCAL_CONFLICT", "INSUFFICIENT_EVIDENCE")
    if global_impression.contradictions:
        negative_ranks = [_SUPPORT_RANK.get(concern_by_domain.get(d, "NONE"), 0) for d in _NEGATIVE_VALENCE_DOMAINS]
        return labels[2] if any(r >= 2 for r in negative_ranks) else labels[1]
    return _valence_vs_concern_consistency(valence, concern_by_domain, labels=labels)


def _final_synthesis(profile: ExpressiveProfile, concerns: tuple[ConcernDomainResult, ...],
                      global_impression: GlobalImpression, consistency: str,
                      global_local_consistency: str) -> str:
    parts = []
    if global_impression.overall_scene:
        parts.append(global_impression.overall_scene)
    if profile.availability == "available" and profile.top_emotion:
        descriptor_labels = ", ".join(d.label for d in profile.richer_descriptors) or profile.top_emotion.lower()
        parts.append(f"The drawing's expressive appearance is mainly classified as {profile.top_emotion} "
                     f"({descriptor_labels}).")
    else:
        parts.append("No expressive-model classification is available for this drawing.")

    notable = [c for c in concerns if c.support_level in ("WEAK", "MODERATE", "STRONG")]
    notable.sort(key=lambda c: _SUPPORT_RANK[c.support_level], reverse=True)
    if notable:
        for c in notable:
            parts.append(f"DOAR found {c.support_level.lower()} support for {c.domain_label.lower()} "
                         f"({c.plain_language_reason})")
    else:
        parts.append("No concern domain currently reaches even weak support from the available evidence.")

    if consistency == "CONFLICT":
        parts.append("The expressive classification and the concern-domain evidence disagree strongly -- "
                     "this result needs closer review rather than a confident conclusion.")
    elif consistency == "MIXED":
        parts.append("The expressive classification and the concern-domain evidence show some disagreement.")
    elif consistency == "INSUFFICIENT_EVIDENCE":
        parts.append("There is not enough evidence yet to state an overall pattern with confidence.")

    if global_local_consistency == "GLOBAL_LOCAL_CONFLICT":
        parts.append("The whole-drawing impression also conflicts with the local/rule-level evidence -- "
                     "treat this as needing closer review, not a confident single conclusion.")
    elif global_local_consistency == "GLOBAL_LOCAL_MIXED":
        parts.append("The whole-drawing impression shows some disagreement with local evidence, worth noting.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 3.6. Gemini concern candidates -- Gemini is a CANDIDATE HYPOTHESIS
# GENERATOR ONLY (see human_interaction.GeminiGlobalObserver for the call
# itself). Every candidate is verified here by looking up the domain's
# ALREADY-COMPUTED, entirely Gemini-blind ConcernDomainResult (produced by
# build_concern_domains() above from governed rule evidence alone) --
# Gemini's own text is NEVER fed back into that computation, so it can
# never itself raise a domain's support_level/support_ratio.
# ---------------------------------------------------------------------------

_SUPPORT_LEVEL_TO_VERIFICATION = {
    "STRONG": "SUPPORTED", "MODERATE": "SUPPORTED", "WEAK": "PARTIALLY_SUPPORTED",
    "NONE": "UNSUPPORTED", "INSUFFICIENT": "INSUFFICIENT_EVIDENCE",
    "NOT_CURRENTLY_ASSESSED": "NOT_CURRENTLY_ASSESSED",
}


@dataclass(frozen=True)
class GeminiConcernCandidate:
    domain: str
    domain_label: str
    gemini_reason: str
    gemini_observations: tuple[str, ...]
    verification_status: str  # SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED/INSUFFICIENT_EVIDENCE/NOT_CURRENTLY_ASSESSED
    governed_support_level: str | None
    why: str

    def to_dict(self) -> dict:
        return {
            "domain": self.domain, "domain_label": self.domain_label, "gemini_reason": self.gemini_reason,
            "gemini_observations": list(self.gemini_observations), "verification_status": self.verification_status,
            "governed_support_level": self.governed_support_level, "why": self.why,
        }


def verify_gemini_concern_candidates(
        candidates: list[dict] | None, concern_domains: tuple[ConcernDomainResult, ...],
) -> tuple[GeminiConcernCandidate, ...]:
    """STEP 1-5 of Gemini-candidate verification: (1) does the domain
    exist in DOAR's registry at all -- `concern_domains` already covers
    every registry domain (build_concern_domains iterates the FULL
    CONCERN_DOMAIN_MAP.json, plus the explicitly-unmapped extras), so a
    domain missing from it entirely is an unrecognized/misspelled name;
    (2)-(4) which relevant rules/observations are actually supported, and
    deterministic verification, are BOTH already done -- that is exactly
    what build_concern_domains()'s INDEPENDENT_FAMILY_RATIO computation
    is; (5) this function is the "judge" step: it maps the REAL,
    Gemini-blind support_level onto a verification status. Gemini's own
    wording never changes that support_level."""
    by_domain = {c.domain: c for c in concern_domains}
    verified = []
    for cand in (candidates or []):
        domain = cand.get("domain")
        governed = by_domain.get(domain)
        if governed is None:
            verified.append(GeminiConcernCandidate(
                domain=domain or "unknown", domain_label=(domain or "unknown").replace("_", " "),
                gemini_reason=cand.get("reason", ""), gemini_observations=tuple(cand.get("observations") or ()),
                verification_status="NOT_CURRENTLY_ASSESSED", governed_support_level=None,
                why="This domain is not in DOAR's approved concern-domain registry -- Gemini's suggestion "
                    "cannot be verified and is not used as evidence."))
            continue
        status = _SUPPORT_LEVEL_TO_VERIFICATION.get(governed.support_level, "INSUFFICIENT_EVIDENCE")
        why = (f"DOAR's own governed evidence for {governed.domain_label} is {governed.support_level} "
              f"-- {governed.plain_language_reason}")
        verified.append(GeminiConcernCandidate(
            domain=domain, domain_label=governed.domain_label, gemini_reason=cand.get("reason", ""),
            gemini_observations=tuple(cand.get("observations") or ()), verification_status=status,
            governed_support_level=governed.support_level, why=why))
    return tuple(verified)


def build_case_interpretation(
        bundle: dict, *, global_provider: GlobalImageRepresentationProvider | None = None,
        visual_judge_audit: dict | None = None, concept_provider=None,
        gemini_global_observation: dict | None = None,
        gemini_concern_candidates: list[dict] | None = None) -> CaseInterpretation:
    """The one entry point -- builds everything above from a single,
    already-governed bundle (the same shape `human_interaction.
    answer_question`/`live_case_bundle.build_live_case_bundle` already
    use). Deterministic, pure, no network call, no fabricated value.
    `global_provider` is the pluggable whole-image representation (see
    `GlobalImageRepresentationProvider`); `visual_judge_audit` is an
    optional, already-run, session-only Visual Consistency Judge result
    (AUDIT_ONLY, never required, never changes any support level).
    `concept_provider` is the pluggable semantic-concept source (see
    `providers.SemanticConceptProvider`; defaults to `build_visual_
    concepts`). `gemini_global_observation`/`gemini_concern_candidates`
    are the OPTIONAL, already-fetched structured output of
    `human_interaction.GeminiGlobalObserver` -- a candidate hypothesis
    generator only; candidates are verified against governed evidence
    (see `verify_gemini_concern_candidates`) and can never themselves
    raise a concern domain's support level."""
    concern_domains = build_concern_domains(bundle)
    concern_by_domain = {c.domain: c.support_level for c in concern_domains}
    concepts = concept_provider.build_concepts(bundle) if concept_provider is not None else build_visual_concepts(bundle)
    concept_names = {c.concept_name for c in concepts}
    profile = build_expressive_profile(bundle, concept_names, concern_by_domain)
    global_impression = build_global_impression(
        bundle, concepts, profile, concern_by_domain,
        provider=global_provider, visual_judge_audit=visual_judge_audit)
    gemini_candidates = verify_gemini_concern_candidates(gemini_concern_candidates, concern_domains)

    contradictions = tuple(
        f"{c.domain_label}: {len(c.contradictory_evidence)} contradicting reference(s)"
        for c in concern_domains if c.contradictory_evidence
    )
    missing_information = tuple(sorted({m for c in concern_domains for m in c.missing_expected_evidence}))
    consistency = _consistency_status(profile.top_emotion, concern_by_domain)
    global_local = _global_local_consistency(global_impression, profile.top_emotion, concern_by_domain)
    synthesis_text = _final_synthesis(profile, concern_domains, global_impression, consistency, global_local)

    return CaseInterpretation(
        visual_concepts=concepts, expressive_profile=profile, concern_domains=concern_domains,
        global_impression=global_impression, contradictions=contradictions,
        missing_information=missing_information, consistency_status=consistency,
        global_local_consistency=global_local, final_synthesis=synthesis_text,
        gemini_global_observation=gemini_global_observation, gemini_candidates=gemini_candidates,
    )


def summarize_for_grounding(interp: CaseInterpretation) -> dict:
    """Compact summary for Ask DOAR's evidence package (human_interaction.
    build_evidence_package) -- so a question like "why is X support
    moderate" or "why do emotion and the deeper analysis conflict" can be
    answered grounded in this synthesis, without handing the answer
    helper/Judge the full (much larger) CaseInterpretation. Only domains
    reaching at least WEAK support are included; NONE/NOT_CURRENTLY_
    ASSESSED domains are omitted here (the fact that a domain is not
    reachable is still visible in the Psychologist/Technical views)."""
    profile = interp.expressive_profile
    return {
        "expressive_top_emotion": profile.top_emotion,
        "expressive_richer_descriptors": [d.label for d in profile.richer_descriptors],
        "notable_concern_domains": [
            {"domain_label": c.domain_label, "support_level": c.support_level,
             "why": c.plain_language_reason, "contradicting_evidence_count": len(c.contradictory_evidence)}
            for c in interp.concern_domains if c.support_level in ("WEAK", "MODERATE", "STRONG")
        ],
        "consistency_status": interp.consistency_status,
        "contradictions": list(interp.contradictions),
        "global_overall_scene": interp.global_impression.overall_scene,
        "global_local_consistency": interp.global_local_consistency,
        "final_synthesis": interp.final_synthesis,
        "gemini_candidate_verifications": [
            {"domain_label": c.domain_label, "verification_status": c.verification_status, "why": c.why}
            for c in interp.gemini_candidates
        ],
    }
