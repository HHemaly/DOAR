"""Structured report planner (DOAR-TRACE Phase 1.5, Section 6 -- full
rewrite of the Phase 1 version; see `docs/AGGREGATION_POLICY.md`).

Produces `structured_analysis.json` deterministically from a real
`analyze_image` result (`Analysis.to_dict()` shape), the 41-rule
`rules_registry_v2.json` (Phase 1.5), and `construct_registry.json`. No
LLM, no invented clinical probability.

Three explicit levels, never conflated:

  LEVEL A -- OBSERVATION: an objective extracted feature/model result,
             shown with no interpretation attached at all.
  LEVEL B -- INDIVIDUAL RULE SUGGESTION: one matched rule's own cautious
             reading, evidence level, source, limitations, alternative
             explanations, and a contextual question. NEVER shown as a
             combined "theme" -- Phase 1's bug (a single triggered rule
             produced a "candidate_drawing_level_themes" entry, e.g.
             coverage_full alone -> "self_esteem") is fixed here: a
             single rule can never appear under
             `combined_drawing_level_hypotheses` below, only under
             `individual_rule_suggestions`.
  LEVEL C -- COMBINED DRAWING-LEVEL HYPOTHESIS: only produced when the
             construct policy in `construct_registry.json` is satisfied:
             >=2 independent evidence families, >=2 distinct evidence
             IDs, no dependency-group double counting, no upstream
             failure, low-confidence evidence suppressed, and (for the
             three "serious" mood/fear/anger constructs) at least one
             contributor above purely speculative grade. Ordinal levels
             0-4 (see `docs/AGGREGATION_POLICY.md`); only levels 2-4 may
             ever appear as a combined hypothesis.

The calibrated Angry/Fear/Happy/Sad expressive-content model is wired in
as its own independent evidence family, `global_expressive_content_model`
-- with mandatory wording ("the visual model found the drawing most
similar to the dataset's <X> expressive-content category"), never "the
child is <X>".
"""

from __future__ import annotations

from typing import Any

from .construct_registry_build import build_construct_registry
from .registry_v2_build import build_registry_v2

STRUCTURED_ANALYSIS_SCHEMA_VERSION = "structured_analysis_v2"  # Phase 1.5 -- distinct from Phase 1's "structured_analysis_v1"

_TRIGGERED_STATUS = "weak_support"  # rules.py::evaluate_rules' real live vocabulary

# global_expressive_content_model: which construct each calibrated top-class
# can contribute to, and the ONLY correct wording for citing it (never "the
# child is X").
MODEL_CLASS_TO_CONSTRUCT = {
    "Angry": "tension_or_anger_pattern",
    "Fear": "fear_or_insecurity_pattern",
    "Happy": "positive_affective_tone",
    "Sad": "low_mood_or_emotional_distress_pattern",
}
MODEL_EVIDENCE_FAMILY = "global_expressive_content_model"
# Below this calibrated confidence, the model's contribution is suppressed
# entirely from aggregation (still shown at Level A as a raw observation,
# but never counted toward a Level C combined hypothesis) -- mirrors
# emotion.py's own "moderate" uncertainty band (>=0.50).
MODEL_MIN_CONFIDENCE_FOR_AGGREGATION = 0.50

# Constructs whose combined hypothesis requires at least one contributing
# rule/family above purely speculative evidence grade before it may ever
# be shown -- the task's explicit "serious low-mood/fear/distress warning"
# requirement. Every other construct only needs the family/evidence-ID
# count thresholds.
_SERIOUS_CONSTRUCTS = {
    "low_mood_or_emotional_distress_pattern", "fear_or_insecurity_pattern", "tension_or_anger_pattern",
}
_SPECULATIVE_ONLY_GRADE = {"Speculative"}

# Construct pairs the construct_registry.json's own possible_contradicting_families
# imply are opposites -- used to flag (never silently drop) a genuine
# cross-construct contradiction.
OPPOSING_CONSTRUCTS = {
    "low_mood_or_emotional_distress_pattern": "positive_affective_tone",
    "positive_affective_tone": "low_mood_or_emotional_distress_pattern",
    "social_distance_or_isolation": "affiliation_or_connection",
    "affiliation_or_connection": "social_distance_or_isolation",
    "high_visual_energy_or_expansiveness": "caution_or_low_visual_energy",
    "caution_or_low_visual_energy": "high_visual_energy_or_expansiveness",
}


def _ordinal_level(n_families: int) -> int:
    """0=no evidence, 1=isolated clue, 2=possible pattern, 3=converging
    drawing pattern, 4=drawing evidence + external context (never reached
    here -- no parent/child/professional-context input exists in this
    pipeline; the value is defined for completeness per the task spec,
    not reachable in this phase)."""
    if n_families <= 0:
        return 0
    if n_families == 1:
        return 1
    if n_families == 2:
        return 2
    return 3


def _model_contribution(emotion: dict[str, Any]) -> dict[str, Any] | None:
    """Returns a synthetic 'rule-like' contribution dict for the expressive
    model if it ran, is confident enough, and its top class maps to a known
    construct -- else None (never fabricated, never counted below threshold)."""
    if emotion.get("status") != "available":
        return None
    top_class = emotion.get("top_class")
    confidence = emotion.get("confidence") or 0.0
    construct = MODEL_CLASS_TO_CONSTRUCT.get(top_class)
    if construct is None or confidence < MODEL_MIN_CONFIDENCE_FOR_AGGREGATION:
        return None
    return {
        "target_construct": construct,
        "evidence_family": MODEL_EVIDENCE_FAMILY,
        "evidence_ids": ["ev_emotion_prediction"],
        "evidence_level_as_written": None,  # not one of the source PDFs' graded terms
        "text": f"The visual model found the drawing most similar to the dataset's {top_class} expressive-content category.",
        "source_rule_id": None,
    }


def build_individual_rule_suggestions(
    rule_evaluations: list[dict[str, Any]], rules_v2_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """LEVEL B. One entry per triggered rule -- never grouped, never shown
    as a combined theme, regardless of whether it happens to map to a
    construct that could also converge at Level C."""
    suggestions = []
    for rule_eval in rule_evaluations:
        if rule_eval.get("status") != _TRIGGERED_STATUS:
            continue
        v2 = rules_v2_by_id.get(rule_eval["rule_id"])
        if v2 is None:
            continue
        suggestions.append({
            "rule_id": v2["rule_id"],
            "observable": v2["observable"],
            "possible_interpretation": v2["possible_interpretation"],
            "evidence_level_as_written": v2["evidence_level_as_written"],
            "evidence_family": v2["evidence_family"],
            "target_construct": v2["target_construct"],
            "evidence_ids": sorted(rule_eval.get("matched_evidence_ids", [])),
            "source_citation": f"{v2['source_document']}, page {v2['source_page']}, \"{v2['source_section']}\"",
            "limitations": list(v2["limitations"]),
            "alternative_explanations": list(v2["alternative_explanations"]),
            "parent_safe_wording": v2["parent_safe_wording"],
            "professional_wording": v2["professional_wording"],
            "contextual_question": v2["question_template"],
            "reference_ids": list(v2["reference_ids"]),
            "requires_clinician_review": True,
        })
    return suggestions


def build_combined_hypotheses(
    rule_evaluations: list[dict[str, Any]], rules_v2_by_id: dict[str, dict[str, Any]],
    emotion: dict[str, Any], constructs_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """LEVEL C. Only ever includes a construct here if its own policy
    (construct_registry.json) is satisfied by >=2 INDEPENDENT contributors
    -- a single rule (Phase 1's bug) can never appear alone."""
    contributions: dict[str, list[dict[str, Any]]] = {}
    for rule_eval in rule_evaluations:
        if rule_eval.get("status") != _TRIGGERED_STATUS:
            continue
        v2 = rules_v2_by_id.get(rule_eval["rule_id"])
        if v2 is None or v2["target_construct"] is None:
            continue
        contributions.setdefault(v2["target_construct"], []).append({
            "target_construct": v2["target_construct"],
            "evidence_family": v2["evidence_family"],
            "evidence_ids": sorted(rule_eval.get("matched_evidence_ids", [])),
            "evidence_level_as_written": v2["evidence_level_as_written"],
            "text": v2["parent_safe_wording"],
            "source_rule_id": v2["rule_id"],
        })

    model_contribution = _model_contribution(emotion)
    if model_contribution is not None:
        contributions.setdefault(model_contribution["target_construct"], []).append(model_contribution)

    hypotheses = []
    for construct_id, items in sorted(contributions.items()):
        construct = constructs_by_id.get(construct_id)
        if construct is None:
            continue
        # Dependency-aware: unique evidence families AND unique evidence IDs
        # -- two rules citing the SAME evidence_id must never be counted as
        # two independent contributors.
        families = sorted({item["evidence_family"] for item in items})
        evidence_ids = sorted({eid for item in items for eid in item["evidence_ids"]})
        min_families = construct["minimum_independent_evidence_families"]
        if len(families) < min_families or len(evidence_ids) < 2:
            continue  # stays at Level B only -- not enough independent evidence to combine
        if construct_id in _SERIOUS_CONSTRUCTS:
            graded: set[str] = set()
            for item in items:
                terms = item["evidence_level_as_written"]
                if isinstance(terms, list):
                    graded.update(terms)
                elif terms:
                    graded.add(terms)
            # None (Arabic source, ungraded) or any non-Speculative English
            # grade clears the bar; an item with NO grade info (Arabic-only
            # or the model) is treated as clearing it too, since "Speculative"
            # is a specific downgrade the source explicitly assigned, not a
            # default absence of information.
            all_graded_are_purely_speculative = bool(graded) and graded <= _SPECULATIVE_ONLY_GRADE
            has_ungraded_contributor = any(item["evidence_level_as_written"] is None for item in items)
            if all_graded_are_purely_speculative and not has_ungraded_contributor:
                continue
        level = _ordinal_level(len(families))
        hypotheses.append({
            "target_construct": construct_id,
            "display_name_en": construct["display_name_en"],
            "display_name_ar": construct["display_name_ar"],
            "ordinal_level": level,
            "level_label": {2: "possible_pattern", 3: "converging_drawing_pattern", 4: "drawing_evidence_plus_external_context"}[level],
            "contributing_evidence_families": families,
            "contributing_evidence_ids": evidence_ids,
            "contributing_rule_ids": sorted({item["source_rule_id"] for item in items if item["source_rule_id"]}),
            "uses_expressive_model": any(item["evidence_family"] == MODEL_EVIDENCE_FAMILY for item in items),
            "supporting_texts": [item["text"] for item in items],
            "allowed_wording": construct["allowed_wording"],
            "prohibited_wording": construct["prohibited_wording"],
            "limitations": construct["limitations"],
            "contextual_questions": construct["contextual_questions"],
            "professional_review_recommendation": construct["professional_review_recommendation"],
            "requires_clinician_review": True,
        })
    return hypotheses


def detect_cross_theme_contradictions(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Never silently drops conflicting evidence: if two combined
    hypotheses' constructs are known opposites and both are present,
    records the contradiction explicitly."""
    by_construct = {h["target_construct"]: h for h in hypotheses}
    contradictions = []
    seen = set()
    for construct, hyp in by_construct.items():
        opposite = OPPOSING_CONSTRUCTS.get(construct)
        if opposite and opposite in by_construct:
            pair_key = tuple(sorted((construct, opposite)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            contradictions.append({
                "construct_a": pair_key[0], "construct_b": pair_key[1],
                "rule_ids_a": by_construct[pair_key[0]]["contributing_rule_ids"],
                "rule_ids_b": by_construct[pair_key[1]]["contributing_rule_ids"],
                "resolution": "not_resolved -- both hypotheses are preserved and shown; no automatic winner is chosen",
            })
    return contradictions


def _suggested_parent_questions(
    individual_suggestions: list[dict[str, Any]], hypotheses: list[dict[str, Any]],
) -> list[str]:
    questions: list[str] = []
    for hyp in hypotheses:
        for q in hyp["contextual_questions"]:
            if q not in questions:
                questions.append(q)
    for suggestion in individual_suggestions:
        q = suggestion["contextual_question"]
        if q and q not in questions:
            questions.append(q)
    return questions


def build_structured_analysis(analysis: dict[str, Any], registry_v2: dict[str, Any] | None = None,
                               construct_registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """`analysis` is a real `Analysis.to_dict()` result (from
    `analyze_image`)."""
    registry_v2 = registry_v2 or build_registry_v2()
    construct_registry = construct_registry or build_construct_registry()
    rules_v2_by_id = {r["rule_id"]: r for r in registry_v2["rules"]}
    constructs_by_id = {c["construct_id"]: c for c in construct_registry["constructs"]}
    rule_evaluations = analysis.get("rule_evaluations", [])
    emotion = analysis.get("emotion", {})

    individual_suggestions = build_individual_rule_suggestions(rule_evaluations, rules_v2_by_id)
    combined_hypotheses = build_combined_hypotheses(rule_evaluations, rules_v2_by_id, emotion, constructs_by_id)
    contradictions = detect_cross_theme_contradictions(combined_hypotheses)

    rule_evaluations_v2 = []
    for rule_eval in rule_evaluations:
        v2 = rules_v2_by_id.get(rule_eval["rule_id"], {})
        rule_evaluations_v2.append({
            **rule_eval,
            "observability_class": v2.get("observability_class"),
            "allowed_output_level": v2.get("allowed_output_level"),
            "target_construct": v2.get("target_construct"),
            "evidence_family": v2.get("evidence_family"),
        })

    model_observation = None
    if emotion.get("status") == "available":
        model_observation = {
            "evidence_family": MODEL_EVIDENCE_FAMILY,
            "text": f"The visual model found the drawing most similar to the dataset's {emotion.get('top_class')} expressive-content category.",
            "confidence": emotion.get("confidence"),
            "calibration_status": emotion.get("calibration_status"),
            "used_in_combined_hypothesis": any(
                h["uses_expressive_model"] for h in combined_hypotheses
            ),
        }

    all_missing_evidence = sorted({m for r in rule_evaluations for m in r.get("missing_evidence", [])})
    all_references = sorted({ref for r in rule_evaluations for ref in r.get("references", [])})
    all_limitations = sorted({lim for r in rule_evaluations for lim in r.get("limitations", [])})

    return {
        "schema_version": STRUCTURED_ANALYSIS_SCHEMA_VERSION,
        "registry_v2_schema_version": registry_v2["schema_version"],
        "construct_registry_schema_version": construct_registry["schema_version"],
        # LEVEL A
        "quality": analysis.get("quality", {}),
        "segmentation": analysis.get("segmentation", {}),
        "objective_features": analysis.get("objective_features", {}),
        "detections": {
            "status": "unavailable", "detections": [],
            "reason": "No object detector is implemented in this release (detectors/ is schema-only scaffolding).",
        },
        "model_output": emotion,
        "expressive_model_observation": model_observation,
        # LEVEL B
        "individual_rule_suggestions": individual_suggestions,
        "rule_evaluations": rule_evaluations_v2,
        # LEVEL C
        "combined_drawing_level_hypotheses": combined_hypotheses,
        "cross_theme_contradictions": contradictions,
        # Cross-cutting
        "supporting_evidence_ids": sorted({eid for h in combined_hypotheses for eid in h["contributing_evidence_ids"]}),
        "contradicting_evidence_ids": [],
        "missing_evidence": all_missing_evidence,
        "references": all_references,
        "limitations": all_limitations,
        "suggested_parent_questions": _suggested_parent_questions(individual_suggestions, combined_hypotheses),
        "safety_disclaimer": analysis.get("safety_disclaimer"),
    }
