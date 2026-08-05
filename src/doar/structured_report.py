"""Structured report planner (DOAR-TRACE 4D).

Builds `structured_analysis.json` deterministically from a real
`analyze_image` result (`Analysis.to_dict()` shape) and the
`rules_registry_v2.json` draft (4C). No LLM, no invented clinical
probability -- every field is either copied from real analysis output or
computed by a pure, tested aggregation function below.

Because no object detector exists (`CURRENT_TO_TARGET_GAP_V2.md`), every
`static_detector`/`process_required` registry-v2 rule remains explicitly
`disabled`/unavailable here -- it is never silently omitted (an omitted
rule would be indistinguishable from "never considered"; a rule present
with `allowed_output_level: disabled` and a reason is an honest, auditable
statement that it was considered and blocked).
"""

from __future__ import annotations

from typing import Any

from .registry_v2_build import build_registry_v2

STRUCTURED_ANALYSIS_SCHEMA_VERSION = "structured_analysis_v1"

# Constructs the current registry treats as pairwise opposites, used only
# to detect a genuine cross-theme contradiction if it ever occurs (today's
# 6 executable rules have mutually exclusive numeric thresholds, so this
# cannot fire on real data yet -- see docs/RULE_PDF_COVERAGE_AUDIT.md and
# the synthetic tests in tests/test_structured_report.py). The mechanism
# exists and is tested regardless, per the task's explicit requirement to
# preserve (never silently drop) contradicting evidence.
OPPOSING_CONSTRUCTS = {
    "introversion": "extraversion",
    "extraversion": "introversion",
    "self_esteem": "instability_fear",
    "instability_fear": "self_esteem",
}

_TRIGGERED_STATUS = "weak_support"  # rules.py::evaluate_rules' real live vocabulary


def aggregate_candidate_themes(
    rule_evaluations: list[dict[str, Any]], rules_v2_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Groups triggered rules by target_construct into candidate drawing-
    level themes. Dependency-aware: the same evidence_id is never counted
    twice within a theme (evidence_ids is a set), and escalation to
    `combined_hypothesis_only` requires >=2 evidence IDs from >=2 distinct
    evidence families -- mirrors concerns.py's existing 2-evidence/
    2-source convergence policy, reimplemented locally (not imported) so
    this aggregator can be tested independently of concerns.py's own
    CONCERNS_ENABLED gate."""
    by_construct: dict[str, list[tuple[dict, dict]]] = {}
    for rule_eval in rule_evaluations:
        if rule_eval.get("status") != _TRIGGERED_STATUS:
            continue
        v2 = rules_v2_by_id.get(rule_eval["rule_id"])
        if v2 is None:
            continue
        by_construct.setdefault(v2["target_construct"], []).append((rule_eval, v2))

    themes = []
    for construct, pairs in sorted(by_construct.items()):
        evidence_ids = sorted({eid for r, _ in pairs for eid in r.get("matched_evidence_ids", [])})
        families = sorted({v2["evidence_family"] for _, v2 in pairs})
        rule_ids = sorted({r["rule_id"] for r, _ in pairs})
        if len(evidence_ids) >= 2 and len(families) >= 2:
            allowed_output = "combined_hypothesis_only"
        elif evidence_ids:
            allowed_output = "question_generating"
        else:
            allowed_output = "observation_only"
        themes.append({
            "target_construct": construct,
            "supporting_rule_ids": rule_ids,
            "supporting_evidence_ids": evidence_ids,
            "evidence_families": families,
            "allowed_output_level": allowed_output,
            "missing_evidence": sorted({m for r, _ in pairs for m in r.get("missing_evidence", [])}),
            "references": sorted({ref for _, v2 in pairs for ref in v2["reference_ids"]}),
            "limitations": sorted({lim for _, v2 in pairs for lim in v2["limitations"]}),
            "alternative_explanations": sorted({alt for _, v2 in pairs for alt in v2["alternative_explanations"]}),
            "requires_clinician_review": True,
        })
    return themes


def detect_cross_theme_contradictions(themes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Never silently drops conflicting evidence: if two candidate themes'
    constructs are known opposites (OPPOSING_CONSTRUCTS) and both are
    present, records the contradiction explicitly rather than picking a
    winner or omitting either theme."""
    by_construct = {t["target_construct"]: t for t in themes}
    contradictions = []
    seen = set()
    for construct, theme in by_construct.items():
        opposite = OPPOSING_CONSTRUCTS.get(construct)
        if opposite and opposite in by_construct:
            pair_key = tuple(sorted((construct, opposite)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            contradictions.append({
                "construct_a": pair_key[0],
                "construct_b": pair_key[1],
                "rule_ids_a": by_construct[pair_key[0]]["supporting_rule_ids"],
                "rule_ids_b": by_construct[pair_key[1]]["supporting_rule_ids"],
                "resolution": "not_resolved -- both themes are preserved and shown; no automatic winner is chosen",
            })
    return contradictions


def _suggested_parent_questions(themes: list[dict[str, Any]], rules_v2_by_id: dict[str, dict[str, Any]]) -> list[str]:
    questions = []
    for theme in themes:
        for rule_id in theme["supporting_rule_ids"]:
            wording = rules_v2_by_id[rule_id]["parent_safe_wording"]
            if wording and wording not in questions:
                questions.append(wording)
    return questions


def build_structured_analysis(analysis: dict[str, Any], registry_v2: dict[str, Any] | None = None) -> dict[str, Any]:
    """`analysis` is a real `Analysis.to_dict()` result (from
    `analyze_image`). `registry_v2` defaults to a fresh
    `build_registry_v2()` call if not supplied."""
    registry_v2 = registry_v2 or build_registry_v2()
    rules_v2_by_id = {r["rule_id"]: r for r in registry_v2["rules"]}

    themes = aggregate_candidate_themes(analysis.get("rule_evaluations", []), rules_v2_by_id)
    contradictions = detect_cross_theme_contradictions(themes)

    rule_evaluations_v2 = []
    for rule_eval in analysis.get("rule_evaluations", []):
        v2 = rules_v2_by_id.get(rule_eval["rule_id"], {})
        rule_evaluations_v2.append({
            **rule_eval,
            "observability_class": v2.get("observability_class"),
            "allowed_output_level": v2.get("allowed_output_level"),
            "target_construct": v2.get("target_construct"),
            "evidence_family": v2.get("evidence_family"),
        })

    all_missing_evidence = sorted({
        m for r in analysis.get("rule_evaluations", []) for m in r.get("missing_evidence", [])
    })
    all_references = sorted({
        ref for r in analysis.get("rule_evaluations", []) for ref in r.get("references", [])
    })
    all_limitations = sorted({
        lim for r in analysis.get("rule_evaluations", []) for lim in r.get("limitations", [])
    })

    return {
        "schema_version": STRUCTURED_ANALYSIS_SCHEMA_VERSION,
        "registry_v2_schema_version": registry_v2["schema_version"],
        "quality": analysis.get("quality", {}),
        "segmentation": analysis.get("segmentation", {}),
        "objective_features": analysis.get("objective_features", {}),
        "detections": {
            "status": "unavailable",
            "detections": [],
            "reason": "No object detector is implemented in this release (detectors/ is schema-only scaffolding).",
        },
        "model_output": analysis.get("emotion", {}),
        "rule_evaluations": rule_evaluations_v2,
        "candidate_drawing_level_themes": themes,
        "cross_theme_contradictions": contradictions,
        "supporting_evidence_ids": sorted({eid for t in themes for eid in t["supporting_evidence_ids"]}),
        "contradicting_evidence_ids": [],  # no rule-vs-rule contradiction within a single theme is possible today (see docstring)
        "missing_evidence": all_missing_evidence,
        "references": all_references,
        "limitations": all_limitations,
        "suggested_parent_questions": _suggested_parent_questions(themes, rules_v2_by_id),
        "safety_disclaimer": analysis.get("safety_disclaimer"),
    }
