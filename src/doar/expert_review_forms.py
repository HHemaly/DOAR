"""Generates the expert-review CSV instruments (original 3: DOAR-TRACE
Phase 1.5, Section 10; 3 more: Phase 2A.1, Section 7). Every row is
built from real registry/construct/case data -- never hand-written.
Reviewer-input columns (ratings, Y/N judgments, free text) are left
blank for a human reviewer to fill in; nothing here fabricates a rating
on a reviewer's behalf.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .construct_registry_build import build_construct_registry
from .registry_v2_build import build_registry_v2
from .rule_engine_v2 import ALL_PAGE_GATED_RULE_IDS, V2_RULE_IDS

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "expert_review"
PHASE2A_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "phase2a"

RULE_FORM_FIELDS = [
    "rule_id", "source_text", "source_and_page", "observable_clarity_1_4",
    "interpretation_relevance_1_4", "assessable_from_static_image_y_n", "evidence_needed",
    "construct_mapping_accepted_y_n", "safe_for_parent_y_n", "safe_wording_revision", "comments",
]

CONSTRUCT_FORM_FIELDS = [
    "construct_id", "definition_clarity_1_4", "contributing_rules_appropriate_y_n",
    "missing_rules", "contradictory_rules", "minimum_evidence_family_requirement_appropriate_y_n",
    "parent_safe_wording", "comments",
]

SENTENCE_FORM_FIELDS = [
    "claim_id", "claim_type", "sentence_text", "construct_id", "evidence_ids", "rule_ids",
    "verification_status", "accurate_y_n", "safe_wording_y_n", "comments",
]


def build_rule_review_rows(registry_v2: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    registry_v2 = registry_v2 or build_registry_v2()
    rows = []
    for rule in registry_v2["rules"]:
        rows.append({
            "rule_id": rule["rule_id"],
            "source_text": rule["faithful_source_quote"],
            "source_and_page": f"{rule['source_document']}, page {rule['source_page']}",
            "observable_clarity_1_4": "",
            "interpretation_relevance_1_4": "",
            "assessable_from_static_image_y_n": "Y" if rule["observability_class"] in ("static_direct", "static_detector", "static_proxy") else "N",
            "evidence_needed": rule["required_detector_or_metadata"] or "none (already computed)",
            "construct_mapping_accepted_y_n": "",
            "safe_for_parent_y_n": "",
            "safe_wording_revision": "",
            "comments": "",
        })
    return rows


def build_construct_review_rows(
    registry_v2: dict[str, Any] | None = None, construct_registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    registry_v2 = registry_v2 or build_registry_v2()
    construct_registry = construct_registry or build_construct_registry()
    rules_by_construct: dict[str, list[str]] = {}
    for rule in registry_v2["rules"]:
        if rule["target_construct"]:
            rules_by_construct.setdefault(rule["target_construct"], []).append(rule["rule_id"])

    rows = []
    for construct in construct_registry["constructs"]:
        rows.append({
            "construct_id": construct["construct_id"],
            "definition_clarity_1_4": "",
            "contributing_rules_appropriate_y_n": "",
            "missing_rules": "",
            "contradictory_rules": "",
            "minimum_evidence_family_requirement_appropriate_y_n": "",
            "parent_safe_wording": construct["allowed_wording"][0] if construct["allowed_wording"] else "",
            "comments": f"Currently mapped rules: {', '.join(rules_by_construct.get(construct['construct_id'], [])) or 'none'}",
        })
    return rows


def build_sentence_review_rows(generated_claims: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for claim in generated_claims.get("claims", []):
        rows.append({
            "claim_id": claim["claim_id"],
            "claim_type": claim["claim_type"],
            "sentence_text": claim["text"],
            "construct_id": claim.get("construct_id") or "",
            "evidence_ids": ", ".join(claim["evidence_ids"]),
            "rule_ids": ", ".join(claim["rule_ids"]),
            "verification_status": claim["status"],
            "accurate_y_n": "",
            "safe_wording_y_n": "",
            "comments": "",
        })
    return rows


# ---------------------------------------------------------------------------
# Phase 2A.1, Section 7: 3 new forms extending expert review to the 4
# newly-activated rules (rule_engine_v2.V2_RULE_IDS) and to the
# page-relative/threshold hardening this phase did.
# ---------------------------------------------------------------------------

PHASE2A1_RULE_FORM_FIELDS = [
    "rule_id", "observable", "source_text", "source_and_page", "possible_interpretation",
    "observable_definition_accepted_y_n", "proxy_interpretation_accepted_y_n",
    "threshold_accepted_y_n", "suggested_threshold_or_decision_rule",
    "parent_safe_wording_current", "parent_safe_wording_accepted_y_n",
    "alternative_explanations_current", "alternative_explanations_missing",
    "safe_for_individual_display_y_n", "safe_for_combined_aggregation_y_n", "comments",
]

PHASE2A1_THRESHOLD_FORM_FIELDS = [
    "rule_id", "observable", "current_threshold_or_rule", "threshold_source",
    "real_trigger_rate_pct_among_assessable", "threshold_accepted_y_n",
    "suggested_threshold_or_decision_rule", "comments",
]

PHASE2A1_PAGE_RULE_FORM_FIELDS = [
    "rule_id", "observable", "current_definition", "page_reference_requirement",
    "page_reference_definition_accepted_y_n", "rule_definition_accepted_y_n",
    "suggested_definition_or_decision_rule", "comments",
]


def _real_trigger_rates() -> dict[str, str]:
    """Reads artifacts/phase2a/rule_trigger_distribution.csv if present --
    real per-rule trigger rates, never fabricated. Returns {} (every
    lookup then honestly blank) if the artifact hasn't been generated in
    this checkout yet, rather than raising."""
    path = PHASE2A_ARTIFACTS_DIR / "rule_trigger_distribution.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {row["rule_id"]: row["trigger_rate_among_assessable_pct"] for row in rows}


def build_phase2a1_rule_review_rows(registry_v2: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Scoped to the 4 rules newly activated in Phase 2A.1's parallel
    rule_engine_v2.py (docs/STATIC_PROXY_RULE_POLICY.md)."""
    registry_v2 = registry_v2 or build_registry_v2()
    rules_by_id = {r["rule_id"]: r for r in registry_v2["rules"]}
    rows = []
    for rule_id in sorted(V2_RULE_IDS):
        rule = rules_by_id[rule_id]
        rows.append({
            "rule_id": rule_id,
            "observable": rule["observable"],
            "source_text": rule["faithful_source_quote"],
            "source_and_page": f"{rule['source_document']}, page {rule['source_page']}",
            "possible_interpretation": rule["possible_interpretation"],
            "observable_definition_accepted_y_n": "",
            "proxy_interpretation_accepted_y_n": "",
            "threshold_accepted_y_n": "",
            "suggested_threshold_or_decision_rule": "",
            "parent_safe_wording_current": rule["parent_safe_wording"],
            "parent_safe_wording_accepted_y_n": "",
            "alternative_explanations_current": " | ".join(rule["alternative_explanations"]),
            "alternative_explanations_missing": "",
            "safe_for_individual_display_y_n": "",
            "safe_for_combined_aggregation_y_n": "",
            "comments": "",
        })
    return rows


def build_phase2a1_threshold_review_rows(registry_v2: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Scoped to every currently-executable rule with a non-directly-sourced
    threshold (i.e. every threshold this phase's own audit found was
    invented, centre-sourced, or empirically exploratory --
    docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md) -- broader than just the
    4 newly-activated rules, since PSY_AR_SIZE_FULL_015's threshold was
    also redefined this phase (Section 4) and deserves the same review."""
    registry_v2 = registry_v2 or build_registry_v2()
    rates = _real_trigger_rates()
    rows = []
    for rule in registry_v2["rules"]:
        if rule["allowed_output_level"] != "individual_heuristic_only":
            continue
        if rule["threshold_source"] == "directly_sourced":
            continue  # has a real numeric anchor from the source -- not this form's concern
        rows.append({
            "rule_id": rule["rule_id"],
            "observable": rule["observable"],
            "current_threshold_or_rule": rule["professional_wording"],
            "threshold_source": rule["threshold_source"],
            "real_trigger_rate_pct_among_assessable": rates.get(rule["rule_id"], ""),
            "threshold_accepted_y_n": "",
            "suggested_threshold_or_decision_rule": "",
            "comments": "",
        })
    return rows


def build_phase2a1_page_rule_review_rows(registry_v2: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Scoped to every page-relative rule (rule_engine_v2.ALL_PAGE_GATED_RULE_IDS
    -- the 6 historical page-coverage/placement rules plus
    EN_COMPILED_PLACEMENT_CENTER_029), the set this phase's page-reference
    model (Section 3) and coverage_full redefinition (Section 4) apply to."""
    registry_v2 = registry_v2 or build_registry_v2()
    rules_by_id = {r["rule_id"]: r for r in registry_v2["rules"]}
    rows = []
    for rule_id in sorted(ALL_PAGE_GATED_RULE_IDS):
        rule = rules_by_id.get(rule_id)
        if rule is None:
            continue
        current_definition = (
            "Margin-based: all 4 page margins <= 0.15 of the confirmed page (Phase 2A.1 Section 4)"
            if rule_id == "PSY_AR_SIZE_FULL_015"
            else rule["professional_wording"]
        )
        rows.append({
            "rule_id": rule_id,
            "observable": rule["observable"],
            "current_definition": current_definition,
            "page_reference_requirement": "Requires page_reference.page_relative_features_assessable=True; not_assessable otherwise (never not_matched).",
            "page_reference_definition_accepted_y_n": "",
            "rule_definition_accepted_y_n": "",
            "suggested_definition_or_decision_rule": "",
            "comments": "",
        })
    return rows


def write_phase2a1_rule_review_form(output_dir: Path = OUTPUT_DIR) -> Path:
    path = output_dir / "phase2a1_rule_review_form.csv"
    _write_csv(path, PHASE2A1_RULE_FORM_FIELDS, build_phase2a1_rule_review_rows())
    return path


def write_phase2a1_threshold_review_form(output_dir: Path = OUTPUT_DIR) -> Path:
    path = output_dir / "phase2a1_threshold_review_form.csv"
    _write_csv(path, PHASE2A1_THRESHOLD_FORM_FIELDS, build_phase2a1_threshold_review_rows())
    return path


def write_phase2a1_page_rule_review_form(output_dir: Path = OUTPUT_DIR) -> Path:
    path = output_dir / "phase2a1_page_rule_review_form.csv"
    _write_csv(path, PHASE2A1_PAGE_RULE_FORM_FIELDS, build_phase2a1_page_rule_review_rows())
    return path


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_rule_review_form(output_dir: Path = OUTPUT_DIR) -> Path:
    path = output_dir / "rule_review_form.csv"
    _write_csv(path, RULE_FORM_FIELDS, build_rule_review_rows())
    return path


def write_construct_review_form(output_dir: Path = OUTPUT_DIR) -> Path:
    path = output_dir / "construct_review_form.csv"
    _write_csv(path, CONSTRUCT_FORM_FIELDS, build_construct_review_rows())
    return path


def write_sentence_review_form(generated_claims: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> Path:
    path = output_dir / "report_sentence_review_form.csv"
    _write_csv(path, SENTENCE_FORM_FIELDS, build_sentence_review_rows(generated_claims))
    return path
