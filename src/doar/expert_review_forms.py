"""Generates the 3 expert-review CSV instruments (DOAR-TRACE Phase 1.5,
Section 10). Every row is built from real registry/construct/case data --
never hand-written. Reviewer-input columns (ratings, Y/N judgments, free
text) are left blank for a human reviewer to fill in; nothing here
fabricates a rating on a reviewer's behalf.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .construct_registry_build import build_construct_registry
from .registry_v2_build import build_registry_v2

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "expert_review"

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
