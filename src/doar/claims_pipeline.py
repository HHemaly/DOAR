"""Wires claim_verifier.py into the normal per-case output path
(DOAR-TRACE Phase 1.5, Section 8).

Turns every parent-facing sentence structured_report.py produced (each
individual rule suggestion's parent_safe_wording, each combined
hypothesis's allowed wording + supporting texts) into a `Claim`, runs the
deterministic verifier against it, and only lets a claim through as
"accepted" if every check passes. A claim that fails verification is
never displayed -- it is downgraded to `rejected_fallback_to_observation_only`
and the specific failure is recorded in `verification_report.json`, never
silently dropped.
"""

from __future__ import annotations

from typing import Any

from .claim_verifier import Claim, verify_claim

GENERATED_CLAIMS_SCHEMA_VERSION = "generated_claims_v1"
VERIFICATION_REPORT_SCHEMA_VERSION = "verification_report_v1"


def _build_raw_claims(structured_analysis: dict[str, Any]) -> list[tuple[str, str, str | None, Claim]]:
    raw: list[tuple[str, str, str | None, Claim]] = []
    counter = 0
    for suggestion in structured_analysis.get("individual_rule_suggestions", []):
        counter += 1
        raw.append((
            f"claim_{counter:03d}", "individual_rule_suggestion", suggestion.get("target_construct"),
            Claim(
                text=suggestion["parent_safe_wording"],
                evidence_ids=list(suggestion["evidence_ids"]),
                rule_ids=[suggestion["rule_id"]],
                source_ids=list(suggestion["reference_ids"]),
            ),
        ))
    for hyp in structured_analysis.get("combined_drawing_level_hypotheses", []):
        counter += 1
        wording = hyp["allowed_wording"][0] if hyp["allowed_wording"] else ""
        text = " ".join([wording, *hyp["supporting_texts"]]).strip()
        raw.append((
            f"claim_{counter:03d}", "combined_hypothesis", hyp["target_construct"],
            Claim(
                text=text,
                evidence_ids=list(hyp["contributing_evidence_ids"]),
                rule_ids=list(hyp["contributing_rule_ids"]),
                source_ids=[],
            ),
        ))
    return raw


def build_claims_and_verification(
    structured_analysis: dict[str, Any], analysis: dict[str, Any], registry_v2: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (generated_claims_document, verification_report_document).
    Every claim in the returned generated_claims document that is not
    `status == "accepted"` must not be rendered as if it were verified --
    callers (report/UI layers) should treat non-accepted claims as
    observation-only fallback text, never the claim's own wording."""
    raw_claims = _build_raw_claims(structured_analysis)
    claims_out = []
    verification_entries = []
    for claim_id, claim_type, construct_id, claim in raw_claims:
        report = verify_claim(claim, analysis, registry_v2)
        accepted = report.passed
        claims_out.append({
            "claim_id": claim_id,
            "claim_type": claim_type,
            "text": claim.text,
            "evidence_ids": claim.evidence_ids,
            "rule_ids": claim.rule_ids,
            "source_ids": claim.source_ids,
            "construct_id": construct_id,
            "status": "accepted" if accepted else "rejected_fallback_to_observation_only",
        })
        verification_entries.append({
            "claim_id": claim_id,
            "passed": accepted,
            "checks": [
                {"check_name": c.check_name, "passed": c.passed, "reason": c.reason}
                for c in report.checks
            ],
        })

    generated_claims = {
        "schema_version": GENERATED_CLAIMS_SCHEMA_VERSION,
        "claim_count": len(claims_out),
        "accepted_count": sum(1 for c in claims_out if c["status"] == "accepted"),
        "rejected_count": sum(1 for c in claims_out if c["status"] != "accepted"),
        "claims": claims_out,
    }
    verification_report = {
        "schema_version": VERIFICATION_REPORT_SCHEMA_VERSION,
        "all_passed": all(e["passed"] for e in verification_entries),
        "claim_count": len(verification_entries),
        "claims": verification_entries,
    }
    return generated_claims, verification_report
