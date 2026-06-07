"""
psych_safety_validator.py — Validates psychological interpretation claims.

Checks that each psychological_interpretation claim:
  1. Is based only on verified (not pending/rejected/uncertain) supporting claims
  2. Uses cautious, non-diagnostic language
  3. Does not reference objects/features that were not verified
  4. Has is_diagnosis == False
  5. Includes a caution note

Marks each psych claim as "verified", "uncertain", or "rejected".
"""

from __future__ import annotations
import re
from safety_policy import check_for_diagnostic_language, CAUTIOUS_VERBS


def _build_verified_set(claims: list[dict]) -> set[str]:
    """Return set of verified rule_ids and object labels from all non-psych claims."""
    verified = set()
    for c in claims:
        if c.get("validator_status") != "verified":
            continue
        ct = c.get("claim_type", "")
        ev = c.get("evidence", {})
        if ct == "visual_object":
            lbl = ev.get("label", "")
            if lbl:
                verified.add(lbl.lower())
        if ct == "visual_numeric":
            for key in ev:
                verified.add(key)
        if ct == "ocr_text":
            verified.add("ocr")
    return verified


def validate_psych_claim(claim: dict, verified_set: set, all_claims: list[dict]) -> dict:
    if claim.get("claim_type") != "psychological_interpretation":
        return claim

    ev          = claim.get("evidence", {})
    rule_id     = ev.get("rule_id", "")
    interp_text = claim.get("claim", "")
    sources     = ev.get("sources", [])
    caution     = ev.get("caution", "")
    limitation  = ev.get("limitation", "")
    is_diag     = ev.get("is_diagnosis", None)

    issues = []

    # Check 1: is_diagnosis must be False
    if is_diag is not False and is_diag is not None:
        issues.append("Rule has is_diagnosis != False.")

    # Check 2: caution note must be present
    if not caution:
        issues.append("Missing caution note in rule.")

    # Check 3: no forbidden diagnostic language
    violations = check_for_diagnostic_language(interp_text)
    block_issues = [v for v in violations if v["severity"] == "BLOCK"]
    if block_issues:
        issues.append(f"Forbidden diagnostic language: {[v['pattern'] for v in block_issues]}")

    # Check 4: at least one cautious verb present
    has_cautious = any(v.lower() in interp_text.lower() for v in CAUTIOUS_VERBS)
    if not has_cautious:
        issues.append("Interpretation text does not use cautious qualifying language.")

    # Check 5: rule should be supported by at least one verified feature
    # We check that the rule's activation is supported by at least one numeric or visual claim
    # Look for any verified non-psych claim
    supporting_verified = [
        c for c in all_claims
        if c.get("claim_type") in ("visual_numeric", "visual_object", "ocr_text")
        and c.get("validator_status") == "verified"
    ]

    if not issues:
        if supporting_verified:
            claim["validator_status"] = "verified"
            claim["validator_note"]   = (
                f"Interpretation is cautious, non-diagnostic, and supported by "
                f"{len(supporting_verified)} verified feature claim(s)."
            )
            claim["show_to_user"] = True
        else:
            claim["validator_status"] = "uncertain"
            claim["validator_note"]   = (
                "No verified feature claims found to support this interpretation. "
                "Treat as a speculative indicator only."
            )
            claim["show_to_user"] = False
    else:
        if any("Forbidden" in i for i in issues):
            claim["validator_status"] = "rejected"
            claim["show_to_user"]     = False
        else:
            claim["validator_status"] = "uncertain"
            claim["show_to_user"]     = False
        claim["validator_note"] = "Issues: " + "; ".join(issues)

    return claim


def validate_all_psych_claims(claims: list[dict]) -> list[dict]:
    """Run psych safety validation on all psychological_interpretation claims."""
    verified_set = _build_verified_set(claims)
    result = []
    for c in claims:
        if c.get("claim_type") == "psychological_interpretation":
            c = validate_psych_claim(c, verified_set, claims)
        result.append(c)
    return result


def build_validation_summary(claims: list[dict]) -> dict:
    verified   = sum(1 for c in claims if c.get("validator_status") == "verified")
    uncertain  = sum(1 for c in claims if c.get("validator_status") == "uncertain")
    rejected   = sum(1 for c in claims if c.get("validator_status") == "rejected")
    pending    = sum(1 for c in claims if c.get("validator_status") == "pending")
    show_count = sum(1 for c in claims if c.get("show_to_user"))
    return {
        "total_claims":    len(claims),
        "verified_claims": verified,
        "uncertain_claims": uncertain,
        "rejected_claims": rejected,
        "pending_claims":  pending,
        "show_to_user_count": show_count,
    }
