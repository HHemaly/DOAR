"""Deterministic claim-verifier (DOAR-TRACE 4F).

Verifies a `Claim` (a candidate sentence about a case, however it was
produced -- deterministic template today, an LLM in a later phase) against
the case's own real, saved evidence. No LLM is used here or required to
fail a claim: every check below is a pure function over structured data
(`analysis.json`, `rules_registry_v2.json`, `structured_analysis.json`).

This is the safety layer `TARGET_APPLICATION_ARCHITECTURE.md` Section 12
and `DOAR_TRACE_MASTER_SPEC.md` Section 1 both call for: reuses
`judges.py`'s existing diagnostic-language regex set directly (not a
duplicated copy) so the forbidden-wording list cannot silently drift
between the two enforcement points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .judges import DIAGNOSTIC_PATTERNS, _ARABIC_DIAGNOSTIC

_UNAVAILABLE_HEDGE_PHRASES = (
    "not evaluated", "unavailable", "no detector", "cannot determine",
    "not observed", "not assessed", "لم يُقيَّم", "غير متاح", "لا يوجد كاشف",
)


@dataclass(frozen=True)
class Claim:
    """One candidate sentence to verify."""

    text: str
    evidence_ids: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    claimed_value: float | int | str | None = None
    claimed_value_evidence_id: str | None = None  # which evidence_id the claimed_value should match


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class VerificationReport:
    claim_text: str
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _known_evidence_ids(analysis: dict[str, Any]) -> set[str]:
    ids = {e.get("evidence_id") for e in analysis.get("evidence", [])}
    for feature in analysis.get("objective_features", {}).values():
        eid = feature.get("evidence_id") if isinstance(feature, dict) else None
        if eid:
            ids.add(eid)
    return {i for i in ids if i}


def _known_rule_ids(analysis: dict[str, Any]) -> set[str]:
    return {r["rule_id"] for r in analysis.get("rule_evaluations", [])}


def verify_evidence_ids(claim: Claim, analysis: dict[str, Any]) -> CheckResult:
    known = _known_evidence_ids(analysis)
    unknown = [eid for eid in claim.evidence_ids if eid not in known]
    if unknown:
        return CheckResult("valid_evidence_ids", False, f"References unknown evidence_id(s): {unknown}")
    return CheckResult("valid_evidence_ids", True, "All cited evidence_ids exist in this case's saved evidence.")


def verify_rule_ids(claim: Claim, analysis: dict[str, Any]) -> CheckResult:
    known = _known_rule_ids(analysis)
    unknown = [rid for rid in claim.rule_ids if rid not in known]
    if unknown:
        return CheckResult("valid_rule_ids", False, f"References unknown rule_id(s): {unknown}")
    return CheckResult("valid_rule_ids", True, "All cited rule_ids exist in this case's rule evaluations.")


def verify_source_ids(claim: Claim, registry_v2: dict[str, Any]) -> CheckResult:
    known = set(registry_v2.get("references", {}).keys())
    unknown = [sid for sid in claim.source_ids if sid not in known]
    if unknown:
        return CheckResult("valid_source_ids", False, f"References unknown source/reference ID(s): {unknown}")
    return CheckResult("valid_source_ids", True, "All cited source IDs exist in the registry's reference list.")


def verify_numeric_agreement(claim: Claim, analysis: dict[str, Any]) -> CheckResult:
    if claim.claimed_value is None or claim.claimed_value_evidence_id is None:
        return CheckResult("numeric_agreement", True, "Claim asserts no specific numeric value.")
    target_id = claim.claimed_value_evidence_id
    for evidence in analysis.get("evidence", []):
        if evidence.get("evidence_id") == target_id:
            actual = evidence.get("value")
            if isinstance(actual, (int, float)) and isinstance(claim.claimed_value, (int, float)):
                if abs(float(actual) - float(claim.claimed_value)) > 1e-6:
                    return CheckResult("numeric_agreement", False,
                                        f"Claim asserts {claim.claimed_value!r} for {target_id}, actual is {actual!r}.")
                return CheckResult("numeric_agreement", True, "Claimed value matches saved evidence.")
            if actual != claim.claimed_value:
                return CheckResult("numeric_agreement", False,
                                    f"Claim asserts {claim.claimed_value!r} for {target_id}, actual is {actual!r}.")
            return CheckResult("numeric_agreement", True, "Claimed value matches saved evidence.")
    for feature_id, feature in analysis.get("objective_features", {}).items():
        if isinstance(feature, dict) and feature.get("evidence_id") == target_id:
            actual = feature.get("value")
            if actual is not None and claim.claimed_value is not None and abs(float(actual) - float(claim.claimed_value)) > 1e-6:
                return CheckResult("numeric_agreement", False,
                                    f"Claim asserts {claim.claimed_value!r} for {feature_id} ({target_id}), actual is {actual!r}.")
            return CheckResult("numeric_agreement", True, "Claimed value matches saved evidence.")
    return CheckResult("numeric_agreement", False, f"claimed_value_evidence_id {target_id!r} not found in this case's evidence.")


def verify_unavailable_wording(claim: Claim, analysis: dict[str, Any]) -> CheckResult:
    """If a claim cites a rule that is missing_detector/not_assessable, the
    claim text must hedge appropriately (not assert presence/absence as fact)."""
    rules_by_id = {r["rule_id"]: r for r in analysis.get("rule_evaluations", [])}
    unavailable_statuses = {"missing_detector", "not_assessable_context_unknown"}
    blocked = [rid for rid in claim.rule_ids if rules_by_id.get(rid, {}).get("status") in unavailable_statuses]
    if not blocked:
        return CheckResult("unavailable_evidence_wording", True, "No cited rule is in an unavailable state.")
    text_lower = claim.text.lower()
    hedged = any(phrase.lower() in text_lower for phrase in _UNAVAILABLE_HEDGE_PHRASES)
    if hedged:
        return CheckResult("unavailable_evidence_wording", True, "Unavailable rule(s) correctly hedged in claim text.")
    return CheckResult(
        "unavailable_evidence_wording", False,
        f"Claim cites unavailable rule(s) {blocked} without hedging language "
        f"(expected one of: {', '.join(_UNAVAILABLE_HEDGE_PHRASES[:4])}, ...).",
    )


def verify_no_diagnostic_claims(claim: Claim) -> CheckResult:
    if _ARABIC_DIAGNOSTIC.search(claim.text):
        return CheckResult("no_diagnostic_claims", False, "Claim text matches a forbidden Arabic diagnostic pattern.")
    for pattern in DIAGNOSTIC_PATTERNS:
        if pattern.search(claim.text):
            return CheckResult("no_diagnostic_claims", False,
                                f"Claim text matches forbidden diagnostic pattern: {pattern.pattern}")
    return CheckResult("no_diagnostic_claims", True, "No diagnostic language detected.")


def verify_no_omitted_contradictions(
    claims: list[Claim], structured_analysis: dict[str, Any] | None,
) -> CheckResult:
    """If structured_analysis.json recorded a cross-theme contradiction,
    at least one claim in the batch must reference both opposing rule IDs
    -- a contradiction must never be silently dropped from the final text."""
    contradictions = (structured_analysis or {}).get("cross_theme_contradictions", [])
    if not contradictions:
        return CheckResult("no_omitted_contradictions", True, "No recorded contradictions to check.")
    all_cited_rule_ids = {rid for claim in claims for rid in claim.rule_ids}
    omitted = []
    for contradiction in contradictions:
        rule_ids = set(contradiction.get("rule_ids_a", [])) | set(contradiction.get("rule_ids_b", []))
        if not rule_ids <= all_cited_rule_ids:
            omitted.append((contradiction["construct_a"], contradiction["construct_b"]))
    if omitted:
        return CheckResult("no_omitted_contradictions", False,
                            f"Recorded contradiction(s) not reflected in any claim: {omitted}")
    return CheckResult("no_omitted_contradictions", True, "All recorded contradictions are reflected in the claims.")


def verify_claim(
    claim: Claim, analysis: dict[str, Any], registry_v2: dict[str, Any],
) -> VerificationReport:
    checks = [
        verify_evidence_ids(claim, analysis),
        verify_rule_ids(claim, analysis),
        verify_source_ids(claim, registry_v2),
        verify_numeric_agreement(claim, analysis),
        verify_unavailable_wording(claim, analysis),
        verify_no_diagnostic_claims(claim),
    ]
    return VerificationReport(claim_text=claim.text, checks=checks)


def verify_claims(
    claims: list[Claim], analysis: dict[str, Any], registry_v2: dict[str, Any],
    structured_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reports = [verify_claim(c, analysis, registry_v2) for c in claims]
    contradiction_check = verify_no_omitted_contradictions(claims, structured_analysis)
    return {
        "all_passed": all(r.passed for r in reports) and contradiction_check.passed,
        "per_claim": [
            {"claim_text": r.claim_text, "passed": r.passed,
             "checks": [{"check_name": c.check_name, "passed": c.passed, "reason": c.reason} for c in r.checks]}
            for r in reports
        ],
        "batch_checks": [{
            "check_name": contradiction_check.check_name, "passed": contradiction_check.passed,
            "reason": contradiction_check.reason,
        }],
    }
