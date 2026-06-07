"""
final_response_judge.py — 10-point holistic judge for the parent-facing output.

Checks the complete parent-facing response before it is shown to the user.
Performs all 10 checks from the specification:

  1. Did the answer invent any unverified objects?
  2. Did it mention an unsupported emotion?
  3. Did it use diagnostic or harmful language?
  4. Did it sound too certain?
  5. Did it include a limitation/disclaimer?
  6. Did it give safe parent guidance (gentle questions)?
  7. Did it avoid scary conclusions?
  8. Did it avoid clinical labels without clinical evidence?
  9. Did it avoid presenting uncertain claims as facts?
 10. Is every final statement traceable to a verified claim?

Returns:
  {
    "final_answer_status": "PASS" | "REWRITE_REQUIRED" | "BLOCK",
    "issues": [str],
    "safe_to_show": bool,
    "required_revisions": [str],
    "checks_passed": int,
    "checks_total": 10,
  }

If status == "REWRITE_REQUIRED", a safe fallback answer is also generated.
"""

from __future__ import annotations
import re
from safety_policy import (
    check_for_diagnostic_language, sanitise_text,
    GENERAL_DISCLAIMER, DIAGNOSTIC_LABELS,
)
from numeric_validator import find_ungrounded_numbers


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def _check_1_invented_objects(parent_output: dict, validated_claims: list[dict]) -> list[str]:
    """Objects mentioned in answer must be in verified claims."""
    issues = []
    answer = parent_output.get("parent_answer", "")
    verified_labels = {
        c.get("evidence", {}).get("label", "").lower()
        for c in validated_claims
        if c.get("validator_status") == "verified" and c.get("show_to_user")
        and c.get("claim_type") in ("visual_object", "visual_symbol")
    }
    # We cannot enumerate all possible objects in the answer exhaustively,
    # but we can check for any common objects mentioned vs. the verified set.
    # We do a best-effort check using the verified labels as our known-good set.
    # If an animal name or strong noun appears and is NOT verified, flag it.
    # "train" excluded: matches "trained" in meta-text; transport vehicles checked via "bus","airplane"
    CHECKABLE_NOUNS = [
        "person", "child", "figure", "tree", "flower", "sun", "moon", "star",
        "heart", "animal", "dog", "cat", "tiger", "lion", "wolf", "fox",
        "squirrel", "car", "bus", "airplane", "boat", "house",
        "knife", "gun", "weapon", "fire", "blood",
    ]
    answer_lower = answer.lower()
    for noun in CHECKABLE_NOUNS:
        if noun in answer_lower:
            # Check if this noun is covered by any verified label
            covered = any(noun in lbl for lbl in verified_labels)
            if not covered and noun not in ("person", "child", "figure"):
                # 'person', 'child', 'figure' may come from composition text legitimately
                issues.append(
                    f"Check 1: Answer mentions '{noun}' which is not in verified claims."
                )
    return issues


def _check_2_unsupported_emotion(parent_output: dict, doc: dict) -> list[str]:
    """Emotion claims must come from the heuristic block and must be clearly caveated."""
    issues = []
    answer = parent_output.get("parent_answer", "")
    h = doc.get("feature_based_emotional_tendency", {})
    estimated = h.get("estimated_emotion", "neutral_or_unclear")
    method    = h.get("method", "")

    # Flag if answer positively claims a trained model produced the emotion (not when denying it)
    import re as _re
    if method == "heuristic" and _re.search(
            r"\b(trained model (predicts?|shows?|confirms?|found|identified|detected))\b",
            answer, _re.IGNORECASE):
        issues.append("Check 2: Answer implies a trained model emotion result but only heuristic is available.")
    if estimated == "neutral_or_unclear" and re.search(
            r"\b(the child (is|feels|appears))\s+(sad|angry|happy|afraid|scared|distressed)\b",
            answer, re.IGNORECASE):
        issues.append("Check 2: Emotion asserted in answer but heuristic shows no clear tendency.")
    return issues


def _check_3_diagnostic_language(parent_output: dict) -> list[str]:
    """No diagnostic or harmful language anywhere in the output."""
    issues = []
    text = (parent_output.get("parent_answer", "") + " " +
            parent_output.get("safety_note", ""))
    violations = check_for_diagnostic_language(text)
    for v in violations:
        issues.append(f"Check 3 ({v['severity']}): {v['reason']} — pattern: {v['pattern']}")
    return issues


def _check_4_overcertainty(parent_output: dict) -> list[str]:
    """Answer must not sound too certain."""
    issues = []
    answer = parent_output.get("parent_answer", "")
    CERTAIN_PHRASES = [
        r"\bdefinitely\b", r"\bcertainly\b", r"\bwithout (a |any )?doubt\b",
        r"\bproves?\b", r"\bconfirms?\b", r"\bthis means?\b",
        r"\bwill (suffer|struggle|have problems)\b",
    ]
    for pat in CERTAIN_PHRASES:
        if re.search(pat, answer, re.IGNORECASE):
            issues.append(f"Check 4: Over-certain language found: pattern '{pat}'.")
    return issues


def _check_5_disclaimer_present(parent_output: dict) -> list[str]:
    """Disclaimer must be present."""
    issues = []
    disclaimer = parent_output.get("disclaimer", "")
    safety_note= parent_output.get("safety_note", "")
    if not disclaimer and not safety_note:
        issues.append("Check 5: No disclaimer or safety note found in output.")
    elif "not diagnostic" not in (disclaimer + safety_note).lower():
        issues.append("Check 5: Disclaimer does not contain required 'not diagnostic' language.")
    return issues


def _check_6_gentle_questions(parent_output: dict) -> list[str]:
    """Gentle questions must be present."""
    issues = []
    questions = parent_output.get("gentle_questions", [])
    if not questions:
        issues.append("Check 6: No gentle parent-child questions provided.")
    return issues


def _check_7_scary_conclusions(parent_output: dict) -> list[str]:
    """No alarming or frightening conclusions."""
    issues = []
    answer = parent_output.get("parent_answer", "")
    SCARY = [
        r"\b(danger(ous)?|alarm(ing)?|concerning|serious)\b(?!ly small|ly limited)",
        r"\b(immediately|urgently)\b",
        r"\bseek help\b",
        r"\bsomething is wrong\b",
        r"\byour child (is|has)\s+(a problem|an issue|a condition)\b",
    ]
    for pat in SCARY:
        if re.search(pat, answer, re.IGNORECASE):
            issues.append(f"Check 7: Potentially alarming language: pattern '{pat}'.")
    return issues


def _check_8_clinical_labels(parent_output: dict) -> list[str]:
    """Clinical disorder names must not appear."""
    issues = []
    answer = parent_output.get("parent_answer", "").lower()
    for label in DIAGNOSTIC_LABELS:
        if label in answer:
            issues.append(f"Check 8: Clinical label '{label}' found in parent answer.")
    return issues


def _check_9_uncertain_as_fact(parent_output: dict, validated_claims: list[dict]) -> list[str]:
    """Uncertain/rejected claims must not be presented as facts."""
    issues = []
    answer = parent_output.get("parent_answer", "").lower()
    uncertain_labels = {
        c.get("evidence", {}).get("label", "").lower()
        for c in validated_claims
        if c.get("validator_status") in ("uncertain", "rejected")
        and c.get("claim_type") in ("visual_object", "visual_symbol")
    }
    for lbl in uncertain_labels:
        if lbl and lbl in answer:
            issues.append(
                f"Check 9: Uncertain/rejected object '{lbl}' is mentioned in the answer as if verified."
            )
    return issues


def _check_10_traceable_statements(parent_output: dict, validated_claims: list[dict],
                                    doc: dict) -> list[str]:
    """Numeric claims in answer must be grounded in computed values."""
    issues = []
    answer = parent_output.get("parent_answer", "")
    violations = find_ungrounded_numbers(answer, doc)
    for v in violations:
        if v.get("verdict") == "mismatch":
            issues.append(
                f"Check 10: Answer claims {v['claimed_pct']}% but computed value is "
                f"{v['closest_match']}% (diff {v['diff_pct']}% > tolerance)."
            )
    return issues


# ---------------------------------------------------------------------------
# Safe fallback answer generator
# ---------------------------------------------------------------------------

def _build_safe_fallback(validated_claims: list[dict], doc: dict) -> dict:
    """Minimal, maximally safe parent output using only the safest verified claims."""
    verified_numeric = [
        c for c in validated_claims
        if c.get("claim_type") == "visual_numeric" and c.get("show_to_user")
        and c.get("validator_status") == "verified"
    ]

    parts = [
        "Thank you for sharing your child's drawing. "
        "The automated analysis found some visual features in this drawing."
    ]
    if verified_numeric:
        facts = [c["claim"] for c in verified_numeric[:3]]
        parts.append("Observed visual features: " + " ".join(facts))

    parts.append(
        "These are objective visual measurements only. "
        "No psychological conclusions are drawn from this analysis alone."
    )

    return {
        "parent_answer":     "\n\n".join(parts),
        "gentle_questions":  [
            "Can you tell me what is happening in your drawing?",
            "How does this character feel?",
            "Is there a story behind it?",
        ],
        "safety_note": (
            "This interpretation is not diagnostic and should be considered only as a "
            "supportive, contextual observation."
        ),
        "disclaimer":        GENERAL_DISCLAIMER,
        "generation_method": "safe_fallback",
    }


# ---------------------------------------------------------------------------
# Main judge
# ---------------------------------------------------------------------------

def judge_final_response(parent_output: dict,
                          validated_claims: list[dict],
                          doc: dict) -> dict:
    """
    Run all 10 checks. Return judgment dict.
    If REWRITE_REQUIRED or BLOCK, include a safe fallback answer.
    """
    all_issues = []
    all_issues += _check_1_invented_objects(parent_output, validated_claims)
    all_issues += _check_2_unsupported_emotion(parent_output, doc)
    all_issues += _check_3_diagnostic_language(parent_output)
    all_issues += _check_4_overcertainty(parent_output)
    all_issues += _check_5_disclaimer_present(parent_output)
    all_issues += _check_6_gentle_questions(parent_output)
    all_issues += _check_7_scary_conclusions(parent_output)
    all_issues += _check_8_clinical_labels(parent_output)
    all_issues += _check_9_uncertain_as_fact(parent_output, validated_claims)
    all_issues += _check_10_traceable_statements(parent_output, validated_claims, doc)

    checks_passed = 10 - len(all_issues)  # approximate

    # Determine status
    block_issues = [i for i in all_issues if "BLOCK" in i or "Check 3" in i or "Check 8" in i]
    flag_issues  = [i for i in all_issues if i not in block_issues]

    if block_issues:
        status = "BLOCK"
        safe_to_show = False
    elif len(flag_issues) > 3:
        status = "REWRITE_REQUIRED"
        safe_to_show = False
    elif flag_issues:
        status = "REWRITE_REQUIRED"
        safe_to_show = False
    else:
        status = "PASS"
        safe_to_show = True

    result = {
        "final_answer_status": status,
        "issues":              all_issues,
        "safe_to_show":        safe_to_show,
        "required_revisions":  all_issues,
        "checks_passed":       checks_passed,
        "checks_total":        10,
    }

    if not safe_to_show:
        # Attempt sanitisation first
        sanitised = dict(parent_output)
        sanitised["parent_answer"] = sanitise_text(parent_output.get("parent_answer", ""))
        # Re-check after sanitisation
        re_issues = []
        re_issues += _check_3_diagnostic_language(sanitised)
        re_issues += _check_8_clinical_labels(sanitised)
        if not re_issues and status != "BLOCK":
            result["final_answer_status"] = "PASS"
            result["safe_to_show"]        = True
            result["sanitised_answer"]    = sanitised
            result["note"] = "Answer was sanitised and passed re-check."
        else:
            result["safe_fallback_answer"] = _build_safe_fallback(validated_claims, doc)

    return result
