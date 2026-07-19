"""
numeric_validator.py — Validates numeric claims against computed feature values.

Fixes the known hallucination-judge weakness where fabricated percentages
were not caught because the judge extracted numbers from text but did not
compare them to actual computed values.

Logic:
  1. For each claim of type "visual_numeric", extract the number from the text.
  2. Look up the ground-truth value from the claim's evidence dict.
  3. Compare with ±tolerance (default ±5%).
  4. Set validator_status to "verified", "uncertain", or "rejected".
  5. For non-numeric claims, pass through unchanged.
"""

from __future__ import annotations
import re, json, os

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.json")
def _cfg():
    try:
        with open(_CFG_PATH) as f: return json.load(f)
    except Exception: return {}

CFG = _cfg()
TOLERANCE = float(CFG.get("numeric_validator", {}).get("tolerance", 0.05))
SHOW_BASIC = float(CFG.get("show_to_user_thresholds", {}).get("basic_visual_fact", 0.60))


def _extract_percent(text: str) -> float | None:
    """Extract the first percentage figure from text as a 0-1 float."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0
    return None


def _extract_integer(text: str) -> int | None:
    """Extract the first plain integer from text (for counts like color diversity)."""
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    return None


def validate_numeric_claim(claim: dict) -> dict:
    """
    Validate a single claim. Modifies and returns the claim dict in-place.
    """
    if claim.get("claim_type") != "visual_numeric":
        return claim

    evidence  = claim.get("evidence", {})
    raw_value = claim.get("raw_value")
    text      = claim.get("claim", "")
    confidence= float(claim.get("confidence", 0.0))

    if raw_value is None:
        # No ground truth available — mark uncertain
        claim["validator_status"] = "uncertain"
        claim["validator_note"]   = "No ground-truth value available for numeric comparison."
        claim["show_to_user"]     = False
        return claim

    # Determine what kind of number we're checking
    ground_truth = float(raw_value)

    # Is the claim expressing a percentage?
    extracted_pct = _extract_percent(text)
    if extracted_pct is not None:
        # ground_truth is 0-1 ratio; extracted_pct is also 0-1 after division
        diff = abs(extracted_pct - ground_truth)
        if diff <= TOLERANCE:
            claim["validator_status"] = "verified"
            claim["validator_note"]   = (
                f"Claimed {extracted_pct*100:.1f}% matches computed "
                f"{ground_truth*100:.1f}% (diff {diff*100:.1f}% ≤ tolerance {TOLERANCE*100:.0f}%)."
            )
            claim["show_to_user"] = (confidence >= SHOW_BASIC)
        else:
            claim["validator_status"] = "rejected"
            claim["validator_note"]   = (
                f"Claimed {extracted_pct*100:.1f}% does NOT match computed "
                f"{ground_truth*100:.1f}% (diff {diff*100:.1f}% > tolerance {TOLERANCE*100:.0f}%). "
                "Possible hallucination or rounding error."
            )
            claim["show_to_user"] = False
        return claim

    # Is it a count / integer?
    extracted_int = _extract_integer(text)
    if extracted_int is not None and isinstance(ground_truth, (int, float)) and ground_truth == int(ground_truth):
        if extracted_int == int(ground_truth):
            claim["validator_status"] = "verified"
            claim["validator_note"]   = f"Claimed count {extracted_int} matches computed {int(ground_truth)}."
            claim["show_to_user"]     = (confidence >= SHOW_BASIC)
        else:
            claim["validator_status"] = "rejected"
            claim["validator_note"]   = (
                f"Claimed count {extracted_int} does NOT match computed {int(ground_truth)}."
            )
            claim["show_to_user"] = False
        return claim

    # Fallback — cannot verify
    claim["validator_status"] = "uncertain"
    claim["validator_note"]   = "Could not extract a verifiable number from claim text."
    claim["show_to_user"]     = False
    return claim


def validate_all_numeric_claims(claims: list[dict]) -> list[dict]:
    """Run numeric validation on all claims. Returns updated list."""
    return [validate_numeric_claim(c) for c in claims]


# ---------------------------------------------------------------------------
# Cross-check helper: scan ANY text for numbers not grounded in the doc
# ---------------------------------------------------------------------------

def find_ungrounded_numbers(text: str, doc: dict, tolerance: float = TOLERANCE) -> list[dict]:
    """
    Scan a free-text string for percentage figures and verify each against
    all known numeric features in the doc.

    Returns a list of violations: {claimed_pct, closest_match, diff, verdict}.
    Used by the final response judge.
    """
    # Collect all ground-truth ratios from the doc
    ground_truths: list[float] = []
    cf = doc.get("color_features",       {})
    cp = doc.get("composition_features", {})
    sf = doc.get("stroke_features",      {})

    for val in [cf.get("dark_dominance"), cf.get("warm_dominance"),
                cp.get("empty_space_ratio"), cp.get("drawn_content_ratio"),
                sf.get("fragmentation_ratio")]:
        if val is not None:
            ground_truths.append(float(val))

    violations = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        claimed = float(m.group(1)) / 100.0
        if not ground_truths:
            violations.append({
                "claimed_pct": round(claimed * 100, 1),
                "verdict": "unverifiable",
                "reason": "No ground-truth numeric features available.",
            })
            continue
        closest = min(ground_truths, key=lambda x: abs(x - claimed))
        diff = abs(claimed - closest)
        if diff > tolerance:
            violations.append({
                "claimed_pct":   round(claimed * 100, 1),
                "closest_match": round(closest * 100, 1),
                "diff_pct":      round(diff * 100, 1),
                "tolerance_pct": round(tolerance * 100, 1),
                "verdict":       "mismatch",
                "reason": (
                    f"Claimed {claimed*100:.1f}% not within ±{tolerance*100:.0f}% "
                    f"of any computed feature value. Closest: {closest*100:.1f}%."
                ),
            })
    return violations
