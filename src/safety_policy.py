"""
safety_policy.py — Central safety policy for DOAR v2.

Defines forbidden diagnostic language, required cautious phrases, confidence
thresholds for sensitive claims, and the general non-diagnostic disclaimer.
All other modules import from here so policy is enforced consistently.
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# Load thresholds from config
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.json")

def _load_cfg():
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

CFG = _load_cfg()
SHOW_THRESHOLDS = CFG.get("show_to_user_thresholds", {})
RULE_WEIGHTS    = CFG.get("rule_weights", {})
PSYCH_THRESHOLDS = CFG.get("psychological_rules", {})
NUMERIC_TOLERANCE = CFG.get("numeric_validator", {}).get("tolerance", 0.05)

# ---------------------------------------------------------------------------
# General disclaimer — must appear in every parent-facing report
# ---------------------------------------------------------------------------
GENERAL_DISCLAIMER = (
    "Drawing-based psychological indicators are not diagnostic on their own. "
    "They should be interpreted cautiously and only as supportive observations "
    "alongside the child's verbal explanation, developmental age, cultural context, "
    "caregiver input, and professional clinical assessment when needed."
)

# ---------------------------------------------------------------------------
# Forbidden phrases — BLOCK if any appear in generated output
# ---------------------------------------------------------------------------
BLOCK_PATTERNS = [
    # Hard diagnoses
    r"\b(has|have|diagnosed with)\s+(depression|anxiety disorder|autism|adhd|ptsd|trauma|schizophrenia|bipolar|ocd|attachment disorder)\b",
    r"\b(is|are)\s+(mentally ill|psychologically disturbed|emotionally disturbed)\b",
    # "diagnosis"/"diagnosed" only when used affirmatively — exclude "not diagnostic", "non-diagnostic"
    r"(?<!not\s)(?<!non[- ])\bdiagnos(ed|is)\b",
    r"\bclinical(ly)?\s+(confirm|prove|establish|diagnose)\b",
    r"\bneeds?\s+(immediate\s+)?(therapy|treatment|psychiatric|clinical)\b",
    r"\bshould\s+see\s+(a\s+)?(therapist|psychiatrist|psychologist)\s+immediately\b",
    r"\bconfirms?\s+(trauma|abuse|depression|disorder)\b",
    r"\bproves?\s+that\b",
    r"\bwithout\s+(a\s+)?doubt\b",
    r"\bthis\s+means?\s+the\s+child\b",
    r"\bthe\s+child\s+(is|has)\s+(aggressive|depressed|traumatized|abused)\b",
    # Abuse/trauma direct claims
    r"\b(signs?\s+of|evidence\s+of)\s+(abuse|trauma|neglect)\b",
]

# Patterns that FLAG (require caution banner and sanitisation) but do not BLOCK
FLAG_PATTERNS = [
    r"\bthe\s+child\s+is\s+(sad|angry|scared|unhappy|distressed|upset)\b",
    r"\bclearly\s+(shows?|indicates?|suggests?|demonstrates?)\b",
    r"\bthis\s+(shows?|proves?|confirms?|indicates?)\b",
    r"\bwithout\s+question\b",
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bwill\s+(suffer|struggle|have\s+problems)\b",
    r"\b(must|have\s+to)\s+be\s+(assessed|evaluated)\b",
]

# Sanitisation replacements for FLAG-level issues
SANITISE_REPLACEMENTS = [
    (r"the\s+child\s+is\s+(sad|unhappy|distressed)",
     r"some features may be associated with \1 affect"),
    (r"clearly\s+(shows?|indicates?|suggests?)",
     r"may \1"),
    (r"this\s+proves?\s+that",
     "this is consistent with the possibility that"),
    (r"without\s+(a\s+)?doubt",
     "with some uncertainty"),
    (r"definitely",
     "possibly"),
    (r"certainly",
     "possibly"),
]

# ---------------------------------------------------------------------------
# Cautious wording helpers
# ---------------------------------------------------------------------------
CAUTIOUS_VERBS = [
    "may suggest",
    "could indicate",
    "might reflect",
    "can be associated with",
    "one possible interpretation is",
    "is sometimes linked to",
]

def cautious_phrase(finding: str, verb: str = "may suggest") -> str:
    """Wrap a finding in cautious, non-diagnostic language."""
    return f"{verb} {finding}"

def add_caveat(text: str, caveat: str = "This is not diagnostic and requires contextual interpretation.") -> str:
    return f"{text} {caveat}"

# ---------------------------------------------------------------------------
# Sensitive claim categories — require higher confidence thresholds
# ---------------------------------------------------------------------------
SENSITIVE_CLAIM_TYPES = {
    "weapon_symbol",
    "blood_mark",
    "threatening_scene",
    "self_harm_symbol",
    "closed_eyes",
    "angry_eyes",
    "animal_type_specific",
    "concerning_symbol",
}

def get_show_threshold(claim_type: str) -> float:
    """Return the minimum confidence required to show a claim to the user."""
    vc = CFG.get("object_detection", {}).get("visual_claim_validator", {})
    if claim_type in ("weapon_symbol", "blood_mark", "threatening_scene", "self_harm_symbol",
                      "concerning_symbol"):
        return vc.get("concerning_symbol_threshold", 0.85)
    if claim_type in ("closed_eyes", "angry_eyes", "eye_feature"):
        return vc.get("eye_feature_threshold", 0.75)
    if claim_type in ("animal_type_specific",):
        return vc.get("animal_type_threshold", 0.72)
    if claim_type in ("visual_object", "visual_symbol"):
        return vc.get("object_symbol_threshold", 0.70)
    if claim_type == "ocr_text":
        return CFG.get("ocr", {}).get("min_confidence", 0.65)
    return SHOW_THRESHOLDS.get("basic_visual_fact", 0.60)

# ---------------------------------------------------------------------------
# Forbidden diagnostic labels (used in final response judge)
# ---------------------------------------------------------------------------
DIAGNOSTIC_LABELS = [
    "depression", "anxiety disorder", "autism", "adhd", "ptsd",
    "trauma", "schizophrenia", "bipolar", "ocd", "attachment disorder",
    "conduct disorder", "oppositional defiant", "emotionally disturbed",
    "mentally ill", "psychologically disturbed",
]

def check_for_diagnostic_language(text: str) -> list[dict]:
    """Return list of violations found in text."""
    text_lower = text.lower()
    violations = []
    for pat in BLOCK_PATTERNS:
        if re.search(pat, text_lower):
            violations.append({"severity": "BLOCK", "pattern": pat,
                                "reason": "Forbidden diagnostic or over-certain language"})
    for pat in FLAG_PATTERNS:
        if re.search(pat, text_lower):
            violations.append({"severity": "FLAG", "pattern": pat,
                                "reason": "Overclaiming or emotionally certain language"})
    return violations

def sanitise_text(text: str) -> str:
    """Apply all sanitisation replacements to text."""
    result = text
    for pattern, replacement in SANITISE_REPLACEMENTS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result
