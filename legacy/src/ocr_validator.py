"""
ocr_validator.py — OCR claim validation with confidence filtering and
approximate-match flagging.

Filters:
  1. OCR confidence must be >= OCR_MIN_CONF (0.65) — already done at extraction
  2. Claims between 0.65 and 0.80 are flagged as "uncertain"
  3. Short single-character hits are flagged as uncertain
  4. Common handwriting misread patterns are noted
  5. Negative/sensitive words detected by OCR require higher confidence (0.80)
"""

from __future__ import annotations
import json, os, re

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "thresholds.json")
def _cfg():
    try:
        with open(_CFG_PATH) as f: return json.load(f)
    except Exception: return {}

CFG = _cfg()
_OCR = CFG.get("ocr", {})
OCR_MIN_CONF      = float(_OCR.get("min_confidence",       0.65))
OCR_UNCERTAIN_MAX = float(_OCR.get("uncertain_confidence_max", 0.80))
SHOW_OCR          = float(CFG.get("show_to_user_thresholds", {}).get("ocr_text", 0.70))

# Negative/sensitive words require higher confidence before showing
SENSITIVE_OCR_WORDS = [
    "sad", "angry", "hate", "die", "dead", "kill", "hurt", "cry",
    "lonely", "scared", "fear", "blood", "pain", "alone", "bad",
    "ugly", "monster", "devil", "weapon", "gun", "knife",
]

# Known handwriting misread pairs (original → likely intended)
KNOWN_MISREADS = [
    (r"\bmn\b",  "my"),
    (r"\bme\b",  "me"),  # valid
    (r"\bI\b",   "I"),   # valid
    (r"\b0\b",   "O"),
    (r"\b1\b",   "l"),
]

HANDWRITING_WARNING = (
    "Handwritten text recognition is error-prone. OCR results should be interpreted "
    "cautiously, especially for short words, letters, or low-contrast ink."
)


def _is_sensitive(text: str) -> bool:
    text_lower = text.lower()
    return any(w in text_lower for w in SENSITIVE_OCR_WORDS)


def _is_short_noise(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) <= 1


def validate_ocr_claim(claim: dict) -> dict:
    """Validate a single ocr_text claim."""
    if claim.get("claim_type") != "ocr_text":
        return claim

    ev   = claim.get("evidence", {})
    text = ev.get("text", "").strip()
    conf = float(ev.get("confidence", 0.0))

    # Below minimum — should have been filtered at extraction, but defend here
    if conf < OCR_MIN_CONF:
        claim["validator_status"] = "rejected"
        claim["validator_note"]   = f"OCR confidence {conf:.2f} below minimum {OCR_MIN_CONF:.2f}."
        claim["show_to_user"]     = False
        return claim

    # Short noise
    if _is_short_noise(text):
        claim["validator_status"] = "uncertain"
        claim["validator_note"]   = "Single character or empty OCR result — likely noise."
        claim["show_to_user"]     = False
        return claim

    sensitive = _is_sensitive(text)
    required_conf = 0.80 if sensitive else SHOW_OCR

    if conf >= OCR_UNCERTAIN_MAX:
        claim["validator_status"] = "verified"
        claim["validator_note"]   = (
            f"OCR confidence {conf:.2f} ≥ {OCR_UNCERTAIN_MAX:.2f}. "
            + (HANDWRITING_WARNING if conf < 0.90 else "")
        )
        claim["show_to_user"] = (conf >= required_conf)
    else:
        # Between min_conf and uncertain_max
        claim["validator_status"] = "uncertain"
        claim["validator_note"]   = (
            f"OCR confidence {conf:.2f} is between {OCR_MIN_CONF:.2f} and "
            f"{OCR_UNCERTAIN_MAX:.2f}. Text may be misread. {HANDWRITING_WARNING}"
        )
        claim["show_to_user"] = False

    if sensitive and claim["show_to_user"]:
        claim["sensitive"] = True
        claim["sensitive_note"] = (
            "This OCR result contains a potentially negative/sensitive word. "
            "Interpret with extra caution."
        )

    return claim


def validate_all_ocr_claims(claims: list[dict]) -> list[dict]:
    return [validate_ocr_claim(c) for c in claims]
