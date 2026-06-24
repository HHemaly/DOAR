"""
arabic_translator.py — Optional Arabic translation for DOAR parent-facing outputs.

Translates English analysis text to Arabic using the deep-translator library
(free, no API key required). Falls back gracefully if the package is missing.

Usage:
    from arabic_translator import translate_output
    arabic = translate_output(parent_output_dict)
"""

from __future__ import annotations

_TRANSLATOR_AVAILABLE = False
try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    pass


def _translate(text: str) -> str:
    """Translate a single English string to Arabic. Returns original on failure."""
    if not text or not _TRANSLATOR_AVAILABLE:
        return text
    try:
        # Split long texts to respect the 5000-char API limit
        if len(text) <= 4500:
            return GoogleTranslator(source="en", target="ar").translate(text)
        parts = [text[i:i+4500] for i in range(0, len(text), 4500)]
        return "".join(GoogleTranslator(source="en", target="ar").translate(p) for p in parts)
    except Exception:
        return text  # silent fallback


def translate_output(parent_output: dict) -> dict:
    """
    Take a parent-facing output dict (English) and return a copy
    with all text fields translated to Arabic.

    Input keys translated: parent_answer, safety_note, disclaimer,
    gentle_questions (list of strings).

    Returns a new dict — the original is not modified.
    """
    if not _TRANSLATOR_AVAILABLE:
        return {
            "_translation_note": (
                "Arabic translation not available. "
                "Install with: pip install deep-translator"
            )
        }

    ar = {}
    ar["parent_answer"] = _translate(parent_output.get("parent_answer", ""))
    ar["safety_note"]   = _translate(parent_output.get("safety_note", ""))
    ar["disclaimer"]    = _translate(parent_output.get("disclaimer", ""))
    ar["gentle_questions"] = [
        _translate(q) for q in parent_output.get("gentle_questions", [])
    ]
    ar["_translation_note"] = "Translated from English using deep-translator (Google Translate API)."
    return ar


# ── Hardcoded fallback translations for key phrases ─────────────────────────
# Used when offline or when the API call fails for specific terms.

ARABIC_LABELS = {
    "neutral_or_unclear":  "محايد أو غير واضح",
    "happy":               "سعيد",
    "sad":                 "حزين",
    "angry":               "غاضب",
    "fear":                "خائف",
    "PASS":                "اجتاز",
    "BLOCK":               "محظور",
    "REWRITE_REQUIRED":    "يتطلب إعادة صياغة",
    "verified":            "تم التحقق",
    "uncertain":           "غير مؤكد",
    "rejected":            "مرفوض",
    "high":                "عالي",
    "low":                 "منخفض",
    "medium":              "متوسط",
    "heuristic":           "استكشافي",
    "template":            "نموذج",
}


def translate_label(label: str) -> str:
    """Translate a short label/keyword to Arabic, with hardcoded fallbacks."""
    if label in ARABIC_LABELS:
        return ARABIC_LABELS[label]
    return _translate(label)
