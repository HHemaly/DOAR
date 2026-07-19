from __future__ import annotations

import re


# English diagnostic / over-claim patterns. Broadened (D7) beyond the original
# verb+noun pair to catch "shows signs of", "suffers from", "is depressed", etc.
_CONDITIONS = r"(anxiety|anxious|depression|depressed|trauma|traumati[sz]ed|aggression|aggressive|autism|autistic|abuse|abused|ptsd|adhd|disorder|mentally ill)"
DIAGNOSTIC_PATTERNS = [
    re.compile(rf"\b(has|had|diagnosed with|suffers? from|proves?|confirms?|exhibits?)\s+(a\s+|an\s+)?{_CONDITIONS}\b", re.IGNORECASE),
    re.compile(rf"\b(shows?|showing|signs? of|evidence of|indicates? clear|clearly indicates?)\s+(of\s+)?{_CONDITIONS}\b", re.IGNORECASE),
    re.compile(rf"\bthe child (is|was|seems clinically)\s+{_CONDITIONS}\b", re.IGNORECASE),
]

# Arabic diagnostic terms — the original judge scanned English only (D7 gap).
# Covers "diagnosis/suffers from/afflicted with" + condition nouns.
_ARABIC_DIAGNOSTIC = re.compile(
    r"(تشخيص|يعاني من|مصاب|مصابة|اضطراب|اكتئاب|قلق مرضي|صدمة نفسية|توحد|عدواني|عدوانية|إساءة|مريض نفسي)"
)


def _diagnostic_hit(text: str) -> bool:
    if not text:
        return False
    if _ARABIC_DIAGNOSTIC.search(text):
        return True
    return any(p.search(text) for p in DIAGNOSTIC_PATTERNS)


def run_judges(analysis: dict) -> dict:
    coverage = analysis["composition"]["foreground_coverage"]
    empty = analysis["composition"]["empty_space_ratio"]
    evidence_ids = {item["evidence_id"] for item in analysis["evidence"]}
    rule_errors = []
    for rule in analysis["rule_evaluations"]:
        unknown = set(rule["matched_evidence_ids"]) - evidence_ids
        if unknown:
            rule_errors.append({"rule_id": rule["rule_id"], "unknown_evidence_ids": sorted(unknown)})
    # Scan ALL rule-derived text (English reasoning, parent-safe wording, and
    # any Arabic wording) — not just English professional_reasoning (D7).
    _text_keys = (
        "professional_reasoning", "parent_safe_wording", "english",
        "english_translation", "arabic", "original_arabic", "interpretation",
    )
    narrative_parts = []
    for rule in analysis["rule_evaluations"]:
        for key in _text_keys:
            val = rule.get(key)
            if val:
                narrative_parts.append(str(val))
    narrative = " ".join(narrative_parts)
    diagnostic_found = _diagnostic_hit(narrative)
    disclaimer_text = analysis["safety_disclaimer"].lower()
    disclaimer_present = any(
        phrase in disclaimer_text
        for phrase in ("not a diagnosis", "not diagnostic", "غير تشخيصي", "ليس تشخيص")
    ) or "غير تشخيصي" in analysis["safety_disclaimer"]
    return {
        "quality_judge": {
            "status": "pass" if analysis["quality"]["supported"] else "requires_review",
            "checks": {"supported_file": analysis["quality"]["supported"]},
        },
        "segmentation_judge": {
            "status": "pass" if abs(coverage + empty - 1) < 1e-6 else "fail",
            "checks": {
                "coverage_complementary": abs(coverage + empty - 1) < 1e-6,
                "not_implausibly_full": coverage < 0.90,
                "candidate_disagreement": analysis["segmentation"]["candidate_disagreement"],
            },
        },
        "feature_judge": {
            "status": "pass",
            "checks": {
                "blank_has_no_centroid": not (
                    coverage == 0 and analysis["composition"]["centroid_normalized"] is not None
                ),
                "bbox_evidence_present": "ev_bbox_coverage" in evidence_ids,
            },
        },
        "rule_judge": {
            "status": "pass" if not rule_errors else "fail",
            "unsupported_evidence_references": rule_errors,
            "active_concerns_from_single_symbol": False,
        },
        "safety_judge": {
            "status": "pass" if (not diagnostic_found and disclaimer_present) else "fail",
            "diagnostic_language_found": diagnostic_found,
            "disclaimer_present": disclaimer_present,
            "scanned_languages": ["en", "ar"],
        },
        "module_availability": {
            "detection": "unavailable",
            "ocr": "unavailable",
            "emotion_model": "unavailable",
            "clinician_review": "not_submitted",
        },
    }
