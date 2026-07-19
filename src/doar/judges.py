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
    # ── Quality judge (3-state, from the real quality gate) ──────────────────
    quality_status = analysis["quality"].get(
        "quality_status", "supported" if analysis["quality"].get("supported") else "unsupported")
    quality_judge_status = {"supported": "pass", "requires_review": "requires_review",
                            "unsupported": "fail"}.get(quality_status, "requires_review")

    # ── Segmentation judge — status reflects ALL critical checks (Item 12) ───
    coverage_complementary = abs(coverage + empty - 1) < 1e-6
    not_implausibly_full = coverage < 0.90
    segmentation_pass = coverage_complementary and not_implausibly_full

    # ── Feature judge — real checks, not hardcoded pass (Item 12) ────────────
    blank_has_no_centroid = not (
        coverage == 0 and analysis["composition"].get("centroid_normalized") is not None)
    bbox_evidence_present = "ev_bbox_coverage" in evidence_ids
    feature_pass = blank_has_no_centroid and bbox_evidence_present

    # ── Emotion judge — did a model actually run successfully? (Item 12) ─────
    emotion_status = analysis.get("emotion", {}).get("status", "unavailable")
    emotion_ran = emotion_status == "available"

    # ── Concern provenance — computed, not hardcoded (Item 12) ───────────────
    concerns = analysis.get("concerns", [])
    single_symbol_concern = any(
        len(c.get("supporting_evidence", [])) < 2 or c.get("source_diversity", 0) < 2
        for c in concerns
    )

    # ── Module availability — detected from the analysis (Item 12) ───────────
    module_exec = analysis.get("module_execution", {})
    suppressed = set(module_exec.get("suppressed", []))
    module_availability = {
        "detection": "unavailable",
        "ocr": "unavailable",
        "emotion_model": ("suppressed_low_quality" if "emotion_model" in suppressed
                          else ("available" if emotion_ran else emotion_status)),
        "psychologist_rules": "suppressed_low_quality" if "psychologist_rules" in suppressed else "available",
        "clinician_review": "not_submitted",
    }

    judges = {
        "quality_judge": {
            "status": quality_judge_status,
            "quality_status": quality_status,
            "checks": {
                "resolution_ok": analysis["quality"].get("resolution_ok"),
                "blur_ok": analysis["quality"].get("blur_ok"),
                "contrast_ok": analysis["quality"].get("contrast_ok"),
            },
            "reasons": analysis["quality"].get("unsupported_reasons", []),
        },
        "segmentation_judge": {
            "status": "pass" if segmentation_pass else "fail",
            "checks": {
                "coverage_complementary": coverage_complementary,
                "not_implausibly_full": not_implausibly_full,
                "candidate_disagreement": analysis["segmentation"].get("candidate_disagreement"),
            },
        },
        "feature_judge": {
            "status": "pass" if feature_pass else "requires_review",
            "checks": {
                "blank_has_no_centroid": blank_has_no_centroid,
                "bbox_evidence_present": bbox_evidence_present,
            },
        },
        "emotion_judge": {
            "status": "pass" if emotion_ran else ("suppressed" if emotion_status == "suppressed" else "unavailable"),
            "emotion_model_ran": emotion_ran,
            "emotion_status": emotion_status,
        },
        "rule_judge": {
            "status": "pass" if not rule_errors else "fail",
            "unsupported_evidence_references": rule_errors,
            "active_concerns_from_single_symbol": single_symbol_concern,
        },
        "safety_judge": {
            "status": "pass" if (not diagnostic_found and disclaimer_present) else "fail",
            "diagnostic_language_found": diagnostic_found,
            "disclaimer_present": disclaimer_present,
            "scanned_languages": ["en", "ar"],
        },
        "module_availability": module_availability,
    }

    # ── Overall case status (Item 12) ───────────────────────────────────────
    statuses = [j.get("status") for j in judges.values() if isinstance(j, dict)]
    hard_fail = (judges["safety_judge"]["status"] == "fail"
                 or judges["rule_judge"]["status"] == "fail"
                 or judges["segmentation_judge"]["status"] == "fail"
                 or single_symbol_concern)
    if hard_fail:
        overall = "fail"
    elif "requires_review" in statuses or quality_status != "supported" or not emotion_ran:
        overall = "requires_review"
    else:
        overall = "pass"
    judges["overall_status"] = overall
    judges["clinical_output_suppressed"] = judges["safety_judge"]["status"] == "fail"
    return judges
