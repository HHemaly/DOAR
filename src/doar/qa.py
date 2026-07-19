from __future__ import annotations


def answer(question: str, analysis: dict, judges: dict, language: str = "en") -> dict:
    q = question.casefold()
    ar = language == "ar"
    if any(word in q for word in ("colour", "color", "لون", "ألوان")):
        value = analysis["colour"]["meaningful_colours"]
        return {
            "answer": ("الألوان المرصودة: " if ar else "Detected foreground colours: ") + ", ".join(value),
            "evidence_ids": ["ev_dominant_colour"],
            "source": "ai_objective_analysis",
        }
    if any(word in q for word in ("rule", "قواعد", "قاعدة")):
        evaluated = [r["rule_id"] for r in analysis["rule_evaluations"] if r["status"] != "not_evaluated"]
        unavailable = [r["rule_id"] for r in analysis["rule_evaluations"] if r["status"] == "not_evaluated"]
        return {
            "answer": (
                f"تم تقييم: {', '.join(evaluated)}. غير متاح: {', '.join(unavailable)}."
                if ar else f"Evaluated: {', '.join(evaluated)}. Unavailable: {', '.join(unavailable)}."
            ),
            "evidence_ids": sorted({
                item for rule in analysis["rule_evaluations"] for item in rule["matched_evidence_ids"]
            }),
            "source": "rule_evaluations",
        }
    if any(word in q for word in ("person", "شخص", "إنسان")):
        return {
            "answer": (
                "غير متاح: وحدة اكتشاف الأشخاص لم تُشغّل."
                if ar else "Unavailable: a verified person detector was not run."
            ),
            "evidence_ids": [],
            "source": "module_availability",
        }
    return {
        "answer": (
            "المعلومة غير متاحة في الأدلة المحفوظة."
            if ar else "The requested information is unavailable in the saved evidence."
        ),
        "evidence_ids": [],
        "source": "saved_evidence_only",
    }
