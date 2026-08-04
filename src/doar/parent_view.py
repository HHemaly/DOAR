"""
parent_view.py -- pure functions that turn real analysis output into
plain-language content for the Parent/User view. No Streamlit import (kept
testable and separate from UI, matching human_review.py's convention).

Every function here only rephrases numbers/statuses that already exist in
analysis.json/rules.json -- nothing here invents a value, calls a model,
or fabricates an interpretation. Templates are deterministic string
formatting, not AI-generated text.
"""
from __future__ import annotations

_DISCLAIMER = {
    "en": ("This is a non-diagnostic research tool. A drawing alone cannot "
           "establish a diagnosis. If you have concerns about your child's "
           "emotional wellbeing, please talk with a pediatrician, school "
           "counselor, or a qualified child psychologist."),
    "ar": ("هذه أداة بحثية غير تشخيصية. لا يمكن للرسمة وحدها إثبات تشخيص. "
           "إذا كانت لديك مخاوف بشأن الصحة النفسية لطفلك، يرجى التحدث مع "
           "طبيب أطفال أو مرشد مدرسي أو أخصائي نفسي مؤهل للأطفال."),
}

_PLACEMENT_TEXT = {
    "en": {
        "middle_center": "roughly centered on the page",
        "top_center": "placed toward the top of the page",
        "bottom_center": "placed toward the bottom of the page",
        "middle_left": "placed toward the left side of the page",
        "middle_right": "placed toward the right side of the page",
        "top_left": "placed in the upper-left area of the page",
        "top_right": "placed in the upper-right area of the page",
        "bottom_left": "placed in the lower-left area of the page",
        "bottom_right": "placed in the lower-right area of the page",
        "unavailable": "impossible to determine placement (page appears blank)",
    },
    "ar": {
        "middle_center": "في منتصف الصفحة تقريباً",
        "top_center": "في الجزء العلوي من الصفحة",
        "bottom_center": "في الجزء السفلي من الصفحة",
        "middle_left": "في الجانب الأيسر من الصفحة",
        "middle_right": "في الجانب الأيمن من الصفحة",
        "top_left": "في الزاوية العلوية اليسرى",
        "top_right": "في الزاوية العلوية اليمنى",
        "bottom_left": "في الزاوية السفلية اليسرى",
        "bottom_right": "في الزاوية السفلية اليمنى",
        "unavailable": "يتعذر تحديد الموضع (يبدو أن الصفحة فارغة)",
    },
}


def disclaimer(language: str = "en") -> str:
    return _DISCLAIMER["ar" if language == "ar" else "en"]


def plain_language_observations(analysis: dict, language: str = "en") -> list[dict]:
    """Real measured composition/colour/quality values, rephrased in plain
    sentences. Each item carries the evidence_id(s) it was derived from."""
    ar = language == "ar"
    comp = analysis.get("composition", {})
    colour = analysis.get("colour", {})
    quality = analysis.get("quality", {})
    out = []

    coverage = comp.get("foreground_coverage")
    if coverage is not None:
        pct = round(coverage * 100)
        text = (f"يغطي محتوى الرسمة نحو {pct}% من الصفحة."
                if ar else f"The drawing's content covers about {pct}% of the page.")
        out.append({"text": text, "evidence_ids": ["ev_seg_coverage"]})

    placement = comp.get("placement")
    if placement:
        phrase = _PLACEMENT_TEXT["ar" if ar else "en"].get(placement, placement)
        text = f"الرسمة {phrase}." if ar else f"The drawing is {phrase}."
        out.append({"text": text, "evidence_ids": ["ev_centroid"]})

    colours = colour.get("meaningful_colours", [])
    if colours:
        joined = "، ".join(colours) if ar else ", ".join(colours)
        text = f"الألوان الملحوظة في الرسمة: {joined}." if ar else f"Noticeable colours in the drawing: {joined}."
        out.append({"text": text, "evidence_ids": ["ev_dominant_colour"]})
    else:
        text = "لم تُرصد ألوان بارزة (رسمة أحادية اللون غالباً)." if ar else "No strongly distinct colours were detected (likely a monochrome drawing)."
        out.append({"text": text, "evidence_ids": ["ev_dominant_colour"]})

    quality_status = quality.get("quality_status")
    if quality_status and quality_status != "supported":
        reasons = "; ".join(quality.get("unsupported_reasons", []))
        text = (f"جودة الصورة '{quality_status}' وقد تحد من موثوقية التحليل ({reasons})."
                if ar else f"Image quality is '{quality_status}', which may limit how reliable this analysis is ({reasons}).")
        out.append({"text": text, "evidence_ids": []})

    return out


_RULE_STATUS_MESSAGE = {
    "weak_support": {
        "en": "Evaluated: this pattern WAS observed. This is a possible, low-confidence observation, not a fact.",
        "ar": "تم التقييم: لوحظ هذا النمط. هذه ملاحظة محتملة بثقة منخفضة، وليست حقيقة مؤكدة.",
    },
    "not_matched": {
        "en": "Evaluated: this pattern was checked and was NOT observed in this drawing.",
        "ar": "تم التقييم: تم فحص هذا النمط ولم يُلاحظ في هذه الرسمة.",
    },
    "missing_detector": {
        "en": "Not evaluated: no detector exists yet for this feature in this release. This is not evidence against it — it simply wasn't checked.",
        "ar": "لم يُقيَّم: لا يوجد كاشف لهذه السمة بعد في هذا الإصدار. هذا لا يعني عدم وجودها، بل لم يتم فحصها.",
    },
    "not_assessable_context_unknown": {
        "en": "Not evaluated: this would require knowing the drawing prompt/context, which is not available.",
        "ar": "لم يُقيَّم: يتطلب هذا معرفة سياق أو تعليمات الرسم، وهي غير متوفرة.",
    },
}


def plain_language_rule_rows(rule_evaluations: list[dict], language: str = "en") -> list[dict]:
    """One row per rule, in the vocabulary a parent can read. Status is
    never hidden or collapsed -- missing_detector is shown distinctly from
    not_matched, per RULE_AND_FEATURE_COVERAGE.md Section 4."""
    ar = language == "ar"
    rows = []
    for rule in rule_evaluations:
        status = rule["status"]
        label = rule["original_arabic"] if ar else rule["english_translation"]
        message = _RULE_STATUS_MESSAGE.get(status, {}).get("ar" if ar else "en", status)
        note = None
        if status == "weak_support":
            note = rule.get("parent_safe_wording")
        rows.append({
            "rule_id": rule["rule_id"], "label": label, "status": status,
            "message": message, "note": note,
            "confidence_ceiling": rule.get("confidence_ceiling"),
            "evidence_ids": rule.get("matched_evidence_ids", []),
        })
    return rows


def overall_interpretation(analysis: dict, language: str = "en") -> str:
    """A deterministic, template-composed summary paragraph -- NOT an
    AI-generated interpretation. Explicitly labeled as such by the caller."""
    ar = language == "ar"
    emotion = analysis.get("emotion", {})
    rules = analysis.get("rule_evaluations", [])
    triggered = [r for r in rules if r["status"] == "weak_support"]
    parts = []

    status = emotion.get("status", "unavailable")
    if status == "available":
        top = emotion.get("top_class")
        conf = emotion.get("confidence", 0.0)
        cal = emotion.get("calibration_status", "uncalibrated")
        if ar:
            parts.append(
                f"توقع النموذج الحالة الانفعالية الأكثر احتمالاً هي '{top}' "
                f"بثقة {conf:.0%} (حالة المعايرة: {cal}). هذا احتمال إحصائي "
                "من نموذج تصنيف، وليس تقييماً نفسياً.")
        else:
            parts.append(
                f"The model's single most likely emotion label is '{top}' "
                f"with {conf:.0%} confidence (calibration: {cal}). This is a "
                "statistical classifier output, not a psychological assessment.")
    elif status == "suppressed":
        parts.append("لم يُشغَّل نموذج التصنيف بسبب ضعف جودة الصورة." if ar
                     else "The classification model did not run because image quality was insufficient.")
    else:
        parts.append("لا يتوفر تنبؤ انفعالي لهذه الحالة." if ar
                     else "No emotion prediction is available for this case.")

    if triggered:
        ids = ", ".join(r["rule_id"] for r in triggered)
        parts.append(
            (f"تم رصد {len(triggered)} نمط/أنماط بصرية ضعيفة الدعم ({ids})، وجميعها فرضيات "
             "غير مُتحقق منها علمياً وسقف ثقتها منخفض متعمد.")
            if ar else
            (f"{len(triggered)} weak-support visual pattern(s) were observed ({ids}) — "
             "these are unvalidated hypotheses with an intentionally low confidence cap.")
        )
    else:
        parts.append("لم تُلاحظ أنماط بصرية ذات دعم في هذه الرسمة." if ar
                     else "No supported visual patterns were observed in this drawing.")

    parts.append(disclaimer(language))
    return " ".join(parts)
