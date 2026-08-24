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

_SOURCE_FRIENDLY_NAMES: dict[str, dict[str, str]] = {
    "child_drawing_rules_compiled.pdf": {
        "en": "a compiled reference guide on children's drawings",
        "ar": "دليل مرجعي مجمّع حول رسومات الأطفال",
    },
    "التحليل النفسي للصور.pdf": {
        "en": "a psychologist's notes on drawing analysis",
        "ar": "ملاحظات أخصائي نفسي حول تحليل الرسومات",
    },
}

# The exact closed set analysis.py's _colour() can emit in
# meaningful_colours (verified against that function directly this pass).
_COLOUR_NAMES: dict[str, dict[str, str]] = {
    "red": {"en": "red", "ar": "أحمر"},
    "green": {"en": "green", "ar": "أخضر"},
    "blue": {"en": "blue", "ar": "أزرق"},
    "yellow": {"en": "yellow", "ar": "أصفر"},
    "dark": {"en": "dark", "ar": "داكن"},
    "none_or_neutral": {"en": "none/neutral", "ar": "لا يوجد / محايد"},
}


def friendly_colour_name(colour: str, language: str = "en") -> str:
    entry = _COLOUR_NAMES.get(colour)
    return entry["ar" if language == "ar" else "en"] if entry else colour


_FAMILY_FRIENDLY_NAMES: dict[str, dict[str, str]] = {
    "size_composition": {"en": "page-space use", "ar": "استخدام مساحة الصفحة"},
    "spatial_placement": {"en": "placement", "ar": "الموضع"},
    "line_intensity_quality": {"en": "line-appearance", "ar": "مظهر الخط"},
    "line_fragmentation_quality": {"en": "line-appearance", "ar": "مظهر الخط"},
}


def friendly_source_name(source_document: str, language: str = "en") -> str:
    """A parent-readable name for a source document -- never the raw
    filename. Falls back to the raw name (never hidden) if a document is
    ever added without a mapping, so an unrecognized source is still
    honestly shown rather than silently dropped."""
    entry = _SOURCE_FRIENDLY_NAMES.get(source_document)
    return entry["ar" if language == "ar" else "en"] if entry else source_document


def friendly_family_name(evidence_family: str, language: str = "en") -> str:
    entry = _FAMILY_FRIENDLY_NAMES.get(evidence_family)
    return entry["ar" if language == "ar" else "en"] if entry else evidence_family.replace("_", " ")


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


# ---------------------------------------------------------------------------
# Milestone 2: simplified Parent View -- "what the parent can do" / "when
# to consider a professional". Deliberately generic, non-case-specific
# guidance (never a fabricated clinical recommendation tied to this one
# drawing) -- static, translated templates only.
# ---------------------------------------------------------------------------

_PARENT_CARE_SUGGESTIONS: dict[str, list[str]] = {
    "en": [
        "Keep drawing and creative activities relaxed, playful, and low-pressure.",
        "Ask your child open, curious questions about their drawing (\"Can you tell me about this part?\") "
        "rather than leading questions.",
        "Notice patterns over time and across several drawings, not just this one picture.",
        "Share anything you notice with your child's teacher, pediatrician, or another trusted adult in "
        "their life if you'd like a second perspective.",
    ],
    "ar": [
        "حافظي/حافظ على أجواء مريحة وغير ضاغطة أثناء الرسم والأنشطة الإبداعية.",
        "اطرحي/اطرح على طفلك أسئلة مفتوحة وفضولية عن رسمته (\"هل يمكنك أن تخبرني عن هذا الجزء؟\") بدلاً من "
        "الأسئلة الموجِّهة.",
        "لاحظي/لاحظ الأنماط عبر الوقت وعدة رسومات، وليس فقط هذه الرسمة الواحدة.",
        "شاركي/شارك ما تلاحظينه/تلاحظه مع معلم طفلك أو طبيب الأطفال أو شخص بالغ آخر تثقين/تثق به إذا "
        "أردت رأياً إضافياً.",
    ],
}

_PROFESSIONAL_REFERRAL_GUIDANCE: dict[str, list[str]] = {
    "en": [
        "The same specific pattern keeps repeating across several drawings over time.",
        "You have concerns about your child's mood, behavior, or development that go beyond a single drawing.",
        "You would value an in-person developmental or psychological assessment for peace of mind.",
        "This tool cannot diagnose anything -- only a licensed professional can.",
    ],
    "ar": [
        "إذا استمر ظهور نفس النمط تحديداً عبر عدة رسومات على مدى فترة من الوقت.",
        "إذا كانت لديك مخاوف بشأن مزاج طفلك أو سلوكه أو نموه تتجاوز رسمة واحدة.",
        "إذا رغبت/رغبتِ في تقييم نمائي أو نفسي حضوري لمزيد من الاطمئنان.",
        "لا يمكن لهذه الأداة تشخيص أي شيء -- فقط أخصائي مرخّص يمكنه ذلك.",
    ],
}


def parent_care_suggestions(language: str = "en") -> list[str]:
    """Generic, non-case-specific things a parent can do -- never a
    fabricated recommendation derived from this one drawing's content."""
    return _PARENT_CARE_SUGGESTIONS["ar" if language == "ar" else "en"]


def professional_referral_guidance(language: str = "en") -> list[str]:
    """Generic, non-case-specific signs that a professional, in-person
    assessment may be worth considering."""
    return _PROFESSIONAL_REFERRAL_GUIDANCE["ar" if language == "ar" else "en"]


def plain_language_observations(analysis: dict, language: str = "en", page_reference: dict | None = None) -> list[dict]:
    """Real measured composition/colour/quality values, rephrased in plain
    sentences. Each item carries the evidence_id(s) it was derived from.

    DOAR-TRACE Phase 2A.2, Section 5: `page_reference` (defaults to
    `analysis.get("page_reference")` when not passed explicitly, for
    backward compatibility with older callers/cases) gates the page-
    coverage/placement sentences -- when no page reference is
    assessable, this function must NOT claim "covers X% of the page" or
    "placed at the top/left/right/centre of the page", since neither
    claim is meaningful without a confirmed page. See
    docs/PARENT_VIEW_INFORMATION_POLICY.md."""
    ar = language == "ar"
    comp = analysis.get("composition", {})
    colour = analysis.get("colour", {})
    quality = analysis.get("quality", {})
    out = []

    if page_reference is None:
        page_reference = analysis.get("page_reference")
    page_assessable = bool((page_reference or {}).get("page_relative_features_assessable"))

    if page_assessable:
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
    else:
        text = (
            "تعذّر تأكيد ظهور الورقة كاملة في هذه الصورة، لذلك لم يتم تفسير استخدام الصفحة أو الموضع."
            if ar else
            "The complete sheet could not be confirmed, so page use and placement were not interpreted."
        )
        out.append({"text": text, "evidence_ids": []})

    colours = [friendly_colour_name(c, language) for c in colour.get("meaningful_colours", [])]
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
        parts.append(
            (f"تم رصد {len(triggered)} نمط/أنماط بصرية ضعيفة الدعم، وجميعها فرضيات "
             "غير مُتحقق منها علمياً وسقف ثقتها منخفض متعمد.")
            if ar else
            (f"{len(triggered)} weak-support visual pattern(s) were observed — "
             "these are unvalidated hypotheses with an intentionally low confidence cap.")
        )
    else:
        parts.append("لم تُلاحظ أنماط بصرية ذات دعم في هذه الرسمة." if ar
                     else "No supported visual patterns were observed in this drawing.")

    parts.append(disclaimer(language))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# DOAR-TRACE Phase 2A.2, Section 3: the "Overall result" summary -- the
# first thing a parent reads. Generated entirely from real per-case data
# (structured_analysis.json + page_reference); never hard-coded to a
# specific case. See docs/PARENT_VIEW_INFORMATION_POLICY.md.
# ---------------------------------------------------------------------------

def build_overall_result_summary(
    structured: dict | None, page_reference: dict | None, language: str = "en",
    *, capabilities: dict | None = None,
) -> list[str]:
    """Returns an ordered list of plain sentences covering, in order: (1)
    whether any combined pattern exists, (2) the expressive-content model
    result when available, (3) how many individual heuristic suggestions
    exist, (4) whether objects/visual elements were automatically scanned
    (derived from `capabilities["visual_detection"]` -- the case's real,
    live module_availability, e.g. from judges.json -- never a hardcoded
    claim; `capabilities=None` keeps the pre-visual-detection wording,
    accurate for a case analyzed before that capability existed), (5)
    whether the complete page was assessable. Every sentence is derived
    from real data for THIS case -- no case value is ever hard-coded
    here."""
    ar = language == "ar"
    structured = structured or {}
    combined = structured.get("combined_drawing_level_hypotheses", [])
    individual = structured.get("individual_rule_suggestions", [])
    model_obs = structured.get("expressive_model_observation")
    page_assessable = bool((page_reference or {}).get("page_relative_features_assessable"))
    sentences: list[str] = []

    if combined:
        sentences.append(
            (f"تم تحديد {len(combined)} نمط/أنماط مُجمَّعة من أدلة مستقلة -- انظر قسم 'المعنى المحتمل' أدناه."
             if ar else
             f"{len(combined)} combined drawing pattern(s) were identified from independent evidence -- see "
             "'Possible meaning' below.")
        )
    else:
        not_assessed = []
        visual_available = (capabilities or {}).get("visual_detection") == "available"
        if not visual_available:
            not_assessed.append("الأجسام والعناصر البصرية" if ar else "objects and visual elements")
        if not page_assessable:
            not_assessed.append("استخدام الصفحة وموضع الرسمة" if ar else "page use and drawing placement")
        if not_assessed:
            sentences.append(
                (f"لم يُحدَّد أي نمط مُجمَّع من الأدلة المتاحة حالياً؛ تعذّر تقييم: {', '.join(not_assessed)}." if ar
                 else f"No combined pattern was identified from the evidence currently available; "
                      f"{', '.join(not_assessed)} could not be assessed for this case.")
            )
        else:
            sentences.append(
                "لم يُحدَّد أي نمط مُجمَّع من الأدلة المتاحة لهذه الحالة." if ar else
                "No combined pattern was identified from the evidence available for this case."
            )

    if model_obs:
        sentences.append(model_obs["text"])

    if individual:
        if len(individual) == 1:
            family = friendly_family_name(individual[0]["evidence_family"], language)
            sentences.append(
                (f"لوحظت ملاحظة واحدة ضعيفة الدعم متعلقة بـ{family}." if ar else
                 f"One weak {family} observation was found.")
            )
        else:
            sentences.append(
                (f"لوحظت {len(individual)} ملاحظات فردية ضعيفة الدعم." if ar else
                 f"{len(individual)} weak, individual observations were found.")
            )
    else:
        sentences.append(
            "لم تُلاحظ أي ملاحظات فردية ضعيفة الدعم." if ar else
            "No individual heuristic observations were found."
        )

    if capabilities and capabilities.get("visual_detection") == "available":
        sentences.append(
            "تم فحص الرسمة تلقائياً بحثاً عن أجسام وعناصر بصرية معروفة؛ لا يزال تحليل العلاقات المكانية بين "
            "العناصر غير متوفر في هذا الإصدار." if ar else
            "The drawing was automatically scanned for known objects and visual elements; spatial "
            "relationships between them are not yet analyzed in this version."
        )
    else:
        sentences.append(
            "لم يتم تحليل الأجسام أو العلاقات بينها في هذا الإصدار." if ar else
            "Objects and relationships were not analyzed in this version."
        )

    if page_assessable:
        sentences.append(
            "تم تأكيد ظهور الصفحة كاملة، لذا شمل التحليل استخدام الصفحة وموضع الرسمة." if ar else
            "The complete sheet was confirmed visible, so page use and placement were included in this analysis."
        )
    else:
        sentences.append(
            "تعذّر تأكيد ظهور الورقة كاملة، لذا لم يتم تفسير استخدام الصفحة أو الموضع." if ar else
            "The complete sheet could not be confirmed, so page use and placement were not interpreted."
        )

    return sentences


# ---------------------------------------------------------------------------
# DOAR-TRACE Phase 2A.2, Section 7: capability status -- honest, not
# aspirational. Every "WORKING" item is genuinely wired for every case;
# "LIMITED"/"NOT AVAILABLE" are real, current gaps, not a roadmap.
# ---------------------------------------------------------------------------

_CAPABILITY_STATUS: dict[str, dict[str, list[str]]] = {
    "working": {
        "en": [
            "Image quality and segmentation", "Objective feature extraction",
            "Page-reference assessment", "Expressive-content model (when a checkpoint is loaded)",
            "Automatic visual object detection and on-demand visual search (when real models are available)",
            "10 executable heuristic rules", "Evidence tracking", "Deterministic verification",
            "Expert review workflow",
        ],
        "ar": [
            "فحص جودة الصورة وتجزئتها", "استخراج السمات الموضوعية",
            "تقييم مرجع الصفحة", "نموذج المحتوى التعبيري (عند تحميل نقطة تفتيش)",
            "الكشف التلقائي عن الأجسام والبحث البصري عند الطلب (عند توفر النماذج الفعلية)",
            "10 قواعد استدلالية قابلة للتنفيذ", "تتبع الأدلة", "التحقق الحتمي",
            "سير عمل مراجعة الخبير",
        ],
    },
    "limited": {
        "en": [
            "Page-relative interpretation (only when the complete sheet is confirmed visible)",
            "Line-appearance proxies (not physical pencil pressure)",
            "Combined drawing-pattern aggregation",
            "Deterministic follow-up answers (no free-form AI chat)",
            "Visual detections outside the frozen validated policy (experimental/unknown evidence -- "
            "stored and searchable, never activates a psychological rule)",
        ],
        "ar": [
            "التفسير المتعلق بالصفحة (فقط عند تأكيد ظهور الورقة كاملة)",
            "مؤشرات مظهر الخط (وليست ضغط القلم الفعلي)",
            "تجميع الأنماط على مستوى الرسمة",
            "إجابات المتابعة الحتمية (بدون محادثة ذكاء اصطناعي حرة)",
            "الكشوفات البصرية خارج السياسة المعتمدة المجمدة (أدلة تجريبية/غير معروفة -- تُحفظ ويمكن "
            "البحث عنها، لكنها لا تُفعّل أي قاعدة نفسية أبداً)",
        ],
    },
    "not_available": {
        "en": [
            "Spatial relationships between detected objects (e.g. count/position/adjacency reasoning)",
            "Parent-context updating", "LLM explanation", "Visual AI consistency judge",
        ],
        "ar": [
            "العلاقات المكانية بين الأجسام المكتشفة (مثل العد أو الموضع أو التجاور)",
            "تحديث سياق الوالدين", "الشرح بواسطة نموذج لغوي كبير", "قاضي الاتساق البصري بالذكاء الاصطناعي",
        ],
    },
}


def capability_status(language: str = "en") -> dict[str, list[str]]:
    """Real capability inventory: {"working": [...], "limited": [...],
    "not_available": [...]}. Static across cases in this release -- every
    listed item's status is genuinely stable, not case-dependent. (For
    the actual runtime state of a SPECIFIC, already-analyzed case, read
    that case's judges.json["module_availability"] instead -- the
    per-case canonical capability-state source, e.g. via
    build_overall_result_summary's `capabilities` argument.)"""
    return {tier: items[language if language in ("en", "ar") else "en"] for tier, items in _CAPABILITY_STATUS.items()}


def capability_status_summary_text(language: str = "en") -> str:
    """One or two plain sentences for the Parent view -- the full
    itemized list belongs in Technical view only."""
    if language == "ar":
        return (
            "تعمل هذه الأداة على تحليل جودة الصورة، السمات الموضوعية، مرجع الصفحة، والنموذج التعبيري، وعشر قواعد "
            "استدلالية. لا تتضمن هذه النسخة كشف الأجسام أو الوجوه أو العلاقات المكانية أو أي شرح بذكاء اصطناعي توليدي."
        )
    return (
        "This tool analyzes image quality, objective features, page reference, the expressive-content model, and "
        "10 heuristic rules. This version does not include object, face, or spatial-relationship detection, or any "
        "generative-AI explanation."
    )
