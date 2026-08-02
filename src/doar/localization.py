"""
localization.py — Arabic localization of dynamic report content (Item 14).

Translates field labels AND values (status strings, emotion classes, uncertainty
levels, module availability, quality reasons, limitations) — not just headings.
Technical identifiers (evidence IDs, rule IDs, hashes, numbers) are kept verbatim.
Unknown values pass through unchanged so nothing is silently mistranslated.
"""

from __future__ import annotations
import re

# Field-label translations (dict keys shown in report tables).
FIELD_LABELS_AR = {
    "width": "العرض", "height": "الارتفاع", "min_dimension": "أصغر بُعد",
    "contrast_std": "تباين", "blur_variance": "حدة", "resolution_ok": "الدقة مقبولة",
    "blur_ok": "الحدة مقبولة", "contrast_ok": "التباين مقبول", "supported": "مدعوم",
    "quality_status": "حالة الجودة", "unsupported_reasons": "أسباب عدم الدعم",
    "status": "الحالة", "confidence": "الثقة", "background_rgb": "لون الخلفية",
    "selected_strategy": "الاستراتيجية المختارة", "candidate_disagreement": "اختلاف المرشحين",
    "foreground_coverage": "نسبة المحتوى", "empty_space_ratio": "المساحة الفارغة",
    "bounding_box_coverage": "تغطية الإطار", "centroid_normalized": "المركز",
    "placement": "الموضع", "dominant_colour": "اللون السائد",
    "meaningful_colours": "الألوان المهمة", "colour_diversity": "تنوع الألوان",
    "top_class": "الفئة الأعلى", "uncertainty": "عدم اليقين",
    "calibration_status": "حالة المعايرة", "model_name": "اسم النموذج",
    "reason": "السبب", "quality_judge": "حكم الجودة", "segmentation_judge": "حكم التقسيم",
    "feature_judge": "حكم الميزات", "emotion_judge": "حكم النموذج",
    "rule_judge": "حكم القواعد", "safety_judge": "حكم السلامة",
    "overall_status": "الحالة العامة", "module_availability": "توفر الوحدات",
}

# Value translations (status enums, emotion classes, uncertainty, availability).
VALUE_MAP_AR = {
    # emotion classes
    "Angry": "غاضب", "Fear": "خائف", "Happy": "سعيد", "Sad": "حزين",
    # statuses
    "verified": "مُتحقق", "uncertain": "غير مؤكد", "supported": "مدعوم",
    "unsupported": "غير مدعوم", "requires_review": "يتطلب مراجعة",
    "available": "متاح", "unavailable": "غير متاح", "suppressed": "مُوقَف",
    "suppressed_low_quality": "موقف لضعف الجودة", "not_submitted": "لم يُقدَّم",
    "pass": "ناجح", "fail": "فاشل",
    "missing_detector": "لا يوجد كاشف", "not_matched": "غير مطابق",
    "not_evaluated": "لم يُقيَّم", "weak_support": "دعم ضعيف",
    "disabled": "معطّل",
    # uncertainty levels
    "high": "عالٍ", "low": "منخفض", "moderate": "متوسط",
    # booleans
    "True": "نعم", "False": "لا", "true": "نعم", "false": "لا", "None": "غير متوفر",
}


def localize_key(key: str, language: str) -> str:
    if language != "ar":
        return str(key)
    return FIELD_LABELS_AR.get(key, str(key))


def localize_value(value, language: str):
    if language != "ar":
        return value
    text = str(value)
    if text in VALUE_MAP_AR:
        return VALUE_MAP_AR[text]
    return _localize_freetext(text)


# Structured English reason templates -> Arabic (quality reasons, limitations).
_REASON_PATTERNS = [
    (re.compile(r"resolution below (\d+)px \(min_dimension=(\d+)\)"),
     lambda m: f"الدقة أقل من {m.group(1)} بكسل (أصغر بُعد={m.group(2)})"),
    (re.compile(r"low sharpness \(blur_variance=([\d.]+) < ([\d.]+)\)"),
     lambda m: f"حدة منخفضة (قيمة الحدة={m.group(1)} < {m.group(2)})"),
    (re.compile(r"low contrast \(contrast_std=([\d.]+) < ([\d.]+)\)"),
     lambda m: f"تباين منخفض (الانحراف={m.group(1)} < {m.group(2)})"),
]


def _localize_freetext(text: str) -> str:
    for pattern, repl in _REASON_PATTERNS:
        m = pattern.fullmatch(text)
        if m:
            return repl(m)
    # Localize list-of-reasons joined by "; "
    if "; " in text:
        return "؛ ".join(_localize_freetext(part) for part in text.split("; "))
    return text


def localize_mapping(mapping: dict, language: str) -> dict:
    """Return a mapping with localized keys and scalar values (Arabic).
    Nested dicts/lists are localized element-wise where scalar."""
    if language != "ar":
        return mapping
    out = {}
    for key, value in mapping.items():
        if isinstance(value, list):
            value = [localize_value(v, language) if not isinstance(v, (dict, list)) else v
                     for v in value]
        elif not isinstance(value, dict):
            value = localize_value(value, language)
        out[localize_key(key, language)] = value
    return out
