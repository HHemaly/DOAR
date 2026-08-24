"""Presentation-layer bilingual (English/Arabic) labels and phrasing for
`CaseInterpretation` output -- built for the pre-doctor stabilization pass.

Strictly presentation only: every dict here maps a STABLE, ALREADY-FROZEN
internal identifier (a concern-domain id, a support-level enum value, an
`observable_feature`/`evidence_family` value from RULE_EVIDENCE_MATRIX.csv)
to a human-readable label in each language. Nothing here reads or writes
CONCERN_DOMAIN_MAP.json/RULE_EVIDENCE_MATRIX.csv, computes a support level,
changes a threshold, or adds/removes a rule -- it only relabels numbers and
ids that `case_interpretation.py` already computed. No Streamlit import,
so every function here is directly unit-testable (mirrors parent_view.py's
existing pure bilingual-function pattern).

The `_LABELS` dicts are closed-vocabulary and exhaustive as of this pass:
DOMAIN_LABELS covers every domain id `case_interpretation.py` can ever
emit (the 9 real CONCERN_DOMAIN_MAP.json domains minus the 2 non-clinical
ones it already skips, plus the 3 `_UNMAPPED_DOMAINS`); OBSERVATION_LABELS/
EVIDENCE_FAMILY_LABELS cover every `observable_feature`/`evidence_family`
value in RULE_EVIDENCE_MATRIX.csv (41 and 15 respectively, verified against
the file directly). An id outside these closed sets (only possible if the
frozen registries themselves change) falls back to an underscore-replaced
label rather than raising -- never a translation, but never a crash either.
"""
from __future__ import annotations

import re

Language = str  # "en" | "ar"


def _pick(entry: dict, language: Language) -> str:
    return entry.get(language) or entry["en"]


DOMAIN_LABELS: dict[str, dict[str, str]] = {
    "positive_affect_or_social_engagement": {
        "en": "Positive affect / social engagement", "ar": "مؤشرات إيجابية / تفاعل اجتماعي"},
    "anxiety_or_stress_related": {
        "en": "Anxiety / stress-related indicators", "ar": "مؤشرات مرتبطة بالقلق أو التوتر"},
    "depressive_or_low_mood_related": {
        "en": "Depressive / low-mood-related indicators", "ar": "مؤشرات مرتبطة بانخفاض المزاج أو الاكتئاب"},
    "aggression_or_threat_related": {
        "en": "Aggression / threat-related indicators", "ar": "مؤشرات مرتبطة بالعدوان أو التهديد"},
    "social_withdrawal_related": {
        "en": "Social withdrawal-related indicators", "ar": "مؤشرات مرتبطة بالانسحاب الاجتماعي"},
    "developmental_or_attention_related": {
        "en": "Attention / developmental-related indicators", "ar": "مؤشرات مرتبطة بالانتباه أو النمو"},
    "maltreatment_or_safety_concern": {
        "en": "Maltreatment / safety concern (never proposed by DOAR on its own)",
        "ar": "مخاوف متعلقة بالإساءة أو السلامة (لا يقترحها DOAR أبداً من تلقاء نفسه)"},
    "neutral_descriptive_only": {"en": "Neutral / descriptive only", "ar": "وصفي محايد فقط"},
    "neutral_descriptive_only_no_construct_proposed": {
        "en": "Neutral / descriptive only", "ar": "وصفي محايد فقط"},
    "trauma_related": {"en": "Trauma-related indicators", "ar": "مؤشرات مرتبطة بالصدمة"},
    "bullying_related": {"en": "Bullying-related indicators", "ar": "مؤشرات مرتبطة بالتنمر"},
    "autism_related": {"en": "Autism-related indicators", "ar": "مؤشرات مرتبطة بطيف التوحد"},
}

SUPPORT_LEVEL_LABELS: dict[str, dict[str, str]] = {
    "STRONG": {"en": "Strong", "ar": "قوي"},
    "MODERATE": {"en": "Moderate", "ar": "متوسط"},
    "WEAK": {"en": "Weak", "ar": "ضعيف"},
    "NONE": {"en": "None", "ar": "لا يوجد"},
    "INSUFFICIENT": {"en": "Insufficient evidence", "ar": "الأدلة غير كافية"},
    "NOT_CURRENTLY_ASSESSED": {"en": "Not currently assessed", "ar": "غير مُقيَّم حالياً"},
}

VERIFICATION_STATUS_LABELS: dict[str, dict[str, str]] = {
    "SUPPORTED": {"en": "Supported", "ar": "مدعوم"},
    "PARTIALLY_SUPPORTED": {"en": "Partially supported", "ar": "مدعوم جزئياً"},
    "UNSUPPORTED": {"en": "Unsupported", "ar": "غير مدعوم"},
    "INSUFFICIENT_EVIDENCE": {"en": "Insufficient evidence", "ar": "الأدلة غير كافية"},
    "NOT_CURRENTLY_ASSESSED": {"en": "Not currently assessed", "ar": "غير مُقيَّم حالياً"},
}

CONSISTENCY_LABELS: dict[str, dict[str, str]] = {
    "CONSISTENT": {"en": "Consistent", "ar": "متسق"},
    "MIXED": {"en": "Mixed / some conflicting signals", "ar": "مختلط / توجد إشارات متعارضة"},
    "CONFLICT": {"en": "Clear conflict", "ar": "تعارض واضح"},
    "INSUFFICIENT_EVIDENCE": {"en": "Insufficient evidence", "ar": "الأدلة غير كافية"},
    "GLOBAL_LOCAL_CONSISTENT": {"en": "Consistent", "ar": "متسق"},
    "GLOBAL_LOCAL_MIXED": {"en": "Mixed / some conflicting signals", "ar": "مختلط / توجد إشارات متعارضة"},
    "GLOBAL_LOCAL_CONFLICT": {"en": "Clear conflict", "ar": "تعارض واضح"},
}

# Every observable_feature value in RULE_EVIDENCE_MATRIX.csv (41, verified
# against the file directly this pass) -- purely descriptive relabeling,
# no interpretation added beyond what the feature name itself already says.
OBSERVATION_LABELS: dict[str, dict[str, str]] = {
    "animal_choice_general": {"en": "Choice of animal in the drawing", "ar": "اختيار نوع الحيوان في الرسمة"},
    "circles": {"en": "Circular shapes", "ar": "أشكال دائرية"},
    "closed_eyes": {"en": "Closed eyes", "ar": "عيون مغلقة"},
    "coverage_about_half": {"en": "About half the page used", "ar": "استخدام حوالي نصف الصفحة"},
    "coverage_full": {"en": "Full page used", "ar": "استخدام الصفحة بالكامل"},
    "coverage_small": {"en": "Small portion of the page used", "ar": "استخدام جزء صغير من الصفحة"},
    "dark_colors_sad_faces_isolation": {
        "en": "Dark colors, sad faces, or isolation theme", "ar": "ألوان داكنة أو وجوه حزينة أو طابع عزلة"},
    "exaggerated_body_parts": {"en": "Exaggerated body parts", "ar": "أجزاء جسدية مبالغ فيها"},
    "excessive_detail_or_fixation": {
        "en": "Excessive detail or fixation", "ar": "تفاصيل مفرطة أو انشغال زائد بعنصر معين"},
    "eyes_missing_or_undetailed": {"en": "Eyes missing or undetailed", "ar": "عيون غائبة أو غير مفصّلة"},
    "face_expression": {"en": "Facial expression", "ar": "تعبير الوجه"},
    "flowers_clouds_sun": {"en": "Flowers, clouds, or sun", "ar": "زهور أو غيوم أو شمس"},
    "fox": {"en": "Fox depicted", "ar": "رسم ثعلب"},
    "hearts": {"en": "Heart shapes", "ar": "أشكال قلوب"},
    "heavy_line_pressure_appearance": {"en": "Heavy/firm line appearance", "ar": "خط قوي/ثقيل الضغط"},
    "house": {"en": "House depicted", "ar": "رسم منزل"},
    "light_line_pressure_appearance": {"en": "Light/faint line appearance", "ar": "خط فاتح/باهت الضغط"},
    "lion": {"en": "Lion depicted", "ar": "رسم أسد"},
    "missing_hands": {"en": "Missing hands", "ar": "غياب اليدين"},
    "missing_mouth": {"en": "Missing mouth", "ar": "غياب الفم"},
    "neglect_of_background": {"en": "Background left mostly empty", "ar": "خلفية شبه خالية"},
    "over_erasing_appearance": {"en": "Heavy erasing appearance", "ar": "آثار محو متكررة"},
    "placement_center": {"en": "Placed in the center", "ar": "موضوع في المنتصف"},
    "placement_left": {"en": "Placed on the left", "ar": "موضوع في الجهة اليسرى"},
    "placement_right": {"en": "Placed on the right", "ar": "موضوع في الجهة اليمنى"},
    "placement_top": {"en": "Placed at the top", "ar": "موضوع في الأعلى"},
    "refusal_to_draw_figures": {"en": "Refusal to draw figures", "ar": "امتناع عن رسم الأشخاص"},
    "repeated_family_conflict": {
        "en": "Repeated family-conflict theme", "ar": "تكرار طابع الخلاف الأسري"},
    "repeated_geometric_shapes": {"en": "Repeated geometric shapes", "ar": "تكرار الأشكال الهندسية"},
    "repeated_isolation_themes": {"en": "Repeated isolation theme", "ar": "تكرار طابع العزلة"},
    "repeated_monsters_danger_injury": {
        "en": "Repeated monsters/danger/injury theme", "ar": "تكرار طابع الوحوش أو الخطر أو الإصابة"},
    "shaky_or_broken_lines": {"en": "Shaky or broken lines", "ar": "خطوط مرتجفة أو متقطعة"},
    "squirrel": {"en": "Squirrel depicted", "ar": "رسم سنجاب"},
    "stars": {"en": "Star shapes", "ar": "أشكال نجوم"},
    "stern_eyes": {"en": "Stern/intense eyes", "ar": "عيون حادة/متوترة النظرة"},
    "tiger_or_wolf": {"en": "Tiger or wolf depicted", "ar": "رسم نمر أو ذئب"},
    "tree": {"en": "Tree depicted", "ar": "رسم شجرة"},
    "very_small_drawing_absolute": {"en": "Very small drawing overall", "ar": "رسمة صغيرة الحجم بشكل عام"},
    "vehicles": {"en": "Vehicles depicted", "ar": "رسم مركبات"},
    "wide_eyes": {"en": "Wide eyes", "ar": "عيون واسعة"},
    "zigzag_lines": {"en": "Zigzag lines", "ar": "خطوط متعرجة (زجزاج)"},
}

# Every evidence_family value in RULE_EVIDENCE_MATRIX.csv (15, verified
# against the file directly this pass).
EVIDENCE_FAMILY_LABELS: dict[str, dict[str, str]] = {
    "animal_symbolism": {"en": "Animal symbolism", "ar": "الرمزية الحيوانية"},
    "colour_mood_flags": {"en": "Colour/mood signals", "ar": "إشارات اللون والمزاج"},
    "composition_omission": {"en": "Composition / omission", "ar": "التكوين والحذف"},
    "detail_fixation": {"en": "Level of detail / fixation", "ar": "درجة التفصيل والانشغال"},
    "facial_feature_style": {"en": "Facial feature style", "ar": "أسلوب رسم ملامح الوجه"},
    "isolation_theme": {"en": "Isolation theme", "ar": "طابع العزلة"},
    "line_fragmentation_quality": {"en": "Line fragmentation quality", "ar": "جودة تقطّع الخط"},
    "line_intensity_quality": {"en": "Line intensity quality", "ar": "جودة كثافة/ضغط الخط"},
    "line_quality": {"en": "Line quality", "ar": "جودة الخط"},
    "missing_body_part": {"en": "Missing body part", "ar": "غياب جزء من الجسد"},
    "object_symbolism": {"en": "Object symbolism", "ar": "الرمزية الشيئية"},
    "shape_symbolism": {"en": "Shape symbolism", "ar": "الرمزية الشكلية"},
    "size_composition": {"en": "Size / composition", "ar": "الحجم والتكوين"},
    "spatial_placement": {"en": "Spatial placement", "ar": "الموضع المكاني"},
    "task_engagement": {"en": "Task engagement", "ar": "مستوى الانخراط في المهمة"},
}

EMOTION_LABELS: dict[str, dict[str, str]] = {
    "Happy": {"en": "Happy", "ar": "سعيد"},
    "Sad": {"en": "Sad", "ar": "حزين"},
    "Fear": {"en": "Fear", "ar": "خائف"},
    "Angry": {"en": "Angry", "ar": "غاضب"},
}


def domain_label(domain_id: str, language: Language = "en") -> str:
    entry = DOMAIN_LABELS.get(domain_id)
    return _pick(entry, language) if entry else domain_id.replace("_", " ")


def support_level_label(level: str, language: Language = "en") -> str:
    entry = SUPPORT_LEVEL_LABELS.get(level)
    return _pick(entry, language) if entry else level.replace("_", " ")


def verification_status_label(status: str, language: Language = "en") -> str:
    entry = VERIFICATION_STATUS_LABELS.get(status)
    return _pick(entry, language) if entry else status.replace("_", " ")


def consistency_label(status: str, language: Language = "en") -> str:
    entry = CONSISTENCY_LABELS.get(status)
    return _pick(entry, language) if entry else status.replace("_", " ")


def observation_label(feature_id: str, language: Language = "en") -> str:
    entry = OBSERVATION_LABELS.get(feature_id)
    return _pick(entry, language) if entry else feature_id.replace("_", " ")


def evidence_family_label(family_id: str, language: Language = "en") -> str:
    entry = EVIDENCE_FAMILY_LABELS.get(family_id)
    return _pick(entry, language) if entry else family_id.replace("_", " ")


# The exact closed set of (label, reason-template) pairs
# `case_interpretation._derive_richer_descriptors` can emit (verified
# against that function directly this pass) -- "negative"'s reason
# embeds the top emotion name dynamically, handled separately below.
DESCRIPTOR_LABELS: dict[str, dict[str, str]] = {
    "cheerful": {"en": "cheerful", "ar": "مبهج"},
    "positive": {"en": "positive", "ar": "إيجابي"},
    "negative": {"en": "negative", "ar": "سلبي"},
    "tense": {"en": "tense", "ar": "متوتر"},
    "withdrawn-looking": {"en": "withdrawn-looking", "ar": "يبدو منسحباً"},
}

_DESCRIPTOR_REASON_AR: dict[str, str] = {
    "Happy expressive classification, together with positive-affect-related visual evidence.":
        "تصنيف تعبيري سعيد، إلى جانب أدلة بصرية مرتبطة بالتأثير الإيجابي.",
    "The expressive classifier's top class is Happy.":
        "الفئة الأعلى في المصنِّف التعبيري هي سعيد.",
    "Line-pressure appearance associated in the literature with tension/energy was observed.":
        "تمت ملاحظة مظهر ضغط الخط المرتبط في الأدبيات بالتوتر/الطاقة.",
    "Placement/composition evidence associated with social withdrawal was observed.":
        "تمت ملاحظة أدلة تتعلق بالموضع أو التكوين مرتبطة بالانسحاب الاجتماعي.",
}


def descriptor_label(label: str, language: Language = "en") -> str:
    entry = DESCRIPTOR_LABELS.get(label)
    return _pick(entry, language) if entry else label


def descriptor_reason(reason: str, language: Language = "en") -> str:
    """Translates one of `_derive_richer_descriptors`'s known reason
    templates. The one dynamic case ("...top class is {Emotion}.", the
    "negative" descriptor -- Sad/Fear/Angry) is pattern-matched and
    rebuilt with the translated emotion name; any reason text outside
    this closed set (only possible if case_interpretation.py adds a new
    descriptor) falls back to the original English rather than a
    mistranslation."""
    if language != "ar":
        return reason
    if reason in _DESCRIPTOR_REASON_AR:
        return _DESCRIPTOR_REASON_AR[reason]
    prefix, suffix = "The expressive classifier's top class is ", "."
    if reason.startswith(prefix) and reason.endswith(suffix):
        emotion = reason[len(prefix):-len(suffix)]
        return f"الفئة الأعلى في المصنِّف التعبيري هي {emotion_label(emotion, 'ar')}."
    return reason


def emotion_label(emotion: str | None, language: Language = "en") -> str:
    if not emotion:
        return ""
    entry = EMOTION_LABELS.get(emotion)
    return _pick(entry, language) if entry else emotion


def concern_domain_reason(concern, language: Language = "en") -> str:
    """Bilingual rephrasing of `ConcernDomainResult.plain_language_reason`
    -- built from the SAME already-computed numbers/ids the English
    version uses (assessable_family_count, independent_evidence_families,
    supporting_observations), never re-derived or re-weighted. Unlike
    `plain_language_reason` itself (which embeds the raw
    `observable_feature` id in parentheses, e.g.
    "light_line_pressure_appearance"), BOTH languages here go through
    `observation_label`/`evidence_family_label` -- item C's "no raw ID as
    the main user-facing label" applies in English too, not only Arabic."""
    ar = language == "ar"
    if concern.assessable_family_count == 0:
        return ("لا يمكن حالياً تقييم أي قاعدة من قواعد DOAR المرتبطة بهذا المجال لهذه الرسمة." if ar else
                "No DOAR rule currently mapped to this domain could be assessed for this drawing.")
    n_support = len(concern.independent_evidence_families)
    if n_support == 0:
        if ar:
            return (f"لم تتحقق أي من أسر الأدلة القابلة للتقييم ({concern.assessable_family_count}) "
                     "لهذا المجال في هذه الرسمة.")
        fam_word = "family" if concern.assessable_family_count == 1 else "families"
        return (f"None of the {concern.assessable_family_count} assessable evidence {fam_word} for this "
                "domain were satisfied for this drawing.")
    families_txt = ", ".join(evidence_family_label(f, language) for f in concern.independent_evidence_families)
    observations_txt = ", ".join(observation_label(o, language) for o in concern.supporting_observations)
    if ar:
        obs_suffix = f" ({observations_txt})" if observations_txt else ""
        return (f"دعمت {n_support} من أصل {concern.assessable_family_count} أسرة أدلة مستقلة قابلة للتقييم "
                f"هذا المجال: {families_txt}{obs_suffix}.")
    fam_word = "family" if concern.assessable_family_count == 1 else "families"
    obs_suffix = f" ({observations_txt})" if observations_txt else ""
    return (f"{n_support} of {concern.assessable_family_count} assessable independent evidence {fam_word} "
            f"support this domain: {families_txt}{obs_suffix}.")


# structured_report.py's `_suggested_parent_questions` draws from
# RULE_EVIDENCE_MATRIX.csv's `clinical_question_to_ask_child_or_caregiver`
# column -- a closed set of 35 English strings (verified against the file
# directly this pass): 4 fixed questions, plus 31 that are ALL the exact
# template "Would you be willing to ask your child about the {feature} in
# this drawing?" where {feature} is one `observable_feature` id with
# underscores replaced by spaces. Both are translated here at the
# presentation layer only -- the CSV itself is never read or modified.
_FIXED_PARENT_QUESTIONS_AR: dict[str, str] = {
    "Can you tell me what is happening in this picture?":
        "هل يمكنك أن تخبرني ماذا يحدث في هذه الصورة؟",
    "What did your child use to draw this picture?":
        "بماذا رسم طفلك هذه الصورة؟",
    "What made your child choose to draw here on the page?":
        "ما الذي جعل طفلك يختار الرسم في هذا المكان من الصفحة؟",
    "What made your child decide how much of the page to use for this drawing?":
        "ما الذي جعل طفلك يقرر مقدار مساحة الصفحة التي استخدمها لهذه الرسمة؟",
}
_TEMPLATED_QUESTION_RE = re.compile(
    r"^Would you be willing to ask your child about the (.+) in this drawing\?$")


def parent_question_label(question: str, language: Language = "en") -> str:
    """Translates one `suggested_parent_questions` entry. Falls back to
    the original English for any question outside the closed set above
    (only possible if RULE_EVIDENCE_MATRIX.csv changes) -- never a
    mistranslation."""
    if language != "ar":
        return question
    if question in _FIXED_PARENT_QUESTIONS_AR:
        return _FIXED_PARENT_QUESTIONS_AR[question]
    m = _TEMPLATED_QUESTION_RE.match(question)
    if m:
        feature_id = m.group(1).replace(" ", "_")
        if feature_id in OBSERVATION_LABELS:
            feature_ar = observation_label(feature_id, "ar")
            return f"هل توافق/ين على سؤال طفلك عن \"{feature_ar}\" في هذه الرسمة؟"
    return question


def synthesis_summary(interp, language: Language = "en") -> str:
    """Item K: a synthesis line that explicitly names the broad expressive
    profile, the most important (WEAK+) concern domains with their
    evidence-family counts, the Gemini whole-image note (if the case has
    one), and the consistency/conflict status -- built entirely from
    fields `CaseInterpretation` already computed (never re-derives a
    support level or re-runs any rule/aggregation). Replaces raw display
    of `interp.final_synthesis` at the UI layer only; `final_synthesis`
    itself (case_interpretation.py) is untouched."""
    profile = interp.expressive_profile
    notable = sorted(
        (c for c in interp.concern_domains if c.support_level in ("WEAK", "MODERATE", "STRONG")),
        key=lambda c: ("STRONG", "MODERATE", "WEAK").index(c.support_level))
    ar = language == "ar"

    if profile.availability == "available" and profile.top_emotion:
        top_prob = (profile.base_emotion_probabilities or {}).get(profile.top_emotion)
        pct = f" ({top_prob:.0%})" if top_prob is not None else ""
        emotion_txt = emotion_label(profile.top_emotion, language)
        expressive_sentence = (
            f"الملف التعبيري العام للرسمة يميل بشكل أساسي إلى: {emotion_txt}{pct}."
            if ar else f"The drawing's broad expressive profile is predominantly {emotion_txt}{pct}.")
    else:
        expressive_sentence = (
            "لا يتوفر نموذج تعبيري لهذه الحالة." if ar else
            "No expressive-model result is available for this case.")

    if notable:
        domain_bits = []
        for c in notable:
            fam_count = len(c.independent_evidence_families)
            level_txt = support_level_label(c.support_level, language)
            if ar:
                domain_bits.append(f"{domain_label(c.domain, 'ar')} ({level_txt}، "
                                    f"{fam_count} أسرة أدلة مستقلة)" if fam_count != 1 else
                                    f"{domain_label(c.domain, 'ar')} ({level_txt}، أسرة أدلة مستقلة واحدة)")
            else:
                fam_word = "independent evidence family" if fam_count == 1 else "independent evidence families"
                domain_bits.append(f"{domain_label(c.domain, 'en')} ({level_txt}, {fam_count} {fam_word})")
        domains_txt = "، ".join(domain_bits) if ar else "; ".join(domain_bits)
        concern_sentence = (
            f"كما رصد DOAR المؤشرات التالية: {domains_txt}." if ar else
            f"DOAR also identified: {domains_txt}.")
    else:
        concern_sentence = (
            "لم يصل أي مجال قلق إلى مستوى دعم يستحق الذكر لهذه الحالة." if ar else
            "No concern domain reached a support level worth mentioning for this case.")

    gemini_sentence = ""
    if interp.gemini_global_observation:
        tone = interp.gemini_global_observation.get("overall_visual_tone")
        if tone:
            gemini_sentence = ((f" رأي Gemini الشامل للصورة وصف النبرة العامة بأنها: {tone}.") if ar else
                                (f" Gemini's whole-image opinion described the overall tone as: {tone}."))

    consistency_txt = consistency_label(interp.consistency_status, language)
    if interp.consistency_status in ("CONFLICT", "MIXED"):
        limitation_sentence = (
            f" نظراً لأن هذه المؤشرات تعتمد على أدلة محدودة وتتعارض جزئياً مع الانطباع التعبيري العام "
            f"({consistency_txt})، فيجب اعتبارها ملاحظات ثانوية وليست استنتاجات تشخيصية."
            if ar else
            f" Because these concerns rely on limited evidence and are {consistency_txt.lower()} with the "
            f"broad expressive profile, they should be treated as secondary observations rather than "
            f"diagnostic conclusions.")
    else:
        limitation_sentence = (
            f" هذه الملاحظات {consistency_txt} مع الانطباع التعبيري العام، لكنها تبقى أنماطاً رصدية وليست "
            "استنتاجات تشخيصية."
            if ar else
            f" These observations are {consistency_txt.lower()} with the broad expressive profile, but "
            "remain observational patterns, not diagnostic conclusions.")

    return expressive_sentence + " " + concern_sentence + gemini_sentence + limitation_sentence
