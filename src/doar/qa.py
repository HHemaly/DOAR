"""
qa.py — deterministic, evidence-grounded question answering (Item 13).

Answers ONLY from a saved case's analysis + judges. Covers quality, segmentation,
composition, colours, emotion probabilities, confidence, uncertainty, evidence
IDs, rule IDs, missing detectors, concerns, review history, and limitations, in
English and Arabic. Every answer returns a standard envelope:
  answer, evidence_ids, source_module, availability, limitations,
  non_diagnostic_warning (+ backward-compatible `source`).
Never invents evidence; unknown questions return a safe "unavailable" answer.
"""

from __future__ import annotations

_NON_DIAG = {
    "en": "This is observational evidence only and is not a diagnosis.",
    "ar": "هذه أدلة رصدية فقط وليست تشخيصاً.",
}


def _env(answer_en, answer_ar, *, language, evidence_ids, source_module,
         availability="available", limitations=None, non_diagnostic=False):
    ar = language == "ar"
    return {
        "answer": answer_ar if ar else answer_en,
        "evidence_ids": evidence_ids,
        "source_module": source_module,
        "source": source_module,                       # backward-compatible
        "availability": availability,
        "limitations": limitations or [],
        "non_diagnostic_warning": (_NON_DIAG["ar"] if ar else _NON_DIAG["en"]) if non_diagnostic else None,
    }


def _match(q, *words):
    return any(w in q for w in words)


def answer(question: str, analysis: dict, judges: dict | None = None,
           language: str = "en") -> dict:
    q = question.casefold()
    judges = judges or {}

    # ── Quality ──────────────────────────────────────────────────────────────
    if _match(q, "quality", "blur", "resolution", "جودة", "دقة", "وضوح"):
        qy = analysis.get("quality", {})
        status = qy.get("quality_status", "supported" if qy.get("supported") else "unsupported")
        reasons = "; ".join(qy.get("unsupported_reasons", [])) or "none"
        return _env(
            f"Image quality is '{status}'. Reasons: {reasons}.",
            f"جودة الصورة: '{status}'. الأسباب: {reasons}.",
            language=language, evidence_ids=[], source_module="quality_gate",
            availability=status,
            limitations=["Quality thresholds are not clinically validated on the real dataset."])

    # ── Segmentation ─────────────────────────────────────────────────────────
    if _match(q, "segment", "coverage", "background", "تقسيم", "خلفية", "تغطية"):
        seg = analysis.get("segmentation", {})
        comp = analysis.get("composition", {})
        return _env(
            f"Segmentation status '{seg.get('status')}' (confidence {seg.get('confidence')}). "
            f"Foreground coverage {comp.get('foreground_coverage')}.",
            f"حالة التقسيم '{seg.get('status')}' (الثقة {seg.get('confidence')}). "
            f"نسبة التغطية {comp.get('foreground_coverage')}.",
            language=language, evidence_ids=["ev_seg_coverage", "ev_bbox_coverage"],
            source_module="segmentation")

    # ── Composition ──────────────────────────────────────────────────────────
    if _match(q, "composition", "placement", "position", "size", "empty",
              "تكوين", "موضع", "حجم", "فراغ", "مكان"):
        comp = analysis.get("composition", {})
        return _env(
            f"Placement '{comp.get('placement')}', bounding-box coverage "
            f"{comp.get('bounding_box_coverage')}, empty space {comp.get('empty_space_ratio')}.",
            f"الموضع '{comp.get('placement')}', تغطية الإطار {comp.get('bounding_box_coverage')}, "
            f"المساحة الفارغة {comp.get('empty_space_ratio')}.",
            language=language, evidence_ids=["ev_bbox_coverage", "ev_centroid"],
            source_module="composition")

    # ── Colours ──────────────────────────────────────────────────────────────
    if _match(q, "colour", "color", "لون", "ألوان"):
        colours = analysis.get("colour", {}).get("meaningful_colours", [])
        return _env(
            "Detected foreground colours: " + (", ".join(colours) or "none"),
            "الألوان المرصودة: " + (", ".join(colours) or "لا يوجد"),
            language=language, evidence_ids=["ev_dominant_colour"], source_module="colour",
            limitations=["Colour naming is coarse and background-dependent."])

    # ── Emotion probabilities ────────────────────────────────────────────────
    if _match(q, "emotion", "predict", "probab", "feel", "انفعال", "مشاعر", "احتمال", "تنبؤ"):
        em = analysis.get("emotion", {})
        status = em.get("status", "unavailable")
        if status != "available":
            return _env(
                f"No emotion prediction available (status: {status}).",
                f"لا يوجد تنبؤ انفعالي (الحالة: {status}).",
                language=language, evidence_ids=[], source_module="emotion_model",
                availability=status,
                limitations=["Model probabilities are not psychological confidence."],
                non_diagnostic=True)
        return _env(
            f"Predicted '{em.get('top_class')}' (confidence {em.get('confidence')}). "
            f"Probabilities: {em.get('probabilities', {})}.",
            f"التنبؤ '{em.get('top_class')}' (الثقة {em.get('confidence')}). "
            f"الاحتمالات: {em.get('probabilities', {})}.",
            language=language, evidence_ids=["ev_emotion_prediction"],
            source_module="emotion_model",
            limitations=["Model probabilities are not psychological or diagnostic confidence."],
            non_diagnostic=True)

    # ── Confidence / uncertainty ─────────────────────────────────────────────
    if _match(q, "confidence", "uncertain", "entropy", "margin", "ثقة", "شك", "يقين"):
        em = analysis.get("emotion", {})
        return _env(
            f"Confidence {em.get('confidence')}, uncertainty '{em.get('uncertainty')}', "
            f"top-two margin {em.get('top_two_margin')}, entropy {em.get('entropy')}, "
            f"calibration {em.get('calibration_status')}.",
            f"الثقة {em.get('confidence')}، عدم اليقين '{em.get('uncertainty')}'، "
            f"الفارق {em.get('top_two_margin')}، الإنتروبيا {em.get('entropy')}، "
            f"المعايرة {em.get('calibration_status')}.",
            language=language, evidence_ids=["ev_emotion_prediction"],
            source_module="uncertainty", availability=em.get("status", "unavailable"),
            non_diagnostic=True)

    # ── Object/person detectors (do not exist) ───────────────────────────────
    if _match(q, "person", "people", "face", "object", "animal", "tree", "house",
              "شخص", "إنسان", "وجه", "حيوان", "شجرة"):
        return _env(
            "Unavailable: a verified object/person detector was not run in this release.",
            "غير متاح: لم يُشغّل كاشف موثوق للأشخاص/الأشياء في هذا الإصدار.",
            language=language, evidence_ids=[], source_module="module_availability",
            availability="missing_detector")

    # ── Missing detectors ────────────────────────────────────────────────────
    if _match(q, "detector", "missing", "detect", "كاشف", "اكتشاف"):
        missing = sorted({m for r in analysis.get("rule_evaluations", [])
                          if r["status"] == "missing_detector" for m in r.get("missing_evidence", [])})
        return _env(
            "No detectors exist for: " + (", ".join(missing) or "none") +
            ". Missing evidence is not treated as negative evidence.",
            "لا توجد كواشف لـ: " + (", ".join(missing) or "لا شيء") +
            ". غياب الدليل لا يُعامل كدليل سلبي.",
            language=language, evidence_ids=[], source_module="module_availability",
            availability="missing_detector")

    # ── Rule IDs ─────────────────────────────────────────────────────────────
    if _match(q, "rule", "قاعدة", "قواعد"):
        evals = analysis.get("rule_evaluations", [])
        by_status: dict[str, list[str]] = {}
        for r in evals:
            by_status.setdefault(r["status"], []).append(r["rule_id"])
        summary = "; ".join(f"{k}: {', '.join(v)}" for k, v in by_status.items())
        return _env(
            "Rule statuses: " + summary,
            "حالات القواعد: " + summary,
            language=language,
            evidence_ids=sorted({e for r in evals for e in r.get("matched_evidence_ids", [])}),
            source_module="rule_evaluations",
            limitations=["Psychologist-supplied rules are unvalidated hypotheses."],
            non_diagnostic=True)

    # ── Evidence IDs ─────────────────────────────────────────────────────────
    if _match(q, "evidence", "دليل", "أدلة"):
        ids = [e.get("evidence_id") for e in analysis.get("evidence", [])]
        return _env(
            "Evidence IDs: " + (", ".join(ids) or "none"),
            "معرّفات الأدلة: " + (", ".join(ids) or "لا يوجد"),
            language=language, evidence_ids=ids, source_module="evidence")

    # ── Concerns ─────────────────────────────────────────────────────────────
    if _match(q, "concern", "risk", "worry", "قلق", "مخاوف", "خطر"):
        concerns = analysis.get("concerns", [])
        if not concerns:
            return _env(
                "No concern profiles are active. Concerns require converging evidence "
                "from independent detectors, which are not available.",
                "لا توجد ملفات قلق نشطة. تتطلب المخاوف أدلة متقاربة من كواشف مستقلة غير متوفرة.",
                language=language, evidence_ids=[], source_module="concerns",
                availability="disabled", non_diagnostic=True)
        return _env(
            f"{len(concerns)} concern profile(s), all requiring clinician review.",
            f"{len(concerns)} ملف قلق، جميعها تتطلب مراجعة مختص.",
            language=language,
            evidence_ids=sorted({e for c in concerns for e in c.get("supporting_evidence", [])}),
            source_module="concerns", non_diagnostic=True)

    # ── Review history ───────────────────────────────────────────────────────
    if _match(q, "review", "reviewer", "clinician", "مراجعة", "مختص"):
        avail = judges.get("module_availability", {}).get("clinician_review", "not_submitted")
        return _env(
            f"Clinician review status: {avail}.",
            f"حالة مراجعة المختص: {avail}.",
            language=language, evidence_ids=[], source_module="clinician_review",
            availability=avail)

    # ── Limitations ──────────────────────────────────────────────────────────
    if _match(q, "limitation", "caveat", "reliab", "حدود", "قيود", "موثوق"):
        return _env(
            "Drawing-based indicators are weak, non-diagnostic and context-dependent; "
            "quality thresholds are not clinically validated; detectors for symbols "
            "do not exist; model probabilities are not psychological confidence.",
            "مؤشرات الرسم ضعيفة وغير تشخيصية وتعتمد على السياق؛ عتبات الجودة غير مُتحقق منها "
            "سريرياً؛ لا توجد كواشف للرموز؛ احتمالات النموذج ليست ثقة نفسية.",
            language=language, evidence_ids=[], source_module="limitations",
            non_diagnostic=True)

    # ── Default: grounded refusal ────────────────────────────────────────────
    return _env(
        "The requested information is unavailable in the saved evidence.",
        "المعلومة غير متاحة في الأدلة المحفوظة.",
        language=language, evidence_ids=[], source_module="saved_evidence_only",
        availability="unavailable")
