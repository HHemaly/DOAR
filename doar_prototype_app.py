"""
DOAR dual-view prototype -- Parent/User view + Technical/Research view.

Built per CURRENT_CAPABILITY_AUDIT.md, END_TO_END_INFERENCE_TRACE.md,
RULE_AND_FEATURE_COVERAGE.md, and TARGET_APPLICATION_ARCHITECTURE.md: every
value shown here comes from a real execution of the existing pipeline
(analyze_image / features.objective_feature_row / qa.answer / chat.py).
Nothing is fabricated to make the interface look more complete than the
pipeline actually is -- absent capabilities (object detection, disabled
concern profiles, unevaluable rules) are shown as explicit, honest
messages, never silently omitted or invented.

This is SEPARATE from phase7b_review_app.py (Phase 7B's duplicate-pair
review tool) and from streamlit_app.py (the existing raw-JSON researcher
tool) -- it does not replace or import either.

Launch:
    .\\.venv\\Scripts\\Activate.ps1
    python -m streamlit run doar_prototype_app.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from doar.chat import respond_to_chat  # noqa: E402
from doar.parent_view import (  # noqa: E402
    disclaimer, overall_interpretation, plain_language_observations, plain_language_rule_rows,
)
from doar.profile import ChildProfile, ALLOWED_AGE_RANGES, load_profile, save_profile  # noqa: E402
from doar.timed_analysis import analyze_image_with_timing  # noqa: E402

CASES_DIR = ROOT / "outputs" / "prototype_cases"
KNOWN_CHECKPOINTS = {
    "efficientnet_b0 (Phase 5, seed 42, calibrated) -- recommended": str(
        ROOT / "outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt"),
    "resnet18 (Phase 5, seed 42, calibrated)": str(
        ROOT / "outputs/phase5/seed42_reference/resnet18_seed_42/best.pt"),
    "mobilenet_v3_small (Phase 5, seed 42, calibrated)": str(
        ROOT / "outputs/phase5/seed42_reference/mobilenet_v3_small_seed_42/best.pt"),
    "No emotion model (segmentation/composition/rules only)": None,
}

st.set_page_config(page_title="DOAR prototype - dual view", layout="wide")
st.title("DOAR v3 -- Dual-View Prototype")
st.warning(
    "Research prototype, non-diagnostic. Every checkpoint currently available was "
    "trained on a duplicate-contaminated, leakage-gate-overridden split -- see "
    "CURRENT_CAPABILITY_AUDIT.md Section 3. Treat all output as preliminary."
)

# ---------------------------------------------------------------------------
# Sidebar: upload + optional child context + checkpoint choice, or reopen.
# ---------------------------------------------------------------------------
st.session_state.setdefault("case_dir", "")

with st.sidebar:
    st.header("New analysis")
    uploaded = st.file_uploader("Upload a child's drawing", type=["png", "jpg", "jpeg"])
    st.caption("Optional context (never used by the model or rules -- display only)")
    age_range = st.selectbox("Age range", ALLOWED_AGE_RANGES, index=len(ALLOWED_AGE_RANGES) - 1)
    gender = st.text_input("Gender (optional)")
    instruction = st.text_input("Drawing instruction/prompt given to the child (optional)")
    concern = st.text_area("Parent's concern or question (optional)")
    checkpoint_choice = st.selectbox("Emotion model", list(KNOWN_CHECKPOINTS.keys()))
    custom_checkpoint = st.text_input("...or a custom checkpoint path (overrides the choice above)")

    if st.button("Analyze", type="primary", disabled=uploaded is None):
        case_name = f"{Path(uploaded.name).stem}_{int(time.time())}"
        case_dir = CASES_DIR / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        image_path = case_dir / uploaded.name
        image_path.write_bytes(uploaded.getvalue())

        checkpoint = custom_checkpoint.strip() or KNOWN_CHECKPOINTS[checkpoint_choice]
        with st.spinner("Running the real analysis pipeline..."):
            analyze_image_with_timing(str(image_path), str(case_dir), checkpoint)
            profile = ChildProfile(
                age_range=age_range, gender=gender or None,
                drawing_instruction=instruction or None,
                date=date.today().isoformat(), parent_concern=concern or None,
            )
            save_profile(case_dir, profile)
        st.session_state["case_dir"] = str(case_dir.resolve())
        st.rerun()

    st.divider()
    previous = sorted((p.name for p in CASES_DIR.iterdir() if p.is_dir()), reverse=True) if CASES_DIR.exists() else []
    if previous:
        picked = st.selectbox("Or reopen a previous case", ["(none)"] + previous)
        if picked != "(none)":
            st.session_state["case_dir"] = str((CASES_DIR / picked).resolve())

case_dir_text = st.sidebar.text_input("Case folder", key="case_dir")
case_dir = Path(case_dir_text) if case_dir_text else None

if not case_dir or not (case_dir / "analysis.json").exists():
    st.info("Upload a drawing in the sidebar and click **Analyze**, or select/enter a case folder.")
    st.stop()

analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
judges = json.loads((case_dir / "judges.json").read_text(encoding="utf-8"))
detections = json.loads((case_dir / "detections.json").read_text(encoding="utf-8"))
timing_path = case_dir / "timing.json"
timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else None
profile = load_profile(case_dir)
# DOAR-TRACE 4E: structured_analysis.json (4D) is written by every
# analyze_image run since 4B/4D landed -- older cases from before that
# change won't have it, so this stays optional rather than a hard error.
structured_path = case_dir / "structured_analysis.json"
structured = json.loads(structured_path.read_text(encoding="utf-8")) if structured_path.exists() else None
objective_features_path = case_dir / "objective_features.json"
objective_features_doc = (
    json.loads(objective_features_path.read_text(encoding="utf-8")) if objective_features_path.exists() else None
)


def artifact(name: str) -> str | None:
    value = analysis.get("artifacts", {}).get(name)
    if not value:
        return None
    p = case_dir / value if not Path(value).is_absolute() else Path(value)
    return str(p) if p.exists() else None


parent_tab, technical_tab = st.tabs(["Parent / User View", "Technical / Research View"])

# ===========================================================================
# PARENT / USER VIEW
# ===========================================================================
with parent_tab:
    language = st.radio("Language / اللغة", ["en", "ar"], horizontal=True, key="parent_lang")
    ar = language == "ar"
    direction = "rtl" if ar else "ltr"
    st.markdown(f'<div dir="{direction}">', unsafe_allow_html=True)

    st.subheader("الرسمة المرفوعة" if ar else "Uploaded drawing")
    norm = artifact("normalized_image")
    if norm:
        st.image(norm, width=320)

    st.subheader("سياق الطفل" if ar else "Child context (parent-provided)")
    if profile:
        cols = st.columns(4)
        cols[0].metric("الفئة العمرية" if ar else "Age range", profile.age_range)
        cols[1].metric("الجنس" if ar else "Gender", profile.gender or ("غير محدد" if ar else "not specified"))
        cols[2].metric("التاريخ" if ar else "Date", profile.date or "-")
        cols[3].metric("التعليمات" if ar else "Instruction", profile.drawing_instruction or ("لا يوجد" if ar else "none"))
        if profile.parent_concern:
            st.caption(("مصدر القلق: " if ar else "Parent's concern: ") + profile.parent_concern)
        st.caption("هذه المعلومات للعرض فقط ولا تُستخدم في النموذج أو القواعد."
                   if ar else "This context is for display only -- it is never used by the model or the rules.")
    else:
        st.caption("لم تُقدَّم معلومات سياقية." if ar else "No context information was provided.")

    st.subheader("ملاحظات مرصودة وقابلة للقياس" if ar else "Visible, measurable observations")
    for obs in plain_language_observations(analysis, language):
        st.write("- " + obs["text"])

    st.subheader("الأجسام المكتشفة" if ar else "Detected objects")
    if detections.get("status") == "unavailable":
        st.info("كاشف الأجسام غير مُطبَّق في هذا الإصدار — لم تُفحص الرسمة لمحتواها."
               if ar else "Detector not implemented in this release -- the drawing's content was not analyzed for objects.")
    else:
        st.json(detections)

    st.subheader("السمات بلغة مبسطة" if ar else "Features, in plain language")
    st.write(
        ("تغطية المحتوى: " if ar else "Content coverage: ")
        + f"{analysis['composition']['foreground_coverage']:.0%}"
    )
    st.write(
        ("مستوى تفاصيل الخطوط: " if ar else "Line-detail level: ")
        + ("غير متاح — يتطلب حساباً إضافياً غير مُفعَّل في هذا العرض" if ar
           else "not computed in this view (see Technical view for the full 59-feature table)")
    )
    st.caption("سمتان (عدد الأشكال المغلقة وتكرارها) غير متاحتين لعدم وجود كاشف أشكال."
               if ar else "Two features (enclosed-shape count, shape repetition) are unavailable -- no shape detector exists yet.")

    st.subheader("فرضيات محتملة على مستوى الرسمة" if ar else "Possible drawing-level themes")
    st.caption(
        "لماذا اقتُرحت كل فرضية: القواعد الداعمة والأدلة والمراجع. هذه ليست تشخيصاً."
        if ar else
        "Why each theme was suggested: the supporting rules, evidence, and references behind it. This is not a diagnosis."
    )
    if structured and structured.get("candidate_drawing_level_themes"):
        for theme in structured["candidate_drawing_level_themes"]:
            with st.expander(f"{theme['target_construct']} ({theme['allowed_output_level']})"):
                st.write(("القواعد الداعمة: " if ar else "Supporting rules: ") + ", ".join(theme["supporting_rule_ids"]))
                st.write(("الأدلة: " if ar else "Evidence: ") + ", ".join(theme["supporting_evidence_ids"]))
                st.write(("المراجع: " if ar else "References: ") + (", ".join(theme["references"]) or ("لا يوجد" if ar else "none")))
                for lim in theme["limitations"]:
                    st.caption(("قيد: " if ar else "Limitation: ") + lim)
                for alt in theme["alternative_explanations"]:
                    st.caption(("تفسير بديل: " if ar else "Alternative explanation: ") + alt)
                if theme["allowed_output_level"] == "combined_hypothesis_only":
                    st.warning(
                        "فرضية مُجمَّعة من عدة أدلة مستقلة — تتطلب مراجعة مختص."
                        if ar else "A combined hypothesis from multiple independent pieces of evidence -- requires clinician review.")
        if structured.get("cross_theme_contradictions"):
            st.warning(("تم رصد فرضيات متعارضة ولم يُختر فائز تلقائياً:" if ar
                       else "Conflicting themes were detected; no automatic winner was chosen:"))
            for c in structured["cross_theme_contradictions"]:
                st.write(f"- {c['construct_a']} ↔ {c['construct_b']}")
    else:
        st.info("لا توجد فرضيات مرشحة على مستوى الرسمة لهذه الحالة." if ar
               else "No candidate drawing-level themes for this case.")

    st.subheader("كل السمات (٥٩ سمة، قابلة للتوسيع)" if ar else "All features (59, expandable)")
    if objective_features_doc:
        with st.expander(("عرض كل السمات الموضوعية" if ar else "Show all objective features")
                         + f" ({objective_features_doc['feature_count']}, "
                         + f"{objective_features_doc['missing_count']} {'غير متاحة' if ar else 'unavailable'})"):
            st.dataframe(objective_features_doc["features"], use_container_width=True)
    else:
        st.caption("سمات موضوعية غير متاحة لهذه الحالة (تحليل سابق للتحديث)." if ar
                  else "Objective features not available for this case (analyzed before this update).")

    st.subheader("القواعد المُقيَّمة" if ar else "Rules evaluated")
    st.caption("كل قاعدة هي فرضية غير مُتحقق منها علمياً، وليست حقيقة مؤكدة."
              if ar else "Every rule is a scientifically-unvalidated hypothesis, never a confirmed fact.")
    for row in plain_language_rule_rows(analysis["rule_evaluations"], language):
        badge = {"weak_support": "OBSERVED", "not_matched": "NOT OBSERVED",
                 "missing_detector": "NOT EVALUATED", "not_assessable_context_unknown": "NOT EVALUABLE"}.get(row["status"], row["status"])
        with st.expander(f"[{badge}] {row['label']}"):
            st.write(row["message"])
            if row["note"]:
                st.write(("الصياغة الآمنة: " if ar else "Parent-safe wording: ") + row["note"])
                st.caption(("سقف الثقة: " if ar else "Confidence ceiling: ") + f"{row['confidence_ceiling']}")

    st.subheader("نتيجة النموذج" if ar else "Model output")
    emotion = analysis["emotion"]
    if emotion["status"] == "available":
        st.bar_chart(emotion["probabilities"])
        st.write(f"{'الأعلى احتمالاً' if ar else 'Most likely'}: **{emotion['top_class']}** "
                 f"({emotion['confidence']:.0%}, {'معايرة' if ar else 'calibration'}: {emotion['calibration_status']})")
    else:
        st.info(f"{'الحالة' if ar else 'Status'}: {emotion['status']} -- {emotion.get('reason', '')}")

    st.subheader("التفسير العام (نص مُركَّب آلياً وليس توليداً ذكياً)" if ar
                 else "Overall interpretation (template-composed, not AI-generated)")
    st.write(overall_interpretation(analysis, language))

    st.subheader("عدم اليقين والحدود" if ar else "Uncertainty and limitations")
    limitations = sorted({lim for e in analysis["evidence"] for lim in e.get("limitations", [])})
    for lim in limitations:
        st.write("- " + lim)
    st.write(("- " if not ar else "- ") + (
        "لم تُتحقق عتبات الجودة سريرياً على هذه البيانات." if ar
        else "Quality-gate thresholds are engineering defaults, not clinically validated on this dataset."))

    st.subheader("إرشادات آمنة للوالدين" if ar else "Safe parent guidance")
    st.info(disclaimer(language))

    st.divider()
    st.subheader("محادثة متابعة" if ar else "Follow-up chat")
    st.caption(
        "الإجابات مبنية حصراً على أدلة هذه الحالة المحفوظة، بدون نموذج لغوي أو اتصال بالإنترنت."
        if ar else "Answers are grounded strictly in this case's saved evidence -- no LLM, no network call."
    )
    st.session_state.setdefault(f"chat_{case_dir}", [])
    question = st.text_input("سؤال عن هذه الحالة" if ar else "Ask a question about this case", key="parent_chat_q")
    if st.button("إرسال" if ar else "Send", key="parent_chat_send") and question:
        response = respond_to_chat(case_dir, question, language)
        st.session_state[f"chat_{case_dir}"].append((question, response))
    for q, r in reversed(st.session_state[f"chat_{case_dir}"]):
        st.markdown(f"**{'أنت' if ar else 'You'}:** {q}")
        st.markdown(f"**{'الرد' if ar else 'Answer'}:** {r.answer}")
        if r.evidence_ids:
            st.caption(("الأدلة: " if ar else "Evidence: ") + ", ".join(r.evidence_ids))
        if r.non_diagnostic_warning:
            st.caption(r.non_diagnostic_warning)

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# TECHNICAL / RESEARCH VIEW
# ===========================================================================
with technical_tab:
    st.subheader("Input and model metadata")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Schema version", analysis["schema_version"])
    meta_cols[1].metric("Model", analysis["emotion"].get("model_name") or "n/a")
    meta_cols[2].metric("Calibration", analysis["emotion"].get("calibration_status") or "n/a")
    meta_cols[3].metric("Processing time (s)", timing["processing_seconds"] if timing else "not recorded")
    st.code(f"checkpoint_sha256: {analysis['emotion'].get('checkpoint_sha256')}\n"
            f"preprocessing_version: {analysis['emotion'].get('preprocessing_version')}\n"
            f"image_path (original): {analysis['image_path']}")

    st.subheader("Image preview and overlays")
    cols = st.columns(4)
    for i, name in enumerate(["normalized_image", "foreground_mask", "feature_overlay", "stroke_map"]):
        path = artifact(name)
        if path:
            cols[i % 4].image(path, caption=name)

    st.subheader("Complete class probabilities")
    if analysis["emotion"]["status"] == "available":
        st.dataframe({
            "class": list(analysis["emotion"]["probabilities"].keys()),
            "calibrated_probability": list(analysis["emotion"]["probabilities"].values()),
            "raw_probability": list(analysis["emotion"].get("raw_probabilities", {}).values() or [None] * 4),
        })
    else:
        st.info(f"No probabilities -- emotion status: {analysis['emotion']['status']}")

    st.subheader("Object detections")
    st.json(detections)
    st.caption("detections.json is a hardcoded, unconditional stub for every case -- "
               "see CURRENT_CAPABILITY_AUDIT.md Section 7. \"Detector not implemented\", not \"no objects found\".")

    st.subheader("Raw and normalized objective features")
    try:
        from doar.features import objective_feature_row, serialize_feature_row
        original_image = case_dir / Path(analysis["image_path"]).name
        feature_row = objective_feature_row(str(original_image), analysis)
        serialized = serialize_feature_row(feature_row)
        st.dataframe([
            {"feature": name, "value": v["value"], "confidence": v["confidence"],
             "missing": v["missing"], "method": v["method"]}
            for name, v in serialized.items()
        ], use_container_width=True)
        n_missing = sum(1 for v in serialized.values() if v["missing"])
        st.caption(f"{len(serialized)} features computed; {n_missing} marked unavailable "
                   "(no fabricated values for missing detectors).")
    except Exception as exc:  # pragma: no cover -- UI guard, real error still surfaced
        st.error(f"Feature computation failed: {exc}")

    st.subheader("Rule coverage table")
    st.dataframe(analysis["rule_evaluations"], use_container_width=True)

    st.subheader("Dependency grouping and aggregation calculation (DOAR-TRACE 4D)")
    if structured:
        st.caption(
            "candidate_drawing_level_themes: rules grouped by target_construct, deduplicated "
            "by evidence_id, escalated to combined_hypothesis_only only when >=2 distinct "
            "evidence IDs from >=2 distinct evidence families converge (structured_report.py)."
        )
        if structured["candidate_drawing_level_themes"]:
            st.dataframe([
                {
                    "target_construct": t["target_construct"],
                    "supporting_rule_ids": ", ".join(t["supporting_rule_ids"]),
                    "supporting_evidence_ids": ", ".join(t["supporting_evidence_ids"]),
                    "evidence_families": ", ".join(t["evidence_families"]),
                    "n_evidence_ids": len(t["supporting_evidence_ids"]),
                    "n_evidence_families": len(t["evidence_families"]),
                    "allowed_output_level": t["allowed_output_level"],
                }
                for t in structured["candidate_drawing_level_themes"]
            ], use_container_width=True)
        else:
            st.caption("No rule triggered on this image -- no theme to aggregate.")
        if structured["cross_theme_contradictions"]:
            st.dataframe(structured["cross_theme_contradictions"], use_container_width=True)
        st.caption(f"structured_analysis schema_version={structured['schema_version']}, "
                   f"registry_v2 schema_version={structured['registry_v2_schema_version']}")
    else:
        st.info("structured_analysis.json not available for this case (analyzed before this update).")

    st.subheader("Evidence IDs")
    st.dataframe([
        {**e, "value": str(e["value"]), "limitations": "; ".join(e.get("limitations", []))}
        for e in analysis["evidence"]
    ], use_container_width=True)

    st.subheader("Missing-capability warnings")
    warnings = [
        "Object/geometry detection: not implemented (schema-only scaffolding in src/doar/detectors/)",
        "shape.enclosed_shape_count, shape.repetition_score: not implemented (no shape detector)",
        "Concern profiles: implemented but disabled by design (CONCERNS_ENABLED = False)",
        "13 of 19 rules (all tier_2): permanently missing_detector until a real detector exists",
    ]
    for w in warnings:
        st.write("- " + w)

    st.subheader("Reliability and safety warnings (deterministic judges)")
    st.json(judges)

    st.subheader("Case / report version")
    st.write(f"schema_version={analysis['schema_version']}, case_dir={case_dir.name}")

    st.subheader("Downloadable structured evidence and reports")
    st.download_button("analysis.json", (case_dir / "analysis.json").read_bytes(), file_name="analysis.json")
    for report in sorted((case_dir / "reports").glob("*.html")):
        st.download_button(report.name, report.read_bytes(), file_name=report.name)

    st.subheader("LLM-generated claims, evidence references, and judge results")
    st.info("No LLM is enabled in this prototype -- see LLM_GROUNDING_AND_SAFETY_DESIGN.md. "
            "All chat responses below are deterministic evidence lookups (src/doar/qa.py), not generated text.")
    chat_log = st.session_state.get(f"chat_{case_dir}", [])
    if chat_log:
        st.dataframe([
            {"question": q, "answer": r.answer, "evidence_ids": ", ".join(r.evidence_ids),
             "availability": r.availability, "escalated": r.escalated}
            for q, r in chat_log
        ], use_container_width=True)
    else:
        st.caption("No chat turns yet this session.")
