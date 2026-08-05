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
    disclaimer, plain_language_observations, plain_language_rule_rows,
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
# DOAR-TRACE Phase 1.5 Section 8: judges_v2/generated_claims/verification_report
# are written automatically by every analyze_image run since this phase --
# older cases won't have them, so these stay optional too.
judges_v2_path = case_dir / "judges_v2.json"
judges_v2_doc = json.loads(judges_v2_path.read_text(encoding="utf-8")) if judges_v2_path.exists() else None
generated_claims_path = case_dir / "generated_claims.json"
generated_claims_doc = json.loads(generated_claims_path.read_text(encoding="utf-8")) if generated_claims_path.exists() else None
verification_report_path = case_dir / "verification_report.json"
verification_report_doc = (
    json.loads(verification_report_path.read_text(encoding="utf-8")) if verification_report_path.exists() else None
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

    norm = artifact("normalized_image")
    if norm:
        st.image(norm, width=320)
    if profile:
        st.caption(
            (f"السياق: {profile.age_range}، {profile.gender or 'غير محدد'}، {profile.date or '-'} "
             "(للعرض فقط، لا يُستخدم في النموذج أو القواعد)") if ar else
            (f"Context: {profile.age_range}, {profile.gender or 'not specified'}, {profile.date or '-'} "
             "(display only, never used by the model or the rules)")
        )

    combined_hyps = (structured or {}).get("combined_drawing_level_hypotheses", [])
    individual_suggestions = (structured or {}).get("individual_rule_suggestions", [])

    # 1. Drawing-level analysis summary --------------------------------------
    st.subheader("١. ملخص التحليل على مستوى الرسمة" if ar else "1. Drawing-level analysis summary")
    if combined_hyps:
        st.write((f"تم رصد {len(combined_hyps)} فرضية/فرضيات مُجمَّعة من أدلة مستقلة، و{len(individual_suggestions)} "
                  f"ملاحظة فردية إضافية." if ar else
                  f"{len(combined_hyps)} combined pattern(s) from independent evidence were found, plus "
                  f"{len(individual_suggestions)} additional individual observation(s)."))
    elif individual_suggestions:
        st.write((f"لم يتقارب دليلان مستقلان على فرضية واحدة؛ توجد {len(individual_suggestions)} ملاحظة فردية ضعيفة الدعم."
                  if ar else
                  f"No two independent pieces of evidence converged on one pattern; {len(individual_suggestions)} "
                  "weak, individual observation(s) were found."))
    else:
        st.write("لم تُلاحَظ أي أنماط أو قواعد بصرية في هذه الرسمة." if ar else "No visual patterns or rules were observed in this drawing.")
    st.caption(disclaimer(language))

    # DOAR-TRACE Phase 2A, Section 3+9: page-frame assessability. Rules
    # about page coverage or placement are silently absent (not
    # "not_matched") from the suggestions above when the page isn't
    # confirmed visible -- stated explicitly here so that absence reads as
    # "we could not check this", never as "nothing was found".
    page_frame = (structured or {}).get("page_frame_assessment") or analysis.get("page_frame") or {}
    pf_status = page_frame.get("page_frame_status")
    if pf_status and pf_status not in ("full_page_detected", "likely_full_page"):
        st.warning(
            "تعذّر التأكد من ظهور الصفحة كاملة في هذه الصورة، لذلك أُخفيت أي ملاحظات عن موضع الرسمة على الصفحة أو "
            "نسبة تغطيتها لها -- وهي غير غائبة لأنها لم تُلاحَظ، بل لأنه لا يمكن تقييمها بثقة."
            if ar else
            "The full page could not be reliably confirmed as visible in this image, so any observations about the "
            "drawing's placement on the page or how much of the page it covers are suppressed for this case -- "
            "not because none were found, but because they cannot be assessed with confidence."
        )
    elif pf_status:
        st.caption(("حالة إطار الصفحة: مؤكَّدة (" if ar else "Page-frame status: confirmed (") + pf_status + ")")

    # 2. Combined patterns, levels 2-4 only ----------------------------------
    st.subheader("٢. الأنماط المُجمَّعة (المستويات ٢-٤ فقط)" if ar else "2. Combined patterns (levels 2-4 only)")
    if combined_hyps:
        for hyp in combined_hyps:
            with st.expander(f"[{hyp['level_label']}] {hyp['display_name_ar'] if ar else hyp['display_name_en']}"):
                for text in hyp["supporting_texts"]:
                    st.write("- " + text)
                st.caption(("مستوى التقارب: " if ar else "Convergence level: ") + f"{hyp['ordinal_level']} ({hyp['level_label']})")
                st.caption(("عائلات الأدلة المستقلة: " if ar else "Independent evidence families: ") + ", ".join(hyp["contributing_evidence_families"]))
                st.warning("فرضية مُجمَّعة من عدة أدلة مستقلة — تتطلب مراجعة مختص، وليست تشخيصاً."
                          if ar else "A combined hypothesis from multiple independent pieces of evidence -- requires clinician review, and is not a diagnosis.")
    else:
        st.info("لا توجد أنماط مُجمَّعة (مستوى ٢ أو أعلى) لهذه الحالة." if ar
               else "No combined pattern (level 2 or higher) for this case.")

    # 3. Individual rule suggestions (Level B -- never combined) ------------
    st.subheader("٣. ملاحظات فردية لكل قاعدة" if ar else "3. Individual rule suggestions")
    st.caption("كل قاعدة تظهر بمفردها فقط، ولا تُعرض أبداً كفرضية مُجمَّعة." if ar
              else "Each rule is shown on its own only -- never displayed as a combined theme.")
    if individual_suggestions:
        for suggestion in individual_suggestions:
            with st.expander(f"{suggestion['rule_id']} -- {suggestion['observable'].replace('_', ' ')}"):
                st.write(suggestion["parent_safe_wording"])
                st.caption(("مستوى الدليل كما ورد في المصدر: " if ar else "Evidence level as written in source: ")
                          + (", ".join(suggestion["evidence_level_as_written"]) if suggestion["evidence_level_as_written"] else "not graded by source"))
                if suggestion["evidence_family"] in ("line_intensity_quality", "line_fragmentation_quality"):
                    st.caption(
                        "هذه ملاحظة عن مظهر الخط المُستخرج من الصورة فقط، وليست قياساً لقوة ضغط القلم الفعلية."
                        if ar else
                        "This is a line-appearance proxy extracted from the image only -- not a measurement of "
                        "actual physical pencil pressure."
                    )
    else:
        st.info("لا توجد ملاحظات فردية لهذه الحالة." if ar else "No individual rule suggestions for this case.")

    # 4. What was measured/detected (Level A) --------------------------------
    st.subheader("٤. ما تم قياسه أو اكتشافه" if ar else "4. What was measured/detected")
    for obs in plain_language_observations(analysis, language):
        st.write("- " + obs["text"])
    if detections.get("status") == "unavailable":
        st.info("كاشف الأجسام غير مُطبَّق في هذا الإصدار — لم تُفحص الرسمة لمحتواها."
               if ar else "Detector not implemented in this release -- the drawing's content was not analyzed for objects.")

    # 5. Expressive-content model result -------------------------------------
    st.subheader("٥. نتيجة نموذج المحتوى التعبيري" if ar else "5. Expressive-content model result")
    model_obs = (structured or {}).get("expressive_model_observation")
    if model_obs:
        st.write(model_obs["text"])
        st.caption(f"confidence={model_obs['confidence']:.0%}, calibration={model_obs['calibration_status']}, "
                   f"used_in_combined_hypothesis={model_obs['used_in_combined_hypothesis']}")
        st.caption("هذا تصنيف إحصائي وليس تقييماً نفسياً لحالة الطفل." if ar
                  else "This is a statistical classification, not a psychological assessment of the child's state.")
    else:
        st.info("لا يتوفر ناتج نموذج تعبيري لهذه الحالة." if ar else "No expressive-content model output for this case.")

    # 6. Why each suggestion was made ----------------------------------------
    st.subheader("٦. لماذا اقتُرحت كل ملاحظة" if ar else "6. Why each suggestion was made")
    for suggestion in individual_suggestions:
        st.write(f"**{suggestion['rule_id']}**: " + suggestion["source_citation"])
    for hyp in combined_hyps:
        st.write(f"**{hyp['target_construct']}**: " + (", ".join(hyp["contributing_rule_ids"]) or "expressive model only")
                 + (" + expressive model" if hyp["uses_expressive_model"] else ""))
    if not individual_suggestions and not combined_hyps:
        st.caption("لا يوجد ما يُفسَّر." if ar else "Nothing to explain.")

    # 7. Contradictions and alternative explanations -------------------------
    st.subheader("٧. التناقضات والتفسيرات البديلة" if ar else "7. Contradictions and alternative explanations")
    contradictions = (structured or {}).get("cross_theme_contradictions", [])
    if contradictions:
        st.warning("تم رصد فرضيات متعارضة ولم يُختر فائز تلقائياً:" if ar
                  else "Conflicting patterns were detected; no automatic winner was chosen:")
        for c in contradictions:
            st.write(f"- {c['construct_a']} ↔ {c['construct_b']}")
    else:
        st.caption("لم تُرصد تناقضات بين الأنماط في هذه الحالة." if ar else "No contradictions between patterns were detected in this case.")
    for suggestion in individual_suggestions:
        for alt in suggestion["alternative_explanations"]:
            st.caption(f"[{suggestion['rule_id']}] " + (("تفسير بديل: " if ar else "Alternative: ")) + alt)

    # 8. Missing/unavailable evidence -----------------------------------------
    st.subheader("٨. الأدلة الناقصة أو غير المتاحة" if ar else "8. Missing/unavailable evidence")
    missing = (structured or {}).get("missing_evidence", [])
    if missing:
        for m in missing:
            st.write("- " + m)
    else:
        st.caption("لا توجد أدلة ناقصة مسجَّلة." if ar else "No missing evidence recorded.")
    st.caption("كاشف الأجسام والعلاقات المكانية غير متاحين في هذا الإصدار." if ar
              else "Object and spatial-relationship detection are unavailable in this release.")

    # 9. Suggested questions ---------------------------------------------------
    st.subheader("٩. أسئلة مقترحة" if ar else "9. Suggested questions")
    questions = (structured or {}).get("suggested_parent_questions", [])
    if questions:
        for q in questions:
            st.write("- " + q)
    else:
        st.caption("لا توجد أسئلة مقترحة لهذه الحالة." if ar else "No suggested questions for this case.")

    # 10. Sources and evidence strength ----------------------------------------
    st.subheader("١٠. المصادر وقوة الأدلة" if ar else "10. Sources and evidence strength")
    for suggestion in individual_suggestions:
        st.write(f"**{suggestion['rule_id']}**: {suggestion['source_citation']} "
                 + (f"({', '.join(suggestion['reference_ids'])})" if suggestion["reference_ids"] else ""))

    # 11. Permanent disclaimer ---------------------------------------------------
    st.subheader("١١. إخلاء المسؤولية الدائم" if ar else "11. Permanent disclaimer")
    st.info(disclaimer(language))

    # 12. Expandable all-features and all-rules tables ---------------------------
    st.subheader("١٢. كل السمات وكل القواعد (قابلة للتوسيع)" if ar else "12. All features and all rules (expandable)")
    if objective_features_doc:
        with st.expander(("عرض كل السمات الموضوعية" if ar else "Show all objective features")
                         + f" ({objective_features_doc['feature_count']}, "
                         + f"{objective_features_doc['missing_count']} {'غير متاحة' if ar else 'unavailable'})"):
            st.dataframe(objective_features_doc["features"], use_container_width=True)
    else:
        st.caption("سمات موضوعية غير متاحة لهذه الحالة (تحليل سابق للتحديث)." if ar
                  else "Objective features not available for this case (analyzed before this update).")
    with st.expander(("عرض كل القواعد الـ٤١ وحالتها" if ar else "Show all 41 rules and their status")):
        for row in plain_language_rule_rows(analysis["rule_evaluations"], language):
            badge = {"weak_support": "OBSERVED", "not_matched": "NOT OBSERVED",
                     "missing_detector": "NOT EVALUATED", "not_assessable_context_unknown": "NOT EVALUABLE"}.get(row["status"], row["status"])
            st.write(f"[{badge}] {row['label']}")

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

    st.subheader("Page-frame assessability (DOAR-TRACE Phase 2A, Section 3)")
    page_frame = analysis.get("page_frame") or {}
    if page_frame:
        pf_cols = st.columns(3)
        pf_cols[0].metric("Status", page_frame.get("page_frame_status", "n/a"))
        pf_cols[1].metric("Confidence", f"{page_frame.get('confidence', 0):.2f}")
        pf_cols[2].metric("Method", page_frame.get("method", "n/a"))
        st.json({
            "detected_page_boundary": page_frame.get("detected_page_boundary"),
            "border_evidence": page_frame.get("border_evidence"),
            "cropping_evidence": page_frame.get("cropping_evidence"),
            "limitations": page_frame.get("limitations"),
        })
        not_assessable_ids = (structured or {}).get("page_relative_rules_not_assessable", [])
        if not_assessable_ids:
            st.caption(f"Page-relative rules gated `not_assessable` for this case: {', '.join(not_assessable_ids)}")
        else:
            st.caption("No page-relative rules were gated `not_assessable` for this case.")
    else:
        st.info("page_frame not available for this case (analyzed before Phase 2A).")

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

    st.subheader("Dependency grouping and aggregation calculation (DOAR-TRACE Phase 1.5 Section 6)")
    if structured:
        st.caption(
            "combined_drawing_level_hypotheses: rules (+ the expressive-content model, confidence-gated) "
            "grouped by target_construct, deduplicated by evidence_id, escalated to a Level-C combined "
            "hypothesis only when >=2 distinct evidence IDs from >=2 distinct evidence families converge "
            "and the construct's own policy (construct_registry.json) is satisfied (structured_report.py)."
        )
        if structured["combined_drawing_level_hypotheses"]:
            st.dataframe([
                {
                    "target_construct": h["target_construct"],
                    "ordinal_level": h["ordinal_level"],
                    "level_label": h["level_label"],
                    "contributing_rule_ids": ", ".join(h["contributing_rule_ids"]),
                    "uses_expressive_model": h["uses_expressive_model"],
                    "contributing_evidence_ids": ", ".join(h["contributing_evidence_ids"]),
                    "evidence_families": ", ".join(h["contributing_evidence_families"]),
                }
                for h in structured["combined_drawing_level_hypotheses"]
            ], use_container_width=True)
        else:
            st.caption("No construct reached the >=2-family combined-hypothesis threshold on this image.")
        st.caption(f"individual_rule_suggestions (Level B, never combined): {len(structured['individual_rule_suggestions'])}")
        if structured["cross_theme_contradictions"]:
            st.dataframe(structured["cross_theme_contradictions"], use_container_width=True)
        st.caption(f"structured_analysis schema_version={structured['schema_version']}, "
                   f"registry_v2 schema_version={structured['registry_v2_schema_version']}, "
                   f"construct_registry schema_version={structured.get('construct_registry_schema_version')}")
    else:
        st.info("structured_analysis.json not available for this case (analyzed before this update).")

    st.subheader("Source catalog coverage")
    catalog_path = ROOT / "resources" / "psychology_sources" / "source_rule_catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        st.write(f"{catalog['entry_count']} total source rows ({catalog['entry_count_by_document']}), "
                 f"{catalog['candidate_rule_entry_count']} candidate-rule rows, "
                 f"{len(catalog['relationships'])} explicit relationships.")
        with st.expander("Show all source rows"):
            st.dataframe(catalog["entries"], use_container_width=True)
        with st.expander("Show all relationships (near_duplicate/expands/related_but_distinct)"):
            st.dataframe(catalog["relationships"], use_container_width=True)
    else:
        st.info("source_rule_catalog.json not found.")

    st.subheader("Every rule and its construct mapping")
    registry_v2_path = ROOT / "resources" / "psychology_sources" / "rules_registry_v2.json"
    if registry_v2_path.exists():
        registry_v2_doc = json.loads(registry_v2_path.read_text(encoding="utf-8"))
        st.write(f"{registry_v2_doc['rule_count']} rules total "
                 f"({registry_v2_doc['rule_count_original_production']} original production + "
                 f"{registry_v2_doc['rule_count_new_from_compiled_pdf']} new from the compiled PDF).")
        st.dataframe([
            {"rule_id": r["rule_id"], "observable": r["observable"], "target_construct": r["target_construct"],
             "observability_class": r["observability_class"], "allowed_output_level": r["allowed_output_level"],
             "evidence_family": r["evidence_family"], "source_document": r["source_document"], "source_page": r["source_page"],
             # DOAR-TRACE Phase 2A, Section 9: threshold provenance, dependency
             # grouping (double-counting safety), and validation status --
             # previously only visible by opening the raw JSON.
             "threshold_source": r.get("threshold_source"), "dependency_group": ", ".join(r.get("dependency_group") or []),
             "validation_status": r.get("validation_status"), "confidence_ceiling": r.get("confidence_ceiling"),
             # DOAR-TRACE Phase 2A.1, Section 8: rule policy after
             # measurement hardening.
             "page_reference_requirement": r.get("page_reference_requirement"),
             "feature_version": r.get("feature_version"),
             "expert_review_status": r.get("expert_review_status"),
             "known_robustness_limitations": r.get("known_robustness_limitations")}
            for r in registry_v2_doc["rules"]
        ], use_container_width=True)
        n_executable = sum(1 for r in registry_v2_doc["rules"] if r["allowed_output_level"] == "individual_heuristic_only")
        st.caption(f"{n_executable} of {registry_v2_doc['rule_count']} rules are actually executable "
                   f"(allowed_output_level=individual_heuristic_only); every other rule is disabled regardless of "
                   f"observability_class. See docs/STATIC_PROXY_RULE_POLICY.md.")
    else:
        st.info("rules_registry_v2.json not found.")

    st.subheader("Measurement validation status (DOAR-TRACE Phase 2A, Sections 4-6)")
    st.caption(
        "Validates measurement implementation only, not psychological validity. Synthetic-ground-truth and "
        "transformation-invariance results are cross-case (computed once on a fixed sample), not per-case."
    )
    phase2a_dir = ROOT / "artifacts" / "phase2a"
    ground_truth_summary_path = phase2a_dir / "feature_ground_truth_summary.json"
    invariance_summary_path = phase2a_dir / "feature_invariance_summary.json"
    threshold_csv_path = phase2a_dir / "threshold_sensitivity.csv"
    mv_cols = st.columns(3)
    if ground_truth_summary_path.exists():
        gt = json.loads(ground_truth_summary_path.read_text(encoding="utf-8"))
        gt_counts = gt.get("status_counts", {})
        mv_cols[0].metric("Ground-truth checks", f"{gt_counts.get('pass', 0)}/{gt.get('n_cases', '?')} pass")
    else:
        mv_cols[0].caption("feature_ground_truth_summary.json not found -- run phase2a_feature_ground_truth.py.")
    if invariance_summary_path.exists():
        inv = json.loads(invariance_summary_path.read_text(encoding="utf-8"))
        counts = inv.get("status_counts", {})
        mv_cols[1].metric("Invariance rows", f"{counts.get('pass', 0)} pass / {counts.get('fail', 0)} fail")
    else:
        mv_cols[1].caption("feature_invariance_summary.json not found -- run phase2a_feature_invariance.py.")
    if threshold_csv_path.exists():
        mv_cols[2].metric("Threshold sensitivity rows", sum(1 for _ in threshold_csv_path.open(encoding="utf-8")) - 1)
    else:
        mv_cols[2].caption("threshold_sensitivity.csv not found -- run phase2a_threshold_sensitivity.py.")
    st.caption("Full results: docs/FEATURE_MEASUREMENT_VALIDATION.md, docs/FEATURE_ROBUSTNESS_RESULTS.md, "
               "docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md.")

    st.subheader("Judge verdicts (judges_v2.json)")
    if judges_v2_doc:
        st.dataframe([
            {"judge_id": jid, "status": v["status"], "confidence": v["confidence"], "reasons": "; ".join(v["reasons"])}
            for jid, v in judges_v2_doc.items()
        ], use_container_width=True)
    else:
        st.info("judges_v2.json not available for this case (analyzed before this update).")

    st.subheader("Claim verification")
    if generated_claims_doc and verification_report_doc:
        st.write(f"{generated_claims_doc['claim_count']} claim(s) generated: "
                 f"{generated_claims_doc['accepted_count']} accepted, {generated_claims_doc['rejected_count']} rejected. "
                 f"all_passed={verification_report_doc['all_passed']}")
        st.dataframe(generated_claims_doc["claims"], use_container_width=True)
        with st.expander("Show per-claim verification checks"):
            st.dataframe([
                {"claim_id": c["claim_id"], "passed": c["passed"],
                 "failed_checks": "; ".join(chk["check_name"] for chk in c["checks"] if not chk["passed"])}
                for c in verification_report_doc["claims"]
            ], use_container_width=True)
    else:
        st.info("generated_claims.json/verification_report.json not available for this case (analyzed before this update).")

    st.subheader("Evidence IDs")
    st.dataframe([
        {**e, "value": str(e["value"]), "limitations": "; ".join(e.get("limitations", []))}
        for e in analysis["evidence"]
    ], use_container_width=True)

    st.subheader("Schema and model versions")
    st.code(
        f"analysis.schema_version: {analysis['schema_version']}\n"
        f"structured_analysis schema_version: {structured.get('schema_version') if structured else 'n/a'}\n"
        f"registry_v2 schema_version: {structured.get('registry_v2_schema_version') if structured else 'n/a'}\n"
        f"model_name: {analysis['emotion'].get('model_name') or 'n/a'}\n"
        f"model_version: {analysis['emotion'].get('model_version') or 'n/a'}\n"
        f"preprocessing_version: {analysis['emotion'].get('preprocessing_version') or 'n/a'}"
    )

    st.subheader("Missing/unavailable capability warnings")
    warnings = [
        "Object/geometry detection: not implemented (schema-only scaffolding in src/doar/detectors/)",
        "shape.enclosed_shape_count, shape.repetition_score: not implemented (no shape detector)",
        "Concern profiles: implemented but disabled by design (CONCERNS_ENABLED = False)",
        "35 of 41 registry-v2 rules: allowed_output_level=disabled (no detector wired to any evaluator)",
        "detection_judge, relation_judge, language_judge: not_implemented (no underlying capability yet)",
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
