"""
DOAR dual-view prototype -- Parent/User view + Technical/Research view.

Built per CURRENT_CAPABILITY_AUDIT.md, END_TO_END_INFERENCE_TRACE.md,
RULE_AND_FEATURE_COVERAGE.md, TARGET_APPLICATION_ARCHITECTURE.md, and
docs/PARENT_VIEW_INFORMATION_POLICY.md (DOAR-TRACE Phase 2A.2): every
value shown here comes from a real execution of the existing pipeline
(analyze_image / features.objective_feature_row / qa.answer / chat.py).
Nothing is fabricated to make the interface look more complete than the
pipeline actually is -- absent capabilities (object detection, disabled
concern profiles, unevaluable rules) are shown as explicit, honest
messages, never silently omitted or invented.

Parent View (Phase 2A.2) is deliberately restricted to 5 sections
(Overall result / What was observed / Possible meaning / Questions and
next steps / Limitations) with all internal IDs, raw tables, and
technical enums moved to Technical View -- see
docs/PARENT_VIEW_INFORMATION_POLICY.md for the full policy this file
implements.

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

from doar.case_artifacts import resolve_analysis_artifacts, resolve_artifact_path  # noqa: E402
from doar.case_output import write_versioned  # noqa: E402
from doar.chat import respond_to_chat  # noqa: E402
from doar.expert_review import REVIEW_ACTIONS, load_review, submit_review  # noqa: E402
from doar.production_config import resolve_production_config  # noqa: E402
from doar.page_reference import (  # noqa: E402
    PARENT_PAGE_DECLARATION_CHOICES, PARENT_PAGE_DECLARATION_LABELS,
    describe_declaration_choice, user_page_declaration_from_choice,
)
from doar.parent_view import (  # noqa: E402
    build_overall_result_summary, capability_status, capability_status_summary_text,
    disclaimer, friendly_family_name, friendly_source_name, plain_language_observations,
)
from doar.phase2c7 import detector_policy as _pol  # noqa: E402
from doar.phase2c7 import runtime as _detector_runtime  # noqa: E402
from doar.profile import ChildProfile, ALLOWED_AGE_RANGES, load_profile, save_profile  # noqa: E402
from doar.registry_v2_build import build_registry_v2  # noqa: E402
from doar.timed_analysis import analyze_image_with_timing  # noqa: E402
from doar.visual_evidence import VisualFinding, build_rule_evidence_trace, run_and_persist_initial_scan  # noqa: E402
from doar.phase2b.technical_view import render_phase2b_pilot_summary  # noqa: E402

VISUAL_POLICY_PATH = ROOT / "artifacts" / "phase2c7" / "visual_detector_policy.json"


@st.cache_resource(show_spinner=False)
def _load_registry_v2() -> dict:
    return build_registry_v2()


@st.cache_resource(show_spinner=False)
def _load_eye_entry():
    """Frozen Phase 2C.7 eye policy entry, read from the committed artifact
    -- never re-evaluated or re-thresholded here."""
    rows = json.loads(VISUAL_POLICY_PATH.read_text(encoding="utf-8"))
    eye_row = next(r for r in rows if r["target"] == "eye")
    return _pol.build_eye_policy_entry(
        status=eye_row["status"], best_model=eye_row["best_model"],
        best_model_checkpoint=eye_row["best_model_checkpoint"], prompt=eye_row["prompt"],
        threshold=eye_row["threshold"], precision=eye_row["precision"], recall=eye_row["recall"],
        balanced_accuracy=eye_row["balanced_accuracy"],
        n_ground_truth_present=eye_row["n_ground_truth_present"],
        localization_validated=eye_row["localization_validated"], rationale=eye_row["rationale"])


@st.cache_resource(show_spinner=False)
def _load_model_predict_fns() -> dict:
    """Loads every real model the frozen visual detector policy needs, ONCE
    per server process (Streamlit's cache_resource) -- real weights, never
    called by the automated test suite. Also loads the dynamic open-vocab
    query model under the "open_vocab_query" key for the broad-scan extras
    and on-demand Q&A search."""
    eye_entry = _load_eye_entry()
    fns = _detector_runtime.build_real_model_predict_fns(eye_entry.best_model)
    fns["open_vocab_query"] = _detector_runtime.load_open_vocab_query_fn()
    return fns

CASES_DIR = ROOT / "outputs" / "prototype_cases"

st.set_page_config(page_title="DOAR prototype - dual view", layout="wide")
st.title("DOAR v3 -- Dual-View Prototype")
st.warning(
    "Research prototype, non-diagnostic. Every checkpoint currently available was "
    "trained on a duplicate-contaminated, leakage-gate-overridden split -- see "
    "CURRENT_CAPABILITY_AUDIT.md Section 3. Treat all output as preliminary."
)

# ---------------------------------------------------------------------------
# Sidebar: upload + optional child context + checkpoint choice + page
# declaration (DOAR-TRACE Phase 2A.2, Section 6), or reopen.
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

    st.caption("Does this image show the complete sheet of paper?")
    page_choice = st.radio(
        "Page declaration", PARENT_PAGE_DECLARATION_CHOICES,
        format_func=lambda c: PARENT_PAGE_DECLARATION_LABELS[c]["en"],
        index=0, label_visibility="collapsed", key="page_declaration_choice",
    )
    st.caption(
        "This is optional -- the system can also try to work this out automatically. Manually marking exact page "
        "corners is not available in this version."
    )

    if st.button("Analyze", type="primary", disabled=uploaded is None):
        case_name = f"{Path(uploaded.name).stem}_{int(time.time())}"
        case_dir = CASES_DIR / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        image_path = case_dir / uploaded.name
        image_path.write_bytes(uploaded.getvalue())

        # DOAR V1.1 Stage 3: the user makes NO model/checkpoint choice --
        # the frozen production configuration is resolved automatically,
        # once, and reused for every case. If the expressive-model
        # checkpoint is not present on this machine, the SAME code path
        # that already handles "no emotion model" runs (emotion.py's own
        # correct unavailable() branch) -- never a fabricated fallback,
        # never a different checkpoint silently substituted.
        production_config = resolve_production_config()
        declaration = user_page_declaration_from_choice(page_choice)
        with st.spinner("Running the real analysis pipeline..."):
            analyze_image_with_timing(str(image_path), str(case_dir), production_config.expressive_model_checkpoint,
                                       user_page_declaration=declaration)
            write_versioned(case_dir / "production_config.json", production_config.to_dict())
            profile = ChildProfile(
                age_range=age_range, gender=gender or None,
                drawing_instruction=instruction or None,
                date=date.today().isoformat(), parent_concern=concern or None,
            )
            save_profile(case_dir, profile)
        with st.spinner("Running the automatic visual object scan (first run loads real "
                         "models and can take a few minutes; cached after that)..."):
            try:
                run_and_persist_initial_scan(
                    case_dir, str(image_path), eye_entry=_load_eye_entry(),
                    registry_v2=_load_registry_v2(), model_predict_fns=_load_model_predict_fns())
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, analysis itself already succeeded
                st.warning(f"Visual object scan failed ({exc}); the rest of the analysis is unaffected. "
                           "detections.json remains the honest 'unavailable' stub for this case.")
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
# DOAR-TRACE Phase 2A.1: page_reference/canonical_features are written by
# every analyze_image run since that phase -- older cases won't have
# them (both default to {} via schemas.py's field default), so downstream
# code always treats a missing/empty dict the same as "not available".
page_reference = analysis.get("page_reference") or {}
canonical_features = analysis.get("canonical_features") or {}


def artifact(name: str) -> str | None:
    value = analysis.get("artifacts", {}).get(name)
    if not value:
        return None
    p = resolve_artifact_path(case_dir, value)
    return str(p) if p.exists() else None


parent_tab, technical_tab = st.tabs(["Parent / User View", "Technical / Research View"])

# ===========================================================================
# PARENT / USER VIEW  (DOAR-TRACE Phase 2A.2: exactly 5 sections --
# see docs/PARENT_VIEW_INFORMATION_POLICY.md)
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
    page_assessable = bool(page_reference.get("page_relative_features_assessable"))

    # 1. Overall result -------------------------------------------------------
    st.subheader("١. النتيجة الإجمالية" if ar else "1. Overall result")
    for sentence in build_overall_result_summary(
            structured, page_reference, language, capabilities=judges.get("module_availability", {})):
        st.write(sentence)
    st.caption(disclaimer(language))

    # 2. What was observed ------------------------------------------------------
    st.subheader("٢. ما تمت ملاحظته" if ar else "2. What was observed")
    for obs in plain_language_observations(analysis, language, page_reference=page_reference):
        st.write("- " + obs["text"])

    # 3. Possible meaning ---------------------------------------------------
    st.subheader("٣. المعنى المحتمل" if ar else "3. Possible meaning")
    if combined_hyps:
        st.caption(
            "الأنماط التالية نتجت عن تقارب أدلة مستقلة، وتتطلب مراجعة مختص." if ar else
            "The pattern(s) below came from independent evidence converging together, and need clinician review."
        )
        for hyp in combined_hyps:
            title = f"[{hyp['level_label']}] {hyp['display_name_ar'] if ar else hyp['display_name_en']}"
            with st.expander(title):
                for text in hyp["supporting_texts"]:
                    st.write("- " + text)
                families = ", ".join(sorted({friendly_family_name(f, language) for f in hyp["contributing_evidence_families"]}))
                st.caption(("عائلات الأدلة المستقلة: " if ar else "Independent evidence areas: ") + families)
                st.warning(
                    "فرضية مُجمَّعة من عدة أدلة مستقلة — تتطلب مراجعة مختص، وليست تشخيصاً." if ar else
                    "A combined hypothesis from multiple independent pieces of evidence -- requires clinician "
                    "review, and is not a diagnosis."
                )
    else:
        st.info("لا توجد أنماط مُجمَّعة لهذه الحالة." if ar else "No combined pattern was found for this case.")

    if individual_suggestions:
        st.caption(
            "كل ملاحظة فردية تُعرض بمفردها فقط، ولا تُعرض أبداً كفرضية مُجمَّعة." if ar else
            "Each individual observation is shown on its own only -- never as a combined pattern."
        )
        for i, suggestion in enumerate(individual_suggestions):
            family_label = friendly_family_name(suggestion["evidence_family"], language)
            title = (f"ملاحظة {i + 1}: {family_label}" if ar else f"Observation {i + 1}: {family_label}")
            with st.expander(title):
                st.write(suggestion["parent_safe_wording"])
                source_doc = suggestion["source_citation"].split(",")[0].strip()
                st.caption(("المصدر: " if ar else "Source: ") + friendly_source_name(source_doc, language))
                st.caption(("مستوى الدليل كما ورد في المصدر: " if ar else "Evidence level as written in source: ")
                          + (", ".join(suggestion["evidence_level_as_written"]) if suggestion["evidence_level_as_written"] else "not graded by source"))
                if suggestion["evidence_family"] in ("line_intensity_quality", "line_fragmentation_quality"):
                    st.caption(
                        "هذه ملاحظة عن مظهر الخط المُستخرج من الصورة فقط، وليست قياساً لقوة ضغط القلم الفعلية." if ar else
                        "This is a line-appearance proxy extracted from the image only -- not a measurement of "
                        "actual physical pencil pressure."
                    )
                for alt in suggestion["alternative_explanations"]:
                    st.caption(("تفسير بديل: " if ar else "Alternative explanation: ") + alt)
                if suggestion["limitations"]:
                    st.caption(("محدودية: " if ar else "Limitation: ") + "; ".join(suggestion["limitations"]))
    else:
        st.info("لا توجد ملاحظات فردية لهذه الحالة." if ar else "No individual observations for this case.")

    contradictions = (structured or {}).get("cross_theme_contradictions", [])
    if contradictions:
        st.warning(
            "تم رصد فرضيات متعارضة ولم يُختر فائز تلقائياً:" if ar else
            "Conflicting patterns were detected; no automatic winner was chosen:"
        )
        for c in contradictions:
            st.write(f"- {c['construct_a']} ↔ {c['construct_b']}")

    # 4. Questions and next steps ---------------------------------------------
    st.subheader("٤. أسئلة وخطوات تالية" if ar else "4. Questions and next steps")
    questions = list((structured or {}).get("suggested_parent_questions", []))
    if not questions:
        questions = ["Can you tell me what is happening in this picture?"] if not ar else \
            ["هل يمكنك أن تخبرني ماذا يحدث في هذه الصورة؟"]
    for q in questions:
        st.write("- " + q)

    # 5. Limitations ------------------------------------------------------------
    st.subheader("٥. المحدوديات" if ar else "5. Limitations")
    st.write(capability_status_summary_text(language))
    if not page_assessable:
        st.warning(
            "تعذّر تأكيد ظهور الورقة كاملة في هذه الصورة، لذلك أُخفيت أي ملاحظات عن موضع الرسمة على الصفحة أو نسبة "
            "تغطيتها لها -- وهي غير غائبة لأنها لم تُلاحَظ، بل لأنه لا يمكن تقييمها بثقة." if ar else
            "The complete sheet could not be confirmed as visible in this image, so any observations about the "
            "drawing's placement on the page or how much of the page it covers are suppressed for this case -- not "
            "because none were found, but because they cannot be assessed with confidence."
        )
    st.info(disclaimer(language))
    with st.expander("More details about this result" if not ar else "مزيد من التفاصيل حول هذه النتيجة"):
        st.caption(
            "التفاصيل التقنية الكاملة (المعرفات الداخلية، السمات الخام، القواعد الأربعون، سجلات التحقق) متاحة في "
            "علامة التبويب 'Technical / Research View'." if ar else
            "The full technical detail (internal IDs, raw features, all 41 rules, verification records) is "
            "available in the 'Technical / Research View' tab."
        )

    st.divider()
    st.subheader("محادثة متابعة" if ar else "Follow-up chat")
    st.caption(
        "الإجابات مبنية حصراً على أدلة هذه الحالة المحفوظة، بدون نموذج لغوي أو اتصال بالإنترنت." if ar else
        "Answers are grounded strictly in this case's saved evidence -- no LLM, no network call."
    )
    st.session_state.setdefault(f"chat_{case_dir}", [])
    question = st.text_input("سؤال عن هذه الحالة" if ar else "Ask a question about this case", key="parent_chat_q")
    if st.button("إرسال" if ar else "Send", key="parent_chat_send") and question:
        response = respond_to_chat(
            case_dir, question, language, registry_v2=_load_registry_v2(),
            open_vocab_predict_fn=_load_model_predict_fns().get("open_vocab_query"))
        st.session_state[f"chat_{case_dir}"].append((question, response))
    for q, r in reversed(st.session_state[f"chat_{case_dir}"]):
        st.markdown(f"**{'أنت' if ar else 'You'}:** {q}")
        st.markdown(f"**{'الرد' if ar else 'Answer'}:** {r.answer}")
        if r.non_diagnostic_warning:
            st.caption(r.non_diagnostic_warning)

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# TECHNICAL / RESEARCH VIEW  (DOAR-TRACE Phase 2A.2, Section 9: organized
# under 11 named subsections -- nothing removed from Phase 2A/2A.1, only
# reorganized and extended with page-reference/canonical-feature/
# capability detail that did not exist before this phase.)
# ===========================================================================
with technical_tab:
    st.header("1. Input and page reference")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Schema version", analysis["schema_version"])
    meta_cols[1].metric("Model", analysis["emotion"].get("model_name") or "n/a")
    meta_cols[2].metric("Calibration", analysis["emotion"].get("calibration_status") or "n/a")
    meta_cols[3].metric("Processing time (s)", timing["processing_seconds"] if timing else "not recorded")
    st.code(f"checkpoint_sha256: {analysis['emotion'].get('checkpoint_sha256')}\n"
            f"preprocessing_version: {analysis['emotion'].get('preprocessing_version')}\n"
            f"image_path (original): {analysis['image_path']}")

    st.subheader("Page-frame assessability (automatic classical-CV signal)")
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
    else:
        st.info("page_frame not available for this case (analyzed before Phase 2A).")

    st.subheader("Page reference (resolved -- Phase 2A.1/2A.2)")
    if page_reference:
        declared_choice = describe_declaration_choice(page_reference)
        pr_cols = st.columns(4)
        pr_cols[0].metric("Mode", page_reference.get("page_reference_mode", "n/a"))
        pr_cols[1].metric("Assessable", str(page_reference.get("page_relative_features_assessable")))
        pr_cols[2].metric("Confidence", f"{page_reference.get('confidence', 0):.2f}")
        pr_cols[3].metric("Parent declaration used", declared_choice)
        st.caption(
            f"obtained_via={page_reference.get('obtained_via')} -- '{declared_choice}' is reconstructed purely "
            "from the saved page_reference fields (obtained_via + page_reference_mode), not a separate file."
        )
        st.json({"page_polygon": page_reference.get("page_polygon"), "limitations": page_reference.get("limitations")})
        not_assessable_ids = (structured or {}).get("page_relative_rules_not_assessable", [])
        st.caption(
            f"Page-relative rules gated `not_assessable` for this case: {', '.join(not_assessable_ids)}"
            if not_assessable_ids else "No page-relative rules were gated `not_assessable` for this case."
        )
        st.caption(
            "API capability note: `user_defined_page_corners` (manual corner marking) exists in page_reference.py "
            "and is fully tested (tests/test_page_reference.py), but is not yet exposed through this prototype's "
            "UI -- only the 4-choice declaration above is available here. See docs/PAGE_REFERENCE_MODEL.md."
        )
    else:
        st.info("page_reference not available for this case (analyzed before Phase 2A.1).")

    st.header("2. Image quality and segmentation")
    quality = analysis.get("quality", {})
    q_cols = st.columns(4)
    q_cols[0].metric("Quality status", quality.get("quality_status", "n/a"))
    q_cols[1].metric("Blur variance", quality.get("blur_variance", "n/a"))
    q_cols[2].metric("Contrast std", quality.get("contrast_std", "n/a"))
    q_cols[3].metric("Min dimension (px)", quality.get("min_dimension", "n/a"))
    if quality.get("unsupported_reasons"):
        st.caption("Unsupported reasons: " + "; ".join(quality["unsupported_reasons"]))
    segmentation = analysis.get("segmentation", {})
    st.json({
        "status": segmentation.get("status"), "confidence": segmentation.get("confidence"),
        "background_stability": segmentation.get("background_stability"),
        "selected_strategy": segmentation.get("selected_strategy"),
    })
    st.subheader("Image preview and overlays")
    cols = st.columns(4)
    for i, name in enumerate(["normalized_image", "foreground_mask", "feature_overlay", "stroke_map"]):
        path = artifact(name)
        if path:
            cols[i % 4].image(path, caption=name)

    st.header("3. Objective features")
    st.caption(
        "Coverage/placement/margin features are computed relative to the UPLOADED IMAGE FRAME, not verified "
        "physical-page usage -- see docs/PARENT_VIEW_INFORMATION_POLICY.md and Section 3 header below."
    )
    try:
        from doar.features import objective_feature_row, serialize_feature_row
        original_image = case_dir / Path(analysis["image_path"]).name
        # analysis.json stores artifact paths case-relative (e.g.
        # "artifacts/foreground_mask.png") -- resolve against case_dir
        # before Image.open() ever sees them; the app's cwd is the repo
        # root (wherever `streamlit run` was launched from), not the case
        # directory, so an unresolved relative path fails here even
        # though the file genuinely exists on disk (DOAR V1.1 Problem A).
        resolved_analysis = resolve_analysis_artifacts(analysis, case_dir)
        feature_row = objective_feature_row(str(original_image), resolved_analysis)
        serialized = serialize_feature_row(feature_row)
        st.caption(
            "Measured relative to the uploaded image frame; not interpreted as physical-page usage."
        )
        st.dataframe([
            {"feature": name, "value": v["value"], "confidence": v["confidence"],
             "missing": v["missing"], "method": v["method"]}
            for name, v in serialized.items()
        ], width="stretch")
        n_missing = sum(1 for v in serialized.values() if v["missing"])
        st.caption(f"{len(serialized)} features computed; {n_missing} marked unavailable "
                   "(no fabricated values for missing detectors).")
    except Exception as exc:  # pragma: no cover -- UI guard, real error still surfaced
        st.error(f"Feature computation failed: {exc}")

    st.header("4. Canonical features (resolution-normalized -- Phase 2A.1 Section 6)")
    if canonical_features:
        canon_info = canonical_features.get("canonicalization", {})
        cc_cols = st.columns(3)
        cc_cols[0].metric("Resized", str(canon_info.get("resized")))
        cc_cols[1].metric("Original size", "x".join(str(v) for v in canon_info.get("original_size", [])) or "n/a")
        cc_cols[2].metric("Canonical size", "x".join(str(v) for v in canon_info.get("canonical_size", [])) or "n/a")
        st.caption(
            "Strictly separate from the objective_features table above -- never merged. Only these 4 features "
            "(the ones Phase 2A found resize-sensitive) have a canonical counterpart; no rule currently reads it."
        )
        st.dataframe([
            {"feature": name, "value": v["value"], "confidence": v["confidence"], "method": v["method"]}
            for name, v in canonical_features.get("features", {}).items()
        ], width="stretch")
    else:
        st.info("canonical_features not available for this case (analyzed before Phase 2A.1).")

    st.header("5. Expressive-content model")
    if analysis["emotion"]["status"] == "available":
        st.dataframe({
            "class": list(analysis["emotion"]["probabilities"].keys()),
            "calibrated_probability": list(analysis["emotion"]["probabilities"].values()),
            "raw_probability": list(analysis["emotion"].get("raw_probabilities", {}).values() or [None] * 4),
        })
    else:
        reason = analysis["emotion"].get("reason") or "no reason recorded"
        st.info(f"No probabilities -- emotion status: {analysis['emotion']['status']}. Reason: {reason}")
        st.caption(
            "No probability values are fabricated when this branch is unavailable -- see "
            "production_config.json below for the checkpoint identifier this pipeline always tries first.")

    production_config_path = case_dir / "production_config.json"
    if production_config_path.exists():
        st.subheader("Production configuration / provenance")
        st.caption(
            "The single, automatically-resolved configuration this case was analyzed with -- never a "
            "per-case user choice. See src/doar/production_config.py.")
        st.json(json.loads(production_config_path.read_text(encoding="utf-8")))
    else:
        st.caption("production_config.json not available for this case (analyzed before DOAR V1.1).")

    st.header("6. Rule evaluations")
    st.dataframe(analysis["rule_evaluations"], width="stretch")

    st.header("7. Combined-pattern calculation")
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
            ], width="stretch")
        else:
            st.caption("No construct reached the >=2-family combined-hypothesis threshold on this image.")
        st.caption(f"individual_rule_suggestions (Level B, never combined): {len(structured['individual_rule_suggestions'])}")
        if structured["cross_theme_contradictions"]:
            st.dataframe(structured["cross_theme_contradictions"], width="stretch")
        st.caption(f"structured_analysis schema_version={structured['schema_version']}, "
                   f"registry_v2 schema_version={structured['registry_v2_schema_version']}, "
                   f"construct_registry schema_version={structured.get('construct_registry_schema_version')}")
    else:
        st.info("structured_analysis.json not available for this case (analyzed before this update).")

    st.header("8. Evidence and provenance")
    st.dataframe([
        {**e, "value": str(e["value"]), "limitations": "; ".join(e.get("limitations", []))}
        for e in analysis["evidence"]
    ], width="stretch")
    st.code(
        f"analysis.schema_version: {analysis['schema_version']}\n"
        f"structured_analysis schema_version: {structured.get('schema_version') if structured else 'n/a'}\n"
        f"registry_v2 schema_version: {structured.get('registry_v2_schema_version') if structured else 'n/a'}\n"
        f"model_name: {analysis['emotion'].get('model_name') or 'n/a'}\n"
        f"model_version: {analysis['emotion'].get('model_version') or 'n/a'}\n"
        f"preprocessing_version: {analysis['emotion'].get('preprocessing_version') or 'n/a'}\n"
        f"case_dir: {case_dir.name}"
    )
    st.subheader("Downloadable structured evidence and reports")
    st.download_button("analysis.json", (case_dir / "analysis.json").read_bytes(), file_name="analysis.json")
    for report in sorted((case_dir / "reports").glob("*.html")):
        st.download_button(report.name, report.read_bytes(), file_name=report.name)

    st.header("9. Judges and verification")
    if judges_v2_doc:
        st.dataframe([
            {"judge_id": jid, "status": v["status"], "confidence": v["confidence"], "reasons": "; ".join(v["reasons"])}
            for jid, v in judges_v2_doc.items()
        ], width="stretch")
    else:
        st.info("judges_v2.json not available for this case (analyzed before this update).")
    if generated_claims_doc and verification_report_doc:
        st.write(f"{generated_claims_doc['claim_count']} claim(s) generated: "
                 f"{generated_claims_doc['accepted_count']} accepted, {generated_claims_doc['rejected_count']} rejected. "
                 f"all_passed={verification_report_doc['all_passed']}")
        st.dataframe(generated_claims_doc["claims"], width="stretch")
        with st.expander("Show per-claim verification checks"):
            st.dataframe([
                {"claim_id": c["claim_id"], "passed": c["passed"],
                 "failed_checks": "; ".join(chk["check_name"] for chk in c["checks"] if not chk["passed"])}
                for c in verification_report_doc["claims"]
            ], width="stretch")
    else:
        st.info("generated_claims.json/verification_report.json not available for this case (analyzed before this update).")
    st.subheader("Reliability and safety warnings (deterministic judges.json)")
    st.json(judges)
    st.subheader("LLM-generated claims, evidence references, and judge results")
    st.info("No LLM is enabled in this prototype -- see LLM_GROUNDING_AND_SAFETY_DESIGN.md. "
            "All chat responses below are deterministic evidence lookups (src/doar/qa.py), not generated text.")
    chat_log = st.session_state.get(f"chat_{case_dir}", [])
    if chat_log:
        st.dataframe([
            {"question": q, "answer": r.answer, "evidence_ids": ", ".join(r.evidence_ids),
             "availability": r.availability, "escalated": r.escalated}
            for q, r in chat_log
        ], width="stretch")
    else:
        st.caption("No chat turns yet this session.")

    st.header("10. Visual object detections")
    st.subheader("Object detections")
    if detections.get("status") == "available":
        findings = detections.get("findings", [])
        st.write(f"{detections.get('n_findings', len(findings))} finding(s), generated "
                 f"{detections.get('generated_at', '?')}.")
        status_order = {"VALIDATED": 0, "EXPERIMENTAL": 1, "UNKNOWN": 2, "DISABLED_FOR_RULES": 3}
        rows = sorted(findings, key=lambda f: (status_order.get(f["validation_status"], 9), -f["confidence"]))
        st.dataframe([
            {"label": f["label"], "free_form_label": f.get("free_form_label"),
             "confidence": round(f["confidence"], 3), "validation_status": f["validation_status"],
             "evidence_status": f["evidence_status"], "rule_mapping_status": f["rule_mapping_status"],
             "related_rule_ids": ", ".join(f.get("related_rule_ids") or []),
             "detector": f["detector"], "bbox": f.get("bbox"), "source": f["source"],
             "query": f.get("query"), "timestamp": f["timestamp"]}
            for f in rows
        ], width="stretch")
        st.caption(
            "VALIDATED = eligible for the validated evidence pipeline. EXPERIMENTAL = shown here and "
            "searchable, but never activates a psychological conclusion. UNKNOWN = detected but never "
            "individually measured against human ground truth (broad-scan extras / on-demand search "
            "results). rule_mapping_status=MAPPED means at least one registry rule mentions this label -- "
            "it does NOT mean the rule was triggered; see the rule-evidence trace below.")

        entities = detections.get("entities", [])
        if entities:
            st.subheader("Rich visual entities (Visual Knowledge V2)")
            st.caption(
                "Superset of the table above: entity_type, aliases, broader categories, geometry, and "
                "colour, plus case_verification_status (from expert review) kept strictly separate from "
                "model_validation_status (a detector-level property). candidate_labels is what the "
                "detector considered -- never auto-promoted into canonical_label.")
            st.dataframe([
                {"entity_id": e["entity_id"], "entity_type": e["entity_type"],
                 "canonical_label": e["canonical_label"],
                 "candidate_labels": ", ".join(f"{lbl}:{conf:.2f}" for lbl, conf in e.get("candidate_labels") or []),
                 "aliases_en": ", ".join(e.get("aliases_en") or []),
                 "broader_categories": ", ".join(e.get("broader_categories") or []),
                 "possible_subtypes": ", ".join(e.get("possible_subtypes") or []),
                 "visual_similarities": ", ".join(e.get("visual_similarities") or []),
                 "dominant_colors": ", ".join(e.get("dominant_colors") or []) if e.get("dominant_colors") else None,
                 "relative_size": e.get("relative_size"), "page_position": e.get("page_position"),
                 "model_validation_status": e["model_validation_status"],
                 "case_verification_status": e["case_verification_status"],
                 "memory_status": e.get("memory_status")}
                for e in entities
            ], width="stretch")
            st.caption(
                "possible_subtypes/visual_similarities are structurally present but intentionally empty "
                "in this phase -- no subtype/similarity computation exists yet; populating them now would "
                "mean fabricating data. memory_status is always 'not_indexed' -- Visual Memory does not "
                "exist yet.")
        else:
            st.caption("entities not available for this case (analyzed before Visual Knowledge V2, or "
                       "the scan produced no findings).")

        trace = build_rule_evidence_trace([VisualFinding.from_dict(f) for f in findings])
        if trace:
            st.subheader("Rule-evidence trace (\"detected\" -- descriptive only)")
            st.dataframe(trace, width="stretch")
            st.caption("can_activate=True only for validated_evidence findings; every rule in this table "
                       "was still gated by its own allowed_output_level (see section 11) -- this is a "
                       "descriptive trace, it does not by itself trigger a rule.")
        else:
            st.caption("No finding currently maps to any registry rule_id.")

        visual_sourced_rules = [r for r in analysis.get("rule_evaluations", [])
                                 if r.get("visual_evidence_sourced")]
        st.subheader("Visual evidence actually used in rule reasoning (\"used\" -- real outcome)")
        if visual_sourced_rules:
            st.dataframe([
                {"rule_id": r["rule_id"], "status": r["status"],
                 "matched_evidence_ids": ", ".join(r.get("matched_evidence_ids") or []),
                 "missing_evidence": ", ".join(r.get("missing_evidence") or []),
                 "rule_confidence": r.get("rule_confidence")}
                for r in visual_sourced_rules
            ], width="stretch")
            st.caption("status='weak_support' means a validated visual finding legitimately satisfied "
                       "this rule and it flowed into aggregation/synthesis below. status='missing_detector' "
                       "means either no matching validated finding existed, or the rule's own "
                       "allowed_output_level in rules_registry_v2.json is still 'disabled' (a real "
                       "registry-curation decision this session does not override) -- see missing_evidence.")
        else:
            st.caption("No visual finding was used by the real rule engine for this case -- either "
                       "nothing detected matched a rule's observable, or (as of this session) every "
                       "matching rule's allowed_output_level is still 'disabled' in rules_registry_v2.json. "
                       "This is the honest, expected state today: see DOAR_V1_RULE_INTEGRATION_REPORT.md.")
    else:
        st.json(detections)
        st.caption("detections.json is the honest 'unavailable' stub for this case -- either it was "
                   "analyzed before the visual scan was wired in, or the scan failed for this case "
                   "(see the warning shown at analysis time). \"Detector not implemented / scan failed\", "
                   "not \"no objects found\".")
    cap = capability_status("en")
    cap_cols = st.columns(3)
    with cap_cols[0]:
        st.markdown("**WORKING**")
        for item in cap["working"]:
            st.write("- " + item)
    with cap_cols[1]:
        st.markdown("**LIMITED**")
        for item in cap["limited"]:
            st.write("- " + item)
    with cap_cols[2]:
        st.markdown("**NOT AVAILABLE**")
        for item in cap["not_available"]:
            st.write("- " + item)
    st.caption("Additional specific gaps:")
    for w in [
        "shape.enclosed_shape_count, shape.repetition_score: not implemented (no shape detector)",
        "Concern profiles: implemented but disabled by design (CONCERNS_ENABLED = False)",
        "31 of 41 registry-v2 rules: allowed_output_level=disabled (no detector wired to any evaluator)",
        "detection_judge, relation_judge, language_judge: not_implemented (no underlying capability yet)",
    ]:
        st.write("- " + w)

    st.header("11. Research / Validation")
    st.caption(
        "Cross-case research and measurement-validation metrics -- NOT specific to this case, and "
        "never mixed into the case-level evidence/rule sections above. Collapsed by default so it is "
        "never confused with current-case analysis.")
    with st.expander("Phase 2B object-evidence pilot + measurement-validation status", expanded=False):
        render_phase2b_pilot_summary(st, ROOT)
        st.subheader("Measurement validation status")
        st.caption(
            "Validates measurement implementation only, not psychological validity. Synthetic-ground-truth "
            "and transformation-invariance results are cross-case (computed once on a fixed sample), not "
            "per-case."
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

    st.header("12. Sources and registry")
    st.subheader("Source catalog coverage")
    catalog_path = ROOT / "resources" / "psychology_sources" / "source_rule_catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        st.write(f"{catalog['entry_count']} total source rows ({catalog['entry_count_by_document']}), "
                 f"{catalog['candidate_rule_entry_count']} candidate-rule rows, "
                 f"{len(catalog['relationships'])} explicit relationships.")
        with st.expander("Show all source rows"):
            st.dataframe(catalog["entries"], width="stretch")
        with st.expander("Show all relationships (near_duplicate/expands/related_but_distinct)"):
            st.dataframe(catalog["relationships"], width="stretch")
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
             "threshold_source": r.get("threshold_source"), "dependency_group": ", ".join(r.get("dependency_group") or []),
             "validation_status": r.get("validation_status"), "confidence_ceiling": r.get("confidence_ceiling"),
             "page_reference_requirement": r.get("page_reference_requirement"),
             "feature_version": r.get("feature_version"),
             "expert_review_status": r.get("expert_review_status"),
             "known_robustness_limitations": r.get("known_robustness_limitations")}
            for r in registry_v2_doc["rules"]
        ], width="stretch")
        n_executable = sum(1 for r in registry_v2_doc["rules"] if r["allowed_output_level"] == "individual_heuristic_only")
        st.caption(f"{n_executable} of {registry_v2_doc['rule_count']} rules are actually executable "
                   f"(allowed_output_level=individual_heuristic_only); every other rule is disabled regardless of "
                   f"observability_class. See docs/STATIC_PROXY_RULE_POLICY.md.")
    else:
        st.info("rules_registry_v2.json not found.")

    st.header("13. Expert review")
    st.caption(
        "Kept strictly separate from the AI's own output -- submitting a review never edits detections.json, "
        "analysis.json, or any other AI-produced artifact (ai_output_preserved stays True always). Also kept "
        "separate from formal ground truth: nothing submitted here feeds any Phase 2C annotation/training pipeline.")
    review = load_review(case_dir)
    st.write(f"Status: **{review['status']}** -- {len(review['history'])} review action(s) so far.")
    if review["history"]:
        st.dataframe(review["history"], width="stretch")
    with st.form("expert_review_form"):
        reviewer_name = st.text_input("Reviewer name")
        action = st.selectbox("Action", REVIEW_ACTIONS)
        target_label = st.text_input(
            "Target finding label (for confirm/reject/rename/mark_missing_evidence -- "
            "e.g. a label shown in section 10's detections table)")
        new_label = st.text_input("New label (only used for the 'rename' action)")
        note = st.text_area("Note")
        submitted = st.form_submit_button("Submit review")
        if submitted:
            if not reviewer_name.strip():
                st.error("Reviewer name is required.")
            else:
                submit_review(case_dir, reviewer_name=reviewer_name.strip(), action=action,
                               target_label=target_label.strip() or None,
                               new_label=new_label.strip() or None, note=note.strip() or None)
                st.success("Review recorded.")
                st.rerun()
