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

import csv
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from doar import human_interaction as hi  # noqa: E402
from doar import case_interpretation as ci  # noqa: E402
from doar import case_presentation as cpres  # noqa: E402
from doar import formal_features as ff  # noqa: E402
from doar.formal_feature_review import (  # noqa: E402
    CLINICALLY_RELEVANT_VALUES, EXPERT_RULE_DECISION_VALUES, EXPERT_SUPPORT_LEVEL_VALUES,
    OBSERVATION_CORRECT_VALUES, OVERALL_SYNTHESIS_AGREEMENT_VALUES, load_formal_feature_review,
    submit_formal_feature_review,
)
from doar.case_artifacts import resolve_analysis_artifacts, resolve_artifact_path  # noqa: E402
from doar.case_output import write_versioned  # noqa: E402
from doar.expert_review import REVIEW_ACTIONS, load_review, submit_review  # noqa: E402
from doar.live_case_bundle import build_live_case_bundle  # noqa: E402
from doar.production_config import resolve_production_config  # noqa: E402
from doar.page_reference import (  # noqa: E402
    PARENT_PAGE_DECLARATION_CHOICES, PARENT_PAGE_DECLARATION_LABELS,
    describe_declaration_choice, user_page_declaration_from_choice,
)
from doar.parent_view import (  # noqa: E402
    capability_status, disclaimer, plain_language_observations, plain_language_rule_rows,
    parent_care_suggestions,
)
from doar.phase2c7 import detector_policy as _pol  # noqa: E402
from doar.phase2c7 import runtime as _detector_runtime  # noqa: E402
from doar.profile import ChildProfile, ALLOWED_AGE_RANGES, load_profile, save_profile  # noqa: E402
from doar.psychologist_feedback import (  # noqa: E402
    FEEDBACK_CATEGORIES, FEEDBACK_VERDICTS, export_all_feedback, load_feedback, submit_categorized_feedback,
)
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

@st.cache_resource(show_spinner=False)
def _load_hi_providers() -> dict:
    """Milestone 1: resolve the Human Interaction Layer's optional Gemini
    providers ONCE per server process -- each already degrades to a safe
    deterministic/disabled default when GEMINI_API_KEY/google-genai is
    unavailable (doar.human_interaction.resolve_default_*)."""
    return {
        "answer_provider": hi.resolve_default_answer_provider(),
        "judge": hi.resolve_default_judge(),
        "research_provider": hi.resolve_default_research_provider(),
        "visual_recheck_provider": hi.resolve_default_visual_recheck_provider(),
        "visual_consistency_judge": hi.resolve_default_visual_consistency_judge(),
        "global_observer": hi.resolve_default_global_observer(),
    }


def _detections_cache_token(case_dir_str: str) -> float:
    """detections.json's own mtime, used as part of the bundle cache key --
    so a bundle built before deep visual analysis ran is never silently
    reused (with no entities) after the "Run deep visual analysis" button
    writes real entities to disk. 0.0 (a stable, cacheable value) when the
    file does not exist yet."""
    path = Path(case_dir_str) / "detections.json"
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _load_live_case_bundle(case_dir_str: str, _detections_token: float) -> dict:
    """Cached per (case_dir, detections.json mtime) -- synthesize_drawing()
    is pure/local (no network), but rebuilding it on every Streamlit rerun
    is wasteful. Keying on the detections mtime (rather than case_dir
    alone) is what makes the deep-scan button's new entities actually
    visible without a stale cache hit."""
    return build_live_case_bundle(case_dir_str)


CASES_DIR = ROOT / "outputs" / "prototype_cases"


# NOTE (Supervisor Demo, feature/supervisor-demo-v2): `_build_full_interpretation`
# and `_render_ask_doar` were moved up here (from their original spot just before
# the Parent/Psychologist/Technical tabs) so the new Supervisor Demo mode -- which
# must be able to render BEFORE the legacy sidebar upload/`case_dir` gate runs --
# can call them too. Neither function's body changed; both are still shared,
# unchanged, by the legacy Parent/Psychologist views below.
def _build_full_interpretation(bundle: dict, case_dir_str: str) -> ci.CaseInterpretation:
    """Shared by Parent and Psychologist views -- rebuilds CaseInterpretation
    (a cheap, deterministic, pure computation) using whatever expensive,
    session-cached results already exist for this case (the Run Full
    Analysis button's Gemini global observation, the standalone Visual
    Consistency Judge audit). Never itself triggers a network call or a
    model load -- those only happen inside their own explicit button
    handlers."""
    full_analysis = st.session_state.get(f"full_analysis_{case_dir_str}")
    gemini_observation = full_analysis.get("gemini_observation") if full_analysis else None
    gemini_candidates = (gemini_observation.get("candidate_concern_hypotheses")
                        if gemini_observation else None)
    visual_audit = st.session_state.get(f"visual_audit_{case_dir_str}")
    return ci.build_case_interpretation(
        bundle, visual_judge_audit=visual_audit,
        gemini_global_observation=gemini_observation, gemini_concern_candidates=gemini_candidates)


def _render_ask_doar(*, audience: str, chat_key: str, widget_prefix: str, is_ar: bool,
                      allow_open_vocab_recheck: bool = True) -> None:
    """Shared Ask DOAR chat box -- reused, unchanged, by both the Parent
    and Psychologist views (Milestone 1's human_interaction.answer_question
    pipeline; only `audience` differs). `chat_key` intentionally keeps the
    Parent view's original `hi_chat_{case_dir}` key so Technical view's
    existing provenance table keeps working unchanged; the Psychologist
    view uses a separate key so the two conversations never mix. Reads the
    module-level `case_dir`/`detections` globals at CALL time (not def
    time) -- the Supervisor Demo view sets these itself before calling in,
    exactly like the legacy sidebar/case-load flow does further down."""
    st.subheader("اسأل DOAR" if is_ar else "Ask DOAR")
    st.caption(
        "الإجابات مبنية حصراً على أدلة هذه الحالة المحفوظة ومصادر DOAR المعتمدة؛ يتم التحقق منها آلياً قبل عرضها."
        if is_ar else
        "Answers are grounded in this case's saved evidence and DOAR's approved sources, and are "
        "verified before being shown."
    )
    st.session_state.setdefault(chat_key, [])
    question = st.text_input("سؤال عن هذه الحالة" if is_ar else "Ask a question about this case",
                              key=f"{widget_prefix}_chat_q")
    if st.button("إرسال" if is_ar else "Send", key=f"{widget_prefix}_chat_send") and question:
        history = [{"role": turn["role"], "content": turn["content"]} for turn in st.session_state[chat_key]]
        with st.spinner("DOAR يراجع أدلة الحالة..." if is_ar else "DOAR is reviewing the case evidence..."):
            try:
                bundle = _load_live_case_bundle(str(case_dir), _detections_cache_token(str(case_dir)))
                providers = _load_hi_providers()
                # Performance fix: only pass a real on-demand open-vocab predict_fn
                # (which would load Grounding DINO/OWLv2 on first use) once deep
                # visual analysis has actually been run for this case -- otherwise
                # an ordinary Ask DOAR question must never trigger a multi-minute
                # model load; answer_question degrades to its existing, safe
                # "on-demand re-check unavailable" wording when this is None.
                # allow_open_vocab_recheck=False (Supervisor Demo) skips this
                # entirely -- Grounding-DINO/OWLv2 can be a slow, first-use-only
                # multi-minute load, which a reliability-first prepared demo
                # must never risk triggering from an ordinary Ask DOAR question.
                open_vocab_fn = (_load_model_predict_fns().get("open_vocab_query")
                                  if allow_open_vocab_recheck and detections.get("status") == "available"
                                  else None)
                answer = hi.answer_question(
                    question, bundle, audience=audience,
                    open_vocab_predict_fn=open_vocab_fn,
                    visual_recheck_provider=providers["visual_recheck_provider"],
                    external_research_provider=providers["research_provider"],
                    answer_provider=providers["answer_provider"], judge=providers["judge"],
                    conversation_history=history,
                )
                answer_text = answer.answer
                provenance = []
                if answer.used_visual_recheck:
                    provenance.append("on-demand visual re-check (experimental, not part of the original analysis)")
                if answer.used_external_research:
                    provenance.append("external research (clearly separate from DOAR's own case evidence)")
                turn_meta = {"judge_verdict": answer.judge_verdict, "judge_mode": answer.judge_mode,
                             "answer_provider": answer.answer_provider, "judge_details": answer.judge_details}
            except Exception as exc:  # noqa: BLE001 -- never leak a raw traceback/key to the user
                answer_text = (
                    "محادثة الذكاء الاصطناعي غير متاحة مؤقتاً. يبقى تحليل DOAR المُعَدّ مسبقاً متاحاً." if is_ar else
                    "AI conversation is temporarily unavailable. The prepared DOAR analysis remains available.")
                provenance = []
                turn_meta = {"judge_verdict": "error", "judge_mode": "n/a", "answer_provider": "n/a", "judge_details": {}}
                st.error(hi.sanitize_error_text(exc))
        st.session_state[chat_key].append({"role": "user", "content": question})
        st.session_state[chat_key].append(
            {"role": "assistant", "content": answer_text, "provenance": provenance, **turn_meta})
    for turn in reversed(st.session_state[chat_key]):
        who = ("أنت" if is_ar else "You") if turn["role"] == "user" else ("الرد" if is_ar else "Answer")
        st.markdown(f"**{who}:** {turn['content']}")
        if turn.get("provenance"):
            st.caption("; ".join(turn["provenance"]))
    if st.session_state[chat_key]:
        st.caption(hi.FOOTER_DISCLAIMER)


# ===========================================================================
# SUPERVISOR DEMO (feature/supervisor-demo-v2): a separate, polished,
# presentation-focused view built for a ~5-7 minute supervisor walkthrough.
# Reuses the SAME pipeline outputs, the SAME Gemini providers
# (human_interaction.py / _load_hi_providers / _render_ask_doar above),
# case_interpretation.py (READ-ONLY -- never modified), and
# case_presentation.py's bilingual label functions as the existing
# Parent/Psychologist/Technical views -- nothing here re-implements
# evidence/rule logic, it only re-presents it. Defaults to three prepared,
# already-analyzed cases under outputs/prototype_cases/ so opening a case
# NEVER calls Gemini and NEVER re-runs the pipeline (see _sd_load_case_docs
# / _sd_drawing_analysis below). The legacy Parent/Psychologist/Technical
# tabs remain fully intact and reachable via the sidebar "View" switch.
# ===========================================================================
SUPERVISOR_DEMO_CASES: dict[str, dict] = {
    "case1": {
        "case_id": "a103_1787479142", "role": "complete",
        "title": {"en": "Full Analysis Walkthrough", "ar": "جولة كاملة في التحليل"},
        "purpose": {
            "en": "The main example: DOAR's full pipeline on one drawing, from observation to governed synthesis.",
            "ar": "المثال الرئيسي: خط أنابيب DOAR الكامل على رسمة واحدة، من الملاحظة إلى التوليف المحكوم.",
        },
    },
    "case2": {
        "case_id": "a111_1787479358", "role": "contrast",
        "title": {"en": "A Different Drawing Profile", "ar": "ملف رسمة مختلف"},
        "purpose": {
            "en": "A visually different drawing: different colours, detected content and expressive profile.",
            "ar": "رسمة مختلفة بصرياً: ألوان ومحتوى مكتشف وملف تعبيري مختلف.",
        },
    },
    "case3": {
        "case_id": "h38_1786305027", "role": "governance",
        "title": {"en": "Uncertainty and Governance", "ar": "عدم اليقين والحوكمة"},
        "purpose": {
            "en": "Why verification matters: weak, contradicted or missing evidence, handled cautiously.",
            "ar": "لماذا يهم التحقق: أدلة ضعيفة أو متعارضة أو ناقصة، تُعالَج بحذر.",
        },
    },
}

_SD_STATUS_COLOR = {"verified": "green", "uncertain": "orange", "rejected": "red",
                     "objective": "blue", "evidence": "violet", "neutral": "gray"}
_SD_FINDING_STATUS_KIND = {
    "VALIDATED": "verified", "EXPERIMENTAL": "uncertain", "UNKNOWN": "uncertain",
    "DISABLED_FOR_RULES": "rejected",
}


def _sd_status_badge(container, text: str, kind: str) -> None:
    container.badge(text, color=_SD_STATUS_COLOR.get(kind, "gray"))


@st.cache_data(show_spinner=False)
def _sd_formal_feature_labels() -> dict:
    """feature_id -> display_name, from the committed
    FORMAL_FEATURE_DEFINITION_TABLE.csv -- Section 3's source of
    human-readable OBJECTIVE_ONLY measurement names (never invented here)."""
    path = ROOT / "FORMAL_FEATURE_DEFINITION_TABLE.csv"
    labels: dict = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                labels[row["feature_id"]] = row["display_name"]
    return labels


@st.cache_data(show_spinner=False)
def _sd_rules_registry_index() -> dict:
    """rule_id -> registry row, from the frozen, active rules_registry_v2.json
    (the real governed 41-rule corpus, 10 enabled at baseline) -- this is
    deliberately NOT MASTER_RULE_FEATURE_REGISTRY_V3.json, which is a
    225-source-entry/206-canonical-observable/21-evidence-family RESEARCH
    organization, not an active psychological rule set (see Section 4)."""
    path = ROOT / "resources" / "psychology_sources" / "rules_registry_v2.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {r["rule_id"]: r for r in doc.get("rules", [])}


@st.cache_data(show_spinner=False)
def _sd_load_case_docs(case_dir_str: str) -> dict:
    """Every optional per-case JSON artifact this view might use, tolerating
    absence (older/lighter cases) -- never a raw FileNotFoundError/
    JSONDecodeError reaching the UI. Read-only: never triggers analysis,
    a model load, or a network call. Cached by case_dir string only --
    these are frozen, prepared demo cases that never change mid-demo."""
    case_dir = Path(case_dir_str)

    def _jload(name: str):
        p = case_dir / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- corrupt/partial cache file, never a crash
            return None

    analysis = _jload("analysis.json")
    return {
        "case_dir": case_dir,
        "analysis": analysis,
        "judges": _jload("judges.json") or {},
        "detections": _jload("detections.json") or {"status": "unavailable"},
        "objective_features": _jload("objective_features.json"),
        "formal_features": _jload("formal_features.json"),
        "structured": _jload("structured_analysis.json"),
        "page_reference": ((analysis or {}).get("page_reference")) or {},
    }


def _sd_normalized_image(docs: dict) -> str | None:
    analysis = docs.get("analysis") or {}
    value = (analysis.get("artifacts") or {}).get("normalized_image")
    if value:
        p = resolve_artifact_path(docs["case_dir"], value)
        if p.exists():
            return str(p)
    image_path = analysis.get("image_path")
    if image_path:
        p = docs["case_dir"] / Path(image_path).name
        if p.exists():
            return str(p)
    return None


def _sd_case_metrics(docs: dict) -> dict:
    analysis = docs.get("analysis") or {}
    det = docs.get("detections") or {}
    findings = det.get("findings", []) if det.get("status") == "available" else []
    n_verified = sum(1 for f in findings if f.get("validation_status") == "VALIDATED")
    ff_doc = docs.get("formal_features")
    n_measurements = (sum(1 for v in ff_doc.get("features", {}).values() if not v.get("missing"))
                       if ff_doc else 0)
    rules = analysis.get("rule_evaluations", [])
    applicable = [r for r in rules if r.get("status") in ("weak_support", "not_matched")]
    reg = _sd_rules_registry_index()
    families = {reg[r["rule_id"]]["evidence_family"] for r in applicable
                if r["rule_id"] in reg and reg[r["rule_id"]].get("evidence_family")}
    return {"n_verified": n_verified, "n_measurements": n_measurements,
            "n_applicable_rules": len(applicable), "n_families": len(families),
            "applicable_rules": applicable}


def _sd_inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #FAF6EF; }
        [data-testid="stSidebar"] { background-color: #F3ECE1; }
        .doar-hero-title { font-size: 3.0rem; font-weight: 700; letter-spacing: 0.05em;
            color: #4A4238; margin-bottom: 0.1em; }
        .doar-hero-sub { font-size: 1.15rem; color: #5B7B6F; margin: 0.1em 0; }
        .doar-hero-sub2 { font-size: 1.0rem; color: #6B6255; margin: 0.1em 0 0.6em 0; }
        .doar-kicker { display: inline-block; font-size: 0.75rem; letter-spacing: 0.08em;
            text-transform: uppercase; color: #7C6F9B; background: #EDE7F6;
            border-radius: 999px; padding: 0.15em 0.9em; margin-bottom: 0.6em; }
        .doar-pipeline { font-size: 1.05rem; font-weight: 600; color: #4A4238;
            background: linear-gradient(90deg, #E9F1EC 0%, #EDE7F6 50%, #FBEFE9 100%);
            border: 1px solid #E4DCCB; border-radius: 14px; padding: 0.9em 1.2em;
            text-align: center; margin: 0.8em 0 1.2em 0; }
        .doar-note { font-size: 0.85rem; color: #7A7264; border-left: 3px solid #C9BFA8;
            padding-left: 0.7em; margin-top: 0.4em; }
        div[data-testid="stButton"] button { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sd_pipeline_banner(language: str) -> None:
    ar = language == "ar"
    text = ("لاحظ &larr; تحقّق &larr; قِس &larr; طابِق الأدلة &larr; استدلّ &larr; اشرح" if ar else
            "Observe &rarr; Verify &rarr; Measure &rarr; Match Evidence &rarr; Reason &rarr; Explain")
    st.markdown(f'<div class="doar-pipeline">{text}</div>', unsafe_allow_html=True)


def _sd_home(language: str) -> None:
    ar = language == "ar"
    st.markdown('<div class="doar-kicker">' +
                ("نموذج بحث لرسالة ماجستير" if ar else "Master's Research Prototype") + '</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="doar-hero-title">DOAR</div>', unsafe_allow_html=True)
    st.markdown('<div class="doar-hero-sub">' +
                ("رصد وتحليل واستدلال في رسومات الأطفال" if ar else
                 "Drawing Observation, Analysis and Reasoning") + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="doar-hero-sub2">' +
                ("تحليل قابل للتتبع ومبني على الأدلة لرسومات الأطفال." if ar else
                 "Evidence-grounded and traceable analysis of children's drawings.") + '</div>',
                unsafe_allow_html=True)
    st.caption("DOAR نموذج بحثي ودعم مهني غير تشخيصي." if ar else
               "DOAR is a non-diagnostic research and professional-support prototype.")
    st.write("")
    st.markdown(
        ("لا يكتفي DOAR بسؤال الذكاء الاصطناعي التوليدي عن تفسير رسمة، بل يجمع بين النماذج البصرية والقياسات "
         "الموضوعية والتحقق والأدلة العلمية والاستدلال المحكوم." if ar else
         "DOAR does not simply ask generative AI to interpret a drawing. It combines visual models, objective "
         "measurements, verification, scientific evidence and governed reasoning.")
    )
    _sd_pipeline_banner(language)

    st.subheader("جرّب عرضاً توضيحياً مُعداً مسبقاً" if ar else "TRY A PREPARED DEMONSTRATION")
    cols = st.columns(3)
    for col, (case_key, cfg) in zip(cols, SUPERVISOR_DEMO_CASES.items()):
        case_dir = CASES_DIR / cfg["case_id"]
        with col:
            with st.container(border=True):
                docs = _sd_load_case_docs(str(case_dir))
                img = _sd_normalized_image(docs)
                if img:
                    st.image(img, width="stretch")
                else:
                    st.caption("(الصورة غير متاحة)" if ar else "(image unavailable)")
                st.markdown(f"**{cfg['title'][language]}**")
                st.caption(cfg["purpose"][language])
                if st.button(("افتح التحليل" if ar else "Open Analysis"),
                             key=f"sd_open_{case_key}", width="stretch", type="primary"):
                    st.session_state["supervisor_active_case"] = case_key
                    st.session_state["supervisor_section"] = "Drawing Analysis"
                    st.rerun()
    st.divider()
    st.caption(
        ("هذه حالات تطويرية مُعدة مسبقاً؛ لا تُستخدم بيانات \"الاختبار المُقفَل\" أبداً في هذا العرض." if ar else
         "These are prepared development-only cases; \"Locked Test\" data is never used in this demonstration.")
    )


def _sd_render_evidence_trace(docs: dict, rule_row: dict, reg_row: dict, language: str) -> None:
    """The evidence-trace chain: Observed feature -> Source/provider ->
    Verification -> Canonical observable -> Eligible governed rule ->
    Evidence source -> Evidence family -> Use in synthesis. Built entirely
    from real fields already on `rule_row`/`reg_row`/the case's own
    detections.json (via build_rule_evidence_trace, unchanged) -- nothing
    invented."""
    ar = language == "ar"
    det = docs.get("detections") or {}
    findings = det.get("findings", []) if det.get("status") == "available" else []
    visual_rows = []
    if findings:
        try:
            trace = build_rule_evidence_trace([VisualFinding.from_dict(f) for f in findings])
            visual_rows = [t for t in trace if t["rule_id"] == rule_row["rule_id"]]
        except Exception:  # noqa: BLE001 -- trace is descriptive only, never fatal
            visual_rows = []
    if visual_rows:
        t = visual_rows[0]
        source_txt = f"{t['detector']} ({t['source']})"
        verification_txt = t["validation_status"]
    else:
        source_txt = "استخلاص سمات حتمي (analysis.py)" if ar else "deterministic feature extraction (analysis.py)"
        verification_txt = "حتمي (بدون نموذج تعلّم)" if ar else "deterministic (no learned model)"
    used_txt = (
        ("استُخدم في التوليف — طابق قاعدة محكومة (دعم ضعيف)" if ar else
         "Used in synthesis -- matched a governed rule (weak support)")
        if rule_row["status"] == "weak_support" else
        ("لم يُستخدم — لم يُلاحظ هذا النمط في هذه الرسمة" if ar else
         "Not used in synthesis -- this pattern was not observed in this drawing")
    )
    chain = [
        ("الميزة المُلاحَظة" if ar else "Observed feature",
         cpres.observation_label(reg_row.get("observable", rule_row["rule_id"]), language)),
        ("المصدر/المزوّد" if ar else "Source / provider", source_txt),
        ("التحقق" if ar else "Verification", verification_txt),
        ("المتغير القابل للملاحظة" if ar else "Canonical observable", reg_row.get("observable", "n/a")),
        ("القاعدة المحكومة المؤهلة" if ar else "Eligible governed rule",
         f"{rule_row['rule_id']} ({reg_row.get('target_construct', 'n/a')})"),
        ("مصدر الدليل" if ar else "Evidence source",
         f"{reg_row.get('source_document', 'n/a')} (p.{reg_row.get('source_page', 'n/a')})"),
        ("أسرة الأدلة" if ar else "Evidence family",
         cpres.evidence_family_label(reg_row.get("evidence_family", ""), language)
         if reg_row.get("evidence_family") else "n/a"),
        ("الاستخدام في التوليف" if ar else "Use in synthesis", used_txt),
    ]
    for k, v in chain:
        st.write(f"**{k}:** {v}")


def _sd_drawing_analysis(language: str, case_key: str) -> None:
    ar = language == "ar"
    cfg = SUPERVISOR_DEMO_CASES[case_key]
    case_dir_local = CASES_DIR / cfg["case_id"]
    docs = _sd_load_case_docs(str(case_dir_local))
    analysis = docs.get("analysis")
    if not analysis:
        st.error("تعذّر تحميل بيانات هذه الحالة." if ar else "Could not load this case's data.")
        return

    # Reuse _render_ask_doar (defined earlier, unchanged) below by setting
    # the SAME module-level globals it reads at call time -- the legacy
    # sidebar/case-load flow sets these identically further down, and this
    # branch always st.stop()s before that legacy code ever runs, so there
    # is no collision within one script run.
    global case_dir, detections
    case_dir = docs["case_dir"]
    detections = docs["detections"]

    bundle = _load_live_case_bundle(str(case_dir), _detections_cache_token(str(case_dir)))
    interp = _build_full_interpretation(bundle, str(case_dir))
    metrics = _sd_case_metrics(docs)
    det = docs["detections"]
    findings = det.get("findings", []) if det.get("status") == "available" else []

    st.markdown(f"## {cfg['title'][language]}")
    top_left, top_right = st.columns([3, 2])
    with top_left:
        img = _sd_normalized_image(docs)
        if img:
            st.image(img, width="stretch",
                      caption=("الرسمة الأصلية" if ar else "Original drawing"))
        else:
            st.info("صورة الرسمة الأصلية غير متاحة لهذه الحالة." if ar else
                    "The original drawing image is unavailable for this case.")
    with top_right:
        with st.container(border=True):
            st.markdown("**" + ("ملخص التحليل" if ar else "Analysis summary") + "**")
            st.write(("✓ " if ar else "✓ ") + ("التحليل متاح" if ar else "Analysis available"))
            st.write(f"{'الملاحظات المتحقق منها' if ar else 'Verified observations'}: {metrics['n_verified']}")
            if metrics["n_measurements"]:
                st.write(f"{'القياسات الموضوعية' if ar else 'Objective measurements'}: {metrics['n_measurements']}")
            st.write(f"{'القواعد المحكومة القابلة للتطبيق' if ar else 'Applicable governed rules'}: "
                     f"{metrics['n_applicable_rules']}")
            if metrics["n_families"]:
                st.write(f"{'أسر الأدلة' if ar else 'Evidence families'}: {metrics['n_families']}")

    st.markdown("---")
    st.markdown("#### " + ("١. الملف التعبيري — ليس تشخيصاً" if ar else
                            "1. EXPRESSIVE CONTENT PROFILE — Not diagnosis."))
    profile = interp.expressive_profile
    if profile.availability == "available" and profile.base_emotion_probabilities:
        prob_cols = st.columns(len(profile.base_emotion_probabilities))
        for c, (k, v) in zip(prob_cols, profile.base_emotion_probabilities.items()):
            c.metric(cpres.emotion_label(k, language), f"{v:.0%}")
        if profile.richer_descriptors:
            descriptors_txt = ("، " if ar else ", ").join(
                cpres.descriptor_label(d.label, language) for d in profile.richer_descriptors)
            st.caption(("أوصاف أغنى: " if ar else "Richer descriptors: ") + descriptors_txt)
        st.markdown(
            '<div class="doar-note">' +
            ("يصف هذا النموذج المحتوى التعبيري في الرسمة ولا يُشخِّص حالة نفسية." if ar else
             "This model describes expressive content in the drawing and does not diagnose a "
             "psychological condition.") + '</div>', unsafe_allow_html=True)
    else:
        reason = f" ({profile.unavailable_reason})" if profile.unavailable_reason else ""
        st.caption(("نموذج المحتوى التعبيري غير متاح لهذه الحالة." if ar else
                    "The expressive-content model is unavailable for this case.") + reason)

    st.markdown("#### " + ("٢. ما لاحظه DOAR" if ar else "2. WHAT DOAR OBSERVED"))
    comp_obs = plain_language_observations(analysis, language, page_reference=docs["page_reference"])
    if findings:
        st.markdown("**" + ("الأجسام والعناصر البصرية" if ar else "Objects / Visual elements") + "**")
        status_order = {"VALIDATED": 0, "EXPERIMENTAL": 1, "UNKNOWN": 2, "DISABLED_FOR_RULES": 3}
        shown = sorted(findings, key=lambda f: (status_order.get(f["validation_status"], 9),
                                                  -f["confidence"]))[:10]
        status_word = {"verified": ("تحقّق" if ar else "VERIFIED"), "uncertain": ("غير مؤكد" if ar else "UNCERTAIN"),
                        "rejected": ("مرفوض" if ar else "REJECTED")}
        for f in shown:
            row_cols = st.columns([3, 1.4, 2])
            label_txt = (f.get("free_form_label") or f["label"]).replace("_", " ")
            row_cols[0].write("- " + label_txt)
            kind = _SD_FINDING_STATUS_KIND.get(f["validation_status"], "uncertain")
            _sd_status_badge(row_cols[1], status_word[kind], kind)
            row_cols[2].caption(f["detector"])
        if len(findings) > 10:
            st.caption(f"+ {len(findings) - 10} " + ("المزيد (انظر الأثر التقني)." if ar else
                                                       "more (see Technical Trace)."))
        with st.expander("المعرّفات الخام" if ar else "Raw identifiers", expanded=False):
            st.dataframe([{"label": f["label"], "validation_status": f["validation_status"],
                            "confidence": round(f["confidence"], 3)} for f in findings],
                         width="stretch", hide_index=True)
    if comp_obs:
        st.markdown("**" + ("التكوين والألوان وجودة الصورة" if ar else
                             "Composition, colours and image quality") + "**")
        for o in comp_obs:
            c1, c2 = st.columns([5, 1])
            c1.write("- " + o["text"])
            _sd_status_badge(c2, "موضوعي" if ar else "OBJECTIVE", "objective")
    if not findings and not comp_obs:
        st.caption("لم يُلاحظ شيء يمكن الإبلاغ عنه في هذه الرسمة." if ar else
                   "Nothing reportable was observed for this drawing.")
    if interp.gemini_candidates:
        st.markdown("**" + ("ملاحظات دلالية بالذكاء الاصطناعي (مُرشَّحة)" if ar else
                             "AI semantic observations (candidates)") + "**")
        gc_kind_map = {"SUPPORTED": "verified", "PARTIALLY_SUPPORTED": "uncertain", "UNSUPPORTED": "rejected",
                       "INSUFFICIENT_EVIDENCE": "uncertain", "NOT_CURRENTLY_ASSESSED": "uncertain"}
        for gc in interp.gemini_candidates:
            kind = gc_kind_map.get(gc.verification_status, "uncertain")
            c1, c2 = st.columns([5, 1])
            c1.write(f"- {cpres.domain_label(gc.domain, language)}: {gc.gemini_reason}")
            _sd_status_badge(c2, cpres.verification_status_label(gc.verification_status, language).upper(), kind)
            if kind in ("rejected", "uncertain"):
                st.caption(("مرشّح من الذكاء الاصطناعي فقط — لم يُستخدم في التفسير النهائي." if ar else
                            "AI candidate only — NOT USED IN FINAL INTERPRETATION."))

    st.markdown("#### " + ("٣. القياسات الموضوعية" if ar else "3. OBJECTIVE MEASUREMENTS"))
    ff_doc = docs.get("formal_features")
    obj_doc = docs.get("objective_features")
    labels_map = _sd_formal_feature_labels()
    shown_ids = ["colour.diversity", "colour.chromatic_coverage", "line.darkness_mean", "line.continuity",
                 "line.orientation_entropy", "symmetry.bilateral", "scene.detail_density"]
    rows = []
    if obj_doc:
        for feat in obj_doc.get("features", []):
            if feat["feature_id"] == "segmentation.foreground_coverage" and not feat.get("missing"):
                rows.append((("تغطية الرسمة" if ar else "Drawing coverage"), f"{feat['value']:.0%}"))
    if ff_doc:
        feats = ff_doc.get("features", {})
        for fid in shown_ids:
            v = feats.get(fid)
            if v and not v.get("missing"):
                rows.append((labels_map.get(fid, fid), f"{v['value']:.2f}"))
    if rows:
        m_cols = st.columns(4)
        for i, (label, value) in enumerate(rows):
            with m_cols[i % 4]:
                st.metric(label, value)
                st.caption("OBJECTIVE")
        st.markdown(
            '<div class="doar-note">' +
            ("هذه قياسات بصرية موضوعية. لا يُسند إليها معنى نفسي تلقائياً." if ar else
             "These are objective visual measurements. Psychological meaning is not assigned automatically.") +
            '</div>', unsafe_allow_html=True)
    else:
        st.caption("لا تتوفر قياسات موضوعية لهذه الحالة." if ar else
                   "No objective measurements are available for this case.")

    st.markdown("#### " + ("٤. الأدلة/القواعد المحكومة القابلة للتطبيق" if ar else
                            "4. APPLICABLE GOVERNED EVIDENCE / RULES"))
    st.markdown(
        '<div class="doar-note">' +
        ("يعرض هذا القسم فقط القواعد المحكومة المؤهلة فعلياً لهذه الرسمة من السجل النشط (النواة الأصلية "
         "المكونة من 41 قاعدة، منها 10 مفعّلة أساساً) — وليس سجل الأبحاث الكامل (225 إدخال مصدر/206 متغير "
         "قابل للملاحظة/21 أسرة أدلة، وهو تنظيم بحثي وليس 225 قاعدة نفسية مفعّلة)." if ar else
         "This section shows only the governed rules actually eligible (evaluated) for THIS drawing, from "
         "the active governed rule set (the original 41-rule corpus, 10 enabled at baseline) -- never the "
         "full research registry (225 source entries / 206 canonical observables / 21 evidence families is "
         "a research organization, not 225 active psychological rules).") + '</div>', unsafe_allow_html=True)
    reg = _sd_rules_registry_index()
    applicable = metrics["applicable_rules"]
    if applicable:
        for r in applicable:
            reg_row = reg.get(r["rule_id"], {})
            status_kind = "verified" if r["status"] == "weak_support" else "rejected"
            with st.container(border=True):
                head_cols = st.columns([4, 1.4])
                obs_label = cpres.observation_label(reg_row.get("observable", r["rule_id"]), language)
                head_cols[0].markdown(f"**{obs_label}**")
                status_txt = (("مطابق" if ar else "MATCHED") if r["status"] == "weak_support" else
                              ("غير مطابق" if ar else "NOT MATCHED"))
                _sd_status_badge(head_cols[1], status_txt, status_kind)
                why_txt = (
                    ("تم التقييم: لوحظ هذا النمط." if ar else "Evaluated: this pattern WAS observed.")
                    if r["status"] == "weak_support" else
                    ("تم التقييم: تم فحص هذا النمط ولم يُلاحظ." if ar else
                     "Evaluated: this pattern was checked and was NOT observed.")
                )
                st.caption(("سبب الأهلية: " if ar else "Why eligible: ") + why_txt)
                fam = reg_row.get("evidence_family")
                fam_cols = st.columns(3)
                fam_cols[0].write(("أسرة الأدلة: " if ar else "Evidence family: ") +
                                  (cpres.evidence_family_label(fam, language) if fam else "n/a"))
                fam_cols[1].write(("الحالة الحاكمة: " if ar else "Governance status: ") +
                                  (reg_row.get("allowed_output_level") or "n/a"))
                src, page = reg_row.get("source_document"), reg_row.get("source_page")
                fam_cols[2].write(("المصدر: " if ar else "Source: ") + (f"{src} (p.{page})" if src else "n/a"))
                cautious = (
                    ("ملاحظة رصدية محتملة بثقة منخفضة، وليست حقيقة أو تشخيصاً." if ar else
                     "A possible, low-confidence observational pattern -- not a fact or a diagnosis.")
                    if r["status"] == "weak_support" else
                    ("غياب هذا النمط هنا ليس بحد ذاته دليلاً ذا دلالة." if ar else
                     "The absence of this pattern here is not itself meaningful evidence of anything.")
                )
                st.caption(("تفسير حذر: " if ar else "Cautious interpretation: ") + cautious)
                with st.expander(("عرض أثر الدليل" if ar else "View evidence trace"), expanded=False):
                    _sd_render_evidence_trace(docs, r, reg_row, language)
    else:
        st.caption("لا توجد قاعدة محكومة قابلة للتطبيق لهذه الحالة." if ar else
                   "No governed rule was applicable (evaluated) for this case.")

    st.markdown("#### " + ("٥. الاتفاق وعدم اليقين" if ar else "5. AGREEMENT & UNCERTAINTY"))
    a_cols = st.columns(4)
    n_agree = 1 if interp.consistency_status == "CONSISTENT" else 0
    n_uncertain = len(interp.missing_information)
    n_rejected = sum(1 for f in findings if f.get("validation_status") == "DISABLED_FOR_RULES")
    n_insufficient = 0 if profile.availability == "available" else 1
    a_cols[0].metric("✓ " + ("متفق" if ar else "Verified agreement"), n_agree)
    a_cols[1].metric("⚠ " + ("غير مؤكد" if ar else "Uncertain evidence"), n_uncertain)
    a_cols[2].metric("✕ " + ("مرفوض" if ar else "Rejected"), n_rejected)
    a_cols[3].metric("? " + ("أدلة غير كافية" if ar else "Insufficient evidence"), n_insufficient)
    if interp.contradictions:
        st.markdown("**" + ("تعارضات" if ar else "Contradictions") + "**")
        for c in interp.contradictions:
            st.warning(c)

    st.markdown("#### " + ("٦. التوليف المحكوم" if ar else "6. GOVERNED SYNTHESIS"))
    with st.container(border=True):
        notable = sorted((c for c in interp.concern_domains if c.support_level in ("WEAK", "MODERATE", "STRONG")),
                          key=lambda c: ("STRONG", "MODERATE", "WEAK").index(c.support_level))
        st.markdown("**" + ("ما لاحظه DOAR" if ar else "What DOAR observed") + "**")
        st.write(cpres.synthesis_summary(interp, language).split(". ")[0] + ".")
        st.markdown("**" + ("ما تقاربت عليه الأدلة" if ar else "What evidence converged") + "**")
        if notable:
            for c in notable:
                st.write(f"- {cpres.domain_label(c.domain, language)}: {cpres.concern_domain_reason(c, language)}")
        else:
            st.write(("لم تتقارب أي أسر أدلة مستقلة على مجال قلق لهذه الرسمة." if ar else
                     "No independent evidence families converged on a concern domain for this drawing."))
        st.markdown("**" + ("ما قد يستحق اهتماماً مهنياً" if ar else "What may merit professional attention") + "**")
        if notable:
            for c in notable:
                st.write(f"- {cpres.domain_label(c.domain, language)} ({cpres.support_level_label(c.support_level, language)})")
        else:
            st.write(("لا شيء في هذه الرسمة يصل حالياً إلى مستوى دعم يستدعي اهتماماً مهنياً بناءً على أدلة "
                     "DOAR وحدها." if ar else
                     "Nothing in this drawing currently reaches a support level that would merit "
                     "professional attention on DOAR's own evidence."))
        st.markdown("**" + ("ما يبقى غير مؤكد" if ar else "What remains uncertain") + "**")
        if interp.missing_information:
            for m in interp.missing_information[:5]:
                st.write("- " + m)
        else:
            st.write(("لم تُسجَّل أوجه عدم يقين محددة بخلاف القيود المعتادة أدناه." if ar else
                     "No specific uncertainty was flagged beyond the usual limitations noted throughout."))
        st.markdown("**" + ("ما لا يمكن الخلوص إليه" if ar else "What cannot be concluded") + "**")
        st.write(
            ("لا تُقدّم هذه الملاحظات تشخيصاً سريرياً؛ فهي تدعم المراجعة المهنية ولا تحل محلها." if ar else
             "These findings support professional review and do not provide a clinical diagnosis.")
        )

    st.markdown("---")
    st.markdown("#### " + ("اسأل DOAR" if ar else "Ask DOAR"))
    st.caption("اسأل عن هذه الرسمة أو الأدلة أو كيف توصّل DOAR إلى استنتاجه." if ar else
               "Ask about this drawing, the evidence, or how DOAR reached its conclusion.")
    suggestions_en = ["What did DOAR notice?", "Why did DOAR reach this conclusion?",
                       "Which observations were verified?", "Which observations were uncertain or rejected?",
                       "Which rules actually applied?", "What evidence supports this interpretation?",
                       "What can DOAR not conclude?", "What might a psychologist ask next?"]
    suggestions_ar = ["ماذا لاحظ DOAR؟", "لماذا توصّل DOAR إلى هذا الاستنتاج؟",
                       "ما الملاحظات التي تم التحقق منها؟", "ما الملاحظات غير المؤكدة أو المرفوضة؟",
                       "ما القواعد التي انطبقت فعلياً؟", "ما الأدلة الداعمة لهذا التفسير؟",
                       "ما الذي لا يمكن لـDOAR الخلوص إليه؟", "ما الذي قد يسأل عنه الأخصائي النفسي لاحقاً؟"]
    chip_q_key = f"sd_{case_key}_chat_q"
    chip_cols = st.columns(4)
    for i, q in enumerate(suggestions_ar if ar else suggestions_en):
        if chip_cols[i % 4].button(q, key=f"sd_chip_{case_key}_{i}", width="stretch"):
            st.session_state[chip_q_key] = q
            st.rerun()
    _render_ask_doar(audience="clinician", chat_key=f"sd_chat_{case_dir}", widget_prefix=f"sd_{case_key}", is_ar=ar,
                      allow_open_vocab_recheck=False)
    st.caption(
        ("تجيب DOAR بالاستناد إلى حزمة أدلة هذه الحالة فقط؛ لا يمكنها أبداً تفعيل قاعدة معطّلة أو تشخيص الطفل."
         if ar else
         "Ask DOAR answers only from this case's saved evidence -- it can never activate a disabled rule "
         "or diagnose the child.")
    )


def _sd_technical_trace(language: str, case_key: str) -> None:
    ar = language == "ar"
    cfg = SUPERVISOR_DEMO_CASES[case_key]
    case_dir_local = CASES_DIR / cfg["case_id"]
    docs = _sd_load_case_docs(str(case_dir_local))
    analysis = docs.get("analysis")
    if not analysis:
        st.error("تعذّر تحميل بيانات هذه الحالة." if ar else "Could not load this case's data.")
        return
    st.markdown(f"## {'الأثر التقني' if ar else 'Technical Trace'} — {cfg['title'][language]}")
    st.caption(
        ("جدول واحد بأدلة القاعدة المحكومة الفعلية لهذه الحالة. التفاصيل الخام متاحة فقط داخل قسم "
         "\"متقدم / تفاصيل خام\" أدناه." if ar else
         "One clean table of this case's real governed rule evidence. Raw JSON is available only "
         "inside \"Advanced / Raw Details\" below.")
    )
    reg = _sd_rules_registry_index()
    det = docs.get("detections") or {}
    findings = det.get("findings", []) if det.get("status") == "available" else []
    visual_trace = []
    if findings:
        try:
            visual_trace = build_rule_evidence_trace([VisualFinding.from_dict(f) for f in findings])
        except Exception:  # noqa: BLE001 -- trace is descriptive only, never fatal
            visual_trace = []
    visual_by_rule = {}
    for t in visual_trace:
        visual_by_rule.setdefault(t["rule_id"], t)

    rows = []
    for r in analysis.get("rule_evaluations", []):
        if r.get("status") not in ("weak_support", "not_matched"):
            continue
        reg_row = reg.get(r["rule_id"], {})
        vt = visual_by_rule.get(r["rule_id"])
        rows.append({
            "Observation": cpres.observation_label(reg_row.get("observable", r["rule_id"]), language),
            "Value": "observed" if r["status"] == "weak_support" else "not observed",
            "Source": (vt["detector"] if vt else "deterministic feature extraction"),
            "Verification": (vt["validation_status"] if vt else "deterministic"),
            "Canonical observable": reg_row.get("observable", "n/a"),
            "Applied rule": r["rule_id"],
            "Evidence family": (cpres.evidence_family_label(reg_row.get("evidence_family", ""), language)
                                 if reg_row.get("evidence_family") else "n/a"),
            "Final use": "Used in synthesis" if r["status"] == "weak_support" else "Not used (no match)",
        })
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("لا توجد صفوف قواعد محكومة قابلة للتطبيق لهذه الحالة." if ar else
                   "No applicable governed rule rows for this case.")

    with st.expander(("متقدم / تفاصيل خام" if ar else "Advanced / Raw Details"), expanded=False):
        st.caption("أدوات JSON خام لهذه الحالة (لأغراض بحثية/تقنية فقط)." if ar else
                   "Raw JSON artifacts for this case (research/technical use only).")
        st.json(analysis, expanded=False)
        if docs.get("detections"):
            st.markdown("**detections.json**")
            st.json(docs["detections"], expanded=False)
        if docs.get("judges"):
            st.markdown("**judges.json**")
            st.json(docs["judges"], expanded=False)


def _sd_research_progress(language: str) -> None:
    ar = language == "ar"
    st.markdown("## " + ("تقدّم البحث" if ar else "Research Progress"))
    st.caption(
        ("حالات محافِظة لا تُبالغ في وصف الاكتمال — منفصلة تماماً عن أي حالة عرض محددة." if ar else
         "Conservative statuses that do not overstate completion -- entirely separate from any specific demo case.")
    )
    rows = [
        ("F0", "Dataset integrity / frozen controls", "COMPLETE"),
        ("E1", "Learned visual representation experiments",
         "BENCHMARK COMPLETED (final configuration still experimentally selected)"),
        ("E2", "Objective / formal measurement", "IN PROGRESS -- measurement QC / foreground validation"),
        ("E3", "Semantic perception", "IN DEVELOPMENT / EXPERIMENTAL"),
        ("--", "Evidence governance", "PROTOTYPE IMPLEMENTED"),
        ("--", "Psychologist review", "PROTOTYPE IMPLEMENTED"),
        ("--", "Expert evaluation", "PLANNED / IN PROGRESS"),
        ("--", "Locked Test", "LOCKED / UNTOUCHED"),
    ]
    st.dataframe([{"Phase": p, "Area": a, "Status": s} for p, a, s in rows], width="stretch", hide_index=True)
    st.caption(
        ("سجل القواعد النشط هو النواة الأصلية المكونة من 41 قاعدة (10 مفعّلة أساساً)؛ سجل الأبحاث (225 إدخال "
         "مصدر/206 متغير قابل للملاحظة/21 أسرة أدلة) هو تنظيم بحثي وليس قواعد نفسية مفعّلة." if ar else
         "The active governed rule set is the original 41-rule corpus (10 enabled at baseline); the research "
         "registry (225 source entries / 206 canonical observables / 21 evidence families) is a research "
         "organization, not 225 active psychological rules.")
    )


def render_supervisor_demo() -> None:
    _sd_inject_css()
    st.session_state.setdefault("supervisor_language", "en")
    st.session_state.setdefault("supervisor_section", "Home")
    st.session_state.setdefault("supervisor_active_case", None)

    with st.sidebar:
        st.markdown("### DOAR")
        language = st.radio("Language / اللغة", ["en", "ar"], horizontal=True, key="supervisor_language")
        ar = language == "ar"
        section_options = ["Home", "Drawing Analysis", "Technical Trace", "Research Progress"]
        section_label_ar = {"Home": "الرئيسية", "Drawing Analysis": "تحليل الرسمة",
                             "Technical Trace": "الأثر التقني", "Research Progress": "تقدّم البحث"}
        st.radio(("الأقسام" if ar else "Sections"), section_options,
                  format_func=lambda s: section_label_ar[s] if ar else s, key="supervisor_section")
        st.divider()
        if st.session_state["supervisor_active_case"]:
            st.caption(("الحالة النشطة: " if ar else "Active case: ") +
                       SUPERVISOR_DEMO_CASES[st.session_state["supervisor_active_case"]]["title"][language])
        gemini_ready = bool(os.environ.get("GEMINI_API_KEY"))
        st.caption("Gemini: " + (("جاهز" if ar else "READY") if gemini_ready else
                                  ("غير مُهيّأ" if ar else "NOT CONFIGURED")))
        st.divider()
        st.caption("DOAR نموذج بحثي ودعم مهني غير تشخيصي." if ar else
                   "DOAR is a non-diagnostic research and professional-support prototype.")

    section = st.session_state["supervisor_section"]
    active_case = st.session_state["supervisor_active_case"]

    if section == "Home":
        _sd_home(language)
        return
    if section in ("Drawing Analysis", "Technical Trace") and not active_case:
        st.info("اختر حالة عرض من الرئيسية أولاً." if language == "ar" else
                "Select a demo case from Home first.")
        pick_cols = st.columns(3)
        for col, (case_key, cfg) in zip(pick_cols, SUPERVISOR_DEMO_CASES.items()):
            if col.button(cfg["title"][language], key=f"sd_quickpick_{case_key}", width="stretch"):
                st.session_state["supervisor_active_case"] = case_key
                st.rerun()
        return
    if section == "Drawing Analysis":
        _sd_drawing_analysis(language, active_case)
    elif section == "Technical Trace":
        _sd_technical_trace(language, active_case)
    elif section == "Research Progress":
        _sd_research_progress(language)


st.set_page_config(page_title="DOAR prototype - dual view", layout="wide")
st.title("DOAR v3 -- Dual-View Prototype")
st.warning(
    "Research prototype, non-diagnostic. Every checkpoint currently available was "
    "trained on a duplicate-contaminated, leakage-gate-overridden split -- see "
    "CURRENT_CAPABILITY_AUDIT.md Section 3. Treat all output as preliminary."
)

# ---------------------------------------------------------------------------
# View switch (feature/supervisor-demo-v2): defaults to the new, polished
# Supervisor Demo (Home / Drawing Analysis / Technical Trace / Research
# Progress, three prepared cases, zero Gemini calls just to open a case).
# The full legacy Parent/Psychologist/Technical research app -- upload,
# reopen-any-case, Run Full Analysis, deep visual scan, expert review, etc.
# -- is completely untouched below and stays reachable via this switch.
# ---------------------------------------------------------------------------
# Smart default: if a case_dir was already pre-seeded (e.g. an existing
# test/script driving this app the original way via
# st.session_state["case_dir"]), default straight to the legacy view so
# that flow keeps working completely unchanged; a fresh launch with no
# pre-seeded case defaults to the new Supervisor Demo.
st.session_state.setdefault(
    "supervisor_view_mode",
    "Full Research App (legacy)" if st.session_state.get("case_dir") else "Supervisor Demo")
_view_mode = st.sidebar.radio(
    "View", ["Supervisor Demo", "Full Research App (legacy)"], key="supervisor_view_mode")
if _view_mode == "Supervisor Demo":
    render_supervisor_demo()
    st.stop()
st.sidebar.divider()

# ---------------------------------------------------------------------------
# Sidebar: upload + optional child context + checkpoint choice + page
# declaration (DOAR-TRACE Phase 2A.2, Section 6), or reopen.
# ---------------------------------------------------------------------------
st.session_state.setdefault("case_dir", "")

with st.sidebar:
    st.session_state.setdefault("language", "en")
    language = st.radio("Language / اللغة", ["en", "ar"], horizontal=True, key="language")
    ar_side = language == "ar"

    st.header("تحليل جديد" if ar_side else "New analysis")
    uploaded = st.file_uploader(
        "ارفع رسمة الطفل" if ar_side else "Upload a child's drawing", type=["png", "jpg", "jpeg"])
    st.caption(
        "سياق اختياري (لا يُستخدم أبداً في النموذج أو القواعد -- للعرض فقط)" if ar_side else
        "Optional context (never used by the model or rules -- display only)")
    age_range = st.selectbox(
        "الفئة العمرية" if ar_side else "Age range", ALLOWED_AGE_RANGES, index=len(ALLOWED_AGE_RANGES) - 1)
    gender = st.text_input("الجنس (اختياري)" if ar_side else "Gender (optional)")
    instruction = st.text_input(
        "تعليمات/طلب الرسم المُعطى للطفل (اختياري)" if ar_side else
        "Drawing instruction/prompt given to the child (optional)")
    concern = st.text_area("مخاوف/سؤال الوالد (اختياري)" if ar_side else "Parent's concern or question (optional)")

    st.caption("هل تُظهر هذه الصورة الورقة كاملة؟" if ar_side else "Does this image show the complete sheet of paper?")
    page_choice = st.radio(
        "Page declaration", PARENT_PAGE_DECLARATION_CHOICES,
        format_func=lambda c: PARENT_PAGE_DECLARATION_LABELS[c]["ar" if ar_side else "en"],
        index=0, label_visibility="collapsed", key="page_declaration_choice",
    )
    st.caption(
        "هذا اختياري -- يمكن للنظام أيضاً محاولة تحديد ذلك تلقائياً. تحديد زوايا الصفحة يدوياً غير متاح في هذا "
        "الإصدار." if ar_side else
        "This is optional -- the system can also try to work this out automatically. Manually marking exact page "
        "corners is not available in this version."
    )

    if st.button("تحليل" if ar_side else "Analyze", type="primary", disabled=uploaded is None):
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
        # Performance fix: Analyze runs ONLY the light/normal DOAR pipeline
        # (objective features, page reference, rules, expressive model) --
        # never the heavy Grounding-DINO/OWLv2 open-vocabulary scan, which
        # `finalize_case` already leaves as an honest {"status":
        # "unavailable"} stub until a reviewer explicitly requests it from
        # the Psychologist view's "Run deep visual analysis" button below.
        with st.spinner("جارٍ تشغيل تحليل DOAR..." if ar_side else "Running the DOAR analysis..."):
            analyze_image_with_timing(str(image_path), str(case_dir), production_config.expressive_model_checkpoint,
                                       user_page_declaration=declaration)
            write_versioned(case_dir / "production_config.json", production_config.to_dict())
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
        picked = st.selectbox("أو أعد فتح حالة سابقة" if ar_side else "Or reopen a previous case",
                               ["(none)"] + previous)
        if picked != "(none)":
            st.session_state["case_dir"] = str((CASES_DIR / picked).resolve())

case_dir_text = st.sidebar.text_input("مجلد الحالة" if ar_side else "Case folder", key="case_dir")
case_dir = Path(case_dir_text) if case_dir_text else None

if not case_dir or not (case_dir / "analysis.json").exists():
    st.info(
        "ارفع رسمة في الشريط الجانبي واضغط **تحليل**، أو اختر/أدخل مجلد حالة." if language == "ar" else
        "Upload a drawing in the sidebar and click **Analyze**, or select/enter a case folder.")
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


ar = language == "ar"


def _rtl_open(is_ar: bool) -> None:
    st.markdown(f'<div dir="{"rtl" if is_ar else "ltr"}">', unsafe_allow_html=True)


def _rtl_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# (_build_full_interpretation and _render_ask_doar now live earlier in this
# file, right after CASES_DIR -- see the NOTE there. Both are unchanged.)
parent_tab, psychologist_tab, technical_tab = st.tabs(
    ["Parent / User View", "Psychologist / Professional View", "Technical / Research View"])

# ===========================================================================
# PARENT / USER VIEW (Milestone 2: simplified to exactly 8 sections -- no
# rule IDs, evidence families, thresholds, implementation statuses, raw
# source details, or research jargon here; that detail lives in the
# Psychologist and Technical views instead).
# ===========================================================================
with parent_tab:
    _rtl_open(ar)

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

    _parent_bundle = _load_live_case_bundle(str(case_dir), _detections_cache_token(str(case_dir)))
    _interp = _build_full_interpretation(_parent_bundle, str(case_dir))
    _profile = _interp.expressive_profile
    _notable_domains = sorted(
        (c for c in _interp.concern_domains if c.support_level in ("WEAK", "MODERATE", "STRONG")),
        key=lambda c: ("STRONG", "MODERATE", "WEAK").index(c.support_level))

    # 1. Overall expressive impression (SECTION 1: EXPRESSIVE PROFILE only --
    # deliberately modest; never DOAR's final answer. See sections 3/6 below
    # for the actual concern-domain reasoning and conclusion.) --------------
    st.subheader("الانطباع التعبيري العام" if ar else "Overall expressive impression")
    if _profile.availability == "available" and _profile.top_emotion:
        feeling_words = (", ".join(cpres.descriptor_label(d.label, language) for d in _profile.richer_descriptors)
                          or cpres.emotion_label(_profile.top_emotion, language))
        st.write(("بشكل أساسي: " if ar else "Mainly: ") + feeling_words)
        if _profile.base_emotion_probabilities:
            st.dataframe({("الفئة" if ar else "class"):
                              [cpres.emotion_label(k, language) for k in _profile.base_emotion_probabilities],
                          ("النسبة" if ar else "probability"):
                              [f"{v:.0%}" for v in _profile.base_emotion_probabilities.values()]},
                         width="stretch", hide_index=True)
        st.caption(
            "هذا مخرج المصنِّف التعبيري الواسع فقط -- وليس الخلاصة النهائية لتحليل DOAR؛ انظر \"مؤشرات تستحق "
            "الانتباه\" و\"الخلاصة العامة\" أدناه." if ar else
            "This is only the broad expressive classifier's output -- not DOAR's final conclusion; see "
            "\"Concern indicators\" and \"Overall conclusion\" below."
        )
    else:
        st.info("نموذج الشعور العام غير متاح لهذه الحالة." if ar else
                "No overall-feeling model result is available for this case.")

    # 2. What DOAR noticed ----------------------------------------------------------
    st.subheader("ما لاحظه DOAR" if ar else "What DOAR noticed")
    why_lines = [cpres.descriptor_reason(d.reason, language) for d in _profile.richer_descriptors]
    why_lines += [o["text"] for o in plain_language_observations(analysis, language, page_reference=page_reference)]
    for line in why_lines[:6]:
        st.write("- " + line)
    if not why_lines:
        st.caption("لم نلاحظ شيئاً يمكن الإبلاغ عنه في هذه الرسمة." if ar else
                   "Nothing reportable was noticed in this drawing.")

    # 3. Concern indicators (SECTION 3: real DOAR concern-domain registry --
    # MODERATE/STRONG prominent, WEAK secondary, NONE/INSUFFICIENT collapsed
    # into a compact expander; never a raw rule ID here) --------------------
    st.subheader("مؤشرات تستحق الانتباه" if ar else "Concern indicators")
    st.caption(
        "هذه أنماط رصدية من سجل DOAR المعتمد -- وليست تشخيصاً." if ar else
        "These are observational patterns from DOAR's approved registry -- not a diagnosis."
    )
    if _notable_domains:
        for c in _notable_domains:
            domain_txt = cpres.domain_label(c.domain, language)
            level_txt = cpres.support_level_label(c.support_level, language)
            reason_txt = cpres.concern_domain_reason(c, language)
            if c.support_level in ("MODERATE", "STRONG"):
                # MODERATE/STRONG: prominent (multiple independent evidence families).
                st.warning(f"**{domain_txt} -- {level_txt}**\n\n" + ("السبب: " if ar else "Why: ") + reason_txt)
            else:
                # WEAK: visible but secondary/subdued (a single evidence family).
                st.caption(f"{domain_txt} -- {level_txt}. " + ("السبب: " if ar else "Why: ") + reason_txt)
    else:
        st.info("لم يصل أي مجال قلق إلى مستوى دعم يستحق الذكر لهذه الحالة." if ar else
                "No concern domain reached a support level worth mentioning for this case.")
    _quiet_domains = [c for c in _interp.concern_domains if c.support_level in ("NONE", "INSUFFICIENT")]
    if _quiet_domains:
        with st.expander("مجالات أخرى لم تصل إلى مستوى دعم يُذكر" if ar else
                          "Other domains with no notable support", expanded=False):
            for c in _quiet_domains:
                st.caption(f"{cpres.domain_label(c.domain, language)} -- {cpres.support_level_label(c.support_level, language)}")

    # 4. Whole-image AI opinion (only shown once Full Analysis/Gemini has run
    # for this case -- concise, candidate-only, never treated as evidence) --
    _gobs = _interp.gemini_global_observation
    if _gobs:
        st.subheader("رأي الذكاء الاصطناعي الشامل للصورة" if ar else "Whole-image AI opinion")
        if _gobs.get("overall_scene"):
            st.write(_gobs["overall_scene"])
        if _gobs.get("overall_visual_tone"):
            st.caption(("النبرة العامة: " if ar else "Overall tone: ") + str(_gobs["overall_visual_tone"]))

    # 5. Agreement / disagreement -----------------------------------------------------
    st.subheader("مدى اتفاق الأدلة" if ar else "Agreement / disagreement")
    if _interp.consistency_status == "CONSISTENT":
        st.write("تتفق معظم الأدلة." if ar else "Most evidence agrees.")
    elif _interp.consistency_status in ("MIXED", "CONFLICT"):
        st.write("بعض الأدلة تشير إلى اتجاهات مختلفة." if ar else "Some evidence points in different directions.")
    else:
        st.caption("لا تتوفر أدلة كافية لتقييم مدى الاتفاق." if ar else
                   "Not enough evidence is available to assess agreement.")

    # 6. Overall conclusion (item K: names the expressive profile, the
    # notable concern domains with their evidence-family counts, the
    # Gemini note if available, and conflicts/limitations -- built from
    # CaseInterpretation's own already-computed fields only) ----------------
    st.subheader("الخلاصة العامة" if ar else "Overall conclusion")
    st.write(cpres.synthesis_summary(_interp, language))

    # 7. Questions you can ask --------------------------------------------------------
    st.subheader("أسئلة يمكنك طرحها" if ar else "Questions you can ask")
    questions = [cpres.parent_question_label(q, language)
                 for q in (structured or {}).get("suggested_parent_questions", [])]
    if not questions:
        questions = ["هل يمكنك أن تخبرني ماذا يحدث في هذه الصورة؟"] if ar else \
            ["Can you tell me what is happening in this picture?"]
    for q in questions:
        st.write("- " + q)
    for item in parent_care_suggestions(language)[:2]:
        st.write("- " + item)

    # 8. Ask DOAR -----------------------------------------------------------------
    st.divider()
    _render_ask_doar(audience="parent", chat_key=f"hi_chat_{case_dir}", widget_prefix="parent", is_ar=ar)

    st.divider()
    _page_assessable = bool(page_reference.get("page_relative_features_assessable"))
    if not _page_assessable or judges.get("module_availability", {}).get("visual_detection") != "available":
        not_assessed = []
        if judges.get("module_availability", {}).get("visual_detection") != "available":
            not_assessed.append("الأجسام والعناصر البصرية" if ar else "objects and visual elements")
        if not _page_assessable:
            not_assessed.append("استخدام الصفحة وموضعها" if ar else "page use and placement")
        st.warning(
            (f"تعذّر تقييم بثقة: {', '.join(not_assessed)} -- هذا لا يعني غيابها." if ar
             else f"Could not be confidently assessed: {', '.join(not_assessed)} -- this does not mean "
                  f"they are absent.")
        )
    st.caption(disclaimer(language))

    _rtl_close()

# ===========================================================================
# PSYCHOLOGIST / PROFESSIONAL VIEW (Milestone 2, new): everything a
# reviewing professional needs -- original drawing, objective observations,
# evidence by verification status, rules considered, supporting evidence,
# sources/evidence strength, combined interpretation, alternative
# explanations, limitations/contradictions, the expressive-model result on
# its own, all Judge results (deterministic + Gemini answer-Judge +
# AUDIT-ONLY Visual Consistency Judge), and a feedback form persisted for
# thesis-level analysis.
# ===========================================================================
with psychologist_tab:
    _rtl_open(ar)

    st.subheader("الرسمة الأصلية" if ar else "Original drawing")
    norm = artifact("normalized_image")
    if norm:
        st.image(norm, width=360)

    if detections.get("status") != "available":
        st.warning(
            "لم يُشغَّل بعد المسح البصري العميق لهذه الحالة -- هذا لا يعني عدم وجود أجسام، بل أنه لم يُطلَب بعد." if ar
            else "Deep visual analysis has not been run yet for this case -- this does not mean no objects "
                 "are present, only that it has not been requested yet."
        )

    st.divider()
    st.markdown("### " + ("تشغيل التحليل الكامل" if ar else "Run Full Analysis"))
    st.caption(
        "يُشغّل الفحص الدلالي العميق (إن لم يكن متاحاً بعد) ورأي Gemini الشامل للصورة، ثم يُعيد بناء التفسير "
        "الكامل -- عند الطلب فقط، أبداً تلقائياً." if ar else
        "Runs the deep semantic/object scan (if not already available) and Gemini's whole-image opinion, then "
        "rebuilds the full interpretation -- on request only, never automatically."
    )
    full_analysis_key = f"full_analysis_{case_dir}"
    if st.button("تشغيل التحليل الكامل" if ar else "Run Full Analysis", key="run_full_analysis", type="primary"):
        image_path = case_dir / Path(analysis["image_path"]).name
        with st.status(
                "جارٍ تحليل الرسمة الكاملة..." if ar else "Analyzing whole drawing...",
                expanded=True) as full_status:
            try:
                providers_hi = _load_hi_providers()
                if detections.get("status") != "available":
                    full_status.update(label="جارٍ اكتشاف العناصر البصرية..." if ar else "Detecting visual elements...")
                    run_and_persist_initial_scan(
                        case_dir, str(image_path), eye_entry=_load_eye_entry(),
                        registry_v2=_load_registry_v2(), model_predict_fns=_load_model_predict_fns())

                full_status.update(label="جارٍ فحص العلاقات..." if ar else "Examining relationships...")
                fresh_bundle = build_live_case_bundle(case_dir)

                full_status.update(
                    label="جارٍ بناء التفسير الدلالي..." if ar else "Building semantic interpretation...")
                full_status.update(
                    label="جارٍ التحقق من أدلة القواعد النفسية..." if ar else "Checking psychological rule evidence...")

                gemini_result = hi.run_gemini_global_observation(fresh_bundle, observer=providers_hi["global_observer"])
                full_status.update(label="جارٍ التحقق من اقتراحات Gemini..." if ar else "Checking Gemini suggestions...")

                full_status.update(label="جارٍ مقارنة الأدلة..." if ar else "Comparing evidence...")
                full_status.update(
                    label="جارٍ بناء التفسير النهائي..." if ar else "Building final interpretation...")
                st.session_state[full_analysis_key] = {
                    "gemini_status": gemini_result["status"], "gemini_reason": gemini_result.get("reason"),
                    "gemini_observation": gemini_result.get("observation"),
                }
                full_status.update(label="اكتمل." if ar else "Complete.", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, prior results are unaffected
                full_status.update(
                    label=("فشل التحليل الكامل." if ar else "Full Analysis failed."), state="error")
                st.error(f"Full Analysis failed ({exc}); previously available results are unaffected.")
        st.rerun()
    _full_analysis_result = st.session_state.get(full_analysis_key)
    if _full_analysis_result and _full_analysis_result["gemini_status"] != "ok":
        # Item F: never silently omit the whole-image AI opinion -- always
        # show a clear, non-technical sentence that it did not run this
        # time, plus the technical reason as a secondary detail.
        st.warning(
            "رأي الذكاء الاصطناعي الشامل للصورة غير متاح لهذا التشغيل." if ar else
            "Whole-image AI opinion unavailable for this run."
        )
        st.caption(f"{_full_analysis_result['gemini_status']}: {_full_analysis_result.get('gemini_reason')}")

    st.markdown("**" + ("التحليل البصري العميق (Grounding-DINO / OWLv2)" if ar
                         else "Deep visual analysis (Grounding-DINO / OWLv2)") + "**")
    st.caption(
        "هذا يُشغِّل نماذج الرؤية الثقيلة مفتوحة المفردات -- عند الطلب فقط، أبداً تلقائياً عند التحليل العادي "
        "أو تبديل التبويبات/اللغة." if ar else
        "This runs the heavy open-vocabulary vision models -- on request only, never automatically "
        "during ordinary analysis or when switching tabs/language."
    )
    deep_scan_label = (
        ("إعادة تشغيل التحليل البصري العميق" if ar else "Re-run deep visual analysis")
        if detections.get("status") == "available" else
        ("تشغيل التحليل البصري العميق" if ar else "Run deep visual analysis")
    )
    if st.button(deep_scan_label, key="run_deep_visual_analysis"):
        image_path = case_dir / Path(analysis["image_path"]).name
        with st.status(
                "جارٍ تحميل نماذج الرؤية..." if ar else "Loading visual models...",
                expanded=True) as deep_scan_status:
            try:
                model_predict_fns = _load_model_predict_fns()
                deep_scan_status.update(label="جارٍ فحص الرسمة..." if ar else "Scanning drawing...")
                eye_entry = _load_eye_entry()
                registry_v2 = _load_registry_v2()
                deep_scan_status.update(
                    label="جارٍ بناء الأدلة البصرية..." if ar else "Building visual evidence...")
                run_and_persist_initial_scan(
                    case_dir, str(image_path), eye_entry=eye_entry,
                    registry_v2=registry_v2, model_predict_fns=model_predict_fns)
                deep_scan_status.update(
                    label="اكتمل." if ar else "Complete.", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, light analysis is unaffected
                deep_scan_status.update(
                    label=("فشل التحليل البصري العميق." if ar else "Deep visual analysis failed."),
                    state="error")
                st.error(f"Deep visual analysis failed ({exc}); the rest of the analysis is unaffected. "
                         "detections.json remains the honest 'unavailable' stub for this case.")
        st.rerun()

    # Phase G0: OBJECTIVE_ONLY formal/graphic-element measurements
    # (line geometry, stroke texture, colour/fill, symmetry/regularity --
    # see src/doar/formal_features.py). Deterministic CV only, no model
    # weights -- cheap, but still gated behind an explicit button (never
    # computed during ordinary Analyze) since it is a NEW, separate,
    # research-only artifact, not yet part of any Parent/Psychologist
    # conclusion. Its output can NEVER affect concern-domain support --
    # see tests/test_formal_features_no_concern_score_effect.py.
    st.markdown("**" + ("قياسات شكلية/بيانية (بحثية، وصفية فقط)" if ar
                         else "Formal/graphic measurements (research, OBJECTIVE_ONLY)") + "**")
    st.caption(
        "قياسات حاسوبية حتمية (سُمك الخط، الاتجاه، التماثل، الانتظام...) -- وصفية فقط، ولا يمكنها أبداً "
        "تغيير درجة دعم أي مجال قلق." if ar else
        "Deterministic computer-vision measurements (line width, orientation, symmetry, regularity...) -- "
        "descriptive only, and can never change any concern domain's support level."
    )
    formal_features_key = f"formal_features_{case_dir}"
    if st.button("حساب القياسات الشكلية/البيانية" if ar else "Compute formal/graphic measurements",
                 key="run_formal_features"):
        try:
            image_path = case_dir / Path(analysis["image_path"]).name
            resolved_analysis = resolve_analysis_artifacts(analysis, case_dir)
            doc = ff.run_and_persist_formal_features(case_dir, image_path, resolved_analysis)
            overlay = ff.render_formal_feature_overlay(str(image_path), resolved_analysis)
            st.session_state[formal_features_key] = {"document": doc, "overlay": overlay}
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, rest of analysis unaffected
            st.error(f"Formal/graphic measurement failed ({exc}); the rest of the analysis is unaffected.")
    _formal_result = st.session_state.get(formal_features_key)
    if _formal_result:
        st.image(_formal_result["overlay"], width=320,
                  caption=("أخضر = مقاطع خط مستقيمة مكتشفة؛ أحمر = نقاط تقاطع/زوايا (تفسير \"الاستقامة\" و\"كثافة "
                           "التقاطع\")" if ar else
                           "Green = detected near-straight segments; red = corner/crossing points (explains "
                           "\"straightness\"/\"crossing density\")"))
        with st.expander("كل القياسات (خام)" if ar else "All measurements (raw)", expanded=False):
            st.dataframe([
                {"feature_id": name, "value": round(v["value"], 4), "missing": v["missing"],
                 "confidence": v["confidence"]}
                for name, v in _formal_result["document"]["features"].items()
            ], width="stretch", hide_index=True)
        with st.expander("مراجعة خبير لقياس واحد (بحثي، لا يُفعّل تلقائياً)" if ar
                          else "Expert review of one measurement (research, never auto-activated)", expanded=False):
            with st.form("formal_feature_review_form"):
                ff_reviewer = st.text_input("اسم المراجع" if ar else "Reviewer name", key="ff_reviewer_name")
                ff_feature_id = st.selectbox("المعرّف" if ar else "Feature", list(_formal_result["document"]["features"]))
                ff_obs_correct = st.radio("هل الملاحظة صحيحة؟" if ar else "Is the observation correct?",
                                          OBSERVATION_CORRECT_VALUES, horizontal=True, key="ff_obs_correct")
                ff_relevant = st.radio("هل هي ذات صلة سريرية؟" if ar else "Clinically relevant?",
                                       CLINICALLY_RELEVANT_VALUES, horizontal=True, key="ff_relevant")
                ff_support = st.radio("مستوى الدعم من وجهة نظر الخبير" if ar else "Expert support level",
                                      EXPERT_SUPPORT_LEVEL_VALUES, horizontal=True, key="ff_support")
                ff_decision = st.radio("القرار المقترح للقاعدة" if ar else "Suggested rule decision",
                                       EXPERT_RULE_DECISION_VALUES, horizontal=True, key="ff_decision")
                ff_agreement = st.radio("الاتفاق مع التوليف العام" if ar else "Agreement with overall synthesis",
                                        OVERALL_SYNTHESIS_AGREEMENT_VALUES, horizontal=True, key="ff_agreement")
                ff_notes = st.text_area("ملاحظات" if ar else "Notes", key="ff_notes")
                if st.form_submit_button("إرسال المراجعة" if ar else "Submit review"):
                    if not ff_reviewer.strip():
                        st.error("اسم المراجع مطلوب." if ar else "Reviewer name is required.")
                    else:
                        submit_formal_feature_review(
                            case_dir, reviewer_name=ff_reviewer.strip(), feature_id=ff_feature_id,
                            observation_correct=ff_obs_correct, clinically_relevant=ff_relevant,
                            expert_support_level=ff_support, expert_rule_decision=ff_decision,
                            overall_synthesis_agreement=ff_agreement,
                            raw_measurement=_formal_result["document"]["features"][ff_feature_id]["value"],
                            notes=ff_notes.strip() or None)
                        st.success("تم حفظ المراجعة." if ar else "Review recorded.")
        existing_ff_reviews = load_formal_feature_review(case_dir).get("entries", [])
        if existing_ff_reviews:
            st.caption(("مراجعات سابقة لهذه الحالة: " if ar else "Prior reviews for this case: ")
                       + str(len(existing_ff_reviews)))

    audit_key = f"visual_audit_{case_dir}"
    _psych_bundle = _load_live_case_bundle(str(case_dir), _detections_cache_token(str(case_dir)))
    _psych_interp = _build_full_interpretation(_psych_bundle, str(case_dir))
    _psych_profile = _psych_interp.expressive_profile

    st.subheader("الملف التعبيري" if ar else "Expressive profile")
    if _psych_profile.availability == "available" and _psych_profile.base_emotion_probabilities:
        st.dataframe({("الفئة" if ar else "class"):
                          [cpres.emotion_label(k, language) for k in _psych_profile.base_emotion_probabilities],
                      ("النسبة" if ar else "probability"):
                          [f"{v:.0%}" for v in _psych_profile.base_emotion_probabilities.values()]},
                     width="stretch", hide_index=True)
        st.caption(("الفئة الأعلى: " if ar else "Top class: ") + str(_psych_profile.top_emotion)
                   + (f", {'المعايرة' if ar else 'calibration'}: {_psych_profile.calibration_status}"
                      if _psych_profile.calibration_status else ""))
    else:
        st.info("نموذج المحتوى التعبيري غير متاح لهذه الحالة." if ar else
                "The expressive-content model is unavailable for this case.")
        if _psych_profile.unavailable_reason:
            st.caption(_psych_profile.unavailable_reason)

    st.subheader("الانطباع التعبيري الأغنى" if ar else "Richer expressive impression")
    if _psych_profile.richer_descriptors:
        for d in _psych_profile.richer_descriptors:
            st.write(f"- **{cpres.descriptor_label(d.label, language)}** -- {cpres.descriptor_reason(d.reason, language)}")
    else:
        st.caption("لا توجد أوصاف تعبيرية أغنى مدعومة لهذه الحالة." if ar else
                   "No richer expressive descriptor is supported for this case.")

    st.subheader("ملاحظات بصرية مهمة" if ar else "Important visual observations")
    if _psych_interp.visual_concepts:
        for c in _psych_interp.visual_concepts:
            st.write(f"- {c.human_readable_label.replace('_', ' ')} ({c.status})")
    else:
        st.caption("لا توجد ملاحظات بصرية بعد." if ar else "No visual observations yet.")

    st.subheader("العلاقات" if ar else "Relationships")
    try:
        from doar.relationship_features import compute_relationship_features
        _rel = compute_relationship_features(_psych_bundle.get("entities") or [])
        st.caption(
            (f"عدد الأشكال المتحقق منها: {_rel['entity_count']}" if ar else
             f"Verified figures/objects: {_rel['entity_count']}")
            + (f" -- {', '.join(sorted(_rel['repeated_labels']))} " + ("مكررة" if ar else "repeated")
               if _rel["repeated_labels"] else "")
            + (f" -- {len(_rel['spatially_separated_entity_ids'])} "
               + ("منفصلة مكانياً" if ar else "spatially separated") if _rel["spatially_separated_entity_ids"] else "")
        )
    except Exception:
        st.caption("لا تتوفر بيانات علاقات كافية." if ar else "Not enough data for relationship analysis.")

    st.subheader("الانطباع العام للرسمة (كاملة)" if ar else "Global (whole-drawing) impression")
    st.caption(
        "حتمي (تركيب/علاقات) بشكل أساسي، مع ملاحظة Gemini الشاملة (إن وُجدت) مُميَّزة بوضوح أدناه." if ar else
        "Primarily deterministic (composition/relationships); Gemini's whole-image note (if run) is shown "
        "clearly separated below."
    )
    _global = _psych_interp.global_impression
    if _global.availability == "available":
        st.write(_global.overall_scene)
        st.caption(("النبرة التعبيرية العامة: " if ar else "Overall expressive tone: ") + _global.expressive_tone)
        for note in _global.contradictions:
            st.warning(note)
        st.markdown(f"**{'الاتساق بين الانطباع العام والأدلة المحلية' if ar else 'Global <-> local consistency'}: "
                    f"{cpres.consistency_label(_psych_interp.global_local_consistency, language)}**")
    else:
        st.caption("لا يتوفر انطباع عام كافٍ للرسمة الكاملة بعد." if ar else
                   "Not enough whole-drawing evidence is available yet for a global impression.")

    if _psych_interp.gemini_global_observation:
        _gobs = _psych_interp.gemini_global_observation
        with st.expander(("رأي Gemini الشامل للصورة (مُرشَّح فقط)" if ar else
                          "Gemini's whole-image opinion (CANDIDATE only)"), expanded=True):
            st.caption(
                "توليد فرضيات مُرشَّحة فقط -- لا يُغيّر أبداً درجات دعم مجالات القلق مباشرة." if ar else
                "Candidate hypothesis generation only -- never directly changes concern-domain support scores."
            )
            st.write(_gobs.get("overall_scene"))
            st.caption(("النبرة: " if ar else "Tone: ") + str(_gobs.get("overall_visual_tone")))
            if _gobs.get("expressive_descriptors"):
                st.write(("أوصاف تعبيرية: " if ar else "Expressive descriptors: ")
                        + ", ".join(_gobs["expressive_descriptors"]))
            if _gobs.get("salient_relationships"):
                st.write(("علاقات بارزة: " if ar else "Salient relationships: ")
                        + "; ".join(_gobs["salient_relationships"]))
            if _gobs.get("important_visual_observations"):
                st.write(("ملاحظات بصرية مهمة: " if ar else "Important visual observations: ")
                        + "; ".join(_gobs["important_visual_observations"]))
            if _gobs.get("uncertainty"):
                st.caption(("عدم اليقين: " if ar else "Uncertainty: ") + _gobs["uncertainty"])

    if _psych_interp.gemini_candidates:
        st.subheader("فرضيات Gemini المُرشَّحة -- بعد التحقق" if ar else "Gemini candidate hypotheses -- verified")
        st.caption(
            "كل فرضية من Gemini يتم التحقق منها مقابل أدلة DOAR المحكومة فقط -- لا يمكن لرأي Gemini أن يرفع "
            "درجة الدعم." if ar else
            "Every Gemini candidate is checked against DOAR's own governed evidence only -- Gemini's opinion "
            "can never raise the support score."
        )
        for gc in _psych_interp.gemini_candidates:
            gc_domain_txt = cpres.domain_label(gc.domain, language)
            gc_status_txt = cpres.verification_status_label(gc.verification_status, language)
            with st.expander(f"{gc_domain_txt} -- {gc_status_txt}"):
                st.caption(("اقتراح Gemini: " if ar else "Gemini suggested: ") + gc.gemini_reason)
                if gc.gemini_observations:
                    st.write(("ملاحظات Gemini: " if ar else "Gemini's observations: ")
                            + "; ".join(gc.gemini_observations))
                if ar and gc.governed_support_level:
                    st.write("لماذا: أدلة DOAR المحكومة الخاصة بهذا المجال هي "
                             + cpres.support_level_label(gc.governed_support_level, "ar") + ".")
                else:
                    st.write(("لماذا: " if ar else "Why: ") + gc.why)

    st.subheader("تصنيفات مجالات القلق" if ar else "Concern-domain classifications")
    st.caption(
        "كل مجال يُعرض بمستوى دعمه الفعلي فقط -- لا يوجد تشخيص، فقط أنماط رصدية." if ar else
        "Every domain is shown at its real, computed support level only -- observational patterns, never a diagnosis."
    )
    _gemini_domains_with_candidates = {gc.domain for gc in _psych_interp.gemini_candidates}
    st.dataframe([
        {
            ("المجال" if ar else "Concern"): cpres.domain_label(c.domain, language),
            ("الدعم" if ar else "Support"): cpres.support_level_label(c.support_level, language),
            ("الأسر المستقلة" if ar else "Independent families"):
                f"{len(c.independent_evidence_families)}/{c.assessable_family_count}",
            ("حالة الأدلة" if ar else "Evidence status"):
                ("تعذّر التقييم" if ar else "not assessable") if c.assessable_family_count == 0
                else ("قابل للتقييم" if ar else "assessable"),
            ("فرضية Gemini؟" if ar else "Gemini candidate?"):
                ("نعم" if ar else "Yes") if c.domain in _gemini_domains_with_candidates else ("لا" if ar else "No"),
        }
        for c in _psych_interp.concern_domains
    ], width="stretch", hide_index=True)

    for c in _psych_interp.concern_domains:
        domain_txt = cpres.domain_label(c.domain, language)
        level_txt = cpres.support_level_label(c.support_level, language)
        reason_txt = cpres.concern_domain_reason(c, language)
        with st.expander(f"{domain_txt} -- {level_txt}"):
            st.write(("لماذا: " if ar else "Why: ") + reason_txt)
            if c.support_ratio is not None:
                st.caption(("نسبة الدعم: " if ar else "Support ratio: ")
                          + f"{c.support_ratio:.2f} ({len(c.independent_evidence_families)}/{c.assessable_family_count})")
            if c.independent_evidence_families:
                st.caption(("الأسر المستقلة الداعمة: " if ar else "Supporting independent families: ")
                          + ", ".join(cpres.evidence_family_label(f, language) for f in c.independent_evidence_families))
            if c.supporting_observations:
                st.markdown("**" + ("الملاحظات الداعمة" if ar else "Supporting observations") + "**")
                for o in c.supporting_observations:
                    st.write("- " + cpres.observation_label(o, language))
            if c.alternative_explanations:
                st.markdown("**" + ("تفسيرات بديلة" if ar else "Alternative explanations") + "**")
                for a in c.alternative_explanations:
                    st.write("- " + a)
            if c.contradictory_evidence:
                st.markdown("**" + ("أدلة معارضة" if ar else "Evidence pointing against it") + "**")
                st.write(f"{len(c.contradictory_evidence)} " + ("مرجع معارض" if ar else "contradicting reference(s)"))
            if c.missing_expected_evidence:
                st.caption(("معلومات ناقصة: " if ar else "Missing information: ")
                          + "; ".join(c.missing_expected_evidence))
            if c.sources:
                st.caption(("المصادر: " if ar else "Sources: ")
                          + "; ".join(s.get("citation_title") or s.get("source_pdf_page_section") or "n/a"
                                      for s in c.sources))

    st.subheader("الاتساق بين الانفعال والقواعد" if ar else "Emotion <-> rules consistency")
    st.markdown(f"**{cpres.consistency_label(_psych_interp.consistency_status, language)}**")
    if _psych_interp.contradictions:
        for cont in _psych_interp.contradictions:
            st.write("- " + cont)

    st.subheader("التوليف العام" if ar else "Overall synthesis")
    st.write(cpres.synthesis_summary(_psych_interp, language))
    if _psych_interp.missing_information:
        st.caption(("معلومات مفقودة / عدم يقين: " if ar else "Missing information / uncertainty: ")
                   + "; ".join(_psych_interp.missing_information))

    with st.expander("الأدلة التقنية" if ar else "Technical evidence", expanded=False):
        st.markdown("**" + ("القواعد المدروسة (معرفات خام)" if ar else "Rules considered (raw IDs)") + "**")
        rule_rows = plain_language_rule_rows(analysis["rule_evaluations"], language)
        st.dataframe([
            {"rule_id": r["rule_id"], "label": r["label"], "status": r["status"], "message": r["message"],
             "confidence_ceiling": r["confidence_ceiling"], "matched_evidence_ids": ", ".join(r["evidence_ids"])}
            for r in rule_rows
        ], width="stretch")
        st.markdown("**" + ("بنية CaseInterpretation الكاملة" if ar else "Full CaseInterpretation structure") + "**")
        st.json(_psych_interp.to_dict())

    st.subheader("نتائج قضاة الذكاء الاصطناعي" if ar else "AI Judge results")
    st.caption(
        "قاضي DOAR الحتمي يعمل دائماً أولاً؛ القاضي الدلالي (Gemini) اختياري ولا يمكنه أبداً تجاوز فشل حتمي." if ar
        else "DOAR's deterministic Judge always runs first; the optional semantic (Gemini) Judge can never "
             "override a deterministic failure."
    )
    hi_chat_log = st.session_state.get(f"hi_chat_{case_dir}", []) + st.session_state.get(f"hi_chat_clinician_{case_dir}", [])
    assistant_turns = [t for t in hi_chat_log if t["role"] == "assistant"]
    if assistant_turns:
        st.dataframe([
            {"answer": t["content"][:120], "judge_verdict": t.get("judge_verdict"),
             "judge_mode": t.get("judge_mode"), "answer_provider": t.get("answer_provider")}
            for t in assistant_turns
        ], width="stretch")
    else:
        st.caption("لا توجد محادثات Ask DOAR بعد في هذه الجلسة." if ar else "No Ask DOAR turns yet this session.")

    st.markdown("**" + ("قاضي الاتساق البصري (تدقيق فقط)" if ar else "Visual Consistency Judge (AUDIT ONLY)") + "**")
    st.caption(
        "هذا تدقيق مستقل بواسطة نموذج بصري لغوي (VLM) لا يُغذّي أبداً محرك الأدلة أو القواعد -- تدقيق فقط لمراجعة "
        "بشرية." if ar else
        "This is an independent VLM audit -- it NEVER feeds back into the evidence/rule engine. "
        "Audit only, for human review."
    )
    if st.button("تشغيل تدقيق الاتساق البصري" if ar else "Run visual consistency audit", key="run_visual_audit"):
        providers = _load_hi_providers()
        bundle = _load_live_case_bundle(str(case_dir), _detections_cache_token(str(case_dir)))
        st.session_state[audit_key] = hi.run_visual_consistency_audit(
            bundle, judge=providers["visual_consistency_judge"])
        st.rerun()
    audit_result = st.session_state.get(audit_key)
    if audit_result is None:
        st.caption("لم يُشغَّل بعد في هذه الجلسة." if ar else "Not yet run this session.")
    elif audit_result["status"] != "ok":
        st.info(f"{audit_result['status']}: {audit_result.get('reason')}")
    else:
        audit = audit_result["audit"]
        au_cols = st.columns(4)
        au_cols[0].markdown("**" + ("مدعومة" if ar else "Supported") + "**")
        for x in audit["supported_observations"]:
            au_cols[0].write("- " + x)
        au_cols[1].markdown("**" + ("مُغفلة محتملة" if ar else "Likely missed") + "**")
        for x in audit["likely_missed_observations"]:
            au_cols[1].write("- " + x)
        au_cols[2].markdown("**" + ("متنازع عليها" if ar else "Disputed") + "**")
        for x in audit["disputed_observations"]:
            au_cols[2].write("- " + x)
        au_cols[3].markdown("**" + ("تعذّر التحقق" if ar else "Unavailable checks") + "**")
        for x in audit["unavailable_checks"]:
            au_cols[3].write("- " + x)
        st.caption(("الاتفاق العام: " if ar else "Overall agreement: ") + audit["overall_agreement"])

    st.divider()
    _render_ask_doar(audience="clinician", chat_key=f"hi_chat_clinician_{case_dir}",
                      widget_prefix="clinician", is_ar=ar)

    st.divider()
    st.subheader("تقييم الأخصائي" if ar else "Psychologist feedback")
    st.caption(
        "يُحفظ هذا التقييم لهذه الحالة لأغراض التحليل البحثي (الرسالة الجامعية) -- منفصل تماماً عن مخرجات DOAR "
        "نفسها." if ar else
        "This feedback is persisted for this case for research/thesis analysis -- kept entirely separate "
        "from DOAR's own output."
    )
    existing_feedback = load_feedback(case_dir).get("entries", [])
    if existing_feedback:
        st.dataframe(existing_feedback, width="stretch")
    verdict_labels = {"agree": ("أوافق", "Agree"), "partially_agree": ("أوافق جزئياً", "Partially agree"),
                       "disagree": ("لا أوافق", "Disagree"), "cannot_assess": ("لا يمكن التقييم", "Cannot assess")}
    category_labels = {
        "overall_interpretation": ("التفسير العام", "Overall interpretation"),
        "emotion_interpretation": ("تفسير الانفعال", "Emotion interpretation"),
        "concern_domain_interpretation": ("تفسير مجالات القلق", "Concern-domain interpretation"),
        "explanation_usefulness": ("فائدة الشرح", "Explanation usefulness"),
    }
    with st.form("psychologist_feedback_form"):
        fb_name = st.text_input("اسم الأخصائي" if ar else "Reviewer name")
        fb_ratings = {}
        for category in FEEDBACK_CATEGORIES:
            fb_ratings[category] = st.radio(
                category_labels[category][0] if ar else category_labels[category][1], FEEDBACK_VERDICTS,
                format_func=lambda v: verdict_labels[v][0] if ar else verdict_labels[v][1],
                key=f"fb_{category}", horizontal=True)
        fb_comment = st.text_area("تعليق" if ar else "Comment")
        if st.form_submit_button("إرسال التقييم" if ar else "Submit feedback"):
            if not fb_name.strip():
                st.error("اسم الأخصائي مطلوب." if ar else "Reviewer name is required.")
            else:
                submit_categorized_feedback(case_dir, reviewer_name=fb_name.strip(), ratings=fb_ratings,
                                            comment=fb_comment.strip() or None)
                st.success("تم حفظ التقييم." if ar else "Feedback recorded.")
                st.rerun()
    all_feedback = export_all_feedback(CASES_DIR)
    if all_feedback:
        st.download_button(
            "تنزيل جميع تقييمات الأخصائيين (كل الحالات، JSON)" if ar else
            "Download all psychologist feedback (all cases, JSON)",
            json.dumps(all_feedback, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="psychologist_feedback_export.json")

    _rtl_close()

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
    with st.expander(f"All {len(analysis['rule_evaluations'])} rule evaluations", expanded=False):
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
    with st.expander(f"All {len(analysis['evidence'])} evidence items", expanded=False):
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
    st.subheader("Ask DOAR -- provenance and judge results (Human Interaction Layer)")
    st.info("Ask DOAR is wired to src/doar/human_interaction.py -- deterministic verification always "
            "runs; the optional Gemini answer/judge providers are used only when GEMINI_API_KEY is set, "
            "and always degrade to a deterministic fallback on failure. See LLM_GROUNDING_AND_SAFETY_DESIGN.md. "
            "The fixed response-Judge checklist (evidence grounding, verified/uncertain/rejected fidelity, "
            "missing-detector-!=-absence, page-assessability gates, rule eligibility, source validity, "
            "no diagnosis/causal invention, no single-weak-observation escalation, external-research and "
            "visual-audit separation, alternatives/limitations preserved, question answered) is enforced in "
            "deterministic_verify()/DeterministicJudge and in GeminiJudge's system prompt.")
    for label, key in (("Parent chat", f"hi_chat_{case_dir}"), ("Psychologist chat", f"hi_chat_clinician_{case_dir}")):
        hi_chat_log = st.session_state.get(key, [])
        assistant_turns = [t for t in hi_chat_log if t["role"] == "assistant"]
        st.caption(label)
        if assistant_turns:
            st.dataframe([
                {"answer": t["content"], "provenance": "; ".join(t.get("provenance") or []) or "original_saved_evidence",
                 "judge_verdict": t.get("judge_verdict"), "judge_mode": t.get("judge_mode"),
                 "answer_provider": t.get("answer_provider")}
                for t in assistant_turns
            ], width="stretch")
            with st.expander(f"Per-turn Judge checklist detail ({label})", expanded=False):
                st.json([t.get("judge_details", {}) for t in assistant_turns])
        else:
            st.caption("No chat turns yet this session.")

    st.subheader("Visual Consistency Judge -- raw audit result (AUDIT ONLY)")
    st.caption("Never written into detections.json/analysis.json/structured_analysis.json -- session-only, "
               "never wired into answer_question/deterministic_verify. Triggered manually from the "
               "Psychologist tab.")
    audit_result = st.session_state.get(f"visual_audit_{case_dir}")
    if audit_result is not None:
        st.json(audit_result)
    else:
        st.caption("Not yet run this session.")

    with st.expander("Psychologist feedback -- raw (this case)", expanded=False):
        st.json(load_feedback(case_dir))

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
                 "source": e.get("source"), "has_crop": bool(e.get("crop_ref")),
                 "memory_status": e.get("memory_status")}
                for e in entities
            ], width="stretch")
            st.caption(
                "possible_subtypes/visual_similarities are structurally present but intentionally empty "
                "in this phase -- no subtype/similarity computation exists yet; populating them now would "
                "mean fabricating data. memory_status is always 'not_indexed' -- Visual Memory does not "
                "exist yet. case_verification_status is one of unreviewed/verified/uncertain/rejected -- "
                "always separate from model_validation_status. source='visual_observer' entities have NO "
                "backing detector finding and are structurally invisible to the rule engine regardless of "
                "verification status.")
            crop_entities = [e for e in entities if e.get("crop_ref")]
            if crop_entities:
                st.caption(f"{len(crop_entities)} entity crop(s) available:")
                crop_cols = st.columns(6)
                for i, e in enumerate(crop_entities):
                    crop_path = resolve_artifact_path(case_dir, e["crop_ref"])
                    if crop_path.exists():
                        crop_cols[i % 6].image(
                            str(crop_path),
                            caption=f"{e['canonical_label']} ({e['case_verification_status']})")
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
        st.caption("detections.json is the honest 'unavailable' stub for this case -- most commonly because "
                   "deep visual analysis (Grounding-DINO/OWLv2) has not been requested yet via the "
                   "Psychologist view's 'Run deep visual analysis' button (Analyze itself never runs it "
                   "automatically, for performance); it can also mean the scan was run and failed. Either "
                   "way this is \"not run / failed\", never \"no objects found\".")
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
        with st.expander(f"All {registry_v2_doc['rule_count']} rules", expanded=False):
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

    st.header("14. Research runtime configuration")
    st.caption(
        "Which experiment/artifact currently backs each scientific role, and its real, current status. "
        "Changing the winner (once Colab selects one) means pointing this registry at a new artifact -- "
        "never rewriting CaseInterpretation. See doar.research_runtime / doar.experiment_manifest.")
    from doar.research_runtime import resolve_research_runtime_configuration
    _runtime_cfg = resolve_research_runtime_configuration()
    st.dataframe([
        {"role": c.role, "experiment": c.experiment, "status": c.status, "model_or_artifact": c.model_or_artifact,
         "implementation": c.implementation, "checkpoint": c.checkpoint, "preprocessing": c.preprocessing}
        for c in _runtime_cfg.components()
    ], width="stretch")
    with st.expander("Full research runtime configuration (raw JSON)", expanded=False):
        st.json(_runtime_cfg.to_dict())
    with st.expander("Example experiment artifact manifest (current MobileNet development artifact)", expanded=False):
        from doar.experiment_manifest import example_mobilenet_v3_small_development_manifest
        st.json(example_mobilenet_v3_small_development_manifest().to_dict())
        st.caption("Contract: doar.experiment_manifest.ExperimentManifest / load_experiment_manifest() -- "
                   "the loader shape future Colab experiment exports should target.")

    st.header("15. Traceability: conclusion -> domain -> aggregation -> evidence family -> rule -> "
              "detection/feature -> model/artifact -> experiment")
    _tech_interp = _build_full_interpretation(
        _load_live_case_bundle(str(case_dir), _detections_cache_token(str(case_dir))), str(case_dir))
    for c in _tech_interp.concern_domains:
        if c.support_level in ("NONE", "NOT_CURRENTLY_ASSESSED"):
            continue
        with st.expander(f"{c.domain} -- {c.support_level} (ratio={c.support_ratio})"):
            st.write(f"1. CONCLUSION: {c.domain_label} = {c.support_level} "
                     f"(support_ratio={c.support_ratio}, assessable_families={c.assessable_family_count})")
            st.write(f"2. AGGREGATION: {_runtime_cfg.evidence_aggregator.model_or_artifact} "
                     f"({_runtime_cfg.evidence_aggregator.implementation})")
            st.write(f"3. EVIDENCE FAMILIES: {', '.join(c.independent_evidence_families) or '(none)'}")
            st.write(f"4. RULES: {', '.join(c.supporting_rule_ids) or '(none)'}")
            st.write(f"5. FEATURES/DETECTIONS: {', '.join(c.supporting_observations) or '(none)'}")
            st.write(f"6. MODEL/ARTIFACT: {_runtime_cfg.semantic_concept_provider.model_or_artifact or 'deterministic rule engine (no learned model)'}")
            st.write(f"7. EXPERIMENT: {_runtime_cfg.semantic_concept_provider.experiment or 'n/a (not experiment-derived)'}")
    if _tech_interp.gemini_candidates:
        st.subheader("Gemini candidate -> verification traceability")
        st.dataframe([c.to_dict() for c in _tech_interp.gemini_candidates], width="stretch")
    if _tech_interp.gemini_global_observation:
        with st.expander("Gemini global observer -- raw structured output", expanded=False):
            st.json(_tech_interp.gemini_global_observation)
