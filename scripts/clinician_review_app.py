#!/usr/bin/env python
"""DOAR Psychologist/Clinician Review App (Streamlit) -- first clinician
usability/design session prototype, NOT clinical validation.

Browses the 15 cached development cases end to end: original drawing ->
raw Observer candidates -> Verifier result -> eligible/ineligible
evidence -> atomic rule -> evidence family -> concern domain -> candidate
hypothesis -> clinician package -> parent package. Every pipeline step is
reused as-is from `reasoning_chain.py` and `scripts/run_development_
benchmark.py` -- this file adds NO new reasoning logic, NO new rule
matching, and never weakens the existing safety gate ("only verified
evidence may trigger a rule" -- `reasoning_chain.check_visual_
preconditions` already enforces this and is called unmodified here).

**No Gemini calls when browsing cases** -- this file never imports or
calls `run_live_observer_and_verifier`, `GeminiVisualObserver`, or
`GeminiVisualVerifier`. It only ever reads already-cached verification
JSON via `run_development_benchmark.find_saved_saved_verification_rows`
(live cache first, then legacy dev-check dirs -- same search order the
benchmark runner uses).

Three views:
  A. Clinician view (default) -- drawing + bbox overlay, observations
     grouped VERIFIED/UNCERTAIN/REJECTED-UNREVIEWED (nothing hidden),
     evidence/reasoning trace, candidate hypothesis (or the explicit
     "no hypothesis" message), and feedback controls.
  B. Parent view preview -- the SAME evidence, parent-safe wording only
     (`reasoning_chain.build_parent_package`, already code-constrained,
     no rule IDs/disorder labels/LLM-invented text).
  C. Technical/audit view -- the full trace, for thesis/debugging.

Launch:
    streamlit run scripts/clinician_review_app.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar import reasoning_chain as rc  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_development_benchmark", ROOT / "scripts" / "run_development_benchmark.py")
rdb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdb)

# DOAR_CLINICIAN_FEEDBACK_DIR lets tests redirect saves to a throwaway
# directory -- unset in normal use, where this is exactly
# ROOT / "outputs" / "clinician_feedback".
FEEDBACK_DIR = Path(os.environ.get("DOAR_CLINICIAN_FEEDBACK_DIR", str(ROOT / "outputs" / "clinician_feedback")))

STATUS_COLORS = {"verified": "#1a9850", "uncertain": "#e08214", "rejected": "#d73027", "unreviewed": "#999999"}
STATUS_GROUP_TITLES = {"verified": "VERIFIED", "uncertain": "UNCERTAIN", "other": "REJECTED / UNREVIEWED"}

CLINICIAN_NO_HYPOTHESIS_MESSAGE = (
    "No candidate clinical hypothesis was raised from the currently eligible drawing evidence. "
    "This does not indicate absence of a condition.")
PARENT_NO_HYPOTHESIS_MESSAGE = (
    "No specific pattern was flagged by this automated review. This does not mean there is nothing "
    "to discuss -- a professional can look at the fuller picture regardless of what an automated "
    "review does or doesn't flag.")
CLINICIAN_ASSESSMENT_DISCLAIMER = "Candidate for professional assessment -- not a diagnosis."

OBSERVATION_FEEDBACK_OPTIONS = ("(no feedback yet)", "Confirm visually", "Reject", "Cannot determine")
RELEVANCE_VERDICT_OPTIONS = ("(no feedback yet)", "Agree -- worth investigating", "Disagree", "Insufficient evidence")


# ---------------------------------------------------------------------------
# Pure data-loading/preparation -- no Streamlit calls below this point until
# the render_* functions, so this half is directly unit-testable.
# ---------------------------------------------------------------------------


def list_case_ids() -> list[str]:
    return [im["image_id"] for im in rdb.load_development_set()]


def load_case_bundle(image_id: str) -> dict | None:
    """Everything one case needs, purely from cache. Returns None if no
    saved Observer/Verifier data exists for this image_id (never makes a
    live call to find out more)."""
    images = rdb.load_development_set()
    image = next((im for im in images if im["image_id"] == image_id), None)
    if image is None:
        return None
    rows, source_path = rdb.find_saved_verification_rows(image_id)
    if rows is None:
        return {"image": image, "rows": None, "source_path": None, "entities": None,
                "conditions": None, "checks": None}
    entities = rdb.entities_from_verification_rows(rows)
    conditions = rdb.run_conditions_for_image(image_id, rows)
    checks = rc.check_visual_preconditions(entities)
    return {"image": image, "rows": rows, "source_path": source_path, "entities": entities,
            "conditions": conditions, "checks": checks}


def group_rows_by_status(rows: list[dict]) -> dict[str, list[dict]]:
    """VERIFIED / UNCERTAIN / REJECTED-UNREVIEWED -- nothing is dropped,
    every candidate lands in exactly one group."""
    groups: dict[str, list[dict]] = {"verified": [], "uncertain": [], "other": []}
    for row in rows:
        status = row["verification_status"]
        if status == "verified":
            groups["verified"].append(row)
        elif status == "uncertain":
            groups["uncertain"].append(row)
        else:
            groups["other"].append(row)
    return groups


def draw_annotated_image(image_path: Path, rows: list[dict]) -> Image.Image:
    """Bbox overlay color-coded by verification status -- a fresh
    in-memory copy, never written to disk. A candidate with no bbox is
    simply not drawn (nothing hidden from the text listing, just not
    drawable)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for row in rows:
        bbox = row["observer_candidate"].get("bbox")
        if not bbox:
            continue
        x, y, bw, bh = bbox
        left, top, right, bottom = x * w, y * h, (x + bw) * w, (y + bh) * h
        color = STATUS_COLORS.get(row["verification_status"], "#000000")
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left + 2, max(0, top - 14)), row["observation_id"], fill=color)
    return img


def build_parent_view_sections(hypotheses: list) -> list[dict]:
    """One parent-safe section per hypothesis, built ENTIRELY from
    `reasoning_chain.build_parent_package` (already code/data-constrained
    -- no rule IDs, no disorder labels, no LLM-invented wording). Empty
    list means the caller should show PARENT_NO_HYPOTHESIS_MESSAGE."""
    return [rc.build_parent_package(h) for h in hypotheses]


def git_commit_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Clinician feedback storage -- one JSON per (session, case), plus a
# rebuilt-every-save CSV summary. Never stores API keys; never trains
# anything from this data.
# ---------------------------------------------------------------------------


def feedback_path(session_id: str, case_id: str) -> Path:
    safe_session = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    out_dir = FEEDBACK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe_session}__{case_id}.json"


def save_feedback(session_id: str, case_id: str, payload: dict) -> Path:
    """Deterministic, idempotent: saving the same (session_id, case_id)
    again overwrites the prior file rather than appending or duplicating.
    Uses the same atomic write already validated for the live cache."""
    record = dict(payload)
    record["session_id"] = session_id
    record["case_id"] = case_id
    record["timestamp"] = record.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%S")
    record["git_commit"] = git_commit_sha()
    path = feedback_path(session_id, case_id)
    rdb._atomic_write_json(path, record)
    rebuild_feedback_summary_csv()
    return path


def load_feedback(session_id: str, case_id: str) -> dict | None:
    path = feedback_path(session_id, case_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild_feedback_summary_csv() -> Path:
    """Rebuilt from scratch from every saved JSON file every time --
    always consistent with what's on disk, never an append-only log that
    could drift or duplicate rows across repeated saves."""
    import csv

    out_dir = FEEDBACK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    rows = []
    for json_path in sorted(out_dir.glob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({
            "session_id": record.get("session_id", ""), "case_id": record.get("case_id", ""),
            "timestamp": record.get("timestamp", ""), "git_commit": record.get("git_commit", ""),
            "num_observations_reviewed": len(record.get("observation_feedback", {})),
            "num_hypotheses_reviewed": len(record.get("hypothesis_feedback", [])),
            "general_comments": record.get("general_comments", ""),
            "missing_information": record.get("missing_information", ""),
            "what_should_be_removed": record.get("what_should_be_removed", ""),
            "what_should_be_added": record.get("what_should_be_added", ""),
            "wording_too_strong": record.get("wording_too_strong", ""),
            "references_useful": record.get("references_useful", ""),
            "evidence_trace_understandable": record.get("evidence_trace_understandable", ""),
        })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["session_id", "case_id", "timestamp", "git_commit", "num_observations_reviewed",
                      "num_hypotheses_reviewed", "general_comments", "missing_information",
                      "what_should_be_removed", "what_should_be_added", "wording_too_strong",
                      "references_useful", "evidence_trace_understandable"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def _new_session_id() -> str:
    return f"anon_{uuid.uuid4().hex[:8]}"


def render_clinician_view(bundle: dict, case_id: str, session_id: str) -> None:
    image_path = ROOT / bundle["image"]["relative_path"]
    rows = bundle["rows"]

    if rows is None:
        st.warning("No cached Observer/Verifier data available for this case.")
        st.image(str(image_path), width='stretch')
        return

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Drawing (bbox overlay, color-coded by Verifier status)")
        st.image(draw_annotated_image(image_path, rows), width='stretch')
        st.caption("🟩 Verified   🟧 Uncertain   🟥 Rejected   ⬜ Unreviewed")

    with col_right:
        st.subheader("Visual observations")
        groups = group_rows_by_status(rows)
        prior_feedback = load_feedback(session_id, case_id) or {}
        observation_feedback = dict(prior_feedback.get("observation_feedback", {}))
        for group_key in ("verified", "uncertain", "other"):
            group_rows = groups[group_key]
            st.markdown(f"**{STATUS_GROUP_TITLES[group_key]}** ({len(group_rows)})")
            if not group_rows:
                st.caption("(none)")
            for row in group_rows:
                oc = row["observer_candidate"]
                with st.container(border=True):
                    st.markdown(f"`{row['observation_id']}` **{oc['label']}**")
                    st.caption(
                        f"bbox: {oc.get('bbox')}  |  observer confidence: {oc.get('confidence')}  |  "
                        f"verifier status: **{row['verification_status']}**  |  "
                        f"verifier label: {row.get('verifier_independent_label') or '(none)'}  |  "
                        f"verifier confidence: {row.get('verifier_confidence')}")
                    key = f"obs_{session_id}_{case_id}_{row['observation_id']}"
                    current = observation_feedback.get(row["observation_id"], {}).get("verdict", OBSERVATION_FEEDBACK_OPTIONS[0])
                    verdict = st.selectbox("Clinician verdict", OBSERVATION_FEEDBACK_OPTIONS,
                                            index=OBSERVATION_FEEDBACK_OPTIONS.index(current) if current in OBSERVATION_FEEDBACK_OPTIONS else 0,
                                            key=key, label_visibility="collapsed")
                    observation_feedback[row["observation_id"]] = {"observer_label": oc["label"], "verdict": verdict}

    st.markdown("---")
    st.subheader("Evidence / reasoning")
    conditions = bundle["conditions"]
    doar = conditions["doar_full_pipeline"]
    matrix = rc.load_rule_matrix()

    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        st.markdown("**Eligible verified observations**")
        st.write(", ".join(doar["verified_labels"]) or "(none)")
        st.markdown("**Triggered atomic rules**")
        if doar["eligible_matches"]:
            for m in doar["eligible_matches"]:
                st.markdown(f"- `{m.rule_id}` ({m.evidence_family} / {m.concern_domain}) <- {list(m.matched_entity_ids)}")
                st.caption(f"Source claim: {m.source_claim}  |  Strength: {matrix[m.rule_id]['evidence_strength_as_written']}")
        else:
            st.caption("(none triggered)")
    with ev_col2:
        st.markdown("**Evidence families represented**")
        st.write(", ".join(doar["evidence_families"].keys()) or "(none)")
        st.markdown("**Concern domains touched**")
        st.write(", ".join(doar["concern_domains"].keys()) or "(none)")
        st.markdown("**Rejected/uncertain/unreviewed evidence (excluded from rule-firing)**")
        excluded = groups["uncertain"] + groups["other"]
        if excluded:
            st.write(", ".join(f"{r['observer_candidate']['label']} ({r['verification_status']})" for r in excluded))
        else:
            st.caption("(none -- every candidate was verified)")

    st.markdown("---")
    st.subheader("Candidate hypothesis")
    hypotheses = doar["hypotheses"]
    hypothesis_feedback = list(prior_feedback.get("hypothesis_feedback", []))
    if not hypotheses:
        st.info(CLINICIAN_NO_HYPOTHESIS_MESSAGE)
    for i, h in enumerate(hypotheses):
        cp = rc.build_clinician_package(h)
        with st.container(border=True):
            st.markdown(f"### {cp['candidate_hypothesis_label']}")
            st.markdown(f"**Support level:** {cp['support_level']}")
            st.markdown("**Why:**")
            st.write(f"- Triggered rules: {', '.join(cp['why']['triggered_rule_ids'])}")
            st.write(f"- Evidence families: {', '.join(cp['why']['evidence_families'])}")
            st.write(f"- Supporting entities: {', '.join(cp['why']['supporting_entity_ids'])}")
            for src, strength in zip(cp["why"]["source_claims"], cp["why"]["evidence_strength_as_written"]):
                st.caption(f"  • {src} (strength: {strength})")
            st.markdown(f"**Contradictions/counterevidence:** {', '.join(cp['counterevidence']) or '(none on file)'}")
            st.markdown(f"**Alternative explanations:** {', '.join(cp['alternative_explanations']) or '(none listed)'}")
            st.markdown("**Missing diagnostic information:**")
            for m in cp["missing_clinical_information"]:
                st.write(f"- {m}")
            st.warning(f"{CLINICIAN_ASSESSMENT_DISCLAIMER} {cp['disclaimer']}")

            existing = next((hf for hf in hypothesis_feedback if hf.get("hypothesis_label") == cp["candidate_hypothesis_label"]), {})
            key = f"hyp_{session_id}_{case_id}_{i}"
            current = existing.get("clinician_relevance_verdict", RELEVANCE_VERDICT_OPTIONS[0])
            verdict = st.selectbox("clinician_relevance_verdict", RELEVANCE_VERDICT_OPTIONS,
                                    index=RELEVANCE_VERDICT_OPTIONS.index(current) if current in RELEVANCE_VERDICT_OPTIONS else 0,
                                    key=key)
            st.caption("clinical_diagnostic_status: assessment_pending (not used in this usability study)")
            comment = st.text_input("Comment on this hypothesis", value=existing.get("comment", ""), key=f"{key}_comment")
            new_entry = {
                "concern_domain": h.concern_domain, "hypothesis_label": cp["candidate_hypothesis_label"],
                "clinician_relevance_verdict": verdict, "clinical_diagnostic_status": "assessment_pending",
                "comment": comment,
            }
            hypothesis_feedback = [hf for hf in hypothesis_feedback if hf.get("hypothesis_label") != cp["candidate_hypothesis_label"]]
            hypothesis_feedback.append(new_entry)

    st.markdown("---")
    st.subheader("Clinician feedback")
    general_comments = st.text_area("General comments", value=prior_feedback.get("general_comments", ""), key=f"gc_{session_id}_{case_id}")
    missing_information = st.text_area("What information is missing?", value=prior_feedback.get("missing_information", ""), key=f"mi_{session_id}_{case_id}")
    what_removed = st.text_area("What should be removed?", value=prior_feedback.get("what_should_be_removed", ""), key=f"wr_{session_id}_{case_id}")
    what_added = st.text_area("What should be added?", value=prior_feedback.get("what_should_be_added", ""), key=f"wa_{session_id}_{case_id}")
    wording_too_strong = st.radio("Is wording too strong?", ("unsure", "yes", "no"),
                                   index=("unsure", "yes", "no").index(prior_feedback.get("wording_too_strong", "unsure")),
                                   key=f"wts_{session_id}_{case_id}", horizontal=True)
    references_useful = st.radio("Are references useful?", ("unsure", "yes", "no"),
                                  index=("unsure", "yes", "no").index(prior_feedback.get("references_useful", "unsure")),
                                  key=f"ru_{session_id}_{case_id}", horizontal=True)
    trace_understandable = st.radio("Is the evidence trace understandable?", ("unsure", "yes", "no"),
                                     index=("unsure", "yes", "no").index(prior_feedback.get("evidence_trace_understandable", "unsure")),
                                     key=f"tu_{session_id}_{case_id}", horizontal=True)

    if st.button("Save feedback for this case", key=f"save_{session_id}_{case_id}", type="primary"):
        payload = {
            "observation_feedback": observation_feedback, "hypothesis_feedback": hypothesis_feedback,
            "general_comments": general_comments, "missing_information": missing_information,
            "what_should_be_removed": what_removed, "what_should_be_added": what_added,
            "wording_too_strong": wording_too_strong, "references_useful": references_useful,
            "evidence_trace_understandable": trace_understandable,
        }
        path = save_feedback(session_id, case_id, payload)
        st.success(f"Saved -> {path}")


def render_parent_view(bundle: dict) -> None:
    image_path = ROOT / bundle["image"]["relative_path"]
    rows = bundle["rows"]
    st.image(str(image_path), width='stretch')
    if rows is None:
        st.warning("No cached data available for this case.")
        return

    doar = bundle["conditions"]["doar_full_pipeline"]
    st.subheader("What was visible in the drawing")
    if doar["verified_labels"]:
        st.write(", ".join(sorted(set(doar["verified_labels"]))))
    else:
        st.caption("(no confidently-identified items)")

    st.markdown("---")
    sections = build_parent_view_sections(doar["hypotheses"])
    if not sections:
        st.info(PARENT_NO_HYPOTHESIS_MESSAGE)
    for pp in sections:
        with st.container(border=True):
            st.markdown(f"**What was observed:** {pp['what_was_observed']}")
            st.markdown(f"**In simple terms:** {pp['simple_explanation']}")
            st.markdown(f"**Uncertainty:** {pp['uncertainty']}")
            if pp["gentle_questions_to_ask"]:
                st.markdown("**Questions you could ask your child:**")
                for q in pp["gentle_questions_to_ask"]:
                    st.write(f"- {q}")
            st.markdown(f"**What to monitor:** {pp['what_to_monitor']}")
            st.markdown(f"**When professional review may help:** {pp['when_professional_review_may_help']}")
    st.warning("This is not a diagnosis.")


def render_technical_view(bundle: dict, case_id: str) -> None:
    rows = bundle["rows"]
    if rows is None:
        st.warning("No cached Observer/Verifier data for this case.")
        return

    st.subheader("Provenance")
    st.write(f"Source file: `{bundle['source_path']}`")
    if rows:
        st.write(f"Observer model: `{rows[0].get('observer_model')}`  |  Verifier model: `{rows[0].get('verifier_model')}`")

    st.subheader("Raw Observer candidates -> Verifier result")
    st.json([
        {"observation_id": r["observation_id"], "observer_candidate": r["observer_candidate"],
         "verifier_independent_label": r.get("verifier_independent_label"),
         "verifier_independent_alternative_labels": r.get("verifier_independent_alternative_labels"),
         "verifier_confidence": r.get("verifier_confidence"), "verification_status": r["verification_status"],
         "verifier_notes": r.get("verifier_notes"), "verifier_source_note": r.get("verifier_source_note")}
        for r in rows
    ])

    st.subheader("Entities (post Observer+Verifier merge)")
    st.json([rdb._entity_to_dict(e) for e in bundle["entities"]])

    st.subheader("Visual precondition results (all 41 rules)")
    checks = bundle["checks"]
    by_status: dict[str, list] = {}
    for c in checks:
        by_status.setdefault(c.status, []).append(c)
    for status, group in sorted(by_status.items()):
        with st.expander(f"{status} ({len(group)})"):
            st.json([{"rule_id": c.rule_id, "reason": c.reason, "matched_entity_ids": list(c.matched_entity_ids)}
                      for c in group])

    st.subheader("Eligible atomic rule matches")
    doar = bundle["conditions"]["doar_full_pipeline"]
    st.json([rdb._match_to_dict(m) for m in doar["eligible_matches"]])

    st.subheader("Candidate hypotheses + full packages")
    for pkg in doar["packages"]:
        st.json({
            "hypothesis": rdb._hypothesis_to_dict(pkg["hypothesis"]),
            "clinician_package": pkg["clinician_package"],
            "parent_package": pkg["parent_package"],
        })
    if not doar["packages"]:
        st.caption("(no hypotheses)")

    st.caption(f"case_id={case_id}  |  git_commit={git_commit_sha()}")


def main() -> None:
    st.set_page_config(page_title="DOAR Clinician Review (prototype)", layout="wide")
    st.title("DOAR -- Clinician Review Prototype")
    st.caption(
        "First clinician usability/design session -- NOT clinical validation. Uses cached development-set "
        "results only; no Gemini calls are made while browsing.")

    if "session_id" not in st.session_state:
        st.session_state.session_id = _new_session_id()
    st.sidebar.text_input("Reviewer/session ID (anonymous)", key="session_id")
    session_id = st.session_state.session_id or _new_session_id()

    case_ids = list_case_ids()
    case_id = st.sidebar.selectbox("Case", case_ids, key="selected_case_id")
    st.sidebar.caption(f"{len(case_ids)} development cases available.")

    bundle = load_case_bundle(case_id)
    if bundle is None:
        st.error(f"Unknown case_id: {case_id!r}")
        return

    tab_clinician, tab_parent, tab_technical = st.tabs(
        ["Clinician View", "Parent View Preview", "Technical / Audit View"])
    with tab_clinician:
        render_clinician_view(bundle, case_id, session_id)
    with tab_parent:
        render_parent_view(bundle)
    with tab_technical:
        render_technical_view(bundle, case_id)


if __name__ == "__main__":
    main()
