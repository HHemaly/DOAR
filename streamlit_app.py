from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="DOAR v3 reviewer", layout="wide")
st.title("DOAR v3 - Psychologist Review")
st.warning("Research decision-support only. A drawing alone cannot establish a diagnosis.")

case_text = st.sidebar.text_input("Case folder", value=str(Path("outputs/cases").resolve()))
case = Path(case_text)
analysis_path = case / "analysis.json"
if not analysis_path.exists():
    st.info("Enter a case folder containing analysis.json.")
    st.stop()

analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
judges = json.loads((case / "judges.json").read_text(encoding="utf-8"))
review_path = case / "clinician_review.json"
review = json.loads(review_path.read_text(encoding="utf-8"))


def artifact(name: str) -> str:
    value = analysis["artifacts"][name]
    return str(case / value)

tabs = st.tabs([
    "Summary", "Original Image", "Quality", "Segmentation", "Composition",
    "Colours", "Lines and Strokes", "Shapes", "Objects", "OCR", "Emotion",
    "Rules", "Suggested Concern Profile", "Judges", "Psychologist Review",
    "Reports", "Q&A", "Experiments",
])
with tabs[0]:
    st.json({
        "schema_version": analysis["schema_version"],
        "segmentation_status": analysis["segmentation"]["status"],
        "concern_count": len(analysis["concerns"]),
        "disclaimer": analysis["safety_disclaimer"],
    })
with tabs[1]:
    st.image(artifact("normalized_image"))
with tabs[2]:
    st.json(analysis["quality"])
with tabs[3]:
    cols = st.columns(3)
    cols[0].image(artifact("foreground_mask"), caption="Selected mask")
    cols[1].image(artifact("foreground_only"), caption="Foreground only")
    cols[2].image(artifact("density_map"), caption="Density")
    st.json(analysis["segmentation"])
with tabs[4]:
    st.image(artifact("feature_overlay"))
    st.json(analysis["composition"])
with tabs[5]:
    st.json(analysis["colour"])
with tabs[6]:
    st.image(artifact("stroke_map"))
with tabs[7]:
    st.info("Shape features are preliminary; verified shape detection is not yet available.")
with tabs[8]:
    st.json(json.loads((case / "detections.json").read_text(encoding="utf-8")))
with tabs[9]:
    st.info("OCR module unavailable in this release.")
with tabs[10]:
    st.json(json.loads((case / "emotion.json").read_text(encoding="utf-8")))
with tabs[11]:
    st.dataframe(analysis["rule_evaluations"], use_container_width=True)
with tabs[12]:
    st.json(analysis["concerns"])
with tabs[13]:
    st.json(judges)
with tabs[14]:
    reviewer = st.text_input("Reviewer ID or anonymized code")
    mask_rating = st.selectbox("Segmentation mask", ["uncertain", "approve", "reject"])
    rule_comment = st.text_area("Rule/detection/concern corrections (Arabic or English)")
    uncertainty = st.checkbox("Mark review as uncertain", value=True)
    if st.button("Append review"):
        event = {
            "reviewer_id": reviewer or "anonymous",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mask_rating": mask_rating,
            "comment": rule_comment,
            "uncertain": uncertainty,
        }
        review.setdefault("history", []).append(event)
        review["status"] = "submitted"
        review["ai_output_preserved"] = True
        review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success("Review appended. Original AI output was not overwritten.")
    st.json(review)
with tabs[15]:
    for report in sorted((case / "reports").glob("*.html")):
        st.download_button(report.name, report.read_bytes(), file_name=report.name)
with tabs[16]:
    st.caption("Answers are grounded ONLY in this case's saved evidence and cite evidence IDs.")
    qa_language = st.selectbox("Language", ["en", "ar"], key="qa_lang")
    qa_question = st.text_input("Question about this case", key="qa_question")
    if qa_question:
        from doar.qa import answer as _qa_answer
        try:
            response = _qa_answer(qa_question, analysis, judges, qa_language)
            st.json(response)
        except Exception as exc:  # pragma: no cover - UI guard
            st.error(f"Could not answer from saved evidence: {exc}")
    st.divider()
    st.caption("Equivalent CLI:")
    st.code('python main.py qa --analysis CASE/analysis.json --question "..." --language en')
with tabs[17]:
    st.info("Load validation_leaderboard.json after local model experiments.")
