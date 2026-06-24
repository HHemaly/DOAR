"""
ui/app.py — Local Gradio demo UI for DOAR v2.

Provides a simple web interface for parents, children, and psychologists
to upload a drawing and receive a safe, non-diagnostic analysis.

Usage:
    python ui/app.py
    python ui/app.py --share   # generates a public URL for sharing

Then open: http://127.0.0.1:7860
"""

from __future__ import annotations
import os
import sys
import json
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Import pipeline ───────────────────────────────────────────────────────────
try:
    from pipeline import run_full_pipeline_v2, OUTPUT_DIR
except ImportError as e:
    print(f"Pipeline import error: {e}")
    sys.exit(1)

# ── Import Arabic translator (optional) ───────────────────────────────────────
try:
    from arabic_translator import translate_output
    _AR_AVAILABLE = True
except ImportError:
    _AR_AVAILABLE = False

# ── Gradio ────────────────────────────────────────────────────────────────────
try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis function (called by Gradio)
# ─────────────────────────────────────────────────────────────────────────────

def analyse(image_path: str, question: str, language: str, run_ocr_flag: bool):
    """
    Run the DOAR v2 pipeline and return formatted UI outputs.

    Returns:
        parent_answer_en  — English analysis for parents
        parent_answer_ar  — Arabic translation (if requested)
        gentle_questions  — formatted list of follow-up questions
        technical_json    — internal JSON (for psychologist view)
        report_image      — path to report card PNG
        status_badge      — safety status string
    """
    if image_path is None:
        return (
            "Please upload a drawing image first.",
            "", "", "{}", None, "No image uploaded",
        )

    # Save uploaded file to a temp location if it's a numpy array (Gradio webcam)
    if not isinstance(image_path, str):
        import numpy as np
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        if _CV2:
            cv2.imwrite(tmp.name, cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR))
        else:
            from PIL import Image as _PIL
            _PIL.fromarray(image_path).save(tmp.name)
        image_path = tmp.name

    session_dir = OUTPUT_DIR
    os.makedirs(session_dir, exist_ok=True)

    try:
        result = run_full_pipeline_v2(
            image_path,
            parent_question=question.strip(),
            run_ocr=run_ocr_flag,
            run_arabic=(language == "Arabic" and _AR_AVAILABLE),
            session_dir=session_dir,
        )
    except Exception as e:
        return (
            f"Analysis error: {e}",
            "", "", "{}", None, "ERROR",
        )

    parent_out = result.get("parent_facing_output", {})
    parent_out_ar = result.get("parent_facing_output_ar") or {}
    judgment   = result.get("final_judgment", {})
    doc        = result.get("internal_technical_json", {})

    # English answer
    answer_en = parent_out.get("parent_answer", "No answer generated.")
    disclaimer = parent_out.get("disclaimer", "")
    safety_note = parent_out.get("safety_note", "")
    answer_en += f"\n\n---\n*Safety note: {safety_note}*\n\n*{disclaimer}*"

    # Arabic answer
    if language == "Arabic" and parent_out_ar:
        answer_ar = parent_out_ar.get("parent_answer", "")
        qs_ar     = parent_out_ar.get("gentle_questions", [])
        if answer_ar:
            disc_ar = parent_out_ar.get("disclaimer", "")
            sn_ar   = parent_out_ar.get("safety_note", "")
            answer_ar += f"\n\n---\n*{sn_ar}*\n\n*{disc_ar}*"
    else:
        answer_ar = ""

    # Gentle questions
    qs_en = parent_out.get("gentle_questions", [])
    qs_text = "\n".join(f"• {q}" for q in qs_en)
    if language == "Arabic" and parent_out_ar.get("gentle_questions"):
        qs_ar_text = "\n".join(f"• {q}" for q in parent_out_ar["gentle_questions"])
        qs_text = f"**English:**\n{qs_text}\n\n**Arabic / العربية:**\n{qs_ar_text}"

    # Technical JSON (compact)
    tech = {
        "judge_status":      judgment.get("final_answer_status"),
        "safe_to_show":      judgment.get("safe_to_show"),
        "checks_passed":     f"{judgment.get('checks_passed', 0)}/{judgment.get('checks_total', 10)}",
        "emotion_tendency":  doc.get("feature_based_emotional_tendency", {}).get("estimated_emotion"),
        "validation":        doc.get("validation_summary"),
        "issues":            judgment.get("issues", []),
        "v1_rules_active":   [r.get("rule_id") for r in doc.get("psychological_rule_activations", []) if r.get("activated")],
        "v2_rules_active":   [r.get("rule_id") for r in doc.get("psychological_rule_activations_v2", []) if r.get("activated")],
        "themes":            [t.get("theme") for t in doc.get("theme_scores_v2", [])],
    }
    tech_str = json.dumps(tech, indent=2, ensure_ascii=False)

    # Report card image
    report_img_path = result.get("saved_paths", {}).get("report_card")
    if not report_img_path:
        report_img_path = _make_quick_chart(image_path, doc, parent_out, judgment, session_dir)

    # Status badge
    status = judgment.get("final_answer_status", "UNKNOWN")
    status_badge = {
        "PASS": "✅ PASS — Safe to show",
        "REWRITE_REQUIRED": "⚠️ REWRITE REQUIRED — Sanitised version shown",
        "BLOCK": "❌ BLOCKED — Safe fallback used",
    }.get(status, f"Status: {status}")

    return answer_en, answer_ar, qs_text, tech_str, report_img_path, status_badge


# ─────────────────────────────────────────────────────────────────────────────
# Quick chart when output_manager didn't produce a report card
# ─────────────────────────────────────────────────────────────────────────────

def _make_quick_chart(image_path, doc, parent_out, judgment, out_dir):
    if not _MPL:
        return None
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#f8f9fa")
        fig.suptitle("DOAR Analysis Summary", fontsize=13, fontweight="bold")

        # Left: drawing thumbnail
        ax_img = axes[0]
        if _CV2:
            img = cv2.imread(image_path)
            if img is not None:
                ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_img.axis("off")
        ax_img.set_title(os.path.basename(image_path), fontsize=9)

        # Right: metrics
        ax_met = axes[1]
        ax_met.axis("off")
        cf = doc.get("color_features", {})
        cp = doc.get("composition_features", {})
        ht = doc.get("feature_based_emotional_tendency", {})
        vs = doc.get("validation_summary", {})
        rows = [
            ["Emotion tendency", ht.get("estimated_emotion", "?")],
            ["Empty space", f"{cp.get('empty_space_ratio', 0)*100:.0f}%"],
            ["Color diversity", str(cf.get("color_diversity_count", "?"))],
            ["Dark tones", f"{cf.get('dark_dominance', 0)*100:.0f}%"],
            ["Verified claims", str(vs.get("verified_claims", "?"))],
            ["Judge", judgment.get("final_answer_status", "?")],
        ]
        tbl = ax_met.table(
            cellText=rows, colLabels=["Metric", "Value"],
            cellLoc="left", loc="center", colWidths=[0.6, 0.4],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#ecf0f1" if r % 2 == 0 else "white")
        ax_met.set_title("Key Metrics", fontsize=10, fontweight="bold")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", dir=out_dir, delete=False)
        fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return tmp.name
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI layout
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
.title-block { background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
               border-radius: 10px; padding: 18px; margin-bottom: 10px; }
.title-block h1 { color: white; font-size: 1.6em; margin: 0; }
.title-block p  { color: #d6eaf8; margin: 4px 0 0; font-size: 0.95em; }
.disclaimer-box { background: #fef9e7; border-left: 4px solid #f39c12;
                  padding: 10px 14px; border-radius: 5px; font-size: 0.85em; }
"""

DISCLAIMER_TEXT = (
    "**Disclaimer:** Drawing-based psychological indicators are NOT diagnostic on their own. "
    "They should be interpreted cautiously alongside the child's verbal explanation, "
    "developmental age, cultural context, caregiver input, and professional clinical "
    "assessment when needed."
)


def build_ui():
    with gr.Blocks(css=CSS, title="DOAR — Drawing Analysis") as demo:

        gr.HTML("""
        <div class="title-block">
          <h1>DOAR — Drawing Observation & Analysis Report</h1>
          <p>Upload a child's drawing to receive a safe, non-diagnostic, parent-friendly analysis.</p>
        </div>
        """)

        gr.Markdown(DISCLAIMER_TEXT, elem_classes=["disclaimer-box"])

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Drawing",
                    type="filepath",
                    image_mode="RGB",
                )
                question_input = gr.Textbox(
                    label="Parent question (optional)",
                    placeholder="e.g. What does the drawing show?  Is the child expressing sadness?",
                    lines=2,
                )
                language_radio = gr.Radio(
                    choices=["English", "Arabic"],
                    value="English",
                    label="Output language",
                )
                ocr_checkbox = gr.Checkbox(
                    label="Run text detection (OCR)",
                    value=True,
                )
                submit_btn = gr.Button("Analyse Drawing", variant="primary", size="lg")

            with gr.Column(scale=2):
                status_output = gr.Textbox(label="Safety Status", interactive=False)

                with gr.Tabs():
                    with gr.Tab("Parent Answer (English)"):
                        answer_en_output = gr.Markdown(label="Analysis")
                    with gr.Tab("Parent Answer (Arabic / العربية)"):
                        answer_ar_output = gr.Markdown(label="Arabic Analysis")
                    with gr.Tab("Gentle Questions"):
                        questions_output = gr.Markdown(label="Follow-up questions for the child")
                    with gr.Tab("Report Card"):
                        report_image_output = gr.Image(
                            label="Visual Report Card",
                            type="filepath",
                            interactive=False,
                        )
                    with gr.Tab("Technical JSON (Psychologist View)"):
                        json_output = gr.Code(language="json", label="Internal Analysis JSON")

        submit_btn.click(
            fn=analyse,
            inputs=[image_input, question_input, language_radio, ocr_checkbox],
            outputs=[
                answer_en_output,
                answer_ar_output,
                questions_output,
                json_output,
                report_image_output,
                status_output,
            ],
            api_name="analyse",
        )

        gr.Examples(
            examples=[
                [None, "What colours is the child using?", "English", False],
                [None, "Is the child showing any emotional patterns?", "Arabic", True],
            ],
            inputs=[image_input, question_input, language_radio, ocr_checkbox],
            label="Example questions (upload your own image first)",
        )

        gr.HTML("""
        <div style="text-align:center; color:#888; margin-top:20px; font-size:0.8em;">
          DOAR v2 | For research and supportive use only | Not a diagnostic tool
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser(description="DOAR Gradio UI")
    p.add_argument("--share",  action="store_true", help="Create public share link")
    p.add_argument("--port",   type=int, default=7860,   help="Port (default 7860)")
    p.add_argument("--host",   default="127.0.0.1", help="Host (default 127.0.0.1)")
    args = p.parse_args()

    demo = build_ui()
    print(f"\nStarting DOAR UI at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
