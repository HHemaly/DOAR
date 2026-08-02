"""
html_reports.py — render AnalysisRecord (see schema.py) into HTML reports.

Four report types, all self-contained (inline CSS, no external assets):
  - technical      : full research view (all 5 levels, evidence, rejected claims)
  - parent_en      : parent-facing English
  - parent_ar      : parent-facing Arabic (RTL); preserves caution, never
                     strengthens a claim
  - psychologist   : review form with Agree / Partially / Disagree / Uncertain /
                     N-A radios per item + free-text correction boxes

A visible SYNTHETIC banner is rendered whenever record["is_synthetic"] is True.
"""

from __future__ import annotations
import html
import os

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;
     background:#f4f6f8;color:#1f2d3d;line-height:1.5}
.wrap{max-width:960px;margin:0 auto;padding:24px}
h1{font-size:1.5em;margin:0 0 4px}h2{font-size:1.15em;margin:22px 0 8px;
   border-bottom:2px solid #e1e8ed;padding-bottom:4px}
.card{background:#fff;border-radius:8px;padding:16px 18px;margin:12px 0;
      box-shadow:0 1px 3px rgba(0,0,0,.08)}
.synthetic{background:#fdecea;border:2px dashed #e74c3c;color:#a5281b;
   padding:10px 14px;border-radius:6px;font-weight:bold;margin:12px 0}
.disclaimer{background:#fef9e7;border-left:4px solid #f39c12;padding:10px 14px;
   border-radius:5px;font-size:.9em}
table{border-collapse:collapse;width:100%;font-size:.9em}
th,td{border:1px solid #e1e8ed;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#2c3e50;color:#fff}
tr:nth-child(even){background:#f8fafb}
.badge{display:inline-block;padding:2px 9px;border-radius:12px;font-size:.8em;
   font-weight:bold;color:#fff}
.verified{background:#27ae60}.uncertain{background:#f39c12}
.rejected{background:#e74c3c}.unavailable{background:#95a5a6}
.pass{background:#27ae60}.rewrite_required{background:#f39c12}.block{background:#e74c3c}
.muted{color:#7f8c8d;font-size:.85em}
.rtl{direction:rtl;text-align:right}
.q{background:#eaf4fb;padding:8px 12px;border-radius:6px;margin:6px 0}
.levels td:first-child{font-weight:bold;width:32%}
label.opt{margin-right:10px;font-size:.85em;white-space:nowrap}
textarea{width:100%;min-height:44px;margin-top:6px;font-family:inherit}
img.shot{max-width:100%;border:1px solid #ddd;border-radius:6px}
</style>"""


def _e(x):
    return html.escape(str(x if x is not None else ""))


def _page(title, body, rtl=False):
    cls = ' class="rtl"' if rtl else ""
    return (f"<!doctype html><html{cls}><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{_CSS}</head><body><div class='wrap'>"
            f"{body}</div></body></html>")


def _synthetic_banner(record):
    if record.get("is_synthetic"):
        return ("<div class='synthetic'>SYNTHETIC EXAMPLE — placeholder data for "
                "testing report structure only. Not a real analysis.</div>")
    return ""


def _status_badge(status):
    s = (status or "unavailable").lower()
    return f"<span class='badge {s}'>{_e(status)}</span>"


def _img_tag(path, base_dir):
    if not path or not os.path.exists(path):
        return "<span class='muted'>[image not available]</span>"
    rel = os.path.relpath(path, base_dir) if base_dir else path
    return f"<img class='shot' src='{_e(rel)}' alt='image'>"


# ---------------------------------------------------------------------------
# Technical report
# ---------------------------------------------------------------------------

def render_technical(record: dict, base_dir: str = None) -> str:
    a = record
    mp = a["model_prediction"]
    img = a["image"]
    art = a.get("artifacts", {})

    parts = [_synthetic_banner(a),
             f"<h1>Technical Analysis Report</h1>",
             f"<p class='muted'>Image: {_e(img['filename'])} "
             f"({_e(img['width'])}x{_e(img['height'])}, quality "
             f"{_e(round(img.get('quality_score',0),3))})</p>"]

    # Images
    imgs = "<div class='card'><h2>1. Images</h2>"
    imgs += "<b>Original</b><br>" + _img_tag(art.get("original"), base_dir)
    if art.get("annotated"):
        imgs += "<br><br><b>Annotated (verified detections only)</b><br>" + _img_tag(art["annotated"], base_dir)
    if art.get("gradcam"):
        imgs += ("<br><br><b>Grad-CAM</b> <span class='muted'>(classifier "
                 "attention, NOT psychological meaning)</span><br>" + _img_tag(art["gradcam"], base_dir))
    imgs += "</div>"
    parts.append(imgs)

    # Level table
    gt = a["ground_truth"]
    lvl = ["<div class='card'><h2>2. Scientific levels (kept separate)</h2>",
           "<table class='levels'>"]
    lvl.append(f"<tr><td>L2 Ground truth</td><td>{_e(gt.get('label') or 'unknown')} "
               f"<span class='muted'>({_e(gt.get('source') or 'n/a')})</span></td></tr>")
    if mp.get("available"):
        corr = mp.get("correct")
        corr_txt = "correct" if corr else ("incorrect" if corr is False else "n/a")
        topk = ", ".join(f"{_e(t['class'])}:{_e(round(t['probability'],3))}"
                         for t in mp.get("top_k", []))
        lvl.append(f"<tr><td>L3 Model prediction</td><td><b>{_e(mp['predicted_class'])}</b> "
                   f"(prob {_e(round(mp['confidence'],3))}, {corr_txt})<br>"
                   f"<span class='muted'>model: {_e(mp['model_name'])} · top-k: {topk}</span><br>"
                   f"<span class='muted'>Probability is a classifier score, not psychological confidence.</span></td></tr>")
    else:
        lvl.append(f"<tr><td>L3 Model prediction</td><td class='muted'>{_e(mp.get('note'))}</td></tr>")
    eh = a.get("emotion_heuristic", {})
    if eh:
        lvl.append(f"<tr><td>Emotion heuristic</td><td>{_e(eh.get('estimated_emotion'))} "
                   f"({_e(eh.get('confidence_label'))}) <span class='muted'>— "
                   f"feature heuristic, NOT a trained model</span></td></tr>")
    lvl.append("</table></div>")
    parts.append("".join(lvl))

    # Objective features
    of = a["objective_features"]
    feat_rows = []
    for grp, d in of.items():
        for k, v in (d or {}).items():
            if isinstance(v, (int, float, str)):
                feat_rows.append(f"<tr><td>{_e(grp)}.{_e(k)}</td><td>{_e(v)}</td></tr>")
    parts.append("<div class='card'><h2>3. Objective features (L1)</h2><table>"
                 + "".join(feat_rows[:40]) + "</table></div>")

    # Detections
    dets = a.get("detections", [])
    if dets:
        rows = ["<table><tr><th>Label</th><th>bbox</th><th>detector score</th>"
                "<th>CLIP sim</th><th>threshold</th><th>method</th><th>status</th>"
                "<th>crop</th></tr>"]
        for d in dets:
            rows.append(
                f"<tr><td>{_e(d['label'])}</td><td>{_e(d.get('bbox'))}</td>"
                f"<td>{_e(d.get('detector_score'))}</td>"
                f"<td>{_e(d.get('clip_similarity'))}</td>"
                f"<td>{_e(d.get('validation_threshold'))}</td>"
                f"<td>{_e(d.get('validation_method'))}</td>"
                f"<td>{_status_badge(d.get('validator_status'))}</td>"
                f"<td>{_e(os.path.basename(d.get('crop_path') or '') or '-')}</td></tr>")
        rows.append("</table><p class='muted'>CLIP similarity is a raw cosine "
                    "similarity, not a probability.</p>")
        parts.append("<div class='card'><h2>4. Detections</h2>" + "".join(rows) + "</div>")

    # OCR
    if a.get("ocr"):
        ocr_rows = "".join(
            f"<tr><td>{_e(o.get('text'))}</td><td>{_e(o.get('confidence'))}</td>"
            f"<td>{_e(o.get('semantic_tag'))}</td></tr>" for o in a["ocr"])
        parts.append("<div class='card'><h2>5. OCR text</h2><table>"
                     "<tr><th>text</th><th>confidence</th><th>tag</th></tr>"
                     + ocr_rows + "</table></div>")

    # Rules (L4)
    rules = a.get("rules", [])
    if rules:
        rr = ["<table><tr><th>Rule</th><th>evidence level</th><th>score</th>"
              "<th>status</th><th>interpretation</th><th>reference</th></tr>"]
        for r in rules:
            rr.append(
                f"<tr><td>{_e(r['rule_id'])}</td><td>{_e(r['evidence_level'])}</td>"
                f"<td>{_e(round(r.get('score',0),3))}</td>"
                f"<td>{_status_badge(r.get('status'))}</td>"
                f"<td>{_e(r['interpretation'])}<br><span class='muted'>{_e(r.get('caveat'))}</span></td>"
                f"<td class='muted'>{_e(r.get('reference'))}</td></tr>")
        rr.append("</table>")
        parts.append("<div class='card'><h2>6. Rule-based indicators (L4, non-diagnostic)</h2>"
                     + "".join(rr) + "</div>")

    # Themes
    if a.get("themes"):
        th = "".join(f"<li>{_e(t.get('theme'))} (score {_e(round(t.get('score',0),3))}, "
                     f"rules: {_e(', '.join(t.get('supporting_rules', [])))})</li>"
                     for t in a["themes"])
        parts.append(f"<div class='card'><h2>7. Themes</h2><ul>{th}</ul></div>")

    # Rejected / uncertain claims
    claims = a.get("claims", [])
    rej = [c for c in claims if c.get("validator_status") in ("rejected", "uncertain")]
    if rej:
        cr = "".join(f"<tr><td>{_e(c.get('claim_type'))}</td>"
                     f"<td>{_e(c.get('claim'))[:160]}</td>"
                     f"<td>{_status_badge(c.get('validator_status'))}</td></tr>" for c in rej[:30])
        parts.append("<div class='card'><h2>8. Rejected / uncertain claims (not shown to parents)</h2>"
                     "<table><tr><th>type</th><th>claim</th><th>status</th></tr>"
                     + cr + "</table></div>")

    # Final judgment + interpretation
    fj = a["final_judgment"]
    parts.append(f"<div class='card'><h2>9. Final safety judgment</h2>"
                 f"<p>Status: {_status_badge(fj.get('status'))} · checks "
                 f"{_e(fj.get('checks_passed'))}/{_e(fj.get('checks_total'))} · "
                 f"safe_to_show={_e(fj.get('safe_to_show'))}</p>"
                 f"<div class='disclaimer'>{_e(a['parent_output_en'].get('disclaimer'))}</div></div>")

    return _page("DOAR Technical Report", "".join(parts))


# ---------------------------------------------------------------------------
# Parent reports (EN / AR)
# ---------------------------------------------------------------------------

def render_parent(record: dict, lang: str = "en") -> str:
    a = record
    rtl = (lang == "ar")
    out = a.get("parent_output_ar") if rtl else a.get("parent_output_en")
    out = out or a.get("parent_output_en", {})
    title = "تقرير الرسم" if rtl else "Drawing Report"

    parts = [_synthetic_banner(a), f"<h1>{_e(title)}</h1>",
             "<div class='card'>",
             f"<p>{_e(out.get('parent_answer'))}</p></div>"]

    qs = out.get("gentle_questions", [])
    if qs:
        qh = "أسئلة لطيفة" if rtl else "Gentle questions to explore together"
        parts.append(f"<div class='card'><h2>{_e(qh)}</h2>"
                     + "".join(f"<div class='q'>{_e(q)}</div>" for q in qs) + "</div>")

    parts.append(f"<div class='card disclaimer'>{_e(out.get('safety_note'))}<br><br>"
                 f"{_e(out.get('disclaimer'))}</div>")
    return _page(title, "".join(parts), rtl=rtl)


# ---------------------------------------------------------------------------
# Psychologist review form
# ---------------------------------------------------------------------------

_OPTS = ["Agree", "Partially agree", "Disagree", "Uncertain", "N/A"]


def _radio_group(name):
    return "".join(
        f"<label class='opt'><input type='radio' name='{_e(name)}' "
        f"value='{_e(o)}'> {_e(o)}</label>" for o in _OPTS)


def render_psychologist(record: dict, base_dir: str = None) -> str:
    a = record
    cid = a["image"]["id"]
    art = a.get("artifacts", {})

    parts = [_synthetic_banner(a),
             f"<h1>Psychologist Review — case {_e(cid)}</h1>",
             "<p class='muted'>Mark each item. Unvalidated AI output is NOT clinical truth. "
             "Your review is stored separately and used for agreement metrics.</p>",
             "<form>"]

    # Reviewer meta
    parts.append("<div class='card'><h2>Reviewer</h2>"
                 "<label>Reviewer ID (or anonymous): <input name='reviewer_id'></label></div>")

    # Images
    parts.append("<div class='card'><h2>Images</h2>"
                 + _img_tag(art.get("original"), base_dir)
                 + (("<br><br>" + _img_tag(art.get("annotated"), base_dir)) if art.get("annotated") else "")
                 + "</div>")

    # Model prediction review
    mp = a["model_prediction"]
    mp_txt = (f"Predicted <b>{_e(mp['predicted_class'])}</b> (prob {_e(round(mp['confidence'],3))})"
              if mp.get("available") else "No model prediction available yet.")
    parts.append(f"<div class='card'><h2>Model prediction</h2><p>{mp_txt} · "
                 f"ground truth: {_e(a['ground_truth'].get('label') or 'unknown')}</p>"
                 f"{_radio_group('review_model_prediction')}"
                 "<textarea name='comment_model' placeholder='Correction / comment'></textarea></div>")

    # Detections review
    for i, d in enumerate(a.get("detections", [])):
        parts.append(f"<div class='card'><h2>Detection: {_e(d['label'])}</h2>"
                     f"<p class='muted'>status {_e(d.get('validator_status'))}, "
                     f"CLIP sim {_e(d.get('clip_similarity'))}</p>"
                     f"{_radio_group('review_det_'+str(i))}"
                     f"<textarea name='comment_det_{i}'></textarea></div>")

    # Rules review
    for i, r in enumerate(a.get("rules", [])):
        parts.append(f"<div class='card'><h2>Rule: {_e(r['rule_id'])}</h2>"
                     f"<p>{_e(r['interpretation'])}</p>"
                     f"<p class='muted'>evidence: {_e(r['evidence_level'])} · {_e(r.get('reference'))}</p>"
                     f"{_radio_group('review_rule_'+str(i))}"
                     f"<textarea name='comment_rule_{i}' placeholder='Corrected interpretation'></textarea></div>")

    # Overall
    parts.append("<div class='card'><h2>Overall interpretation & report</h2>"
                 + _radio_group("review_overall")
                 + "<textarea name='comment_overall'></textarea></div>")

    parts.append("<div class='card disclaimer'>"
                 + _e(a['parent_output_en'].get('disclaimer')) + "</div>")
    parts.append("</form>")
    return _page(f"Psychologist Review {cid}", "".join(parts))
