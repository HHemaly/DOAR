from __future__ import annotations

import html
from pathlib import Path


def _rows(mapping: dict) -> str:
    return "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in mapping.items()
    )


def _rule_rows(analysis: dict, language: str) -> str:
    rows = []
    for rule in analysis["rule_evaluations"]:
        wording = rule["original_arabic"] if language == "ar" else rule["english_translation"]
        if rule["status"] == "not_matched":
            reasoning = (
                "لم يُلاحظ الشرط البصري المطلوب، ولذلك لم يُنتج أي تفسير نفسي."
                if language == "ar"
                else "The required visual condition was not observed. No psychological interpretation was produced."
            )
        elif rule["status"] == "not_evaluated":
            missing = ", ".join(rule["missing_evidence"])
            reasoning = (
                f"لم تُقيّم القاعدة بسبب غياب الوحدة أو الدليل: {missing}."
                if language == "ar"
                else f"Not evaluated because the required module or evidence is missing: {missing}."
            )
        elif rule["status"] == "missing_detector":
            missing = ", ".join(rule["missing_evidence"])
            reasoning = (
                f"لا يوجد كاشف لهذه السمة في هذا الإصدار؛ لا يُعامل غياب الدليل كدليل سلبي. ({missing})"
                if language == "ar"
                else (f"No detector exists for this feature in this release, so the "
                      f"rule was not evaluated. Missing evidence is not treated as "
                      f"negative evidence. ({missing})")
            )
        else:
            reasoning = rule.get("parent_safe_wording") or ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(rule['rule_id'])}</td>"
            f"<td>{html.escape(rule['status'])}</td>"
            f"<td>{html.escape(wording)}</td>"
            f"<td>{html.escape(reasoning)}</td>"
            f"<td>{html.escape(', '.join(rule['references']))}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_report(analysis: dict, judges: dict, language: str = "en", parent: bool = False) -> str:
    ar = language == "ar"
    title = "تقرير تحليل الرسم - غير تشخيصي" if ar else "Drawing analysis report - non-diagnostic"
    disclaimer = (
        "هذا التقرير يصف أدلة بصرية فقط، ولا يمثل تشخيصاً. يجب مراجعة النتائج بواسطة مختص مؤهل."
        if ar else analysis["safety_disclaimer"]
    )
    direction = "rtl" if ar else "ltr"
    rule_section = "" if parent else f"""
      <h2>{"تقييم القواعد" if ar else "Rule evaluations"}</h2>
      <table><tr><th>ID</th><th>Status</th><th>Source wording</th><th>Safe reasoning</th><th>References</th></tr>
      {_rule_rows(analysis, language)}</table>"""
    inner = _report_inner(analysis, judges, language, parent)
    return _document(inner, language, direction)


_DOC_STYLE = (
    "body{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5}"
    ".warning{background:#fff3cd;padding:1rem;border-left:5px solid #c78b00}"
    "section{margin-bottom:2.5rem}"
    "table{border-collapse:collapse;width:100%;margin-bottom:1rem}"
    "th,td{border:1px solid #ccc;padding:.45rem;text-align:start}th{background:#f2f2f2}"
)


def _document(inner: str, language: str, direction: str) -> str:
    """Wrap a body fragment in one valid HTML document."""
    return (
        f'<!doctype html><html lang="{language}" dir="{direction}"><head>'
        f'<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<style>{_DOC_STYLE}</style></head><body>{inner}</body></html>"
    )


def _report_inner(analysis: dict, judges: dict, language: str = "en", parent: bool = False) -> str:
    """Return the body fragment (no <html>/<body> wrapper) for one language."""
    ar = language == "ar"
    title = "تقرير تحليل الرسم - غير تشخيصي" if ar else "Drawing analysis report - non-diagnostic"
    disclaimer = (
        "هذا التقرير يصف أدلة بصرية فقط، ولا يمثل تشخيصاً. يجب مراجعة النتائج بواسطة مختص مؤهل."
        if ar else analysis["safety_disclaimer"]
    )
    direction = "rtl" if ar else "ltr"
    rule_section = "" if parent else f"""
      <h2>{"تقييم القواعد" if ar else "Rule evaluations"}</h2>
      <table><tr><th>ID</th><th>Status</th><th>Source wording</th><th>Safe reasoning</th><th>References</th></tr>
      {_rule_rows(analysis, language)}</table>"""
    emotion = analysis["emotion"]
    return f"""<section lang="{language}" dir="{direction}">
    <h1>{title}</h1><div class="warning">{html.escape(disclaimer)}</div>
    <h2>{"جودة الصورة" if ar else "Image quality"}</h2><table>{_rows(analysis["quality"])}</table>
    <h2>{"التقسيم والخلفية" if ar else "Segmentation"}</h2><table>{_rows(analysis["segmentation"])}</table>
    <h2>{"التكوين المكاني" if ar else "Composition"}</h2><table>{_rows(analysis["composition"])}</table>
    <h2>{"الألوان" if ar else "Colours"}</h2><table>{_rows(analysis["colour"])}</table>
    <h2>{"نموذج الانفعال" if ar else "Emotion model"}</h2><table>{_rows(emotion)}</table>
    {rule_section}
    <h2>{"المراجعات الآلية" if ar else "Deterministic judges"}</h2><table>{_rows({k:v.get("status",v) if isinstance(v,dict) else v for k,v in judges.items()})}</table>
    <p>{html.escape(disclaimer)}</p></section>"""


def render_bilingual(analysis: dict, judges: dict) -> str:
    """One valid HTML document containing an English (LTR) and an Arabic (RTL)
    section — replaces the previous two-doctype concatenation (D8)."""
    en = _report_inner(analysis, judges, "en")
    ar = _report_inner(analysis, judges, "ar")
    return _document(en + "<hr>" + ar, "en", "ltr")


def save_reports(analysis: dict, judges: dict, output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    reports = {
        "professional_en": render_report(analysis, judges, "en"),
        "professional_ar": render_report(analysis, judges, "ar"),
        "parent_en": render_report(analysis, judges, "en", parent=True),
        "parent_ar": render_report(analysis, judges, "ar", parent=True),
    }
    paths = {}
    for name, content in reports.items():
        path = output / f"{name}.html"
        path.write_text(content, encoding="utf-8")
        paths[name] = str(path)
    bilingual = output / "bilingual.html"
    bilingual.write_text(render_bilingual(analysis, judges), encoding="utf-8")
    paths["bilingual"] = str(bilingual)
    return paths
