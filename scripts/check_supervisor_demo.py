"""Lightweight supervisor-demo health check (feature/supervisor-demo-v2).

Deliberately NOT an elaborate audit system -- just confirms the handful of
things the supervisor demo actually needs before a live walkthrough:
  - doar_prototype_app.py imports cleanly
  - each of the 3 prepared demo cases has a real image + cached analysis.json
  - evidence-trace data (detections.json findings) exists where the demo
    advertises "View evidence trace"
  - whether GEMINI_API_KEY is configured (Ask DOAR works either way -- the
    prepared demo itself never depends on it)

Run: .venv\\Scripts\\python.exe scripts\\check_supervisor_demo.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASES_DIR = ROOT / "outputs" / "prototype_cases"

# Keep in sync with SUPERVISOR_DEMO_CASES in doar_prototype_app.py.
DEMO_CASES = [
    ("Case 1", "a103_1787479142"),
    ("Case 2", "a111_1787479358"),
    ("Case 3", "h38_1786305027"),
]


def _status_line(label: str, ok: bool, ready_word: str = "READY", fail_word: str = "MISSING") -> str:
    return f"{label:<16}{ready_word if ok else fail_word}"


def check_app_imports() -> tuple[bool, str]:
    try:
        import importlib
        # Compile-check only (avoid running module-level Streamlit calls
        # such as st.set_page_config outside of a real Streamlit runtime).
        source = (ROOT / "doar_prototype_app.py").read_text(encoding="utf-8")
        compile(source, "doar_prototype_app.py", "exec")
        # Also confirm the real doar package (its Gemini providers,
        # case_interpretation, case_presentation, formal_features, etc.)
        # imports without error -- this is what would actually break the
        # app at runtime, not just a syntax check.
        for mod in ("doar.human_interaction", "doar.case_interpretation",
                    "doar.case_presentation", "doar.formal_features"):
            importlib.import_module(mod)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc.__class__.__name__}: {exc}"


def check_case(case_id: str) -> tuple[bool, bool, bool, str]:
    """Returns (image_ok, analysis_ok, evidence_trace_ok, detail)."""
    case_dir = CASES_DIR / case_id
    if not case_dir.is_dir():
        return False, False, False, f"case directory not found: {case_dir}"
    images = [p for p in case_dir.iterdir()
              if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file()]
    image_ok = len(images) > 0
    analysis_path = case_dir / "analysis.json"
    analysis_ok = False
    detail = ""
    if analysis_path.exists():
        try:
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            analysis_ok = bool(analysis.get("rule_evaluations")) or bool(analysis.get("evidence"))
        except Exception as exc:  # noqa: BLE001
            detail = f"analysis.json unreadable: {exc}"
    else:
        detail = "analysis.json not found"
    detections_path = case_dir / "detections.json"
    evidence_trace_ok = False
    if detections_path.exists():
        try:
            detections = json.loads(detections_path.read_text(encoding="utf-8"))
            evidence_trace_ok = (detections.get("status") == "available"
                                  and len(detections.get("findings", [])) > 0)
        except Exception:  # noqa: BLE001
            evidence_trace_ok = False
    return image_ok, analysis_ok, evidence_trace_ok, detail


def main() -> int:
    print("DOAR Supervisor Demo")
    print("-" * 20)

    app_ok, app_detail = check_app_imports()
    print(_status_line("App", app_ok))
    if not app_ok:
        print(f"  detail: {app_detail}")

    all_cases_ready = True
    case_details = []
    for label, case_id in DEMO_CASES:
        image_ok, analysis_ok, trace_ok, detail = check_case(case_id)
        case_ok = image_ok and analysis_ok
        all_cases_ready = all_cases_ready and case_ok
        print(_status_line(label, case_ok))
        if not case_ok:
            case_details.append(f"  {label} ({case_id}): image={image_ok} analysis={analysis_ok} ({detail})")
        elif not trace_ok:
            case_details.append(f"  {label} ({case_id}): evidence trace data not available (detections.json)")

    gemini_configured = bool(os.environ.get("GEMINI_API_KEY"))
    print(_status_line("Gemini", gemini_configured, ready_word="READY", fail_word="NOT CONFIGURED"))

    prepared_demo_ready = app_ok and all_cases_ready
    print(_status_line("Prepared Demo", prepared_demo_ready))

    if case_details:
        print()
        print("Details:")
        for d in case_details:
            print(d)

    return 0 if prepared_demo_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
