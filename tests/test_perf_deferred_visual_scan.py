"""Performance-fix focused tests: the heavy Grounding-DINO/OWLv2 scan must
never run automatically during Analyze, must be reachable only via the
Psychologist view's explicit "Run deep visual analysis" button, must load
its models only once per process, and must never be re-triggered by an
ordinary rerun (tab switch, language switch, feedback submission, Ask
DOAR). No real model weights are loaded in this file -- the heavy loader
functions are patched with lightweight fakes/call-counters throughout, so
this suite stays fast and network-free.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

from doar.timed_analysis import analyze_image_with_timing


def _never_call(*_args, **_kwargs):
    raise AssertionError("a heavy Grounding-DINO/OWLv2 loader was called when it should not have been")


def _build_light_case(tmp_dir: str) -> Path:
    """The light/normal pipeline only -- no deep visual scan (mirrors what
    Analyze now does after the performance fix; detections.json stays the
    honest {"status": "unavailable"} stub `finalize_case` always writes)."""
    case_dir = Path(tmp_dir) / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "drawing.png"
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
    image.save(path)
    analyze_image_with_timing(str(path), str(case_dir), None)
    return case_dir


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class AnalyzeStaysLightTests(unittest.TestCase):
    def test_analyze_button_never_loads_heavy_detectors_and_parent_renders(self):
        with tempfile.TemporaryDirectory() as d:
            image = Image.new("RGB", (300, 300), "white")
            ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
            image_path = Path(d) / "drawing.png"
            image.save(image_path)

            with patch("doar.phase2c7.runtime.build_real_model_predict_fns", side_effect=_never_call), \
                 patch("doar.phase2c7.runtime.load_open_vocab_query_fn", side_effect=_never_call), \
                 patch("doar.visual_evidence.run_and_persist_initial_scan", side_effect=_never_call):
                at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
                at.run(timeout=60)
                uploader = at.sidebar.file_uploader[0]
                uploader.upload("drawing.png", image_path.read_bytes(), "image/png")
                at.run(timeout=60)
                analyze_buttons = [b for b in at.sidebar.button if b.label in ("Analyze", "تحليل")]
                self.assertEqual(len(analyze_buttons), 1)
                analyze_buttons[0].click().run(timeout=90)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

            case_dir = Path(at.session_state["case_dir"])
            self.assertTrue((case_dir / "analysis.json").exists())
            detections = json.loads((case_dir / "detections.json").read_text(encoding="utf-8"))
            self.assertEqual(detections["status"], "unavailable")
            self.assertEqual(len(at.tabs), 3)


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class DeepVisualAnalysisOnDemandTests(unittest.TestCase):
    def _fake_scan(self, call_log):
        def _scan(case_dir, image_path, *, eye_entry, registry_v2, model_predict_fns):
            call_log.append(1)
            from doar.case_output import write_versioned
            write_versioned(Path(case_dir) / "detections.json", {
                "status": "available", "generated_at": "2026-08-23T00:00:00", "n_findings": 0,
                "findings": [], "entities": [],
            })
        return _scan

    def test_deep_scan_runs_once_and_reruns_never_repeat_it(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_light_case(d)
            call_log = []
            with patch("doar.phase2c7.runtime.build_real_model_predict_fns", return_value={}), \
                 patch("doar.phase2c7.runtime.load_open_vocab_query_fn", return_value=lambda *_a, **_k: []), \
                 patch("doar.visual_evidence.run_and_persist_initial_scan", side_effect=self._fake_scan(call_log)):
                at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
                at.session_state["case_dir"] = str(case_dir.resolve())
                at.run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

                deep_buttons = [b for b in at.tabs[1].button if b.label == "Run deep visual analysis"]
                self.assertEqual(len(deep_buttons), 1)
                deep_buttons[0].click().run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
                self.assertEqual(len(call_log), 1)

                detections = json.loads((case_dir / "detections.json").read_text(encoding="utf-8"))
                self.assertEqual(detections["status"], "available")

                # Ordinary reruns (tab switch / language switch / a plain rerun) must
                # NEVER re-trigger the scan -- the button click above was the only
                # explicit request.
                at.run(timeout=60)
                self.assertEqual(len(call_log), 1)
                at.session_state["language"] = "ar"
                at.run(timeout=60)
                self.assertEqual(len(call_log), 1)

                # The button now offers a re-run, but does not fire on its own.
                rerun_buttons = [b for b in at.tabs[1].button if "Re-run deep visual analysis" in b.label
                                 or "إعادة تشغيل" in b.label]
                self.assertEqual(len(rerun_buttons), 1)

    def test_ask_doar_does_not_load_heavy_detector_before_deep_scan(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_light_case(d)
            with patch("doar.phase2c7.runtime.build_real_model_predict_fns", side_effect=_never_call), \
                 patch("doar.phase2c7.runtime.load_open_vocab_query_fn", side_effect=_never_call), \
                 patch("doar.visual_evidence.run_and_persist_initial_scan", side_effect=_never_call), \
                 patch("doar.human_interaction.resolve_default_answer_provider",
                       return_value=__import__("doar.human_interaction", fromlist=["DeterministicAnswerProvider"])
                       .DeterministicAnswerProvider()):
                at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
                at.session_state["case_dir"] = str(case_dir.resolve())
                at.run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

                parent_tab = at.tabs[0]
                q_widgets = [w for w in parent_tab.text_input if w.key == "parent_chat_q"]
                self.assertEqual(len(q_widgets), 1)
                q_widgets[0].set_value("What did you find in this drawing?")
                send_buttons = [b for b in parent_tab.button if b.key == "parent_chat_send"]
                self.assertEqual(len(send_buttons), 1)
                send_buttons[0].click().run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])


if __name__ == "__main__":
    unittest.main()
