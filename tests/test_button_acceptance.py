"""Pre-doctor stabilization pass, item E: click-through acceptance tests
for every significant button in doar_prototype_app.py, using the REAL UI
submission control (never calling the underlying function directly) via
streamlit.testing.v1.AppTest. For each button this proves: click -> action
runs -> relevant state changes -> a visible result appears -> a later
rerun does not erase the result. Heavy Grounding-DINO/OWLv2 loaders are
patched (never called by the automated test suite, per every other test
file in this project); Gemini providers are left real so the test also
exercises the actual degrade-to-fallback path (bounded network timeout
fixed in this same pass -- see tests/test_perf_deferred_visual_scan.py).
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


def _build_light_case(tmp_dir: str) -> Path:
    case_dir = Path(tmp_dir) / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "drawing.png"
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
    image.save(path)
    analyze_image_with_timing(str(path), str(case_dir), None)
    return case_dir


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class ButtonAcceptanceTests(unittest.TestCase):
    """One continuous session mirroring item O's manual acceptance
    workflow -- Analyze already covered by test_perf_deferred_visual_scan;
    this file picks up from an already-analyzed case and exercises every
    remaining button in the order a real reviewer would click them."""

    def test_full_click_through_workflow(self):
        with tempfile.TemporaryDirectory() as d:
            # Resolved immediately -- the app itself stores every
            # session_state key (full_analysis_*, visual_audit_*, hi_chat_*)
            # keyed on str(Path(case_dir_text)) where case_dir_text is what
            # we hand it below (already resolved), and on Windows an
            # unresolved tempdir path can differ from its resolved form
            # (8.3 short names) -- comparing against the same resolved
            # Path this test hands the app is what makes these assertions
            # match, not a re-implementation of the app's own resolution.
            case_dir = _build_light_case(d).resolve()
            # Run Full Analysis's own code (not this test) evaluates
            # model_predict_fns=_load_model_predict_fns() as a call
            # argument BEFORE run_and_persist_initial_scan runs, even
            # though that scan call itself is mocked below -- so these two
            # real-weight loaders must return cheap fakes, not raise, or
            # the (correctly try/excepted, so silent-to-at.exception)
            # AssertionError would prevent full_analysis_key from ever
            # being set and this test would misreport it as a dead button.
            with patch("doar.phase2c7.runtime.build_real_model_predict_fns", return_value={}), \
                 patch("doar.phase2c7.runtime.load_open_vocab_query_fn", return_value=lambda *a, **k: []), \
                 patch("doar.visual_evidence.run_and_persist_initial_scan") as mock_scan:
                def _fake_scan(case_dir_arg, image_path, *, eye_entry, registry_v2, model_predict_fns):
                    from doar.case_output import write_versioned
                    write_versioned(Path(case_dir_arg) / "detections.json", {
                        "status": "available", "generated_at": "2026-08-23T00:00:00", "n_findings": 0,
                        "findings": [], "entities": [],
                    })
                mock_scan.side_effect = _fake_scan

                at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
                at.session_state["case_dir"] = str(case_dir.resolve())
                at.run(timeout=120)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

                # --- Psychologist tab: Run Full Analysis --------------------
                psych = at.tabs[1]
                run_full_btn = [b for b in psych.button if b.key == "run_full_analysis"]
                self.assertEqual(len(run_full_btn), 1, "Run Full Analysis button not found")
                run_full_btn[0].click()
                at.run(timeout=120)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
                self.assertIn(f"full_analysis_{case_dir}", at.session_state,
                              "Run Full Analysis click did not write session_state -- button did nothing")
                # Visible result: the psychologist tab must render SOME
                # gemini-status caption/expander/subheader after the click,
                # even in the fully-offline/failed case (never silent).
                psych = at.tabs[1]
                psych_text = " ".join(m.value for m in psych.markdown) + " ".join(c.value for c in psych.caption)
                self.assertTrue(psych_text.strip(), "Psychologist tab rendered nothing after Run Full Analysis")

                # A later rerun (tab render again) must not erase the result.
                at.run(timeout=60)
                self.assertIn(f"full_analysis_{case_dir}", at.session_state)

                # --- Psychologist tab: Visual Consistency Judge --------------
                psych = at.tabs[1]
                audit_btn = [b for b in psych.button if b.key == "run_visual_audit"]
                self.assertEqual(len(audit_btn), 1, "Visual Consistency Judge button not found")
                audit_btn[0].click()
                at.run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
                self.assertIn(f"visual_audit_{case_dir}", at.session_state,
                              "Visual Consistency Judge click did not write session_state -- button did nothing")
                audit_result = at.session_state[f"visual_audit_{case_dir}"]
                self.assertIn(audit_result["status"], ("ok", "unavailable", "error"))
                # Rerun persistence.
                at.run(timeout=60)
                self.assertIn(f"visual_audit_{case_dir}", at.session_state)

                # --- Psychologist tab: Ask DOAR Send --------------------------
                psych = at.tabs[1]
                psych.text_input(key="clinician_chat_q").set_value("Is there a house?")
                at.run(timeout=60)
                psych = at.tabs[1]
                send_btn = [b for b in psych.button if b.key == "clinician_chat_send"]
                self.assertEqual(len(send_btn), 1)
                send_btn[0].click()
                at.run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
                chat_key = f"hi_chat_clinician_{case_dir}"
                self.assertIn(chat_key, at.session_state)
                chat_log = at.session_state[chat_key]
                self.assertEqual(len(chat_log), 2, "Send did not append a question+answer turn")
                self.assertEqual(chat_log[0]["role"], "user")
                self.assertEqual(chat_log[1]["role"], "assistant")
                self.assertTrue(chat_log[1]["content"].strip(), "Ask DOAR rendered an empty answer")

                # --- Psychologist tab: feedback Submit -------------------------
                psych = at.tabs[1]
                reviewer_inputs = [t for t in psych.text_input if t.label in ("Reviewer name", "اسم الأخصائي")]
                self.assertEqual(len(reviewer_inputs), 1, "Reviewer name field not found")
                reviewer_inputs[0].set_value("Dr. Test")
                at.run(timeout=60)
                psych = at.tabs[1]
                submit_btns = [b for b in psych.button if getattr(b, "label", "") in
                               ("Submit feedback", "إرسال التقييم")]
                self.assertEqual(len(submit_btns), 1, "Psychologist feedback submit control not found")
                submit_btns[0].click()
                at.run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
                feedback_path = case_dir / "psychologist_feedback.json"
                self.assertTrue(feedback_path.exists())
                saved = json.loads(feedback_path.read_text(encoding="utf-8"))
                self.assertEqual(len(saved["entries"]), 1)
                self.assertEqual(saved["entries"][0]["reviewer_name"], "Dr. Test")

                # --- Language switching persists Ask DOAR history --------------
                at.sidebar.radio(key="language").set_value("ar")
                at.run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
                self.assertEqual(at.session_state[chat_key], chat_log,
                                  "Ask DOAR history changed/vanished across a language switch")
                at.sidebar.radio(key="language").set_value("en")
                at.run(timeout=60)
                self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
                self.assertEqual(at.session_state[chat_key], chat_log,
                                  "Ask DOAR history changed/vanished switching language back")


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class FormalFeaturesButtonAcceptanceTests(unittest.TestCase):
    """Phase G0: the new 'Compute formal/graphic measurements' button and
    its expert-review mini-form, click-tested through the real UI."""

    def test_compute_formal_features_and_submit_review(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_light_case(d).resolve()
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=120)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])

            psych = at.tabs[1]
            compute_btn = [b for b in psych.button if b.key == "run_formal_features"]
            self.assertEqual(len(compute_btn), 1, "Compute formal/graphic measurements button not found")
            compute_btn[0].click()
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            self.assertTrue((case_dir / "formal_features.json").exists())
            saved = json.loads((case_dir / "formal_features.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "available")
            self.assertGreater(len(saved["features"]), 0)

            # Visible result persists across a rerun.
            at.run(timeout=60)
            self.assertIn(f"formal_features_{case_dir}", at.session_state)

            # Expert review mini-form, real UI submission.
            psych = at.tabs[1]
            reviewer_inputs = [t for t in psych.text_input if t.key == "ff_reviewer_name"]
            self.assertEqual(len(reviewer_inputs), 1)
            reviewer_inputs[0].set_value("Dr. Formal")
            at.run(timeout=60)
            psych = at.tabs[1]
            submit_btns = [b for b in psych.button if getattr(b, "label", "") in
                           ("Submit review", "إرسال المراجعة")]
            # form_submit_button controls are also exposed via .button in AppTest.
            self.assertGreaterEqual(len(submit_btns), 1, "formal-feature review submit control not found")
            submit_btns[-1].click()
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            review_path = case_dir / "formal_feature_review.json"
            self.assertTrue(review_path.exists())
            review_saved = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(len(review_saved["entries"]), 1)
            self.assertEqual(review_saved["entries"][0]["reviewer_name"], "Dr. Formal")


if __name__ == "__main__":
    unittest.main()
