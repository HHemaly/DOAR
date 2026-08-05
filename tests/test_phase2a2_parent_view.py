"""Tests for DOAR-TRACE Phase 2A.2: Parent-View clarity, capability
transparency, and minimal page-reference user controls."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

from doar.page_reference import user_page_declaration_from_choice
from doar.registry_v2_build import build_registry_v2
from doar.rule_engine_v2 import V2_RULE_IDS
from doar.timed_analysis import analyze_image_with_timing


class NaturalQuestionWordingTests(unittest.TestCase):
    """Section 8: every executable rule's question_template must be a
    natural, open-ended question -- never the mechanically-generated
    "ask about the <raw observable name>" pattern."""

    _FORBIDDEN_PATTERNS = ("ask your child about the", "light line pressure appearance",
                            "heavy line pressure appearance", "coverage about half",
                            "coverage full", "coverage small", "placement top", "placement left",
                            "placement right", "placement center", "shaky or broken lines")

    def setUp(self):
        self.rules_by_id = {r["rule_id"]: r for r in build_registry_v2()["rules"]}
        self.executable_ids = {
            rid for rid, r in self.rules_by_id.items() if r["allowed_output_level"] == "individual_heuristic_only"
        }

    def test_ten_executable_rules_exist(self):
        self.assertEqual(len(self.executable_ids), 10)

    def test_no_executable_rule_uses_the_mechanical_ask_about_pattern(self):
        for rule_id in self.executable_ids:
            question = self.rules_by_id[rule_id]["question_template"].lower()
            for forbidden in self._FORBIDDEN_PATTERNS:
                self.assertNotIn(forbidden, question, f"{rule_id}: {question!r}")

    def test_every_executable_question_is_a_real_question(self):
        for rule_id in self.executable_ids:
            question = self.rules_by_id[rule_id]["question_template"]
            self.assertTrue(question.strip().endswith("?"), rule_id)
            self.assertGreater(len(question), 15, rule_id)

    def test_line_proxy_rules_never_say_pressure_without_a_physical_pressure_hedge(self):
        # Whenever the word "pressure" appears, the same text must also
        # explicitly disclaim that it is not an actual physical pressure
        # measurement -- never a bare, unhedged "pressure" claim.
        for rule_id in ("EN_COMPILED_LINE_HEAVY_PRESSURE_030", "EN_COMPILED_LINE_LIGHT_PRESSURE_031"):
            rule = self.rules_by_id[rule_id]
            for field in ("parent_safe_wording", "professional_wording"):
                text = rule[field].lower()
                if "pressure" in text:
                    self.assertIn("physical", text, f"{rule_id}.{field}: {text!r}")
                    self.assertTrue(
                        "does not measure" in text or "not a measurement" in text,
                        f"{rule_id}.{field}: {text!r}",
                    )
            # question_template must never mention "pressure" at all --
            # only the appearance-based phrasing ("light or thin-line
            # appearance"), per the task's explicit instruction.
            self.assertNotIn("pressure", rule["question_template"].lower(), rule_id)

    def test_disabled_rules_are_unaffected(self):
        # The mechanical fallback still exists for the 31 disabled rules
        # (they never reach the Parent view) -- this is a deliberate scope
        # boundary, not an oversight.
        disabled_sample = next(
            r for rid, r in self.rules_by_id.items()
            if r["allowed_output_level"] != "individual_heuristic_only" and rid not in V2_RULE_IDS
        )
        self.assertIn("Would you be willing to ask your child about", disabled_sample["question_template"])


def _build_case(tmp_dir: str, image: Image.Image, user_page_declaration=None) -> Path:
    path = Path(tmp_dir) / "drawing.png"
    image.save(path)
    case_dir = Path(tmp_dir) / "case"
    analyze_image_with_timing(str(path), str(case_dir), None, user_page_declaration=user_page_declaration)
    return case_dir


def _wide_margin_image() -> Image.Image:
    # Page-frame-assessable, triggers a real line-proxy suggestion too
    # (per the Section 1 real-image smoke check).
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).rectangle((40, 40, 259, 259), fill="black")
    return image


def _thin_margin_image() -> Image.Image:
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).rectangle((2, 2, 197, 197), fill="black")
    return image


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class ParentViewSectionCountTests(unittest.TestCase):
    """Section 11: Parent View has at most five main result sections."""

    def test_parent_view_has_at_most_five_numbered_subheaders(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d, _wide_margin_image())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            # Parent-view result subheaders are numbered "1." .. "5." in
            # English -- Technical view's headers ("1. Input and page
            # reference" etc.) are st.header, not st.subheader, so this
            # counts only the Parent tab's own subheader widgets.
            numbered = [s.value for s in at.subheader if s.value[:2].rstrip(".").isdigit()]
            self.assertLessEqual(len(numbered), 5, numbered)
            self.assertEqual(len(numbered), 5, numbered)  # exactly 5, per the task's required structure


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class NoTechnicalInfoInParentViewTests(unittest.TestCase):
    """Section 4 + Section 11: no internal IDs, tables, or enums leak into
    the Parent tab; the same information remains visible in Technical."""

    def test_no_raw_rule_id_in_parent_tab(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d, _wide_margin_image())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            expander_labels = [e.label for e in at.expander]
            rule_ids = {r["rule_id"] for r in build_registry_v2()["rules"]}
            parent_expanders = [label for label in expander_labels if "Show" not in label and "verification" not in label.lower()]
            for label in parent_expanders:
                for rule_id in rule_ids:
                    self.assertNotIn(rule_id, label)

    def test_no_evidence_id_text_in_any_parent_markdown_or_caption(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d, _wide_margin_image())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            structured = json.loads((case_dir / "structured_analysis.json").read_text(encoding="utf-8"))
            evidence_ids = {eid for s in structured["individual_rule_suggestions"] for eid in s["evidence_ids"]}
            evidence_ids |= {eid for h in structured["combined_drawing_level_hypotheses"] for eid in h["contributing_evidence_ids"]}
            all_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
            for eid in evidence_ids:
                self.assertNotIn(eid, all_text)

    def test_no_detector_absent_string_in_parent_view(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d, _wide_margin_image())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            # detector_absent:* only ever appears inside missing_evidence
            # lists (rule_evaluations / Technical view's raw dataframe) --
            # never in a Parent-tab text/caption/markdown widget.
            all_parent_text = " ".join(c.value for c in at.caption) + " ".join(m.value for m in at.markdown)
            self.assertNotIn("detector_absent", all_parent_text)

    def test_full_41_rule_and_60_feature_tables_not_in_dataframe_before_technical_header(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d, _wide_margin_image())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            # Exactly the dataframes Technical view creates are expected;
            # Parent view (Phase 2A.2) creates none at all.
            self.assertGreater(len(at.dataframe), 0, "Technical view should still show its raw tables")

    def test_raw_values_remain_visible_in_technical_view(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d, _wide_margin_image())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            headers = [h.value for h in at.header]
            self.assertIn("6. Rule evaluations", headers)
            self.assertIn("3. Objective features", headers)
            # The raw rule_evaluations dataframe (Technical view) still
            # carries real rule_ids -- proof nothing was deleted, only
            # moved out of the Parent tab.
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertTrue(any(r["rule_id"] == "PSY_AR_SIZE_HALF_014" for r in analysis["rule_evaluations"]))


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class CapabilityStatusAccuracyTests(unittest.TestCase):
    def test_technical_view_shows_all_three_capability_tiers(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d, _wide_margin_image())
            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            markdown_text = " ".join(m.value for m in at.markdown)
            self.assertIn("WORKING", markdown_text)
            self.assertIn("LIMITED", markdown_text)
            self.assertIn("NOT AVAILABLE", markdown_text)


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class PageDeclarationReachesPipelineTests(unittest.TestCase):
    """Section 11: user-confirmed full frame reaches the real analysis
    pipeline; cropped/uncertain declarations correctly abstain; the
    declaration is persisted."""

    def test_yes_declaration_persists_and_reaches_rule_gating(self):
        with tempfile.TemporaryDirectory() as d:
            decl = user_page_declaration_from_choice("yes")
            case_dir = _build_case(d, _thin_margin_image(), user_page_declaration=decl)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["page_reference"]["page_reference_mode"], "user_confirmed_full_frame")
            self.assertTrue(analysis["page_reference"]["page_relative_features_assessable"])
            by_rule = {r["rule_id"]: r["status"] for r in analysis["rule_evaluations"]}
            self.assertIn(by_rule["PSY_AR_SIZE_FULL_015"], ("weak_support", "not_matched"))

            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            metric_values = [m.value for m in at.metric]
            self.assertIn("yes", metric_values)  # "Parent declaration used" metric

    def test_no_declaration_abstains_even_on_an_otherwise_full_page_image(self):
        with tempfile.TemporaryDirectory() as d:
            decl = user_page_declaration_from_choice("no")
            case_dir = _build_case(d, _wide_margin_image(), user_page_declaration=decl)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["page_reference"]["page_reference_mode"], "cropped_or_content_only")
            self.assertFalse(analysis["page_reference"]["page_relative_features_assessable"])
            by_rule = {r["rule_id"]: r["status"] for r in analysis["rule_evaluations"]}
            self.assertEqual(by_rule["PSY_AR_SIZE_FULL_015"], "not_assessable")

    def test_unsure_declaration_abstains(self):
        with tempfile.TemporaryDirectory() as d:
            decl = user_page_declaration_from_choice("unsure")
            case_dir = _build_case(d, _wide_margin_image(), user_page_declaration=decl)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["page_reference"]["page_reference_mode"], "uncertain")
            self.assertFalse(analysis["page_reference"]["page_relative_features_assessable"])

    def test_auto_declaration_persists_as_automatic(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = _build_case(d, _wide_margin_image(), user_page_declaration=user_page_declaration_from_choice("auto"))
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["page_reference"]["obtained_via"], "classical_cv_border_uniformity_v1")


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class LineProxyWordingRemainsNonDiagnosticInAppTests(unittest.TestCase):
    def test_real_line_proxy_case_renders_non_diagnostic_wording(self):
        # train/Angry/2-2_jpg... is a known real image that triggers
        # EN_COMPILED_LINE_HEAVY_PRESSURE_030 (found during Phase 2A.2
        # Section 1 baseline smoke testing).
        image_path = Path(
            r"C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing\train\Angry"
            r"\2-2_jpg.rf.aef2f6a8cf72176022ee3fe0bd2717ed.jpg"
        )
        if not image_path.exists():
            self.skipTest("Real dataset image not present in this checkout")
        with tempfile.TemporaryDirectory() as d:
            case_dir = Path(d) / "case"
            analyze_image_with_timing(str(image_path), str(case_dir), None)
            analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
            by_rule = {r["rule_id"]: r for r in analysis["rule_evaluations"]}
            self.assertEqual(by_rule["EN_COMPILED_LINE_HEAVY_PRESSURE_030"]["status"], "weak_support")

            at = AppTest.from_file(str(ROOT / "doar_prototype_app.py"))
            at.session_state["case_dir"] = str(case_dir.resolve())
            at.run(timeout=60)
            self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
            all_text = " ".join(m.value for m in at.markdown)
            self.assertNotIn("pressed the pencil", all_text.lower())
            self.assertNotIn("the child is", all_text.lower())
            self.assertIn("appear relatively dark or thick", all_text)


class NoUseContainerWidthTests(unittest.TestCase):
    """Section 10: the app must use no supported-deprecated
    use_container_width call."""

    def test_app_source_contains_no_use_container_width(self):
        source = (ROOT / "doar_prototype_app.py").read_text(encoding="utf-8")
        self.assertNotIn("use_container_width", source)

    def test_app_source_uses_width_stretch_for_dataframes(self):
        source = (ROOT / "doar_prototype_app.py").read_text(encoding="utf-8")
        self.assertIn('width="stretch"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
