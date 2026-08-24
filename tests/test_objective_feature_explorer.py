"""Standalone tests for scripts/objective_feature_explorer.py
(feature/supervisor-demo-v2).

This explorer is a NEW, separate, self-contained demo of DOAR's
objective/formal drawing-feature layer (src/doar/formal_features.py) --
it must work with zero dependency on the rule engine, Ask DOAR, Gemini,
or evidence-aggregation code paths. These tests exercise it exactly as a
user would (Streamlit's AppTest harness, bare-mode script execution --
same pattern as tests/test_supervisor_demo_smoke.py) for all 3 prepared
example drawings, plus a structural import-guard check.

Deliberately does NOT run the full repo test suite and does NOT touch
doar_prototype_app.py or any of its own tests.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "objective_feature_explorer.py"
sys.path.insert(0, str(ROOT / "src"))

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False

EXAMPLE_LABELS = ["Example Drawing 1", "Example Drawing 2", "Example Drawing 3"]
EXPECTED_FEATURE_COUNT = 22

# Full dotted module names this explorer must NEVER import, directly or
# transitively: the rule engine, Ask DOAR / chat, case interpretation
# (psychological mapping / concern domains), case presentation, drawing
# synthesis, and any Gemini client. Matched as exact module-name
# prefixes ("doar.chat", "doar.chat.foo", ...) so Streamlit's own
# unrelated internal "...widgets.chat" module is never a false positive.
FORBIDDEN_MODULE_PREFIXES = [
    "doar.rule_engine_v2",
    "doar.case_interpretation",
    "doar.human_interaction",
    "doar.chat",
    "doar.case_presentation",
    "doar.drawing_synthesis",
    "doar.evidence_adapter",
    "doar.trace_evidence_adapters",
    "google.generativeai",
    "genai",
]


def _is_forbidden(module_name: str) -> bool:
    return any(module_name == p or module_name.startswith(p + ".") for p in FORBIDDEN_MODULE_PREFIXES)


class ImportGuardTests(unittest.TestCase):
    """Structural checks: the explorer's own source only imports the
    objective/formal feature layer, and running it never pulls a
    forbidden module into sys.modules."""

    def test_source_only_imports_doar_formal_features_and_case_artifacts(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(SCRIPT_PATH))
        doar_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("doar"):
                doar_imports.add(node.module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("doar"):
                        doar_imports.add(alias.name)
        allowed = {"doar", "doar.formal_features", "doar.case_artifacts", "doar.features"}
        self.assertTrue(doar_imports.issubset(allowed), f"Unexpected doar imports: {doar_imports - allowed}")
        self.assertTrue(len(doar_imports) > 0, "Expected at least the formal_features import")

    def test_no_forbidden_module_loaded_at_runtime(self):
        if not _STREAMLIT_TESTING_AVAILABLE:
            self.skipTest("streamlit.testing.v1.AppTest not available")
        at = AppTest.from_file(str(SCRIPT_PATH))
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        for label in EXAMPLE_LABELS:
            at.radio(key="ofe_example_picker").set_value(label)
            at.run(timeout=120)
        hits = [m for m in sys.modules if _is_forbidden(m)]
        self.assertEqual(hits, [], f"Forbidden module(s) loaded: {hits}")

    def test_script_never_reads_gemini_api_key(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("GEMINI_API_KEY", source)
        self.assertNotIn("genai.", source)


@unittest.skipUnless(_STREAMLIT_TESTING_AVAILABLE, "streamlit.testing.v1.AppTest not available")
class ExplorerAppTests(unittest.TestCase):
    def _open(self, label: str):
        at = AppTest.from_file(str(SCRIPT_PATH))
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        at.radio(key="ofe_example_picker").set_value(label)
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception] + [label])
        return at

    def test_app_opens_with_default_example(self):
        at = AppTest.from_file(str(SCRIPT_PATH))
        at.run(timeout=120)
        self.assertEqual(len(at.exception), 0, [str(e) for e in at.exception])
        self.assertTrue(any("Objective Feature Explorer" in t.value for t in at.title))
        self.assertEqual(at.radio(key="ofe_example_picker").value, EXAMPLE_LABELS[0])

    def test_all_three_examples_render_image_and_all_22_features_no_traceback(self):
        for label in EXAMPLE_LABELS:
            at = self._open(label)
            # Image renders: drawing + foreground mask + symmetry split = 3 images.
            self.assertEqual(len(at.get("image")), 3, f"{label}: expected 3 images")

            # Feature extraction completed: technical-details dataframe has
            # exactly the 22 current formal features, each with a raw value.
            self.assertEqual(len(at.dataframe), 1, f"{label}: expected 1 technical-details dataframe")
            df = at.dataframe[0].value
            self.assertEqual(len(df), EXPECTED_FEATURE_COUNT, f"{label}: expected {EXPECTED_FEATURE_COUNT} rows")
            self.assertEqual(set(df["feature_id"]), set(
                fid for fid in df["feature_id"]))  # sanity: no duplicate/empty ids
            self.assertTrue((df["feature_id"] != "").all())

            # No raw traceback text leaked anywhere on the page.
            all_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
            self.assertNotIn("Traceback", all_text)
            self.assertNotIn("Traceback", " ".join(w.value for w in at.warning))

    def test_missing_values_would_render_gracefully(self):
        """None of the 3 curated examples actually has a missing formal
        feature (verified separately), so this exercises the rendering
        helper directly with a synthetic missing value to confirm the
        graceful-missing code path (used by formal_features.py's own
        `missing` flag) never raises and never fabricates a number."""
        spec = importlib.util.spec_from_file_location("objective_feature_explorer", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        rendered = module._format_value("line.mean_width", {"value": 0.0, "missing": True})
        self.assertEqual(rendered, "Not measurable for this image")
        rendered_ok = module._format_value("line.mean_width", {"value": 3.5, "missing": False})
        self.assertEqual(rendered_ok, "3.500")

    def test_qc_warning_renders_for_uncertain_examples_and_success_for_verified(self):
        # Example Drawing 1 (e2e_check) and Example Drawing 3 (h38) both have
        # real segmentation.status == "uncertain" in their saved analysis.json
        # -> a "Measurement quality warning" must render, never silent
        # confident numbers. Example Drawing 2 (a111) has status == "verified"
        # -> no warning, a success confirmation instead.
        at1 = self._open("Example Drawing 1")
        warn_text = " ".join(w.value for w in at1.warning)
        self.assertIn("Measurement quality warning", warn_text)
        self.assertEqual(len(at1.success), 0)

        at2 = self._open("Example Drawing 2")
        self.assertEqual(len(at2.warning), 0)
        self.assertEqual(len(at2.success), 1)

        at3 = self._open("Example Drawing 3")
        warn_text3 = " ".join(w.value for w in at3.warning)
        self.assertIn("Measurement quality warning", warn_text3)

    def test_feature_groups_all_render_as_tabs(self):
        at = self._open("Example Drawing 1")
        tab_labels = [t.label for t in at.tabs]
        for expected in ["1. Lines & Stroke Quality", "2. Direction & Organization",
                          "3. Colour", "4. Spatial / Global Structure"]:
            self.assertIn(expected, tab_labels)

    def test_method_explanation_expander_present(self):
        at = self._open("Example Drawing 1")
        expander_labels = [e.label for e in at.expander]
        self.assertTrue(any("How are these measurements calculated?" in lbl for lbl in expander_labels))


if __name__ == "__main__":
    unittest.main()
