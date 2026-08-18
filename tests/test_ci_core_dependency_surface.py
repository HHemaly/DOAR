"""Regression test for the 2026-08 GitHub Actions CORE-job failure.

Root cause: several test modules called production code paths that need
optional dependencies (scikit-learn/joblib via experiments.py::_deps(),
pypdf, matplotlib) without an availability guard. This was invisible
locally because the dev .venv has every optional extra installed, but
the CORE job installs only `.[cv,dev]` -- no `ml`, `ingest`, `deep`, or
`embeddings` extra, and no matplotlib at all (it is not declared as a
dependency of any extra). This test pins both halves of that fact: the
CORE job's real install surface, and that every test module known to
touch one of these optional dependencies still defines its skip-guard
flag -- so a future edit cannot silently drop a guard and reintroduce
the same CI break.

2026-08 update (separate CI-portability fix): the CORE job now ALSO
installs `ui`/`phase2c6` (both pre-existing pyproject.toml extras) --
`unittest discover` was collecting Streamlit-based UI test files
(test_annotate_ui.py, test_clinician_review_app.py, test_generate_
synthesis_acceptance_report.py, test_phase2c6_canvas_compat.py) that
failed at IMPORT time with neither installed, which is a different
failure mode than the guarded-at-call-time optional-dependency modules
below (those already degrade gracefully via their own `_SK`/`_JOBLIB`/
`_PYPDF`/`_MPL`/`_STREAMLIT_TESTING_AVAILABLE`-style flags when a
dependency is missing). `ml`/`ingest`/`deep`/`embeddings` remain
deliberately absent from the CORE install -- this test still pins that.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class CoreJobInstallSurfaceTests(unittest.TestCase):
    def _core_install_step(self) -> str:
        text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        # Keep this a plain substring check (no yaml dependency required)
        # since `dev` itself does not declare PyYAML.
        marker = 'pip install -e ".[cv,dev,ui,phase2c6]"'
        self.assertIn(marker, text,
                       "CORE job's install step changed -- if it now installs "
                       "different extras, some skip-guards below may be safe "
                       "to remove; if it installs fewer, more guards are needed.")
        return marker

    def test_core_job_installs_cv_dev_ui_and_phase2c6_extras(self):
        self._core_install_step()

    def test_ml_ingest_deep_embeddings_extras_not_in_core_install(self):
        text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        core_section = text.split("ml_cpu:")[0]
        for extra in ("[ml", "[ingest", "[deep", "[embeddings"):
            self.assertNotIn(extra, core_section,
                             f"CORE job now installs the {extra!r} extra -- "
                             "re-check whether the skip-guards this test "
                             "checks for are still necessary.")

    def test_ui_and_phase2c6_extras_are_in_core_install(self):
        # Positive control for the CI-portability fix: `unittest discover`
        # collects Streamlit-based UI test files (test_annotate_ui.py,
        # test_clinician_review_app.py, test_generate_synthesis_
        # acceptance_report.py, test_phase2c6_canvas_compat.py) that need
        # `streamlit`/`streamlit-drawable-canvas` importable at collection
        # time, not just at call time -- unlike the guarded modules below.
        text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        core_section = text.split("ml_cpu:")[0]
        for extra in (",ui,", ",phase2c6]"):
            self.assertIn(extra, core_section, f"CORE job no longer installs an extra matching {extra!r}.")

    def test_matplotlib_not_declared_in_any_optional_dependency(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("matplotlib", text,
                         "matplotlib is now a declared dependency -- "
                         "tests/test_thesis_cross_family.py's _MPL guard "
                         "may no longer be necessary.")


# (module, attribute) pairs that must exist and gate the specific test
# method(s) that call an optional-dependency code path, so the guard
# cannot be silently deleted by a future edit.
_GUARDED_MODULES = [
    ("test_stage0_runner", "_SK"),
    ("test_new_experiment_cli", "_SK"),
    ("test_handcrafted_comparison", "_SK"),
    ("test_embedding_classifier", "_SK"),
    ("test_evaluate_checkpoint_dispatch", "_JOBLIB"),
    ("test_source_catalog_and_registry_v2", "_PYPDF"),
    ("test_thesis_cross_family", "_MPL"),
    ("test_smoke_experiment", "_SK"),
]


class OptionalDependencyGuardsPresentTests(unittest.TestCase):
    """Every test module that reaches an optional-dependency code path
    (sklearn/joblib/pypdf/matplotlib) must still define its skip-guard
    flag, and that flag must be a plain bool reflecting real importability
    -- not a hardcoded True that would defeat the guard."""

    def test_every_known_optional_dependency_module_defines_its_guard_flag(self):
        sys.path.insert(0, str(ROOT / "tests"))
        for module_name, flag_name in _GUARDED_MODULES:
            module = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(module, flag_name),
                f"{module_name}.py no longer defines {flag_name} -- the "
                "CORE-job dependency guard for this file was removed.")
            self.assertIsInstance(getattr(module, flag_name), bool)


if __name__ == "__main__":
    unittest.main()
