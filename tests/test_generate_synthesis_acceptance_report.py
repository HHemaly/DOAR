"""Tests for scripts/generate_synthesis_acceptance_report.py -- the
repository-level, reproducible acceptance-report generator (post-
correction acceptance audit). Reuses clinician_review_app.py's own
cache-first case loading -- no live Gemini calls, no new scoring logic.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "generate_synthesis_acceptance_report", ROOT / "scripts" / "generate_synthesis_acceptance_report.py")
report_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report_gen)


class NoLiveGeminiCallsTests(unittest.TestCase):
    def test_module_never_calls_the_live_observer_or_verifier(self):
        source = (ROOT / "scripts" / "generate_synthesis_acceptance_report.py").read_text(encoding="utf-8")
        self.assertNotIn("GeminiVisualObserver(", source)
        self.assertNotIn("GeminiVisualVerifier(", source)
        self.assertNotIn("run_live_observer_and_verifier(", source)


class BuildCaseReportTests(unittest.TestCase):
    def test_unknown_case_id_returns_none(self):
        self.assertIsNone(report_gen.build_case_report("not_a_real_case"))

    def test_p2b_0003_report_has_expected_shape(self):
        report = report_gen.build_case_report("p2b_0003")
        self.assertIsNotNone(report)
        for key in ("image_id", "objective_profile_categories", "unified_evidence_count",
                    "literature_linked_associations", "overall_synthesis", "candidate_hypotheses",
                    "page_reference", "parent_view"):
            self.assertIn(key, report)
        self.assertEqual(report["image_id"], "p2b_0003")
        self.assertIn("level", report["overall_synthesis"])
        self.assertIn("populated", report["parent_view"])
        self.assertIn("rule_id_leak", report["parent_view"])

    def test_parent_view_never_leaks_a_rule_id_for_any_real_case(self):
        for case_id in report_gen.app.list_case_ids():
            report = report_gen.build_case_report(case_id)
            if report is None:
                continue
            self.assertFalse(report["parent_view"]["rule_id_leak"], f"{case_id}: rule ID leaked into parent view")

    def test_parent_view_is_populated_for_every_real_case(self):
        for case_id in report_gen.app.list_case_ids():
            report = report_gen.build_case_report(case_id)
            if report is None:
                continue
            self.assertTrue(report["parent_view"]["populated"], f"{case_id}: parent view not populated")


class CsvRowTests(unittest.TestCase):
    def test_to_csv_row_has_all_declared_columns(self):
        report = report_gen.build_case_report("p2b_0003")
        row = report_gen.to_csv_row(report)
        self.assertEqual(set(row.keys()), set(report_gen.CSV_COLUMNS))


class MainCliTests(unittest.TestCase):
    def test_single_case_run_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(sys, "argv", [
                    "generate_synthesis_acceptance_report.py", "--image-id", "p2b_0003", "--output-dir", tmp]):
                report_gen.main()
            self.assertTrue((Path(tmp) / "report.json").exists())
            self.assertTrue((Path(tmp) / "report.csv").exists())
            import json
            data = json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["image_id"], "p2b_0003")

    def test_unknown_case_id_run_writes_empty_report_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(sys, "argv", [
                    "generate_synthesis_acceptance_report.py", "--image-id", "not_a_real_case", "--output-dir", tmp]):
                report_gen.main()  # must not raise
            import json
            data = json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(data, [])


if __name__ == "__main__":
    unittest.main()
