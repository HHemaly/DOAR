"""Tests for the 5 DOAR-TRACE Phase 2A experiment scripts (Sections 3-6,
10): each writes a real CSV/JSON from a real, non-test sample, never
`split == "test"`. Uses small sample sizes to keep this suite fast --
full-sample reproducibility (identical output across two independent
runs, seeded sampling only, no other randomness) was verified manually
for every script and is reported in PHASE2A_IMPLEMENTATION_REPORT.md,
not re-verified here on every test run."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.page_frame import STATUSES as PAGE_FRAME_STATUSES
from doar.phase2a_feature_ground_truth import CSV_FIELDS as GT_FIELDS, write_ground_truth
from doar.phase2a_feature_invariance import CSV_FIELDS as INVARIANCE_FIELDS
from doar.phase2a_page_frame_audit import DEFAULT_MANIFEST as _AUDIT_MANIFEST
from doar.phase2a_page_frame_audit import FIELDS as AUDIT_FIELDS, run_audit
from doar.phase2a_rule_trigger_distribution import CSV_FIELDS as TRIGGER_FIELDS, run_trigger_distribution
from doar.phase2a_threshold_sensitivity import CSV_FIELDS as THRESHOLD_FIELDS, run_threshold_sensitivity

# outputs/phase5/manifest.csv is derived from the real, private local
# drawing dataset -- it is gitignored (see .gitignore) and never
# committed, so it does not exist on a fresh GitHub Actions checkout.
# These scripts' *default* manifest_path points there; tests below that
# never override manifest_path are local-only integration tests.
_MANIFEST_AVAILABLE = _AUDIT_MANIFEST.exists()
_NO_MANIFEST_REASON = (
    "requires the local, private outputs/phase5/manifest.csv "
    "(gitignored, not available in CI)")


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


class PageFrameAuditScriptTests(unittest.TestCase):
    """Every test here passes an explicit `output_path` in a temp
    directory -- these small samples must never overwrite the canonical,
    full-sample artifact committed at artifacts/phase2a/dataset_page_frame_audit.csv."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out_path = Path(self.temp.name) / "audit.csv"

    @unittest.skipUnless(_MANIFEST_AVAILABLE, _NO_MANIFEST_REASON)
    def test_writes_expected_columns_and_statuses(self):
        summary = run_audit(per_class=2, seed=1, output_path=self.out_path)
        self.assertEqual(summary["n_images"], 8)  # 2 per class x 4 classes
        self.assertEqual(summary["output_path"], str(self.out_path))
        rows = _read_csv(self.out_path)
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows[0].keys()), set(AUDIT_FIELDS))
        for row in rows:
            self.assertIn(row["page_frame_status"], PAGE_FRAME_STATUSES)

    @unittest.skipUnless(_MANIFEST_AVAILABLE, _NO_MANIFEST_REASON)
    def test_never_reads_the_test_split(self):
        summary = run_audit(per_class=2, seed=1, output_path=self.out_path)
        rows = _read_csv(Path(summary["output_path"]))
        self.assertTrue(all(row["source_split"] != "test" for row in rows))


class FeatureGroundTruthScriptTests(unittest.TestCase):
    def test_writes_expected_columns_and_mostly_passes(self):
        summary = write_ground_truth()
        self.assertEqual(GT_FIELDS, [
            "case_id", "feature_id", "expected_value", "measured_value", "absolute_error",
            "relative_error", "tolerance", "status", "limitations",
        ])
        # Validates measurement implementation only -- a small number of
        # documented real fails/not_applicable rows is expected and fine;
        # the overwhelming majority must pass, or the harness itself (or
        # the pipeline) has regressed.
        counts = summary["status_counts"]
        self.assertGreaterEqual(counts.get("pass", 0), 15)


class FeatureInvarianceScriptSchemaTests(unittest.TestCase):
    """The full 880-row invariance audit (4 images x 20 transforms x 11
    features) takes ~40s -- too slow to run twice per CI invocation, and
    unnecessary: no randomness is involved anywhere in this script (fixed
    image sample, deterministic transforms), so reproducibility is
    structural, not merely observed. This test only checks the schema."""

    def test_csv_fields_are_stable(self):
        self.assertEqual(INVARIANCE_FIELDS, [
            "image_id", "transform", "feature_id", "original_value", "transformed_value", "absolute_change",
            "percentage_change", "status", "should_theoretically_preserve", "limitations",
        ])


class ThresholdSensitivityScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out_path = Path(self.temp.name) / "threshold.csv"

    @unittest.skipUnless(_MANIFEST_AVAILABLE, _NO_MANIFEST_REASON)
    def test_writes_expected_columns_for_every_executable_rule(self):
        summary = run_threshold_sensitivity(n_per_class=3, seed=1, output_path=self.out_path)
        rows = _read_csv(Path(summary["output_path"]))
        self.assertEqual(set(rows[0].keys()), set(THRESHOLD_FIELDS))
        rule_ids = {row["rule_id"] for row in rows}
        # The 6 original tier-1 rules -- every one must have a documented
        # threshold_source, never silently invented.
        self.assertEqual(rule_ids, {
            "PSY_AR_SIZE_SMALL_016", "PSY_AR_SIZE_HALF_014", "PSY_AR_SIZE_FULL_015",
            "PSY_AR_PLACE_TOP_017", "PSY_AR_PLACE_LEFT_018", "PSY_AR_PLACE_RIGHT_019",
        })
        for row in rows:
            self.assertTrue(row["source_status"], row["rule_id"])


class RuleTriggerDistributionScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    @unittest.skipUnless(_MANIFEST_AVAILABLE, _NO_MANIFEST_REASON)
    def test_writes_all_registry_rules_with_correct_executable_split(self):
        out_path = Path(self.temp.name) / "trigger1.csv"
        summary = run_trigger_distribution(per_class=2, seed=1, output_path=out_path)
        rows = _read_csv(Path(summary["output_path"]))
        self.assertEqual(set(rows[0].keys()), set(TRIGGER_FIELDS))
        executable_rows = [r for r in rows if r["executable"] == "True"]
        disabled_rows = [r for r in rows if r["executable"] == "False"]
        self.assertEqual(len(executable_rows), 10)
        self.assertTrue(len(disabled_rows) > 0)
        for row in executable_rows:
            self.assertNotEqual(row["trigger_count"], "")
        for row in disabled_rows:
            self.assertEqual(row["trigger_count"], "")

    @unittest.skipUnless(_MANIFEST_AVAILABLE, _NO_MANIFEST_REASON)
    def test_reproducible_on_a_small_sample(self):
        out_path1 = Path(self.temp.name) / "trigger_run1.csv"
        out_path2 = Path(self.temp.name) / "trigger_run2.csv"
        summary1 = run_trigger_distribution(per_class=2, seed=3, output_path=out_path1)
        rows1 = _read_csv(Path(summary1["output_path"]))
        summary2 = run_trigger_distribution(per_class=2, seed=3, output_path=out_path2)
        rows2 = _read_csv(Path(summary2["output_path"]))
        self.assertEqual(rows1, rows2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
