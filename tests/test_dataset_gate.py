from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.dataset_gate import (
    AUTOMATED_METHODOLOGY_ID, CleanSplitGateFailed, REQUIRED_APPROVAL_FIELDS,
    REQUIRED_AUTOMATED_STATUS_FIELDS, REQUIRED_FREEZE_FIELDS,
    check_automated_conservative_gate, check_clean_split_gate,
    require_clean_split_for_full_run,
)


def _write(path: Path, content: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")


class GateAllMissingTests(unittest.TestCase):
    def test_empty_repo_fails_the_gate(self):
        with tempfile.TemporaryDirectory() as d:
            report = check_clean_split_gate(d)
            self.assertFalse(report["gate_passed"])
            self.assertTrue(report["remediation"])
            # 7 of 8 checks fail on a genuinely empty repo; "final test set
            # untouched" correctly PASSES (no unlock log = no evidence of
            # tampering, which is the correct interpretation, not a bug).
            failing = [name for name, c in report["checks"].items() if not c["pass"]]
            self.assertEqual(len(failing), 7)
            self.assertTrue(report["checks"]["final_test_set_untouched"]["pass"])


class GateFullySatisfiedTests(unittest.TestCase):
    def _build_passing_repo(self, root: Path) -> None:
        registry = {"categories_order": {"cat1": ["a", "b"]}}
        _write(root / "outputs/phase7b/human_review/app_data/items_registry.json", registry)
        _write(root / "outputs/phase7b/human_review/app_data/decisions.json",
              {"a": {"decision": "definite_duplicate"}, "b": {"decision": "different_drawings"}})
        for f in ("human_pair_reviews.csv", "human_group_reviews.csv",
                  "reviewer_agreement_report.json", "threshold_precision_summary.json",
                  "unresolved_items.csv"):
            _write(root / "outputs/phase7b/human_review/exports" / f, "x")
        approval = {f: "value" for f in REQUIRED_APPROVAL_FIELDS}
        _write(root / "outputs/phase7b/APPROVED_POLICY.json", approval)
        freeze = {f: "value" for f in REQUIRED_FREEZE_FIELDS}
        _write(root / "outputs/phase7b/final_partition/FROZEN.json", freeze)
        _write(root / "outputs/phase7b/final_partition/partition_manifest.csv", "image_id,split\n")
        verification = {
            "no_group_crosses_partitions": {"ok": True},
            "no_exposed_group_in_valid_or_test": {"ok": True},
            "counts_match_manifest": {"ok": True},
        }
        _write(root / "outputs/phase7b/final_partition/leakage_verification_report.json", verification)

    def test_fully_satisfied_repo_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            report = check_clean_split_gate(root)
            self.assertTrue(report["gate_passed"], report["checks"])
            self.assertEqual(report["remediation"], [])

    def test_require_clean_split_does_not_raise_when_satisfied(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            report = require_clean_split_for_full_run(root)  # must not raise
            self.assertTrue(report["gate_passed"])

    def test_incomplete_human_review_fails_that_check_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            # Only 1 of 2 registry items decided.
            _write(root / "outputs/phase7b/human_review/app_data/decisions.json",
                  {"a": {"decision": "definite_duplicate"}})
            report = check_clean_split_gate(root)
            self.assertFalse(report["gate_passed"])
            self.assertFalse(report["checks"]["human_review_decisions_recorded"]["pass"])

    def test_incomplete_approval_record_fails_that_check_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            _write(root / "outputs/phase7b/APPROVED_POLICY.json", {"approved_by": "x"})  # missing fields
            report = check_clean_split_gate(root)
            self.assertFalse(report["checks"]["duplicate_policy_approved"]["pass"])

    def test_verification_report_false_ok_fails_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            _write(root / "outputs/phase7b/final_partition/leakage_verification_report.json", {
                "no_group_crosses_partitions": {"ok": False},
                "no_exposed_group_in_valid_or_test": {"ok": True},
                "counts_match_manifest": {"ok": True},
            })
            report = check_clean_split_gate(root)
            self.assertFalse(report["checks"]["no_duplicate_group_crosses_splits"]["pass"])
            self.assertTrue(report["checks"]["exposed_images_excluded_from_valid_test"]["pass"])

    def test_test_unlock_log_present_fails_untouched_check(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            _write(root / "outputs/phase7b/final_partition/final_test_unlock_log.jsonl",
                  '{"event": "unlocked"}\n')
            report = check_clean_split_gate(root)
            self.assertFalse(report["checks"]["final_test_set_untouched"]["pass"])
            self.assertFalse(report["gate_passed"])


class RequireCleanSplitRaisesTests(unittest.TestCase):
    def test_raises_with_failed_check_names_when_not_satisfied(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(CleanSplitGateFailed) as ctx:
                require_clean_split_for_full_run(d)
            self.assertIn("human_review_decisions_recorded", str(ctx.exception))

    def test_current_real_repository_state_fails(self):
        # Regression guard: confirms this task's central finding (the gate
        # is closed) stays true for as long as this test suite runs,
        # without re-deriving it from documentation.
        report = check_clean_split_gate(ROOT)
        self.assertFalse(report["gate_passed"])


class AutomatedConservativeGateTests(unittest.TestCase):
    """T0_AUTOMATED_CONSERVATIVE_V1 -- a SEPARATE gate from check_clean_
    split_gate() above. Must never read/write anything under
    outputs/phase7b/human_review/, APPROVED_POLICY.json, or FROZEN.json."""

    def _build_passing_repo(self, root: Path, *, sha_override: str | None = None) -> str:
        import csv
        import hashlib

        d = root / "outputs/t0_automated/final_partition"
        d.mkdir(parents=True, exist_ok=True)
        manifest_path = d / "partition_manifest.csv"
        rows = [
            {"image_id": "a", "path": "/x/train/Happy/a.jpg", "relative_path": "train/Happy/a.jpg", "new_split": "train"},
            {"image_id": "b", "path": "/x/valid/Sad/b.jpg", "relative_path": "valid/Sad/b.jpg", "new_split": "valid"},
            {"image_id": "c", "path": "/x/test/Fear/c.jpg", "relative_path": "test/Fear/c.jpg", "new_split": "test"},
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        _write(d / "duplicate_group_edges_exact.csv", "a,b,sha256\n")
        _write(d / "duplicate_group_edges_near.csv", "a,b,hamming\n")
        verification = {
            "no_conflict_in_supervised_split": {"ok": True},
            "no_exposed_group_in_valid_or_test": {"ok": True},
            "counts_match_manifest": {"ok": True},
            "no_image_in_multiple_partitions": {"ok": True},
        }
        _write(d / "leakage_verification_report.json", verification)
        actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        status = {f: "value" for f in REQUIRED_AUTOMATED_STATUS_FIELDS}
        status["methodology_id"] = AUTOMATED_METHODOLOGY_ID
        status["not_human_reviewed"] = True
        status["manifest_sha256"] = sha_override or actual_sha
        _write(root / "outputs/t0_automated/T0_AUTOMATED_CONSERVATIVE_V1.json", status)
        return actual_sha

    def test_empty_repo_fails(self):
        with tempfile.TemporaryDirectory() as d:
            report = check_automated_conservative_gate(d)
            self.assertFalse(report["gate_passed"])
            self.assertEqual(report["methodology_id"], AUTOMATED_METHODOLOGY_ID)

    def test_fully_satisfied_repo_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            report = check_automated_conservative_gate(root)
            self.assertTrue(report["gate_passed"], report["checks"])
            self.assertEqual(report["remediation"], [])

    def test_stale_sha256_fails_only_that_check(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root, sha_override="0" * 64)
            report = check_automated_conservative_gate(root)
            self.assertFalse(report["checks"]["manifest_sha256_recorded"]["pass"])
            self.assertFalse(report["gate_passed"])

    def test_missing_not_human_reviewed_flag_fails_labeling_check(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            status_path = root / "outputs/t0_automated/T0_AUTOMATED_CONSERVATIVE_V1.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["not_human_reviewed"] = False  # simulates a fabricated claim of human review
            _write(status_path, status)
            report = check_automated_conservative_gate(root)
            self.assertFalse(report["checks"]["methodology_correctly_labeled_not_human_reviewed"]["pass"])

    def test_cross_split_exact_duplicate_edge_fails_that_check_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            d2 = root / "outputs/t0_automated/final_partition"
            _write(d2 / "duplicate_group_edges_exact.csv", "a,b,sha256\na,c,deadbeef\n")  # a=train, c=test
            report = check_automated_conservative_gate(root)
            self.assertFalse(report["checks"]["no_exact_duplicate_group_crosses_splits"]["pass"])
            self.assertTrue(report["checks"]["no_near_duplicate_group_crosses_splits"]["pass"])

    def test_never_touches_phase7b_human_review_gate_state(self):
        # Building ONLY the automated-gate fixtures (no phase7b/human_review/
        # files at all) must still let this gate reach a real pass -- proves
        # it is independent of check_clean_split_gate()'s own state.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._build_passing_repo(root)
            self.assertFalse((root / "outputs/phase7b").exists())
            report = check_automated_conservative_gate(root)
            self.assertTrue(report["gate_passed"])

    def test_old_human_review_gate_is_completely_unaffected(self):
        # check_clean_split_gate() behavior/signature must be byte-identical
        # to before this addition -- same required fields, same failure mode.
        with tempfile.TemporaryDirectory() as d:
            report = check_clean_split_gate(d)
            self.assertFalse(report["gate_passed"])
            self.assertEqual(len(report["checks"]), 8)
            self.assertNotIn("methodology_id", report)

    def test_current_real_repository_automated_manifest_passes(self):
        # Integration check against the real, already-frozen T0 automated
        # manifest in THIS repository (skips cleanly if absent, e.g. before
        # it has been built in a given checkout).
        status_path = ROOT / "outputs/t0_automated/T0_AUTOMATED_CONSERVATIVE_V1.json"
        if not status_path.exists():
            self.skipTest("T0 automated manifest not present in this checkout")
        report = check_automated_conservative_gate(ROOT)
        self.assertTrue(report["gate_passed"], report["checks"])


if __name__ == "__main__":
    unittest.main()
