"""Item 1 — leakage gate: detection, blocking, override audit, subject grouping."""

from __future__ import annotations
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _rows(**overrides):
    base = [
        {"image_id": "a", "path": "/x/a.png", "split": "train", "class": "Happy",
         "sha256": "h1", "phash": "0f0f0f0f0f0f0f0f"},
        {"image_id": "b", "path": "/x/b.png", "split": "valid", "class": "Sad",
         "sha256": "h2", "phash": "ffff0000ffff0000"},
        {"image_id": "c", "path": "/x/c.png", "split": "test", "class": "Angry",
         "sha256": "h3", "phash": "00ff00ff00ff00ff"},
    ]
    return base


class AssessTests(unittest.TestCase):
    def test_clean_dataset_passes(self):
        from doar.leakage import assess_leakage
        r = assess_leakage(_rows())
        self.assertTrue(r["leakage_ok"])

    def test_exact_cross_split_detected(self):
        from doar.leakage import assess_leakage
        rows = _rows()
        rows[1]["sha256"] = "h1"  # b(valid) == a(train)
        r = assess_leakage(rows)
        self.assertFalse(r["leakage_ok"])
        self.assertTrue(r["exact_cross_split_leakage"])

    def test_conflicting_labels_detected(self):
        from doar.leakage import assess_leakage
        rows = _rows()
        rows[1]["sha256"] = "h1"        # same image bytes
        rows[1]["split"] = "train"      # same split (not cross-split) ...
        rows[1]["class"] = "Angry"      # ... but different label
        r = assess_leakage(rows)
        self.assertTrue(r["conflicting_labels"])

    def test_near_duplicate_cross_split_detected(self):
        from doar.leakage import assess_leakage
        rows = _rows()
        rows[0]["phash"] = "0f0f0f0f0f0f0f0f"
        rows[1]["phash"] = "0f0f0f0f0f0f0f0e"  # 1 bit from a, different split
        r = assess_leakage(rows)
        self.assertTrue(r["near_cross_split_leakage"])

    def test_subject_level_leakage_detected(self):
        from doar.leakage import assess_leakage
        rows = _rows()
        for row in rows:
            row["subject_id"] = "child_7"   # same child across all splits
        r = assess_leakage(rows)
        self.assertTrue(r["subject_grouping_available"])
        self.assertTrue(r["subject_cross_split_leakage"])


class GateTests(unittest.TestCase):
    def _manifest(self, d, rows):
        p = Path(d) / "manifest.csv"
        fields = ["image_id", "path", "split", "class", "sha256", "phash", "subject_id"]
        with open(p, "w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return p

    def test_gate_blocks_on_leakage(self):
        from doar.leakage import enforce_leakage_gate, LeakageError
        with tempfile.TemporaryDirectory() as d:
            rows = _rows()
            rows[1]["sha256"] = "h1"
            m = self._manifest(d, rows)
            with self.assertRaises(LeakageError):
                enforce_leakage_gate(m, Path(d) / "gate", timestamp="2026-01-01T00:00:00Z")

    def test_gate_override_requires_justification_and_audits(self):
        from doar.leakage import enforce_leakage_gate, LeakageError
        with tempfile.TemporaryDirectory() as d:
            rows = _rows()
            rows[1]["sha256"] = "h1"
            m = self._manifest(d, rows)
            gate = Path(d) / "gate"
            # override without justification still blocks
            with self.assertRaises(LeakageError):
                enforce_leakage_gate(m, gate, allow_override=True,
                                     timestamp="2026-01-01T00:00:00Z")
            # override with justification passes + writes audit + clean manifest
            report = enforce_leakage_gate(
                m, gate, allow_override=True,
                override_justification="approved for pilot; duplicates are known scans",
                initiated_by="ahmed", timestamp="2026-01-01T00:00:00Z")
            self.assertEqual(report["gate"], "overridden")
            audit = (gate / "leakage_override_audit.jsonl").read_text().strip()
            self.assertIn("approved for pilot", audit)
            self.assertTrue((gate / "clean_manifest.csv").exists())
            self.assertTrue((gate / "quarantine.csv").exists())

    def test_gate_passes_clean(self):
        from doar.leakage import enforce_leakage_gate
        with tempfile.TemporaryDirectory() as d:
            m = self._manifest(d, _rows())
            report = enforce_leakage_gate(m, Path(d) / "gate",
                                          timestamp="2026-01-01T00:00:00Z")
            self.assertEqual(report["gate"], "passed")
            self.assertTrue(report["leakage_ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
