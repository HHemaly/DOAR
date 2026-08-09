"""Technical View rendering tests for Phase 2B's pilot summary. Uses a
minimal fake `st` object rather than a real Streamlit AppTest -- this
module deliberately takes `st` as a parameter precisely so it can be
unit-tested without a running app (see technical_view.py's docstring)."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.technical_view import render_phase2b_pilot_summary


class _FakeStreamlit:
    def __init__(self):
        self.calls = []

    def header(self, text):
        self.calls.append(("header", text))

    def subheader(self, text):
        self.calls.append(("header", text))

    def caption(self, text):
        self.calls.append(("caption", text))

    def write(self, text):
        self.calls.append(("write", text))

    def dataframe(self, data, **kwargs):
        self.calls.append(("dataframe", data))


class RenderPhase2bPilotSummaryTests(unittest.TestCase):
    def test_absent_model_checkpoint_behavior_no_artifacts_present(self):
        with tempfile.TemporaryDirectory() as d:
            fake = _FakeStreamlit()
            render_phase2b_pilot_summary(fake, Path(d))  # no artifacts/phase2b/ at all
            kinds = [c[0] for c in fake.calls]
            self.assertIn("header", kinds)
            self.assertIn("write", kinds)
            # Must not crash and must not fabricate a dataframe from nothing.
            self.assertNotIn("dataframe", kinds)

    def test_renders_a_dataframe_when_real_metrics_exist(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metrics_dir = root / "artifacts" / "phase2b"
            metrics_dir.mkdir(parents=True)
            with (metrics_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["class", "sufficient_support"])
                writer.writeheader()
                writer.writerow({"class": "person", "sufficient_support": "True"})
                writer.writerow({"class": "hand", "sufficient_support": "False"})
            fake = _FakeStreamlit()
            render_phase2b_pilot_summary(fake, root)
            kinds = [c[0] for c in fake.calls]
            self.assertIn("dataframe", kinds)

    def test_never_claims_a_result_is_psychologically_validated(self):
        with tempfile.TemporaryDirectory() as d:
            fake = _FakeStreamlit()
            render_phase2b_pilot_summary(fake, Path(d))
            all_text = " ".join(str(c[1]) for c in fake.calls if c[0] in ("header", "caption", "write"))
            for forbidden in ("depress", "anxiety", "diagnos"):
                self.assertNotIn(forbidden, all_text.lower())


if __name__ == "__main__":
    unittest.main()
