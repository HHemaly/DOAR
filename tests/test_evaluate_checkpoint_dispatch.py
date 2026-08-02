"""Regression tests for evaluate_model()'s checkpoint-type guard.

Found during a full pipeline run (2026-08-02): `evaluate --deep-comparison`
resolved to a PyTorch `.pt` checkpoint and passed it straight to
`evaluate_model()`, which unconditionally called `joblib.load()` -- crashing
with a confusing `UnpicklingError` instead of a clear message. `evaluate` only
ever supported the plain sklearn whole-image baseline produced by `train`;
deep and fusion checkpoints must be evaluated via
`export-probabilities` + `evaluate-predictions` instead (see
CURRENT_STATE_AUDIT.md and RUN_GUIDE_WINDOWS.md 7.8).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class EvaluateCheckpointDispatchTests(unittest.TestCase):
    def test_pt_checkpoint_raises_clear_error_not_unpickling_error(self):
        from doar.models import evaluate_model
        with tempfile.TemporaryDirectory() as tmp:
            fake_pt = Path(tmp) / "best.pt"
            fake_pt.write_bytes(b"not a real torch checkpoint")
            with self.assertRaises(ValueError) as ctx:
                evaluate_model(
                    manifest=str(Path(tmp) / "manifest.csv"),
                    checkpoint=str(fake_pt),
                    output=str(Path(tmp) / "out"),
                    split="valid",
                    unlock_test=False,
                )
            message = str(ctx.exception)
            self.assertIn("export-probabilities", message)
            self.assertIn("evaluate-predictions", message)

    def test_fusion_bundle_raises_clear_error(self):
        from doar.models import evaluate_model
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            fake_bundle = Path(tmp) / "fusion.joblib"
            joblib.dump({"checkpoint_type": "doar_fusion_bundle_v1"}, fake_bundle)
            with self.assertRaises(ValueError) as ctx:
                evaluate_model(
                    manifest=str(Path(tmp) / "manifest.csv"),
                    checkpoint=str(fake_bundle),
                    output=str(Path(tmp) / "out"),
                    split="valid",
                    unlock_test=False,
                )
            message = str(ctx.exception)
            self.assertIn("export-probabilities", message)


if __name__ == "__main__":
    unittest.main()
