"""
Tests for D1 — calibration wiring. Exercises the numpy paths (temperature
softmax + validation-only fit). The torch-dependent collect/calibrate paths are
smoke-imported only (they require the [deep] extra + a checkpoint + dataset).
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


class CalibrationWiringTests(unittest.TestCase):
    def test_temperature_softens_confidence(self):
        from doar.deep.inference import _softmax
        logits = np.array([3.0, 0.5, 0.2, 0.0])
        p1 = _softmax(logits, 1.0)
        p2 = _softmax(logits, 3.0)
        self.assertAlmostEqual(float(p1.sum()), 1.0, places=9)
        self.assertAlmostEqual(float(p2.sum()), 1.0, places=9)
        self.assertGreater(p1.max(), p2.max())  # higher T -> less confident

    def test_fit_temperature_is_validation_only_and_reports_ece(self):
        from doar.deep.calibration import fit_temperature
        rng = np.random.RandomState(1)
        logits = rng.randn(300, 4) * 5.0
        labels = logits.argmax(1)
        result = fit_temperature(logits, labels)
        self.assertEqual(result["fit_split"], "valid")
        self.assertGreater(result["temperature"], 0.0)
        for key in ("validation_ece_before", "validation_ece_after",
                    "validation_nll_before", "validation_nll_after",
                    "validation_brier_before", "validation_brier_after"):
            self.assertIn(key, result)

    def test_calibrate_checkpoint_importable_without_torch(self):
        # Function exists and imports; torch is loaded lazily inside it.
        from doar.deep.calibration import calibrate_checkpoint, collect_validation_logits
        self.assertTrue(callable(calibrate_checkpoint))
        self.assertTrue(callable(collect_validation_logits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
