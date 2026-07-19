"""Item 7 — fusion calibration numerics (validation-only), CPU."""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def _overconfident(n=300, seed=0):
    """Overconfident probabilities: correct class but too-peaked -> high ECE."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 4, n)
    logits = rng.randn(n, 4)
    logits[np.arange(n), y] += 1.0            # mildly correct
    logits *= 4.0                             # sharpen -> overconfident
    exp = np.exp(logits - logits.max(1, keepdims=True))
    proba = exp / exp.sum(1, keepdims=True)
    return y, proba


class FusionCalibrationTests(unittest.TestCase):
    def test_temperature_fit_is_validation_only_and_reports_before_after(self):
        from doar.fusion.calibrate import fit_fusion_temperature
        y, proba = _overconfident()
        r = fit_fusion_temperature(proba, y)
        self.assertEqual(r["fit_split"], "valid")
        self.assertFalse(r["test_used"])
        self.assertTrue(r["raw_preserved"])
        self.assertGreater(r["temperature"], 0)
        for key in ("validation_ece_before", "validation_ece_after",
                    "validation_brier_before", "validation_brier_after",
                    "validation_nll_before", "validation_nll_after"):
            self.assertIn(key, r)

    def test_calibration_reduces_ece_on_overconfident(self):
        from doar.fusion.calibrate import fit_fusion_temperature
        y, proba = _overconfident()
        r = fit_fusion_temperature(proba, y)
        self.assertLessEqual(r["validation_ece_after"], r["validation_ece_before"] + 1e-9)

    def test_apply_temperature_preserves_shape_and_normalizes(self):
        from doar.fusion.calibrate import apply_temperature
        _, proba = _overconfident(n=10)
        out = apply_temperature(proba, 2.0)
        self.assertEqual(out.shape, proba.shape)
        np.testing.assert_allclose(out.sum(1), np.ones(10), atol=1e-6)

    def test_reliability_bins(self):
        from doar.fusion.calibrate import reliability_bins
        y, proba = _overconfident(n=100)
        bins = reliability_bins(y, proba, bins=10)
        self.assertEqual(len(bins), 10)
        self.assertTrue(all("mean_confidence" in b and "accuracy" in b for b in bins))


if __name__ == "__main__":
    unittest.main(verbosity=2)
