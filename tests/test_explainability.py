"""Item 15 — objective-feature importance (CPU) + Grad-CAM layer selection.

Grad-CAM heatmap generation needs torch (real model); here we test its pure
layer-selection logic and that visual vs tabular attribution are kept separate.
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


class PermutationImportanceTests(unittest.TestCase):
    def test_informative_feature_ranks_above_noise(self):
        from doar.explain.feature_importance import permutation_importance
        rng = np.random.RandomState(0)
        n, d = 200, 4
        X = rng.randn(n, d)
        # class depends strongly on feature 0, weakly on others (noise).
        y = (X[:, 0] > 0).astype(int) * 2 + (X[:, 1] > 0.5).astype(int)
        y = np.clip(y, 0, 3)

        def predict_proba(Xin):
            # deterministic scorer keyed on feature 0 (the informative one)
            score = np.zeros((len(Xin), 4))
            for k in range(4):
                center = [-1.5, -0.5, 0.5, 1.5][k]
                score[:, k] = -np.abs(Xin[:, 0] - center)
            e = np.exp(score - score.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)

        result = permutation_importance(predict_proba, X, y, ["f0", "f1", "f2", "f3"], n_repeats=3)
        self.assertEqual(result["attribution_type"], "tabular_objective_features")
        top = result["importances"][0]["feature"]
        self.assertEqual(top, "f0")            # informative feature most important
        self.assertIn("does NOT localize", result["disclaimer"])

    def test_local_contributions_and_coefficients(self):
        from doar.explain.feature_importance import local_contributions, linear_coefficients
        coef = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0, 0, 0], [0, 0, 0]])
        names = ["a", "b", "c"]
        classes = ["Angry", "Fear", "Happy", "Sad"]
        lc = local_contributions(coef, x=[1.0, 0.0, 1.0], feature_mean=[0, 0, 0],
                                 feature_names=names, class_names=classes,
                                 predicted_class_index=0)
        self.assertEqual(lc["contributions"][0]["feature"], "a")   # |2*1| largest
        coeffs = linear_coefficients(coef, names, classes)
        self.assertIn("Angry", coeffs["per_class"])


class GradCamSeparationTests(unittest.TestCase):
    def test_target_layer_names(self):
        from doar.explain.gradcam import target_layer_name
        self.assertEqual(target_layer_name("resnet18"), "layer4")
        self.assertEqual(target_layer_name("efficientnet_b0"), "features")

    def test_visual_and_tabular_are_separate_modules_with_distinct_disclaimers(self):
        from doar.explain.gradcam import DISCLAIMER as CAM
        from doar.explain.feature_importance import DISCLAIMER as FEAT
        self.assertIn("Classifier", CAM) if "Classifier" in CAM else self.assertIn("classifier", CAM.lower())
        self.assertIn("does NOT localize", FEAT)
        self.assertNotEqual(CAM, FEAT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
