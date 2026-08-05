from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.deep.embedding_classifier import (
    DEFAULT_MODELS, run_embedding_classifier_experiment,
)

try:
    import sklearn  # noqa: F401
    _SK = True
except Exception:
    _SK = False


def _write_embeddings(root: Path, *, n_per_class=8, dim=12, seed=0):
    from doar.dataset import CLASSES
    rng = np.random.RandomState(seed)
    ids, splits, labels, embs = [], [], [], []
    for ci, cls in enumerate(CLASSES):
        centroid = rng.randn(dim) * 4
        for i in range(n_per_class):
            ids.append(f"{cls}_{i}")
            splits.append("train" if i < n_per_class - 2 else "valid")
            labels.append(cls)
            embs.append(centroid + rng.randn(dim) * 0.3)
    npz_path = root / "embeddings.npz"
    np.savez_compressed(npz_path, embeddings=np.array(embs, dtype="float32"),
                        image_ids=np.array(ids), splits=np.array(splits), labels=np.array(labels))
    (root / "embedding_metadata.json").write_text(json.dumps({
        "backbone": "dinov2_vits14", "embedding_dimension": dim,
    }), encoding="utf-8")
    return npz_path


class EmbeddingClassifierTests(unittest.TestCase):
    def test_default_models_are_linear_probe_and_small_mlp(self):
        self.assertEqual(DEFAULT_MODELS, ("logistic_regression", "mlp_small"))

    @unittest.skipUnless(_SK, "sklearn not installed")
    def test_well_separated_embeddings_classify_well(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            npz = _write_embeddings(root)
            result = run_embedding_classifier_experiment(npz, root / "out", seeds=(42,))
            self.assertFalse(result["backbone_fine_tuned"])
            self.assertEqual(result["backbone"], "dinov2_vits14")
            top = result["leaderboard"][0]
            self.assertGreater(top["mean_macro_f1"], 0.8)

    @unittest.skipUnless(_SK, "sklearn not installed")
    def test_test_split_never_referenced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            npz = _write_embeddings(root)
            result = run_embedding_classifier_experiment(npz, root / "out", seeds=(42,))
            for run in result["runs"]:
                self.assertTrue(run["test_used"] is False)

    @unittest.skipUnless(_SK, "sklearn not installed")
    def test_missing_split_raises(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            from doar.dataset import CLASSES
            npz_path = root / "embeddings.npz"
            n = 4
            np.savez_compressed(
                npz_path,
                embeddings=np.random.randn(n, 8).astype("float32"),
                image_ids=np.array([f"id{i}" for i in range(n)]),
                splits=np.array(["train"] * n),  # no valid split at all
                labels=np.array(list(CLASSES[:n])),
            )
            with self.assertRaises(ValueError):
                run_embedding_classifier_experiment(npz_path, root / "out", seeds=(42,))

    @unittest.skipUnless(_SK, "sklearn not installed")
    def test_custom_model_menu_falls_back_to_shared_classifier_library(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            npz = _write_embeddings(root)
            result = run_embedding_classifier_experiment(
                npz, root / "out", models=["random_forest"], seeds=(42,))
            self.assertEqual({r["model"] for r in result["leaderboard"]}, {"random_forest"})

    @unittest.skipUnless(_SK, "sklearn not installed")
    def test_predictions_csv_and_checkpoint_written(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            npz = _write_embeddings(root)
            out = root / "out"
            run_embedding_classifier_experiment(npz, out, models=["logistic_regression"], seeds=(42,))
            run_dir = out / "runs" / "logistic_regression_seed_42"
            self.assertTrue((run_dir / "predictions.csv").exists())
            self.assertTrue((run_dir / "model.joblib").exists())
            self.assertTrue((run_dir / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
