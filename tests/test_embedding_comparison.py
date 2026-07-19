"""Item 5 — 5-way embedding comparison assembly, aligned by sample_id (CPU)."""

from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def _block(ids, dim, labels, seed=0):
    """Build {split: {id: (vec, label)}} for a single 'valid' split."""
    rng = np.random.RandomState(seed)
    return {"valid": {sid: (rng.randn(dim), labels[i]) for i, sid in enumerate(ids)}}


class AssembleTests(unittest.TestCase):
    def test_concatenates_blocks_aligned_by_id(self):
        from doar.fusion.embedding_comparison import assemble_config
        ids = ["s0", "s1", "s2"]
        labels = [0, 1, 2]
        feat = _block(ids, 5, labels, seed=1)
        # embeddings in DIFFERENT order but same ids/labels
        emb = {"valid": {}}
        rng = np.random.RandomState(2)
        for sid, lab in zip(["s2", "s0", "s1"], [2, 0, 1]):
            emb["valid"][sid] = (rng.randn(8), lab)
        sample_ids, X, y = assemble_config([feat, emb], "valid")
        self.assertEqual(sample_ids, ["s0", "s1", "s2"])
        self.assertEqual(X.shape, (3, 13))          # 5 + 8 concatenated
        self.assertEqual(y.tolist(), [0, 1, 2])

    def test_intersection_only(self):
        from doar.fusion.embedding_comparison import assemble_config
        feat = _block(["s0", "s1", "s2"], 4, [0, 1, 2], seed=1)
        emb = _block(["s0", "s1"], 4, [0, 1], seed=2)
        ids, X, y = assemble_config([feat, emb], "valid")
        self.assertEqual(ids, ["s0", "s1"])         # s2 dropped (not in both)
        self.assertEqual(X.shape[0], 2)

    def test_label_disagreement_fails(self):
        from doar.fusion.embedding_comparison import assemble_config
        feat = _block(["s0", "s1"], 4, [0, 1], seed=1)
        emb = _block(["s0", "s1"], 4, [0, 2], seed=2)   # s1 label differs
        with self.assertRaises(ValueError):
            assemble_config([feat, emb], "valid")

    def test_five_configurations_defined(self):
        from doar.fusion.embedding_comparison import CONFIGURATIONS
        self.assertEqual(set(CONFIGURATIONS), {
            "objective_only", "generic_embeddings_only", "finetuned_embeddings_only",
            "objective_plus_generic", "objective_plus_finetuned"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
