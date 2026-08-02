"""Item 4 — deep comparison runner: aggregation + validation selection (CPU).

Real training needs the dataset + GPU; here a synthetic trainer is injected so
the orchestration/selection logic is verified deterministically on CPU. The
synthetic results are clearly NOT real metrics.
"""

from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class AggregateTests(unittest.TestCase):
    def test_winner_is_highest_mean_valid_macro_f1(self):
        from doar.deep.compare import aggregate_and_select
        runs = [
            {"model": "a", "seed": 42, "valid_macro_f1": 0.50},
            {"model": "a", "seed": 123, "valid_macro_f1": 0.52},
            {"model": "b", "seed": 42, "valid_macro_f1": 0.70},
            {"model": "b", "seed": 123, "valid_macro_f1": 0.66},
        ]
        summary = aggregate_and_select(runs)
        self.assertEqual(summary["winner"], "b")
        self.assertFalse(summary["test_used"])
        b = next(r for r in summary["leaderboard"] if r["model"] == "b")
        self.assertAlmostEqual(b["mean_valid_macro_f1"], 0.68, places=6)
        self.assertGreater(b["std_valid_macro_f1"], 0.0)


class RunnerTests(unittest.TestCase):
    def test_runner_with_injected_synthetic_trainer(self):
        from doar.deep.compare import run_deep_comparison

        # Deterministic synthetic trainer — NOT real training.
        synthetic_scores = {"small_cnn": 0.40, "resnet18": 0.61}

        def fake_trainer(dataset, model, out, *, seed, epochs, batch_size,
                         image_size, device, grad_accum_steps, calibration):
            score = synthetic_scores[model] + (seed % 3) * 0.01
            Path(out).mkdir(parents=True, exist_ok=True)
            return {"best_valid_macro_f1": score, "checkpoint": str(Path(out) / "best.pt"),
                    "history": [{"epoch": 1, "valid_macro_f1": score}]}

        with tempfile.TemporaryDirectory() as d:
            summary = run_deep_comparison(
                "dataset_dir", d, models=["small_cnn", "resnet18"], seeds=(42, 123),
                batch_size=4, trainer=fake_trainer)
            self.assertEqual(summary["winner"], "resnet18")
            self.assertEqual(summary["safe_defaults"]["batch_size"], 4)
            self.assertTrue(summary["safe_defaults"]["mixed_precision"])
            self.assertTrue((Path(d) / "deep_comparison.json").exists())
            saved = json.loads((Path(d) / "deep_comparison.json").read_text())
            self.assertEqual(len(saved["runs"]), 4)  # 2 models x 2 seeds
            self.assertEqual(saved["winning_run"]["model"], "resnet18")

    def test_completed_run_is_skipped_and_incomplete_run_resumes(self):
        from doar.deep.compare import run_deep_comparison
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            done = root / "runs" / "a_seed_1"
            done.mkdir(parents=True)
            (done / "best.pt").write_bytes(b"done")
            (done / "training_result.json").write_text(json.dumps({
                "best_valid_macro_f1": .8, "checkpoint": str(done / "best.pt"),
                "history": [],
            }))
            incomplete = root / "runs" / "a_seed_2"
            incomplete.mkdir(parents=True)
            (incomplete / "last.pt").write_bytes(b"resume")
            calls = []

            def fake(dataset, model, out, **kwargs):
                calls.append(kwargs)
                return {"best_valid_macro_f1": .7,
                        "checkpoint": str(Path(out) / "best.pt"), "history": []}

            summary = run_deep_comparison(
                "dataset", d, models=["a"], seeds=(1, 2), trainer=fake)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["resume"], str(incomplete / "last.pt"))
            self.assertEqual(len(summary["runs"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
