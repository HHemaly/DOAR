"""CLI-level tests (real subprocess invocations of main.py, not direct
Python calls) for this session's 4 new commands: extract-hog-features,
train-handcrafted-groups, train-embedding-classifier, run-stage0."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def _run(*args, timeout=120):
    return subprocess.run([sys.executable, "main.py", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)


def _stratified_manifest(root: Path, n_per_class: int = 3) -> Path:
    img_dir = root / "imgs"
    img_dir.mkdir()
    classes = ["Angry", "Fear", "Happy", "Sad"]
    rows = []
    for ci, cls in enumerate(classes):
        for i in range(n_per_class):
            split = "train" if i < n_per_class - 1 else "valid"
            colour = ["red", "green", "blue", "black"][ci]
            img = Image.new("RGB", (50, 50), "white")
            ImageDraw.Draw(img).ellipse((5, 5, 45, 45), fill=colour)
            p = img_dir / f"{cls}_{i}.png"
            img.save(p)
            rows.append({"image_id": f"{cls}_{i}", "path": str(p), "split": split,
                        "class": cls, "readable": "True"})
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return manifest


class GateCheckOnlyCliTests(unittest.TestCase):
    def test_gate_check_only_reports_current_repo_gate_state(self):
        with tempfile.TemporaryDirectory() as d:
            result = _run("run-stage0", "--manifest", "outputs/phase5/manifest.csv",
                          "--output", str(Path(d) / "out"), "--gate-check-only")
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertIn("gate_passed", payload)
            self.assertIn("remediation", payload)


class ExtractHogFeaturesCliTests(unittest.TestCase):
    def test_extract_hog_features_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            out = root / "hog_out"
            result = _run("extract-hog-features", "--manifest", str(manifest), "--output", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["processed"], 12)
            self.assertTrue((out / "hog_features.csv").exists())


class TrainHandcraftedGroupsCliTests(unittest.TestCase):
    def test_train_handcrafted_groups_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            obj_out = root / "objective"
            _run("extract-features", "--manifest", str(manifest), "--output", str(obj_out))
            hog_out = root / "hog"
            _run("extract-hog-features", "--manifest", str(manifest), "--output", str(hog_out))
            comp_out = root / "comparison"
            result = _run(
                "train-handcrafted-groups",
                "--objective-features", str(obj_out / "features.csv"),
                "--hog-features", str(hog_out / "hog_features.csv"),
                "--output", str(comp_out),
                "--models", "logistic_regression", "--seeds", "42",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(len(payload["leaderboard"]), 4)  # 4 group configs


class TrainEmbeddingClassifierCliTests(unittest.TestCase):
    def test_train_embedding_classifier_cli(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rng = np.random.RandomState(0)
            classes = ["Angry", "Fear", "Happy", "Sad"]
            ids, splits, labels, embs = [], [], [], []
            for ci, cls in enumerate(classes):
                centroid = rng.randn(8) * 3
                for i in range(6):
                    ids.append(f"{cls}_{i}")
                    splits.append("train" if i < 4 else "valid")
                    labels.append(cls)
                    embs.append(centroid + rng.randn(8) * 0.3)
            npz_path = root / "embeddings.npz"
            np.savez_compressed(npz_path, embeddings=np.array(embs, dtype="float32"),
                                image_ids=np.array(ids), splits=np.array(splits), labels=np.array(labels))
            out = root / "out"
            result = _run("train-embedding-classifier", "--embeddings", str(npz_path),
                          "--output", str(out), "--models", "logistic_regression", "--seeds", "42")
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertGreater(payload["leaderboard"][0]["mean_macro_f1"], 0.0)


class RunStage0CliTests(unittest.TestCase):
    def test_run_stage0_smoke_only_stage_a(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            out = root / "stage0"
            result = _run("run-stage0", "--manifest", str(manifest), "--output", str(out),
                          "--smoke", "--only", "A_objective_features")
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["results"]["A_objective_features"]["status"], "completed")

    def test_run_stage0_full_mode_blocked_by_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            out = root / "stage0"
            result = _run("run-stage0", "--manifest", str(manifest), "--output", str(out))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CleanSplitGateFailed", result.stderr)


if __name__ == "__main__":
    unittest.main()
