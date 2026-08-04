from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.dataset_gate import CleanSplitGateFailed
from doar.stage0_runner import StageSpec, build_real_stage0_plan, run_stage0


def _fake_stage(sid: str, calls: list, *, fail: bool = False):
    def run(stage_dir: Path, smoke: bool):
        calls.append(sid)
        if fail:
            raise ValueError(f"{sid} failed on purpose")
        (stage_dir / "out.txt").write_text("ok")
        return {"sid": sid, "smoke": smoke}
    return run


class OrchestratorCoreTests(unittest.TestCase):
    def test_full_mode_blocked_when_gate_fails(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            stages = [StageSpec("A", "a", _fake_stage("A", calls))]
            with self.assertRaises(CleanSplitGateFailed):
                run_stage0(stages, Path(d) / "out", smoke=False, repo_root=d)
            self.assertEqual(calls, [])  # never even attempted

    def test_smoke_mode_always_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            stages = [StageSpec("A", "a", _fake_stage("A", calls))]
            manifest = run_stage0(stages, Path(d) / "out", smoke=True, repo_root=d)
            self.assertEqual(calls, ["A"])
            self.assertFalse(manifest["gate_report"]["gate_passed"])
            self.assertTrue(manifest["smoke_mode"])

    def test_dependency_failure_propagates_and_skips_dependents(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            stages = [
                StageSpec("X", "x", _fake_stage("X", calls, fail=True)),
                StageSpec("Y", "y", _fake_stage("Y", calls), depends_on=("X",)),
                StageSpec("Z", "z independent", _fake_stage("Z", calls)),
            ]
            manifest = run_stage0(stages, Path(d) / "out", smoke=True, repo_root=d)
            self.assertEqual(manifest["results"]["X"]["status"], "failed")
            self.assertEqual(manifest["results"]["Y"]["status"], "skipped_dependency_failed")
            self.assertEqual(manifest["results"]["Z"]["status"], "completed")
            self.assertEqual(calls, ["X", "Z"])

    def test_resume_skips_already_complete_stage(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            calls = []
            stages = [StageSpec("A", "a", _fake_stage("A", calls))]
            run_stage0(stages, out, smoke=True, repo_root=d)
            calls.clear()
            manifest = run_stage0(stages, out, smoke=True, repo_root=d, resume=True)
            self.assertEqual(calls, [])
            self.assertEqual(manifest["results"]["A"]["status"], "skipped_already_complete")

    def test_no_resume_reruns_completed_stage(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            calls = []
            stages = [StageSpec("A", "a", _fake_stage("A", calls))]
            run_stage0(stages, out, smoke=True, repo_root=d)
            calls.clear()
            run_stage0(stages, out, smoke=True, repo_root=d, resume=False)
            self.assertEqual(calls, ["A"])

    def test_circular_dependency_raises(self):
        with tempfile.TemporaryDirectory() as d:
            stages = [
                StageSpec("A", "a", lambda sd, s: {}, depends_on=("B",)),
                StageSpec("B", "b", lambda sd, s: {}, depends_on=("A",)),
            ]
            with self.assertRaises(ValueError):
                run_stage0(stages, Path(d) / "out", smoke=True, repo_root=d)

    def test_unknown_dependency_raises(self):
        with tempfile.TemporaryDirectory() as d:
            stages = [StageSpec("A", "a", lambda sd, s: {}, depends_on=("NOPE",))]
            with self.assertRaises(ValueError):
                run_stage0(stages, Path(d) / "out", smoke=True, repo_root=d)

    def test_run_manifest_written_with_git_commit_and_stage_order(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            out = Path(d) / "out"
            stages = [StageSpec("A", "a", _fake_stage("A", calls))]
            run_stage0(stages, out, smoke=True, repo_root=d)
            import json
            manifest = json.loads((out / "stage0_run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("git_commit", manifest)
            self.assertEqual(manifest["stage_order"], ["A"])


def _stratified_manifest(root: Path, n_per_class: int = 4) -> Path:
    img_dir = root / "imgs"
    img_dir.mkdir()
    classes = ["Angry", "Fear", "Happy", "Sad"]
    rows = []
    for ci, cls in enumerate(classes):
        for i in range(n_per_class):
            split = "train" if i < n_per_class - 1 else "valid"
            colour = ["red", "green", "blue", "black"][ci]
            img = Image.new("RGB", (60, 60), "white")
            ImageDraw.Draw(img).ellipse((10, 10, 50, 50), fill=colour)
            p = img_dir / f"{cls}_{i}.png"
            img.save(p)
            rows.append({"image_id": f"{cls}_{i}", "path": str(p), "relative_path": p.name,
                        "split": split, "class": cls, "sha256": "x", "phash": "x",
                        "width": 60, "height": 60, "file_size": 100, "readable": "True",
                        "provenance": "synthetic"})
    manifest = root / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return manifest


class RealPlanSmokeTests(unittest.TestCase):
    """Exercises build_real_stage0_plan()'s actual wiring (not fake stages)
    against small synthetic data -- proves the real functions are called
    with the right arguments, never a scientific result."""

    def test_stage_a_objective_features_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            plan = build_real_stage0_plan(manifest, root / "out")
            stage_a = next(s for s in plan if s.stage_id == "A_objective_features")
            result = run_stage0([stage_a], root / "out", smoke=True, repo_root=d)
            self.assertEqual(result["results"]["A_objective_features"]["status"], "completed")

    def test_stage_b_handcrafted_groups_smoke_after_a(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            plan = build_real_stage0_plan(manifest, root / "out")
            stage_a = next(s for s in plan if s.stage_id == "A_objective_features")
            stage_b = next(s for s in plan if s.stage_id == "B_handcrafted_groups")
            result = run_stage0([stage_a, stage_b], root / "out", smoke=True, repo_root=d)
            self.assertEqual(result["results"]["B_handcrafted_groups"]["status"], "completed")

    def test_stage_b_without_a_fails_with_clear_message(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            plan = build_real_stage0_plan(manifest, root / "out")
            stage_b = next(s for s in plan if s.stage_id == "B_handcrafted_groups")
            result = run_stage0([stage_b], root / "out", smoke=True, repo_root=d)
            self.assertEqual(result["results"]["B_handcrafted_groups"]["status"], "failed")
            self.assertIn("Stage A", result["results"]["B_handcrafted_groups"]["error"])

    def test_stage_c_dinov2_smoke_uses_synthetic_embeddings_not_network(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            plan = build_real_stage0_plan(manifest, root / "out")
            stage_c = next(s for s in plan if s.stage_id == "C_dinov2_embeddings")
            result = run_stage0([stage_c], root / "out", smoke=True, repo_root=d)
            self.assertEqual(result["results"]["C_dinov2_embeddings"]["status"], "completed")
            self.assertIn("SMOKE", result["results"]["C_dinov2_embeddings"]["detail"]["note"])

    def test_declared_stage_order_matches_experiment_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = _stratified_manifest(root)
            plan = build_real_stage0_plan(manifest, root / "out")
            self.assertEqual(
                [s.stage_id for s in plan],
                ["A_objective_features", "B_handcrafted_groups", "C_dinov2_embeddings",
                 "D_densenet121", "E_convnext_tiny_probe", "F_fusion"],
            )
            f_stage = next(s for s in plan if s.stage_id == "F_fusion")
            self.assertEqual(f_stage.depends_on, ("A_objective_features", "C_dinov2_embeddings"))


if __name__ == "__main__":
    unittest.main()
