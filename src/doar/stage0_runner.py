"""
stage0_runner.py -- config-driven, resumable, gate-checked orchestrator for
the Stage 0 experiment sequence (EXPERIMENT_PROTOCOL.md Phase 6):

  A. objective-feature classical baseline
  B. HOG + colour + geometry handcrafted-feature comparison
  C. DINOv2 frozen-embedding classifier
  D. DenseNet121 (staged freeze -> unfreeze transfer learning)
  E. ConvNeXt-Tiny feasibility probe
  F. image + objective-feature fusion (depends on A and C/D)

The sequencing/resume/gate-enforcement core (`run_stage0`, `StageSpec`) is
deliberately decoupled from what each stage actually does -- exactly like
deep/compare.py's tests inject a fake trainer, this module can be driven
with injected fake StageSpecs for fast, real, non-network unit testing.
`build_real_stage0_plan()` wires the ACTUAL Stage 0 pipeline using the
already-implemented, already-tested functions in extract.py, experiments.py,
hog_features.py, handcrafted_comparison.py, deep/embeddings.py,
deep/embedding_classifier.py, deep/trainers.py, and fusion/trainer.py -- no
new model logic lives in this file, only orchestration.

FULL (non-smoke) runs refuse to start unless
dataset_gate.check_clean_split_gate() passes -- see dataset_gate.py. SMOKE
runs (synthetic/tiny data, a handful of images, 1 epoch) are always
permitted, since they exercise pipeline code, not a scientific result.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .dataset_gate import CleanSplitGateFailed, check_clean_split_gate

RUNNER_VERSION = "doar_stage0_runner_v1"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    description: str
    run: Callable[[Path, bool], dict]      # (stage_output_dir, smoke) -> result dict
    depends_on: tuple[str, ...] = ()


def _topological_order(stages: list[StageSpec]) -> list[StageSpec]:
    by_id = {s.stage_id: s for s in stages}
    for s in stages:
        for dep in s.depends_on:
            if dep not in by_id:
                raise ValueError(f"Stage {s.stage_id!r} depends on unknown stage {dep!r}")
    ordered: list[StageSpec] = []
    visited: set[str] = set()

    def visit(stage: StageSpec, stack: tuple[str, ...]):
        if stage.stage_id in visited:
            return
        if stage.stage_id in stack:
            raise ValueError(f"Circular Stage 0 dependency involving {stage.stage_id!r}")
        for dep in stage.depends_on:
            visit(by_id[dep], stack + (stage.stage_id,))
        visited.add(stage.stage_id)
        ordered.append(stage)

    for s in stages:
        visit(s, ())
    return ordered


def run_stage0(
    stages: list[StageSpec], output_root: str | Path, *, smoke: bool, resume: bool = True,
    repo_root: str | Path = ".", run_id: str | None = None,
) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    gate_report = check_clean_split_gate(repo_root)
    if not smoke and not gate_report["gate_passed"]:
        raise CleanSplitGateFailed(
            "Clean-split gate failed; refusing to run Stage 0 in FULL mode. "
            "Run with smoke=True for a pipeline-verification-only pass, or "
            "complete the remediation steps in check_clean_split_gate()['remediation']."
        )

    run_id = run_id or datetime.now(timezone.utc).strftime("stage0_%Y%m%dT%H%M%S%fZ")
    ordered = _topological_order(stages)
    results: dict[str, dict] = {}
    failed_or_skipped: set[str] = set()

    for stage in ordered:
        stage_dir = output_root / stage.stage_id
        marker = stage_dir / "STAGE_COMPLETE.json"
        blocked_by = [d for d in stage.depends_on if d in failed_or_skipped]
        if blocked_by:
            results[stage.stage_id] = {
                "status": "skipped_dependency_failed", "blocked_by": blocked_by,
            }
            failed_or_skipped.add(stage.stage_id)
            continue
        if resume and marker.exists():
            results[stage.stage_id] = {
                "status": "skipped_already_complete",
                "detail": json.loads(marker.read_text(encoding="utf-8")),
            }
            continue
        stage_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        try:
            detail = stage.run(stage_dir, smoke)
            marker.write_text(json.dumps({
                "stage_id": stage.stage_id, "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "smoke": smoke, "detail": detail,
            }, indent=2, default=str), encoding="utf-8")
            results[stage.stage_id] = {"status": "completed", "detail": detail}
        except Exception as exc:
            results[stage.stage_id] = {
                "status": "failed", "error": str(exc), "error_type": type(exc).__name__,
            }
            failed_or_skipped.add(stage.stage_id)

    run_manifest = {
        "runner_version": RUNNER_VERSION, "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "smoke_mode": smoke, "resume": resume,
        "gate_report": gate_report,
        "stage_order": [s.stage_id for s in ordered],
        "results": results,
    }
    (output_root / "stage0_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")
    return run_manifest


# ---------------------------------------------------------------------------
# The REAL Stage 0 plan -- pure wiring over already-implemented, already-
# tested functions. No new model/training logic lives below this line.
# ---------------------------------------------------------------------------
def build_real_stage0_plan(
    manifest_csv: str | Path, output_root: str | Path,
    *, dinov2_backbone: str = "dinov2_vits14",
) -> list[StageSpec]:
    manifest_csv = Path(manifest_csv)
    output_root = Path(output_root)

    def stage_a(stage_dir: Path, smoke: bool) -> dict:
        from .extract import extract_features
        from .experiments import run_feature_experiment
        features_dir = stage_dir / "features"
        extract_summary = extract_features(manifest_csv, features_dir)
        seeds = (42,) if smoke else (42, 123, 2026)
        models = ["logistic_regression"] if smoke else None
        result = run_feature_experiment(features_dir / "features.csv", stage_dir / "classical",
                                        models=models, seeds=seeds)
        return {"extract": extract_summary, "leaderboard": result["leaderboard"]}

    def stage_b(stage_dir: Path, smoke: bool) -> dict:
        from .hog_features import extract_hog_features
        from .handcrafted_comparison import run_handcrafted_group_comparison
        objective_features = output_root / "A_objective_features" / "features" / "features.csv"
        if not objective_features.exists():
            raise RuntimeError(
                f"Stage B depends on Stage A's features.csv ({objective_features}); "
                "run Stage A first (it is not a formal `depends_on` dependency because "
                "HOG extraction from raw images is independent of it, but the group "
                "comparison needs both CSVs).")
        hog_dir = stage_dir / "hog"
        hog_summary = extract_hog_features(manifest_csv, hog_dir)
        seeds = (42,) if smoke else (42, 123, 2026)
        models = ["logistic_regression"] if smoke else None
        result = run_handcrafted_group_comparison(
            objective_features, hog_dir / "hog_features.csv", stage_dir / "comparison",
            models=models, seeds=seeds)
        return {"hog_extraction": hog_summary, "leaderboard": result["leaderboard"]}

    def stage_c(stage_dir: Path, smoke: bool) -> dict:
        from .deep.embedding_classifier import run_embedding_classifier_experiment
        if smoke:
            # DINOv2 extraction needs live network access to torch.hub
            # (facebookresearch/dinov2) -- not attempted in smoke mode.
            # The classifier-training code path is proven with a small,
            # explicitly-synthetic embeddings cache instead.
            import numpy as np
            from .dataset import CLASSES
            rng = np.random.RandomState(0)
            n = 8
            embs = np.concatenate([rng.randn(n, 8) + i * 3 for i in range(len(CLASSES))])
            labels = np.repeat(CLASSES, n)
            splits = np.tile(["train"] * (n - 2) + ["valid"] * 2, len(CLASSES))
            ids = np.array([f"synthetic_{i}" for i in range(len(embs))])
            npz_path = stage_dir / "synthetic_embeddings.npz"
            np.savez_compressed(npz_path, embeddings=embs.astype("float32"),
                                image_ids=ids, splits=splits, labels=labels)
            (stage_dir / "embedding_metadata.json").write_text(json.dumps({
                "backbone": f"{dinov2_backbone}_SMOKE_SYNTHETIC_NOT_REAL",
                "embedding_dimension": 8,
            }), encoding="utf-8")
            result = run_embedding_classifier_experiment(npz_path, stage_dir / "classifier", seeds=(42,))
            return {"note": "SMOKE MODE: synthetic embeddings, not real DINOv2 extraction",
                    "leaderboard": result["leaderboard"]}
        from .deep.embeddings import extract_embeddings
        embed_dir = stage_dir / "embeddings"
        embed_summary = extract_embeddings(manifest_csv, embed_dir, backbone=dinov2_backbone)
        result = run_embedding_classifier_experiment(
            embed_dir / "embeddings.npz", stage_dir / "classifier", seeds=(42, 123, 2026))
        return {"extraction": embed_summary, "leaderboard": result["leaderboard"]}

    def _infer_dataset_root() -> Path:
        """train_image_model needs a torchvision-ImageFolder root
        (root/train/<class>/*, root/valid/<class>/*), NOT the manifest CSV
        itself -- derive it from the manifest's own `path` column, which
        for the real dataset looks like `<root>/<split>/<class>/<file>`."""
        with open(manifest_csv, newline="", encoding="utf-8") as handle:
            import csv as _csv
            first = next(_csv.DictReader(handle))
        p = Path(first["path"])
        split_dir = next((parent for parent in p.parents if parent.name == first["split"]), None)
        if split_dir is None:
            raise RuntimeError(
                f"Could not infer an ImageFolder dataset root from manifest row "
                f"path={p} split={first['split']!r} -- expected a "
                f"<root>/<split>/<class>/<file> layout.")
        return split_dir.parent

    def _tiny_synthetic_dataset(root: Path, per_class: int = 4, size: int = 32) -> None:
        """SMOKE MODE ONLY. A flat synthetic manifest (used by stages A-C)
        has no train/<class>/ physical layout to reuse, so this builds a
        dedicated tiny ImageFolder tree instead -- same pattern as
        tests/test_trainer_regression.py::_tiny_dataset."""
        import numpy as np
        from PIL import Image
        from .dataset import CLASSES
        for split in ("train", "valid"):
            for ci, cls in enumerate(CLASSES):
                class_dir = root / split / cls
                class_dir.mkdir(parents=True, exist_ok=True)
                for i in range(per_class):
                    arr = (np.random.RandomState(ci * 100 + i).rand(size, size, 3) * 255).astype("uint8")
                    Image.fromarray(arr).save(class_dir / f"{i}.png")

    def _deep_stage(model_name: str, epochs_smoke: int, epochs_full: int):
        def run(stage_dir: Path, smoke: bool) -> dict:
            from .deep.trainers import train_image_model
            if smoke:
                dataset_root = stage_dir / "synthetic_imagefolder"
                _tiny_synthetic_dataset(dataset_root)
            else:
                dataset_root = _infer_dataset_root()
            result = train_image_model(
                dataset=str(dataset_root), model_name=model_name, output=str(stage_dir / "training"),
                seed=42, image_size=32 if smoke else 224, batch_size=4,
                epochs=epochs_smoke if smoke else epochs_full,
                device="cpu" if smoke else "auto", augmentation="conservative",
                freeze_epochs=1 if smoke else 3, class_weighting=True,
                patience=1 if smoke else 5, pretrained_weights="none" if smoke else "DEFAULT",
            )
            return {"best_valid_macro_f1": result.get("best_valid_macro_f1"),
                    "smoke": smoke, "model": model_name,
                    "dataset_root": str(dataset_root)}
        return run

    def stage_f(stage_dir: Path, smoke: bool) -> dict:
        from .fusion.trainer import train_primary_fusion
        features_csv = output_root / "A_objective_features" / "features" / "features.csv"
        embed_dir = output_root / "C_dinov2_embeddings" / (
            "synthetic_embeddings.npz" if smoke else "embeddings/embeddings.npz")
        if not features_csv.exists() or not embed_dir.exists():
            raise RuntimeError("Stage F depends on Stage A's features.csv and "
                               "Stage C's embeddings -- both must complete first.")
        seeds = (42,) if smoke else (42, 123, 2026)
        methods = ["early_scaled_concat"] if smoke else None
        result = train_primary_fusion(features_csv, embed_dir, stage_dir, methods=methods, seeds=seeds)
        return {"leaderboard": result["leaderboard"]}

    return [
        StageSpec("A_objective_features", "Experiment A: objective-feature classical baseline", stage_a),
        StageSpec("B_handcrafted_groups", "Experiment B: HOG/colour/geometry comparison", stage_b),
        StageSpec("C_dinov2_embeddings", "Experiment C: DINOv2 frozen-embedding classifier", stage_c),
        StageSpec("D_densenet121", "Experiment D: DenseNet121 transfer learning",
                  _deep_stage("densenet121", epochs_smoke=1, epochs_full=50)),
        StageSpec("E_convnext_tiny_probe", "Experiment E: ConvNeXt-Tiny feasibility probe",
                  _deep_stage("convnext_tiny", epochs_smoke=1, epochs_full=15)),
        StageSpec("F_fusion", "Experiment F: image + objective-feature fusion", stage_f,
                  depends_on=("A_objective_features", "C_dinov2_embeddings")),
    ]
