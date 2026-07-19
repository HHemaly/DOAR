"""
compare.py — multi-seed deep-model comparison runner (Item 4).

Trains a fixed line-up (small_cnn, mobilenet_v3_small, resnet18, efficientnet_b0;
heavier models optional) across seeds 42/123/2026 on the SAME leak-safe split,
selects the winner by mean VALIDATION macro-F1 (test never used), and aggregates
mean ± std per model. Reuses deep.trainers.train_image_model (freeze->fine-tune,
class weights, differential LRs, early stopping, mixed precision, resume).

Safe defaults for a 6 GB GPU (Quadro P3200): batch_size 4, mixed precision (in the
trainer), gradient accumulation configurable. device="auto" with CPU fallback.

The trainer is injectable so the orchestration/aggregation is unit-tested on CPU
without torch; real training requires the dataset + GPU.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

DEFAULT_MODELS = ["small_cnn", "mobilenet_v3_small", "resnet18", "efficientnet_b0"]
DEFAULT_SEEDS = (42, 123, 2026)


def aggregate_and_select(runs: list[dict]) -> dict:
    """Pure aggregation + selection. runs: [{model, seed, valid_macro_f1}].
    Winner = highest mean validation macro-F1 (tie-break: lower std)."""
    by_model: dict[str, list[float]] = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(float(r["valid_macro_f1"]))
    leaderboard = []
    for model, scores in by_model.items():
        arr = np.asarray(scores, dtype=float)
        leaderboard.append({
            "model": model,
            "seeds_run": len(scores),
            "mean_valid_macro_f1": float(arr.mean()),
            "std_valid_macro_f1": float(arr.std()),
            "per_seed_valid_macro_f1": scores,
        })
    leaderboard.sort(key=lambda r: (-r["mean_valid_macro_f1"], r["std_valid_macro_f1"]))
    winner = leaderboard[0]["model"] if leaderboard else None
    return {
        "selection_metric": "mean_validation_macro_f1",
        "test_used": False,
        "winner": winner,
        "leaderboard": leaderboard,
    }


def run_deep_comparison(
    dataset: str, output: str, models=None, seeds=DEFAULT_SEEDS,
    batch_size: int = 4, image_size: int = 224, device: str = "auto",
    epochs: int = 30, grad_accum_steps: int = 1, calibration: str | None = None,
    trainer=None,
) -> dict:
    """Train each (model, seed), aggregate, select winner on validation.

    `trainer` defaults to deep.trainers.train_image_model; inject a stub for
    CPU tests. Each trainer call must return a dict containing
    'best_valid_macro_f1' (and ideally 'history' + 'checkpoint')."""
    models = models or DEFAULT_MODELS
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    if trainer is None:  # pragma: no cover - real training needs torch + data
        from .trainers import train_image_model as trainer

    runs, run_records = [], []
    for model in models:
        for seed in seeds:
            run_out = out / "runs" / f"{model}_seed_{seed}"
            result = trainer(
                dataset, model, str(run_out), seed=seed, epochs=epochs,
                batch_size=batch_size, image_size=image_size, device=device,
                grad_accum_steps=grad_accum_steps, calibration=calibration,
            )
            vmf1 = float(result.get("best_valid_macro_f1", 0.0))
            runs.append({"model": model, "seed": seed, "valid_macro_f1": vmf1})
            run_records.append({"model": model, "seed": seed,
                                "best_valid_macro_f1": vmf1,
                                "checkpoint": result.get("checkpoint"),
                                "history": result.get("history", [])})

    summary = aggregate_and_select(runs)
    summary["dataset"] = dataset
    summary["seeds"] = list(seeds)
    summary["safe_defaults"] = {"batch_size": batch_size, "grad_accum_steps": grad_accum_steps,
                                "mixed_precision": True, "device": device}
    summary["runs"] = run_records

    (out / "deep_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_curves_and_bars(out, run_records, summary)
    return summary


def _plot_curves_and_bars(out: Path, run_records, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Training curves per model (valid macro-F1 vs epoch, averaged where possible).
    for rec in run_records:
        hist = rec.get("history") or []
        if not hist:
            continue
        epochs = [h.get("epoch") for h in hist]
        vmf1 = [h.get("valid_macro_f1") for h in hist]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, vmf1, marker="o")
        plt.title(f"{rec['model']} seed {rec['seed']} — valid macro-F1")
        plt.xlabel("epoch")
        plt.ylabel("valid macro-F1")
        plt.grid(alpha=.3)
        for ext in ("png", "svg"):
            plt.savefig(fig_dir / f"curve_{rec['model']}_seed_{rec['seed']}.{ext}",
                        dpi=140, bbox_inches="tight")
        plt.close()

    lb = summary["leaderboard"]
    if lb:
        plt.figure(figsize=(8, 5))
        names = [r["model"] for r in lb]
        means = [r["mean_valid_macro_f1"] for r in lb]
        stds = [r["std_valid_macro_f1"] for r in lb]
        colors = ["#27ae60" if n == summary["winner"] else "#3498db" for n in names]
        plt.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="white")
        plt.title("Deep model comparison — mean ± std validation macro-F1\n"
                  f"(winner: {summary['winner']}; selected on validation)")
        plt.ylabel("validation macro-F1")
        plt.ylim(0, 1)
        plt.xticks(rotation=20, ha="right")
        for ext in ("png", "svg"):
            plt.savefig(fig_dir / f"deep_comparison.{ext}", dpi=150, bbox_inches="tight")
        plt.close()
