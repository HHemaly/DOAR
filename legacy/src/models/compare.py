"""
compare.py — train a baseline + two transfer models on the SAME leak-safe split,
select the winner on VALIDATION performance, then evaluate the winner ONCE on the
untouched test set.

Model selection rule (scientifically important):
    * All models train on the identical split.csv (no re-splitting).
    * The final model is chosen by best VALIDATION accuracy — never test results.
    * Only the chosen model is evaluated on the test set, exactly once.

Default line-up (chosen for this dataset + Colab runtime + explainability):
    baseline     : small CNN (sanity floor, fast)
    mobilenet    : MobileNetV3-Small — light, fast on Colab, Grad-CAM friendly
    resnet18     : ResNet18 — strong, well-understood, Grad-CAM friendly
(EfficientNet-B0 is available via --models if you want a heavier third transfer.)

Writes:
    <out>/model_comparison/comparison.csv          (per-model val metrics)
    <out>/model_comparison/comparison.json
    <out>/model_comparison/figures/model_comparison.(png|svg)
    <out>/model_comparison/selected_model.json
    <out>/training/<model>/...                      (per-model training artefacts)
    <out>/evaluation/...                            (WINNER's test evaluation)
"""

from __future__ import annotations
import os
import csv
import json


DEFAULT_MODELS = ["baseline", "mobilenet", "resnet18"]


def run_comparison(split_csv: str, out_dir: str, timestamp: str,
                   models=None, epochs: int = 25, batch_size: int = 32,
                   lr: float = 1e-3, seed: int = 42) -> dict:
    from src.models.train import train_model
    from src.models.evaluate import evaluate_model

    models = models or DEFAULT_MODELS
    comp_dir = os.path.join(out_dir, "model_comparison")
    fig_dir = os.path.join(comp_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    results = []
    for name in models:
        print(f"\n{'='*60}\n[compare] Training model: {name}\n{'='*60}")
        # Each model writes into its own subfolder so checkpoints don't clash
        model_out = os.path.join(out_dir, "training", name)
        os.makedirs(model_out, exist_ok=True)
        # train_model writes to <model_out>/training/... — nest under model_out
        res = train_model(split_csv, model_out, timestamp, model_name=name,
                          epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
        history = res["history"]
        best_val_acc = max((h["val_acc"] for h in history), default=0.0)
        best_val_loss = res["best_val_loss"]
        results.append({
            "model": name,
            "best_val_acc": round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_epoch": res["best_epoch"],
            "checkpoint": res["best_model"],
        })

    # ── Select winner on VALIDATION accuracy (never test) ────────
    winner = max(results, key=lambda r: r["best_val_acc"])
    print(f"\n[compare] Selected on validation: {winner['model']} "
          f"(val_acc={winner['best_val_acc']})")

    # Write comparison table
    with open(os.path.join(comp_dir, "comparison.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "best_val_acc", "best_val_loss",
                                          "best_epoch", "checkpoint"])
        w.writeheader(); w.writerows(results)
    with open(os.path.join(comp_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"models": results, "selection_metric": "best_val_acc",
                   "winner": winner["model"]}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(comp_dir, "selected_model.json"), "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2, ensure_ascii=False)

    _plot_comparison(fig_dir, results, winner["model"])

    # ── Evaluate ONLY the winner on the untouched test set ───────
    print(f"\n[compare] Evaluating winner '{winner['model']}' on TEST set (once)...")
    metrics = evaluate_model(split_csv, winner["checkpoint"], out_dir, timestamp,
                             batch_size=batch_size)

    return {"comparison": results, "winner": winner, "test_metrics": metrics}


def _plot_comparison(fig_dir, results, winner_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [r["model"] for r in results]
    accs = [r["best_val_acc"] for r in results]
    colors = ["#27ae60" if n == winner_name else "#3498db" for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_title("Model comparison — best validation accuracy\n"
                 f"(winner: {winner_name}; selected on validation, not test)",
                 fontweight="bold")
    ax.set_ylabel("Best validation accuracy"); ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"model_comparison.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
