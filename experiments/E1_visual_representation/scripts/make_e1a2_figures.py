#!/usr/bin/env python
"""Builds E1A_F1-F6 from the E1A_T2/T3/T4 tables (must run after
build_e1a2_tables.py). PNG + PDF + source CSV for each. Skips gracefully
(prints why) if a required table is absent -- never fabricates data."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parents[1]


def _read_csv(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def f1_macro_f1_comparison(t2, figures_dir: Path):
    ok_rows = sorted((r for r in t2 if r["status"] == "ok"), key=lambda r: -float(r["val_macro_f1"]))
    if not ok_rows:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [r["model"] for r in ok_rows]
    values = [float(r["val_macro_f1"]) for r in ok_rows]
    ax.barh(models, values, color="#4C72B0")
    ax.set_xlabel("Validation Macro-F1")
    ax.set_title("E1-A2 Standardized Linear-Probe Screening: Macro-F1 by model")
    ax.invert_yaxis()
    _save(fig, figures_dir / "E1A_F1_macro_f1_comparison")
    with open(figures_dir / "E1A_F1_macro_f1_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "val_macro_f1"])
        w.writerows(zip(models, values))


def f2_per_class_f1(t3, figures_dir: Path):
    if not t3:
        return
    models = sorted({r["model"] for r in t3})
    classes = ("Angry", "Fear", "Happy", "Sad")
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.2
    import numpy as np
    x = np.arange(len(models))
    for i, cls in enumerate(classes):
        values = [next((float(r["f1"]) for r in t3 if r["model"] == m and r["class"] == cls), 0) for m in models]
        ax.bar(x + i * width, values, width, label=cls)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 by model")
    ax.legend()
    _save(fig, figures_dir / "E1A_F2_per_class_f1")


def f3_macro_f1_vs_latency(t2, figures_dir: Path):
    ok_rows = [r for r in t2 if r["status"] == "ok" and r.get("latency")]
    if not ok_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in ok_rows:
        ax.scatter(float(r["latency"]) * 1000, float(r["val_macro_f1"]), s=80)
        ax.annotate(r["model"], (float(r["latency"]) * 1000, float(r["val_macro_f1"])), fontsize=8)
    ax.set_xlabel("End-to-end latency (ms/image)")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Macro-F1 vs. latency trade-off")
    _save(fig, figures_dir / "E1A_F3_macro_f1_vs_latency")


def f4_learning_curves(t4, figures_dir: Path):
    if not t4:
        return
    models = sorted({r["model"] for r in t4})
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        rows = sorted((r for r in t4 if r["model"] == m), key=lambda r: int(r["epoch"]))
        epochs = [int(r["epoch"]) for r in rows]
        f1s = [float(r["val_macro_f1"]) for r in rows]
        ax.plot(epochs, f1s, marker="o", markersize=3, label=m)
        best = next((r for r in rows if r["is_best_epoch"] in ("True", True)), None)
        if best:
            ax.scatter([int(best["epoch"])], [float(best["val_macro_f1"])], marker="*", s=150, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Learning curves (star = best epoch)")
    ax.legend(fontsize=7)
    _save(fig, figures_dir / "E1A_F4_learning_curves")


def f5_train_vs_val_loss(t4, figures_dir: Path):
    if not t4:
        return
    models = sorted({r["model"] for r in t4})
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        rows = sorted((r for r in t4 if r["model"] == m), key=lambda r: int(r["epoch"]))
        epochs = [int(r["epoch"]) for r in rows]
        ax.plot(epochs, [float(r["train_loss"]) for r in rows], label="train")
        ax.plot(epochs, [float(r["val_loss"]) for r in rows], label="val")
        ax.set_title(m, fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle("Train vs. validation loss (convergence/overfitting check)")
    _save(fig, figures_dir / "E1A_F5_train_vs_validation_loss")


def f6_best_epoch_by_model(t2, figures_dir: Path):
    ok_rows = [r for r in t2 if r["status"] == "ok" and r.get("best_epoch") is not None]
    if not ok_rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    models = [r["model"] for r in ok_rows]
    best_epochs = [int(r["best_epoch"]) for r in ok_rows]
    ax.bar(models, best_epochs, color="#55A868")
    ax.set_ylabel("Best epoch")
    ax.set_title("Best (early-stopping-selected) epoch by model")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    _save(fig, figures_dir / "E1A_F6_best_epoch_by_model")


def main(output_root=None, exp_dir=None):
    exp_dir = Path(exp_dir) if exp_dir else EXP_DIR
    tables_dir = exp_dir / "tables"
    figures_dir = exp_dir / "figures"
    t2 = _read_csv(tables_dir / "E1A_T2_standardized_linear_probe.csv")
    t3 = _read_csv(tables_dir / "E1A_T3_per_class_metrics.csv")
    t4 = _read_csv(tables_dir / "E1A_T4_epoch_history.csv")
    if t2 is None:
        print("E1A_T2 not found -- no E1-A2 GPU results yet, skipping figure generation.")
        return
    f1_macro_f1_comparison(t2, figures_dir)
    f2_per_class_f1(t3, figures_dir)
    f3_macro_f1_vs_latency(t2, figures_dir)
    f4_learning_curves(t4, figures_dir)
    f5_train_vs_val_loss(t4, figures_dir)
    f6_best_epoch_by_model(t2, figures_dir)
    print("Figure generation complete.")


if __name__ == "__main__":
    output_root = sys.argv[1] if len(sys.argv) > 1 else None
    main(output_root)
