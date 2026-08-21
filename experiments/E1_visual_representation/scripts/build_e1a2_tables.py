#!/usr/bin/env python
"""Builds all E1A_T1-T6 tables. T1 comes from the preserved E1-A1 CPU
development results (experiments/E1_visual_representation/raw/*_result.json,
the OLD e1a_common.py schema). T2-T6 come from the E1-A2 standardized
linear-probe run (output_root/raw/*_result.json, the NEW e1a2_common.py
schema) -- these are two different schemas, read separately, never mixed
into one row. Small CSV outputs only (Section 18) -- written to the repo's
own tables/ directory so they can be committed; never writes checkpoints/
embeddings anywhere."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]

CNN_VIT_KEYS = ("mobilenet_v3_small", "resnet18", "efficientnet_b0", "densenet121", "convnext_tiny", "vit_b_16")
FOUNDATION_KEYS = ("clip_vit_b32", "dinov2_vits14", "siglip2_vit_b16")
ALL_KEYS = CNN_VIT_KEYS + FOUNDATION_KEYS


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_t1(exp_dir: Path) -> None:
    """E1-A1 (CPU development, preserved, secondary) -- from the OLD
    e1a_common.py-produced raw/*_result.json files."""
    rows = []
    for key in ALL_KEYS:
        result_path = exp_dir / "raw" / f"{key}_result.json"
        smoke_path = exp_dir / "raw" / f"{key}_smoke.json"
        if result_path.exists():
            d = json.loads(result_path.read_text(encoding="utf-8"))
            meta = d.get("meta", {})
            rows.append({
                "model": key, "experiment": "E1-A1_transfer_head_dev_screening", "status": "complete",
                "best_epoch": d.get("best_epoch"), "val_macro_f1": round(d.get("best_valid_macro_f1", 0), 4),
                "params": meta.get("n_params"), "trainable_params": meta.get("n_trainable"),
                "training_seconds": round(d.get("training_seconds", 0), 1),
                "latency_ms": d.get("latency_ms_per_image"), "peak_vram": "n/a_cpu_only",
            })
        elif smoke_path.exists():
            rows.append({
                "model": key, "experiment": "E1-A1_transfer_head_dev_screening", "status": "unfinished_or_omitted",
                "best_epoch": None, "val_macro_f1": None, "params": None, "trainable_params": None,
                "training_seconds": None, "latency_ms": None, "peak_vram": None,
            })
    if not rows:
        return
    _write_csv(exp_dir / "tables" / "E1A_T1_transfer_head_screening.csv", rows, list(rows[0].keys()))
    print(f"Wrote E1A_T1 ({len(rows)} rows)")


def build_t2_t6(output_root: Path, exp_dir: Path) -> None:
    manifest_path = Path(output_root) / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        print("No RUN_MANIFEST.json found under output_root -- skipping T2-T6 (no E1-A2 GPU results yet).")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    t2_rows, t3_rows, t4_rows, t5_rows, t6_rows = [], [], [], [], []
    for key, entry in manifest.get("models", {}).items():
        if entry.get("status") != "trained":
            t2_rows.append({"model": key, "status": entry.get("status", "unknown"),
                            "reason": entry.get("error", ""), "best_epoch": None, "val_macro_f1": None,
                            "balanced_accuracy": None, "accuracy": None, "params": None,
                            "trainable_params": None, "training_time": None, "latency": None,
                            "peak_vram": None})
            continue
        result_path = Path(output_root) / "raw" / f"{key}_result.json"
        history_path = Path(output_root) / "raw" / f"{key}_epoch_history.json"
        if not result_path.exists():
            continue
        d = json.loads(result_path.read_text(encoding="utf-8"))
        ra = d["resource_accounting"]
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        best_rec = next((h for h in history if h["epoch"] == d["best_epoch"]), history[-1] if history else {})

        t2_rows.append({
            "model": key, "status": "ok", "best_epoch": d["best_epoch"],
            "val_macro_f1": round(d["best_valid_macro_f1"], 4),
            "balanced_accuracy": round(best_rec.get("val_balanced_accuracy", 0), 4),
            "accuracy": round(best_rec.get("val_accuracy", 0), 4),
            "params": ra["encoder_full_parameter_count"], "trainable_params": ra["trainable_head_parameter_count"],
            "training_time": round(ra["linear_probe_training_seconds"], 2),
            "latency": ra["end_to_end_latency_seconds_per_image"], "peak_vram": ra["peak_gpu_vram_mb"],
        })

        # T3: per-class metrics from best_valid_predictions.csv
        pred_path = Path(output_root) / "checkpoints" / key / "best_valid_predictions.csv"
        if pred_path.exists():
            from collections import defaultdict
            tp, fp, fn, support = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
            with open(pred_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    t, p = row["true_class"], row["predicted_class"]
                    support[t] += 1
                    if t == p:
                        tp[t] += 1
                    else:
                        fn[t] += 1
                        fp[p] += 1
            for cls in ("Angry", "Fear", "Happy", "Sad"):
                precision = tp[cls] / max(1, tp[cls] + fp[cls])
                recall = tp[cls] / max(1, tp[cls] + fn[cls])
                f1 = 2 * precision * recall / max(1e-12, precision + recall)
                t3_rows.append({"model": key, "class": cls, "precision": round(precision, 4),
                                "recall": round(recall, 4), "f1": round(f1, 4), "support": support[cls]})

        # T4: full epoch history
        for h in history:
            t4_rows.append({
                "model": key, "epoch": h["epoch"], "train_loss": round(h["train_loss"], 4),
                "val_loss": round(h["val_loss"], 4), "train_accuracy": round(h["train_accuracy"], 4),
                "val_accuracy": round(h["val_accuracy"], 4),
                "val_balanced_accuracy": round(h["val_balanced_accuracy"], 4),
                "val_macro_f1": round(h["val_macro_f1"], 4), "learning_rate": h["learning_rate"],
                "runtime_seconds": round(h["runtime_seconds"], 3), "is_best_epoch": h["is_best_epoch"],
            })

        # T5: epoch sensitivity -- 5/10/20/30/best, never fabricated if absent
        by_epoch = {h["epoch"]: h for h in history}
        row = {"model": key}
        for target in (4, 9, 19, 29):  # 0-indexed epoch numbers for "epoch 5/10/20/30"
            label = {4: "epoch_5", 9: "epoch_10", 19: "epoch_20", 29: "epoch_30"}[target]
            row[f"{label}_val_macro_f1"] = round(by_epoch[target]["val_macro_f1"], 4) if target in by_epoch else None
        row["best_epoch"] = d["best_epoch"]
        row["best_epoch_val_macro_f1"] = round(d["best_valid_macro_f1"], 4)
        t5_rows.append(row)

        # T6: resource comparison
        t6_rows.append({
            "model": key, "params": ra["encoder_full_parameter_count"],
            "trainable_params": ra["trainable_head_parameter_count"],
            "training_seconds": round(ra["linear_probe_training_seconds"], 2),
            "seconds_per_epoch": round(ra["linear_probe_training_seconds"] / max(1, d["epochs_completed"]), 4),
            "latency_ms": round((ra["end_to_end_latency_seconds_per_image"] or 0) * 1000, 4),
            "throughput_img_per_sec": round(1 / ra["end_to_end_latency_seconds_per_image"], 2)
                if ra["end_to_end_latency_seconds_per_image"] else None,
            "peak_vram_mb": ra["peak_gpu_vram_mb"],
            "model_size_mb": round(ra["checkpoint_size_bytes"] / 1e6, 4) if ra["checkpoint_size_bytes"] else None,
            "embedding_cache_mb": round(ra["embedding_cache_bytes"] / 1e6, 4),
        })

    if t2_rows:
        _write_csv(exp_dir / "tables" / "E1A_T2_standardized_linear_probe.csv", t2_rows, list(t2_rows[0].keys()))
        print(f"Wrote E1A_T2 ({len(t2_rows)} rows)")
    if t3_rows:
        _write_csv(exp_dir / "tables" / "E1A_T3_per_class_metrics.csv", t3_rows, list(t3_rows[0].keys()))
        print(f"Wrote E1A_T3 ({len(t3_rows)} rows)")
    if t4_rows:
        _write_csv(exp_dir / "tables" / "E1A_T4_epoch_history.csv", t4_rows, list(t4_rows[0].keys()))
        print(f"Wrote E1A_T4 ({len(t4_rows)} rows)")
    if t5_rows:
        _write_csv(exp_dir / "tables" / "E1A_T5_epoch_sensitivity.csv", t5_rows, list(t5_rows[0].keys()))
        print(f"Wrote E1A_T5 ({len(t5_rows)} rows)")
    if t6_rows:
        _write_csv(exp_dir / "tables" / "E1A_T6_resource_comparison.csv", t6_rows, list(t6_rows[0].keys()))
        print(f"Wrote E1A_T6 ({len(t6_rows)} rows)")


def main(output_root=None, exp_dir=None):
    exp_dir = Path(exp_dir) if exp_dir else EXP_DIR
    build_t1(exp_dir)
    if output_root:
        build_t2_t6(Path(output_root), exp_dir)


if __name__ == "__main__":
    output_root = sys.argv[1] if len(sys.argv) > 1 else None
    main(output_root)
