#!/usr/bin/env python
"""E1-A2 one-command runner -- portable, resumable, GPU-ready.

python run_e1a_all.py \
  --dataset-root <path> --output-root <path> --device cuda \
  --seed 42 --max-epochs 30 --patience 5 --resume

Never touches Test performance. Continues past a failed OPTIONAL
candidate rather than aborting the whole run. Writes RUN_MANIFEST.json
recording completion status for every model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1a2_common as ec  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]

DEFAULTS = dict(head_lr=3e-4, weight_decay=1e-4, batch_size=16)


def load_run_manifest(output_root: Path) -> dict:
    path = Path(output_root) / "RUN_MANIFEST.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": "e1a2_run_manifest_v1", "models": {}}


def save_run_manifest(output_root: Path, manifest: dict) -> None:
    path = Path(output_root) / "RUN_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def measure_classifier_only_latency(model, device: str, embed_dim: int, n: int = 50) -> float:
    import torch
    model.eval()
    x = torch.randn(1, embed_dim, device=device)
    with torch.no_grad():
        model(x)  # warm-up
        started = time.perf_counter()
        for _ in range(n):
            model(x)
        return (time.perf_counter() - started) / n


def run_one_candidate(
    model_key: str, split_rows: dict, manifest_sha: str, output_root: Path, device: str,
    max_epochs: int, patience: int, resume: bool,
) -> dict:
    import numpy as np
    import torch

    class_to_idx = {c: i for i, c in enumerate(ec.CLASSES)}
    train_rows, valid_rows = split_rows["train"], split_rows["valid"]

    t_extract0 = time.perf_counter()
    train_cache = ec.extract_and_cache_embeddings(model_key, train_rows, "train", manifest_sha, output_root, device)
    valid_cache = ec.extract_and_cache_embeddings(model_key, valid_rows, "valid", manifest_sha, output_root, device)
    extraction_seconds = time.perf_counter() - t_extract0

    x_train = train_cache["embeddings"]
    y_train = np.asarray([class_to_idx[str(c)] for c in train_cache["labels"]])
    x_valid = valid_cache["embeddings"]
    y_valid = np.asarray([class_to_idx[str(c)] for c in valid_cache["labels"]])
    valid_ids = valid_cache["image_ids"]
    class_weights = ec.compute_class_weights(y_train.tolist())

    ec.set_seed(ec.SEED)
    out_dir = Path(output_root) / "checkpoints" / model_key
    started = time.perf_counter()
    history, best_epoch, best_f1 = ec.run_linear_probe(
        x_train, y_train, x_valid, y_valid, valid_ids, device=device, max_epochs=max_epochs, patience=patience,
        head_lr=DEFAULTS["head_lr"], weight_decay=DEFAULTS["weight_decay"], class_weights=class_weights,
        output_dir=out_dir, model_key=model_key, resume=resume,
    )
    training_seconds = time.perf_counter() - started

    encoder_params = ec.encoder_parameter_count(model_key, device)
    head = torch.nn.Linear(x_train.shape[1], len(ec.CLASSES)).to(device)
    head_params = sum(p.numel() for p in head.parameters())
    classifier_only_latency_s = measure_classifier_only_latency(head, device, x_train.shape[1])

    embed_dim = train_cache["embedding_dimension"]
    embed_cache_bytes = (Path(output_root) / "embeddings" / model_key / "train.npz").stat().st_size + \
        (Path(output_root) / "embeddings" / model_key / "valid.npz").stat().st_size
    checkpoint_size_bytes = (out_dir / "best.pt").stat().st_size if (out_dir / "best.pt").exists() else None

    peak_vram_mb = None
    if device.startswith("cuda"):
        import torch as _torch
        peak_vram_mb = _torch.cuda.max_memory_allocated() / 1e6

    result = {
        "model_key": model_key, "status": "ok", "seed": ec.SEED,
        "best_epoch": best_epoch, "epochs_completed": len(history), "best_valid_macro_f1": best_f1,
        "max_epochs": max_epochs, "patience": patience,
        "head_learning_rate": DEFAULTS["head_lr"], "weight_decay": DEFAULTS["weight_decay"],
        "augmentation": "none_standardized_linear_probe",
        "device": device, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resource_accounting": {
            "encoder_full_parameter_count": encoder_params,
            "trainable_head_parameter_count": head_params,
            "embedding_dimension": embed_dim,
            "pretrained_weights_id": train_cache["weights_id"],
            "one_time_embedding_extraction_seconds_total": extraction_seconds,
            "encoder_inference_latency_seconds_per_image": train_cache.get("encoder_latency_seconds_per_image_mean"),
            "classifier_only_latency_seconds_per_image": classifier_only_latency_s,
            "end_to_end_latency_seconds_per_image": (
                (train_cache.get("encoder_latency_seconds_per_image_mean") or 0) + classifier_only_latency_s),
            "peak_gpu_vram_mb": peak_vram_mb,
            "embedding_cache_bytes": embed_cache_bytes,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "linear_probe_training_seconds": training_seconds,
        },
        "preprocessing_spec": train_cache["preprocessing_spec"],
    }
    (Path(output_root) / "raw" / f"{model_key}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (Path(output_root) / "raw" / f"{model_key}_epoch_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", default=None, help="Comma-separated subset (default: all candidates)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny subset (16 train / 8 valid), 2 epochs -- pipeline validation only, "
                             "never used for the real screening result.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    (output_root / "raw").mkdir(parents=True, exist_ok=True)
    (output_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (output_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    print("=== 1. Verify T0 ===")
    split_rows = ec.load_split_rows(args.dataset_root)
    manifest_sha = split_rows["_manifest_sha256"]
    for split in ("train", "valid"):
        missing = [r for r in split_rows[split] if not Path(r["path"]).exists()]
        if missing:
            print(f"STOP: {len(missing)} {split} paths do not resolve. Exact missing dataset root: {args.dataset_root}",
                  file=sys.stderr)
            sys.exit(1)
    print(f"OK -- manifest SHA256={manifest_sha}, train={len(split_rows['train'])}, "
          f"valid={len(split_rows['valid'])}, test={len(split_rows['test'])} (locked, unused)")

    print("=== 2. Detect hardware ===")
    device = ec.resolve_device(args.device)
    hw = ec.hardware_report(device)
    print(json.dumps(hw, indent=2))

    models = args.models.split(",") if args.models else list(ec.ALL_CANDIDATES)
    max_epochs = 2 if args.smoke else args.max_epochs
    train_rows = split_rows["train"][:16] if args.smoke else split_rows["train"]
    valid_rows = split_rows["valid"][:8] if args.smoke else split_rows["valid"]
    run_split_rows = {"train": train_rows, "valid": valid_rows}

    manifest = load_run_manifest(output_root)
    manifest["hardware"] = hw
    manifest["dataset_root"] = str(args.dataset_root)
    manifest["manifest_sha256"] = manifest_sha
    manifest["smoke_mode"] = args.smoke

    print(f"=== 3-8. Run {len(models)} candidates sequentially ===")
    for model_key in models:
        entry = manifest["models"].get(model_key, {})
        config_sig = f"{max_epochs}_{args.patience}_{args.seed}_{manifest_sha}"
        if args.resume and entry.get("status") == "trained" and entry.get("config_sig") == config_sig:
            print(f"[{model_key}] SKIP -- already trained under this exact configuration (--resume).")
            continue
        print(f"[{model_key}] running...")
        try:
            result = run_one_candidate(
                model_key, run_split_rows, manifest_sha, output_root, device, max_epochs, args.patience,
                resume=args.resume,
            )
            manifest["models"][model_key] = {
                "status": "trained", "config_sig": config_sig,
                "best_valid_macro_f1": result["best_valid_macro_f1"], "best_epoch": result["best_epoch"],
                "timestamp": result["timestamp"],
            }
            print(f"[{model_key}] OK -- best_epoch={result['best_epoch']} macro_f1={result['best_valid_macro_f1']:.4f}")
        except Exception as exc:
            manifest["models"][model_key] = {
                "status": "failed", "error": str(exc), "traceback": traceback.format_exc(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            print(f"[{model_key}] FAILED -- {exc} -- continuing with remaining candidates.", file=sys.stderr)
        save_run_manifest(output_root, manifest)

    print("=== 9-10. Tables and figures ===")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import build_e1a2_tables
        build_e1a2_tables.main(output_root, EXP_DIR)
        import make_e1a2_figures
        make_e1a2_figures.main(output_root, EXP_DIR)
    except Exception as exc:
        print(f"Table/figure generation failed (non-fatal, raw results are still saved): {exc}", file=sys.stderr)

    print("=== DONE. Test set was never accessed for labels/predictions. ===")
    save_run_manifest(output_root, manifest)


if __name__ == "__main__":
    main()
