#!/usr/bin/env python
"""One-off throughput calibration for every CNN/ViT candidate: 200 train +
100 valid images, 2 epochs, steady-state (2nd epoch) seconds/image used to
project full-dataset (2883 images) per-epoch time. Not part of the real
E1-A run -- purely for the throughput/cost stop-rule estimate."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1a_common as ec  # noqa: E402
from run_e1a_candidate import CNN_VIT_MODELS, build_cnn_vit_pipeline  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]


def main():
    rows = ec.load_manifest_rows(EXP_DIR / "raw" / "e1a_train_valid_manifest.csv")
    train_rows = [r for r in rows if r["split"] == "train"][:200]
    valid_rows = [r for r in rows if r["split"] == "valid"][:100]
    results = {}
    for key in CNN_VIT_MODELS:
        try:
            ec.set_seed(42)
            t_build0 = time.perf_counter()
            model, train_loader, valid_loader, train_ds, meta = build_cnn_vit_pipeline(
                CNN_VIT_MODELS[key], train_rows, valid_rows, "cpu")
            build_seconds = time.perf_counter() - t_build0
            cw = ec.compute_class_weights(train_ds.targets)
            history, _, _ = ec.run_epoch_loop(
                model, train_loader, valid_loader, device="cpu", max_epochs=2, patience=5,
                head_lr=3e-4, weight_decay=1e-4, class_weights=cw,
                output_dir=Path("/tmp") / f"calib_{key}", model_key=key)
            steady = history[-1]["runtime_seconds"]
            per_image = steady / (len(train_rows) + len(valid_rows))
            projected_full_epoch = per_image * 2883
            results[key] = {
                "status": "ok", "build_seconds": build_seconds, "steady_epoch_seconds_200img": steady,
                "per_image_seconds": per_image, "projected_full_epoch_seconds": projected_full_epoch,
                "projected_full_epoch_minutes": projected_full_epoch / 60,
                "n_params": meta["n_params"], "n_trainable": meta["n_trainable"],
            }
            print(f"{key}: {projected_full_epoch/60:.1f} min/epoch (projected), "
                  f"{meta['n_params']:,} params")
        except Exception as exc:
            import traceback
            results[key] = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            print(f"{key}: FAILED -- {exc}")
    import json
    (EXP_DIR / "raw" / "throughput_calibration.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
