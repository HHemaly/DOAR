#!/usr/bin/env python
"""Phase 2C.4: run one open-vocabulary detector against all 80 private
blinded pilot images, writing raw per-class presence/score predictions.
Mirrors scripts/phase2b_run_baseline.py's structure exactly. Writes to a
private, gitignored staging location under outputs/ -- never directly to
artifacts/ -- so the caller can inspect/verify before promoting a copy to
the committed artifacts/phase2c4/ directory (same two-step discipline
already used for outputs/phase2c2/raw_predictions_80_private.csv ->
artifacts/phase2c2/raw_predictions_80.csv).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.ontology import CLASS_NAMES  # noqa: E402
from doar.phase2c4 import detectors as det_mod  # noqa: E402

LOADERS = {
    "owlv2": det_mod.load_real_owlv2,
    "grounding_dino": det_mod.load_real_grounding_dino,
    "florence2": det_mod.load_real_florence2,
    "yolo_world": det_mod.load_real_yolo_world,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(LOADERS))
    parser.add_argument("--images-dir", type=Path, default=ROOT / "outputs/phase2c1/private_images")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (ROOT / f"outputs/phase2c4/raw_predictions_{args.model}_private.csv")

    t0 = time.time()
    detector = LOADERS[args.model]()
    load_time = time.time() - t0
    print(f"{args.model} loaded in {load_time:.1f}s", flush=True)

    image_paths = sorted(args.images_dir.glob("p2b_*"))
    rows = []
    t0 = time.time()
    for i, path in enumerate(image_paths):
        pilot_id = path.stem
        img_t0 = time.time()
        predictions = detector.predict_image(str(path))
        row = {"pilot_id": pilot_id}
        for cls in CLASS_NAMES:
            detected, score = predictions[cls]
            row[f"{cls}__detected"] = detected
            row[f"{cls}__score"] = round(score, 4)
        rows.append(row)
        elapsed_img = time.time() - img_t0
        print(f"[{i + 1}/{len(image_paths)}] {pilot_id} took {elapsed_img:.1f}s", flush=True)
    total_time = time.time() - t0
    print(f"inference over {len(rows)} images took {total_time:.1f}s "
          f"({total_time / len(rows):.2f}s/image mean)", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    print(f"SUMMARY model={args.model} load_time_s={load_time:.1f} "
          f"total_inference_s={total_time:.1f} mean_s_per_image={total_time / len(rows):.2f}")


if __name__ == "__main__":
    main()
