#!/usr/bin/env python
"""Phase 2B: run the two chosen baselines (CLIP zero-shot, classical-CV
circularity) against a directory of blinded pilot images, writing raw
predictions to artifacts/phase2b/raw_predictions.csv.

Requires network access on first run (or a pre-populated open_clip cache
directory) to download the CLIP checkpoint -- see
docs/PHASE2B_MODEL_SELECTION.md for the real download-speed finding
(~600MB, ~3.5 minutes at ~2.9 MB/s in this environment). This script is
never invoked by the automated test suite (see tests/test_phase2b_inference.py,
which uses an injected synthetic backend instead).
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.phase2b.inference import ZeroShotClipDetector, detect_circles_classical  # noqa: E402
from doar.phase2b.ontology import CLASS_NAMES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="directory of blinded p2b_XXXX.<ext> images "
                             "(output of scripts/phase2b_audit_dataset.py)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="open_clip weight cache directory")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "phase2b" / "raw_predictions.csv")
    args = parser.parse_args()

    t0 = time.time()
    detector = ZeroShotClipDetector.load_real(cache_dir=str(args.cache_dir) if args.cache_dir else None)
    print(f"model loaded in {time.time() - t0:.1f}s")

    image_paths = sorted(glob.glob(str(args.images_dir / "p2b_*")))
    results = []
    t0 = time.time()
    for path in image_paths:
        pilot_id = Path(path).stem
        row = {"pilot_id": pilot_id}
        for pred in detector.predict_image(path):
            row[f"{pred.class_name}__status"] = pred.status
            row[f"{pred.class_name}__similarity"] = round(pred.similarity, 4)
        circle = detect_circles_classical(path)
        row["circle_classical__detected"] = circle.detected
        row["circle_classical__circularity"] = circle.circularity
        row["circle_classical__count"] = circle.count
        results.append(row)
    print(f"inference over {len(results)} images took {time.time() - t0:.2f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} rows to {args.output}")
    assert set(CLASS_NAMES).issubset({k.split("__")[0] for k in results[0]})


if __name__ == "__main__":
    main()
