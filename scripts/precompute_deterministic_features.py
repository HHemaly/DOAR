#!/usr/bin/env python
"""Precomputes and caches deterministic (color/line/composition) features
for the 15-image development set.

Pure local classical-CV computation (`drawing_synthesis.compute_
deterministic_features`, itself reusing `analysis.py::_segment/_
composition/_colour` and `features.py::objective_feature_row` verbatim)
-- no network call, no Gemini, no model weights. Mirrors `run_
development_benchmark.py`'s cache-first pattern for the Gemini live
cache: `scripts/clinician_review_app.py` and `drawing_synthesis.py`
itself both check this cache before computing anything, so running this
script once keeps the app instant and every run byte-reproducible.

Usage:
    python scripts/precompute_deterministic_features.py
    python scripts/precompute_deterministic_features.py --image-id p2b_0003
    python scripts/precompute_deterministic_features.py --force
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar import drawing_synthesis as ds  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_development_benchmark", ROOT / "scripts" / "run_development_benchmark.py")
rdb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute deterministic (color/line/composition) features for the development set.")
    parser.add_argument("--image-id", default=None, help="Restrict to a single image_id.")
    parser.add_argument("--force", action="store_true", help="Recompute even if a cached result already exists.")
    args = parser.parse_args()

    images = rdb.load_development_set()
    if args.image_id:
        images = [im for im in images if im["image_id"] == args.image_id]
        if not images:
            print(f"Unknown image_id: {args.image_id!r}")
            return

    for i, image in enumerate(images, start=1):
        image_id = image["image_id"]
        cache_file = ds.deterministic_cache_path(image_id)
        if cache_file.exists() and not args.force:
            print(f"[{i}/{len(images)}] {image_id}: cached -> {cache_file}")
            continue
        image_path = ROOT / image["relative_path"]
        if args.force and cache_file.exists():
            cache_file.unlink()
        result = ds.load_or_compute_deterministic_features(image_id, image_path)
        print(f"[{i}/{len(images)}] {image_id}: computed {len(result['objective_features'])} features -> {cache_file}")

    print(f"\nDone. Cache directory: {ds.DEFAULT_DETERMINISTIC_CACHE_DIR}")


if __name__ == "__main__":
    main()
