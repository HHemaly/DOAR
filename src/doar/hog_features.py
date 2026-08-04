"""
hog_features.py -- dependency-light HOG (Histogram of Oriented Gradients)
feature extraction for Experiment B (RQ2: do HOG, colour and geometry
features improve on the objective-feature baseline?).

Neither opencv (this environment's cv2 5.0.0 build ships without
`HOGDescriptor`, confirmed empirically) nor scikit-image (not installed)
provides HOG here, so this is a from-scratch, pure-numpy implementation of
the standard Dalal-Triggs HOG descriptor -- matching the project's existing
convention of dependency-light, hand-rolled implementations (see
`dataset.py::_average_hash`/`_difference_hash`, `analysis.py`'s
segmentation). Parameters are FIXED a priori (never searched), per
PHASE7_EXPERIMENT_MATRIX.csv row C4's own stated rationale: "to avoid a
hidden validation-leakage path."

Fixed configuration: 64x64 grayscale, 8x8-pixel cells, 9 unsigned
orientation bins (0-180 degrees), 2x2-cell blocks with 1-cell stride,
L2 block normalization -- yields exactly 49 blocks x 36 = 1764 dims.
"""
from __future__ import annotations

import csv
import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .features import FeatureValue

HOG_VERSION = "hog_v1"
_IMAGE_SIZE = 64
_CELL = 8
_BLOCK_CELLS = 2
_NBINS = 9
_CELLS_PER_SIDE = _IMAGE_SIZE // _CELL           # 8
_BLOCKS_PER_SIDE = _CELLS_PER_SIDE - _BLOCK_CELLS + 1  # 7
DESCRIPTOR_LENGTH = _BLOCKS_PER_SIDE * _BLOCKS_PER_SIDE * _BLOCK_CELLS * _BLOCK_CELLS * _NBINS  # 1764


def _cell_histogram(magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """9-bin unsigned-orientation histogram for one cell, magnitude-weighted."""
    bin_width = 180.0 / _NBINS
    bins = np.minimum((orientation / bin_width).astype(np.int64), _NBINS - 1)
    hist = np.zeros(_NBINS, dtype=np.float64)
    for b in range(_NBINS):
        hist[b] = magnitude[bins == b].sum()
    return hist


def compute_hog_descriptor(gray: np.ndarray) -> np.ndarray:
    """gray: 2D float array, exactly (_IMAGE_SIZE, _IMAGE_SIZE). Returns a
    1764-dim, L2-block-normalized HOG descriptor (deterministic, no RNG)."""
    if gray.shape != (_IMAGE_SIZE, _IMAGE_SIZE):
        raise ValueError(f"Expected a {_IMAGE_SIZE}x{_IMAGE_SIZE} array, got {gray.shape}")
    gy, gx = np.gradient(gray.astype(np.float64))
    magnitude = np.hypot(gx, gy)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0  # unsigned: fold to [0, 180)

    cell_hist = np.zeros((_CELLS_PER_SIDE, _CELLS_PER_SIDE, _NBINS))
    for cy in range(_CELLS_PER_SIDE):
        for cx in range(_CELLS_PER_SIDE):
            sl = (slice(cy * _CELL, (cy + 1) * _CELL), slice(cx * _CELL, (cx + 1) * _CELL))
            cell_hist[cy, cx] = _cell_histogram(magnitude[sl], orientation[sl])

    blocks = []
    for by in range(_BLOCKS_PER_SIDE):
        for bx in range(_BLOCKS_PER_SIDE):
            block = cell_hist[by:by + _BLOCK_CELLS, bx:bx + _BLOCK_CELLS].flatten()
            norm = np.sqrt(float((block ** 2).sum()) + 1e-6 ** 2)
            blocks.append(block / norm)
    descriptor = np.concatenate(blocks)
    assert descriptor.shape == (DESCRIPTOR_LENGTH,), descriptor.shape
    return descriptor


def hog_feature_row(image_path: str | Path) -> dict[str, FeatureValue]:
    """Real, image-derived HOG features only -- never label-derived. Confidence
    is 1.0 (deterministic given the fixed pixel grid, unlike the segmentation-
    dependent objective features in features.py)."""
    image = Image.open(image_path).convert("L").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.BILINEAR)
    gray = np.asarray(image, dtype=np.float64)
    descriptor = compute_hog_descriptor(gray)
    return {
        f"hog.bin_{i:04d}": FeatureValue(
            value=float(descriptor[i]), valid_min=0.0, valid_max=1.0, confidence=1.0,
            method=HOG_VERSION, evidence_id=f"ev_hog_bin_{i:04d}", missing=False,
            version=HOG_VERSION,
        )
        for i in range(DESCRIPTOR_LENGTH)
    }


def serialize_hog_row(row: dict[str, FeatureValue]) -> dict:
    return {name: asdict(value) for name, value in row.items()}


def extract_hog_features(manifest: str | Path, output: str | Path) -> dict:
    """Mirrors extract.py::extract_features's shape (manifest in, features.csv
    out) but for HOG only -- kept as a separate CSV/command so Experiment A's
    objective-feature extraction is never coupled to Experiment B's HOG
    extraction; handcrafted_comparison.py joins the two by image_id."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(manifest, newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    flat_rows, failures = [], []
    for record in manifest_rows:
        if record.get("readable", "").lower() not in ("true", "1"):
            failures.append({"image_id": record["image_id"], "path": record["path"],
                             "error": "manifest_unreadable"})
            continue
        try:
            row = hog_feature_row(record["path"])
            flat_rows.append({
                "image_id": record["image_id"], "path": record["path"],
                "split": record["split"], "class": record["class"],
                **{name: item.value for name, item in row.items()},
            })
        except Exception as exc:
            failures.append({"image_id": record.get("image_id"), "path": record.get("path"),
                             "error": str(exc), "traceback": traceback.format_exc()})
    fields = ["image_id", "path", "split", "class"] + [f"hog.bin_{i:04d}" for i in range(DESCRIPTOR_LENGTH)]
    with (output / "hog_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    with (output / "hog_failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "path", "error", "traceback"])
        writer.writeheader()
        writer.writerows(failures)
    summary = {
        "manifest": str(Path(manifest).resolve()), "processed": len(flat_rows),
        "failures": len(failures), "descriptor_length": DESCRIPTOR_LENGTH,
        "hog_version": HOG_VERSION,
        "splits": sorted({row["split"] for row in flat_rows}),
    }
    (output / "hog_extraction_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary
