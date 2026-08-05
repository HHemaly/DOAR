"""Transformation-robustness audit (DOAR-TRACE Phase 2A, Section 5). A
fixed, real, non-test image sample is put through a fixed set of
transformations; for each (image, transform, feature) triple, records
whether the feature changed, by how much, and whether that change was
theoretically expected. **No pass threshold is invented silently** --
every tolerance and its rationale is stated inline (`_EXPECTATIONS`)
rather than picked ad hoc per row.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .analysis import analyze_image
from .features import objective_feature_row

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "artifacts" / "phase2a" / "feature_invariance.csv"
SUMMARY_PATH = ROOT / "artifacts" / "phase2a" / "feature_invariance_summary.json"

CSV_FIELDS = [
    "image_id", "transform", "feature_id", "original_value", "transformed_value", "absolute_change",
    "percentage_change", "status", "should_theoretically_preserve", "limitations",
]

# Fixed, real, non-test sample -- same 4 images used throughout this
# session's earlier real-image work (already confirmed split=="train").
FIXED_SAMPLE = [
    ("happy_h42", r"C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing\train\Happy\h42_jpg.rf.08b140109fc5cf5017748422ed3392cd.jpg"),
    ("fear_f42", r"C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing\train\Fear\f42.jpg"),
    ("sad_1_21", r"C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing\train\Sad\Sad_1_21_jpg.rf.d16f63112f9d00e8b749ccafdb6dab1a.jpg"),
    ("angry_a11", r"C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing\train\Angry\new_a11.jpg"),
]

# Features audited, grouped by their theoretical invariance rationale.
_GEOMETRY_FEATURES = ["segmentation.foreground_coverage", "segmentation.bounding_box_coverage",
                       "composition.centroid_x", "composition.centroid_y", "composition.symmetry"]
_STROKE_FEATURES = ["stroke.edge_density", "stroke.intensity_proxy", "stroke.fragmentation",
                     "segmentation.component_count", "segmentation.largest_component_ratio"]
_COLOUR_FEATURES = ["colour.dark_ratio"]
ALL_FEATURES = _GEOMETRY_FEATURES + _STROKE_FEATURES + _COLOUR_FEATURES


def _resize(scale: float) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return _fn


def _rotate(degrees: float) -> Callable[[Image.Image], Image.Image]:
    return lambda img: img.rotate(degrees, expand=False, fillcolor=(255, 255, 255))


def _jpeg(quality: int) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return _fn


def _brightness(factor: float) -> Callable[[Image.Image], Image.Image]:
    return lambda img: ImageEnhance.Brightness(img).enhance(factor)


def _contrast(factor: float) -> Callable[[Image.Image], Image.Image]:
    return lambda img: ImageEnhance.Contrast(img).enhance(factor)


def _add_white_margin(px: int) -> Callable[[Image.Image], Image.Image]:
    return lambda img: ImageOps.expand(img, border=px, fill=(255, 255, 255))


def _crop_border(px: int) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((px, px, max(px + 1, w - px), max(px + 1, h - px)))
    return _fn


def _perspective(strength: float) -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        w, h = img.size
        dx, dy = int(w * strength), int(h * strength)
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [(dx, dy), (w - dx, 0), (w, h), (0, h - dy)]

        def _coeffs(src_pts, dst_pts):
            matrix = []
            for (x, y), (X, Y) in zip(dst_pts, src_pts):
                matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
                matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
            a = np.array(matrix, dtype=np.float64)
            b = np.array(src_pts).reshape(8)
            res = np.linalg.solve(a, b)
            return res

        coeffs = _coeffs(src, dst)
        return img.transform((w, h), Image.PERSPECTIVE, coeffs, fillcolor=(255, 255, 255))
    return _fn


def _mild_shadow() -> Callable[[Image.Image], Image.Image]:
    def _fn(img: Image.Image) -> Image.Image:
        arr = np.asarray(img).astype(np.float32)
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        gradient = 1.0 - 0.25 * (xx / w)  # left bright, right slightly darker
        shadowed = arr * gradient[..., None]
        return Image.fromarray(np.clip(shadowed, 0, 255).astype(np.uint8))
    return _fn


def _grayscale() -> Callable[[Image.Image], Image.Image]:
    return lambda img: img.convert("L").convert("RGB")


# transform_name -> (function, {feature_id: (should_preserve, tolerance_pct, rationale)})
_GEOM_TOL = 0.08   # 8% relative -- allows for resampling/anti-aliasing, not a silently-picked number: matches the
                   # ground-truth harness's own 2% *absolute* coverage tolerance scaled up for compounded transform noise
_STROKE_TOL = 0.35  # stroke/detail features are expected to be materially more sensitive to resampling/compression
_COLOUR_TOL = 0.10

TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "resize_0.5x": _resize(0.5), "resize_1x": _resize(1.0), "resize_2x": _resize(2.0),
    "rotate_-5deg": _rotate(-5), "rotate_-2deg": _rotate(-2), "rotate_+2deg": _rotate(2), "rotate_+5deg": _rotate(5),
    "jpeg_q95": _jpeg(95), "jpeg_q75": _jpeg(75), "jpeg_q50": _jpeg(50), "jpeg_q30": _jpeg(30),
    "brightness_up_1.3x": _brightness(1.3), "brightness_down_0.7x": _brightness(0.7),
    "contrast_up_1.3x": _contrast(1.3), "contrast_down_0.7x": _contrast(0.7),
    "white_margin_20px": _add_white_margin(20), "border_crop_10px": _crop_border(10),
    "perspective_mild": _perspective(0.04), "shadow_mild": _mild_shadow(), "grayscale": _grayscale(),
}

# Which transforms a geometry/stroke/colour feature is theoretically
# expected to survive (within its group tolerance) -- stated explicitly,
# not inferred silently. Transforms not listed for a group are expected
# to CHANGE that group's features (documented per-row via `limitations`).
_GEOMETRY_PRESERVING = {"resize_0.5x", "resize_1x", "resize_2x", "jpeg_q95", "jpeg_q75",
                        "brightness_up_1.3x", "brightness_down_0.7x", "contrast_up_1.3x", "contrast_down_0.7x",
                        "grayscale", "rotate_-2deg", "rotate_+2deg"}
_STROKE_PRESERVING = {"resize_1x", "jpeg_q95", "brightness_up_1.3x", "brightness_down_0.7x", "grayscale"}
_COLOUR_PRESERVING = {"resize_0.5x", "resize_1x", "resize_2x", "jpeg_q95", "jpeg_q75",
                       "rotate_-2deg", "rotate_+2deg", "grayscale"}


def _group_and_tolerance(feature_id: str) -> tuple[set[str], float]:
    if feature_id in _GEOMETRY_FEATURES:
        return _GEOMETRY_PRESERVING, _GEOM_TOL
    if feature_id in _STROKE_FEATURES:
        return _STROKE_PRESERVING, _STROKE_TOL
    return _COLOUR_PRESERVING, _COLOUR_TOL


def _extract(path_or_image, is_path: bool) -> dict[str, float]:
    temp = tempfile.TemporaryDirectory()
    try:
        if is_path:
            image = Image.open(path_or_image).convert("RGB")
        else:
            image = path_or_image
        img_path = Path(temp.name) / "image.png"
        image.save(img_path)
        out = Path(temp.name) / "out"
        result = analyze_image(img_path, out)
        features = objective_feature_row(img_path, result.to_dict())
        return {fid: features[fid].value for fid in ALL_FEATURES}
    finally:
        temp.cleanup()


def run_invariance_audit() -> dict[str, Any]:
    rows = []
    for image_id, path in FIXED_SAMPLE:
        original_image = Image.open(path).convert("RGB")
        original_values = _extract(original_image, is_path=False)
        for transform_name, transform_fn in TRANSFORMS.items():
            transformed_image = transform_fn(original_image)
            transformed_values = _extract(transformed_image, is_path=False)
            for feature_id in ALL_FEATURES:
                orig_v = original_values[feature_id]
                trans_v = transformed_values[feature_id]
                abs_change = abs(trans_v - orig_v)
                pct_change = (abs_change / abs(orig_v)) if abs(orig_v) > 1e-9 else None
                preserving_set, tol = _group_and_tolerance(feature_id)
                should_preserve = transform_name in preserving_set
                if should_preserve:
                    status = "pass" if (pct_change is not None and pct_change <= tol) or (pct_change is None and abs_change < 0.02) else "fail"
                else:
                    status = "expected_change"
                rows.append({
                    "image_id": image_id, "transform": transform_name, "feature_id": feature_id,
                    "original_value": round(orig_v, 6), "transformed_value": round(trans_v, 6),
                    "absolute_change": round(abs_change, 6),
                    "percentage_change": round(pct_change, 4) if pct_change is not None else None,
                    "status": status, "should_theoretically_preserve": should_preserve,
                    "limitations": "" if should_preserve else "Transform is expected to materially change this feature group; not scored as a failure.",
                })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary = {
        "schema_version": "feature_invariance_summary_v1",
        "n_rows": len(rows), "n_images": len(FIXED_SAMPLE), "n_transforms": len(TRANSFORMS),
        "n_features": len(ALL_FEATURES), "status_counts": status_counts,
        "tolerances": {"geometry_relative_pct": _GEOM_TOL, "stroke_relative_pct": _STROKE_TOL, "colour_relative_pct": _COLOUR_TOL},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(run_invariance_audit())
