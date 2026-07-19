from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class FeatureValue:
    value: float
    valid_min: float | None
    valid_max: float | None
    confidence: float
    method: str
    evidence_id: str
    missing: bool = False
    version: str = "3.1.0"


def _entropy(values: np.ndarray) -> float:
    hist = np.histogram(values, bins=64, range=(0, 255))[0].astype(float)
    probabilities = hist[hist > 0] / max(1, hist.sum())
    return float(-(probabilities * np.log2(probabilities)).sum())


def _connected_components(mask: np.ndarray) -> list[int]:
    # Component statistics are computed on a capped mask for predictable runtime.
    step = max(1, int(math.ceil(max(mask.shape) / 512)))
    sample = mask[::step, ::step]
    visited = np.zeros_like(sample, dtype=bool)
    sizes: list[int] = []
    h, w = sample.shape
    for y, x in np.argwhere(sample):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and sample[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(size * step * step)
    return sizes


def objective_feature_row(image_path: str | Path, analysis: dict) -> dict[str, FeatureValue]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    mask = np.asarray(Image.open(analysis["artifacts"]["foreground_mask"]).convert("L")) > 0
    conf = float(analysis["segmentation"]["confidence"])
    comp = analysis["composition"]
    fg = rgb[mask]
    gx, gy = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    edges = magnitude > max(8.0, float(np.percentile(magnitude, 80)))
    component_sizes = _connected_components(mask)
    total_fg = max(1, int(mask.sum()))
    hsv_saturation = ((rgb.max(axis=2) - rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), 1))
    quadrants = [
        mask[: mask.shape[0] // 2, : mask.shape[1] // 2],
        mask[: mask.shape[0] // 2, mask.shape[1] // 2 :],
        mask[mask.shape[0] // 2 :, : mask.shape[1] // 2],
        mask[mask.shape[0] // 2 :, mask.shape[1] // 2 :],
    ]
    densities = [float(q.mean()) for q in quadrants]
    centroid = comp["centroid_normalized"] or [math.nan, math.nan]
    margins = comp["margins_normalized"] or [math.nan] * 4
    background = np.asarray(analysis["segmentation"]["background_rgb"], dtype=float)
    foreground_mean = fg.mean(axis=0) if len(fg) else np.array([math.nan] * 3)
    dark_ratio = float((fg.mean(axis=1) < 80).mean()) if len(fg) else 0.0
    colour_bins = analysis["colour"].get("colour_proportions", {})

    values = {
        "quality.width": image.width,
        "quality.height": image.height,
        "quality.aspect_ratio": image.width / image.height,
        "quality.resolution": image.width * image.height,
        "quality.blur_score": float(np.var(np.diff(gray, axis=0))),
        "quality.contrast": float(gray.std()),
        "quality.brightness": float(gray.mean() / 255),
        "quality.exposure": float(((gray > 245) | (gray < 10)).mean()),
        "quality.saturation": float(hsv_saturation.mean()),
        "quality.entropy": _entropy(gray),
        "quality.noise_proxy": float(np.abs(gray - np.asarray(image.convert("L").filter(ImageFilter.MedianFilter(3)))).mean() / 255),
        "segmentation.page_confidence": analysis["segmentation"]["background_stability"],
        "segmentation.foreground_coverage": comp["foreground_coverage"],
        "segmentation.empty_space_ratio": comp["empty_space_ratio"],
        "segmentation.bounding_box_coverage": comp["bounding_box_coverage"],
        "segmentation.confidence": conf,
        "segmentation.component_count": len(component_sizes),
        "segmentation.largest_component_ratio": max(component_sizes, default=0) / total_fg,
        "segmentation.small_component_ratio": sum(v < max(8, total_fg * .001) for v in component_sizes) / max(1, len(component_sizes)),
        "segmentation.border_touch_ratio": float(np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])).mean()),
        "segmentation.extent": comp["foreground_coverage"] / max(comp["bounding_box_coverage"], 1e-9),
        "composition.centroid_x": centroid[0],
        "composition.centroid_y": centroid[1],
        "composition.left_margin": margins[0],
        "composition.top_margin": margins[1],
        "composition.right_margin": margins[2],
        "composition.bottom_margin": margins[3],
        "composition.horizontal_balance": 1 - abs(densities[0] + densities[2] - densities[1] - densities[3]),
        "composition.vertical_balance": 1 - abs(densities[0] + densities[1] - densities[2] - densities[3]),
        "composition.occupied_quadrants": sum(v > 0.001 for v in densities),
        **{f"composition.quadrant_{i + 1}_density": value for i, value in enumerate(densities)},
        "composition.center_occupancy": float(mask[mask.shape[0] // 4: 3 * mask.shape[0] // 4, mask.shape[1] // 4: 3 * mask.shape[1] // 4].mean()),
        "composition.symmetry": float(1 - np.logical_xor(mask, np.fliplr(mask)).mean()),
        "composition.spatial_dispersion": float(np.std(np.argwhere(mask), axis=0).mean() / max(mask.shape)) if mask.any() else 0.0,
        "colour.foreground_mean_r": foreground_mean[0],
        "colour.foreground_mean_g": foreground_mean[1],
        "colour.foreground_mean_b": foreground_mean[2],
        "colour.background_mean": float(background.mean() / 255),
        "colour.red_ratio": colour_bins.get("red", 0.0),
        "colour.green_ratio": colour_bins.get("green", 0.0),
        "colour.blue_ratio": colour_bins.get("blue", 0.0),
        "colour.yellow_ratio": colour_bins.get("yellow", 0.0),
        "colour.dark_ratio": dark_ratio,
        "colour.diversity": analysis["colour"]["colour_diversity"],
        "colour.entropy": _entropy(fg.mean(axis=1)) if len(fg) else 0.0,
        "colour.monochrome_flag": float(analysis["colour"]["colour_diversity"] <= 1),
        "colour.grayscale_flag": float(len(fg) == 0 or np.abs(fg - fg.mean(axis=1, keepdims=True)).mean() < 8),
        "stroke.edge_density": float(edges.mean()),
        "stroke.horizontal_ratio": float((np.abs(gy) > 2 * np.abs(gx))[edges].mean()) if edges.any() else 0.0,
        "stroke.vertical_ratio": float((np.abs(gx) > 2 * np.abs(gy))[edges].mean()) if edges.any() else 0.0,
        "stroke.diagonal_ratio": float((np.abs(gx - gy) < .4 * (np.abs(gx) + np.abs(gy) + 1))[edges].mean()) if edges.any() else 0.0,
        "stroke.fragmentation": min(1.0, len(component_sizes) / max(1, total_fg / 50)),
        "stroke.intensity_proxy": float((255 - fg.mean()) / 255) if len(fg) else 0.0,
        "shape.contour_proxy_count": len(component_sizes),
        "shape.enclosed_shape_count": 0.0,
        "shape.repetition_score": 0.0,
    }
    result = {}
    for name, value in values.items():
        numeric = float(value)
        evidence_id = "ev_bbox_coverage" if name == "segmentation.bounding_box_coverage" else f"ev_feature_{name.replace('.', '_')}"
        result[name] = FeatureValue(
            value=numeric,
            valid_min=None,
            valid_max=None,
            confidence=conf if not name.startswith("quality.") else 1.0,
            method="objective_features_v3_1",
            evidence_id=evidence_id,
            missing=not math.isfinite(numeric),
        )
    return result


def serialize_feature_row(row: dict[str, FeatureValue]) -> dict:
    return {name: asdict(value) for name, value in row.items()}
