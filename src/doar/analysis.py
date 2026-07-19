from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .schemas import Analysis, Evidence
from .dataset import CLASSES
from .rules import evaluate_rules
from .case_output import finalize_case
from .emotion import predict as predict_emotion


DISCLAIMER = (
    "This research output describes visible image evidence only. It is not a "
    "diagnosis and requires qualified professional review."
)


def _estimate_background(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    return np.median(border.astype(np.float32), axis=0)


def _candidate_score(mask: np.ndarray, edge_strength: np.ndarray) -> float:
    coverage = float(mask.mean())
    if coverage < 0.00005:
        coverage_score = 1.0
    elif coverage > 0.85:
        coverage_score = max(0.0, 1.0 - (coverage - 0.85) * 5)
    else:
        coverage_score = 1.0
    alignment = float(edge_strength[mask].mean() / (edge_strength.mean() + 1e-6)) if mask.any() else 1.0
    border_ratio = float(np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])).mean())
    return float(np.clip(0.45 * coverage_score + 0.40 * min(alignment, 1.0) + 0.15 * (1 - border_ratio), 0, 1))


def _segment(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, dict[str, np.ndarray], dict]:
    bg = _estimate_background(rgb)
    distance = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    gray = rgb.mean(axis=2)
    gx, gy = np.gradient(gray)
    edge_strength = np.hypot(gx, gy)
    colour_threshold = max(12.0, float(np.percentile(distance, 75)) * 0.45)
    candidate_colour = distance >= colour_threshold
    local_mean = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.BoxBlur(radius=9)),
        dtype=np.float32,
    )
    candidate_adaptive = gray < (local_mean - 8.0)
    if float(bg.mean()) < 100:
        candidate_gray = gray > min(245.0, float(bg.mean()) + 45.0)
    else:
        candidate_gray = gray < max(65.0, float(bg.mean()) - 35.0)
    candidates = {
        "colour_distance": candidate_colour,
        "adaptive_grayscale": candidate_adaptive,
        "global_grayscale": candidate_gray,
    }
    scores = {name: _candidate_score(value, edge_strength) for name, value in candidates.items()}
    selected_name = max(scores, key=scores.get)
    mask = candidates[selected_name].copy()
    # Dependency-free opening/closing style cleanup.
    for minimum in (3, 4):
        padded = np.pad(mask.astype(np.uint8), 1)
        neighbours = sum(
            padded[y:y + mask.shape[0], x:x + mask.shape[1]]
            for y in range(3) for x in range(3)
        )
        mask = neighbours >= minimum
    border_distance = np.linalg.norm(
        np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1])).astype(np.float32) - bg,
        axis=1,
    )
    background_stability = float(np.clip(1.0 - np.std(border_distance) / 80.0, 0.0, 1.0))
    pairwise = []
    values = list(candidates.values())
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            union = np.logical_or(values[left], values[right]).sum()
            pairwise.append(float(np.logical_xor(values[left], values[right]).sum() / max(1, union)))
    disagreement = float(np.mean(pairwise))
    confidence = float(np.clip(
        0.45 * scores[selected_name] + 0.35 * background_stability + 0.20 * (1 - disagreement),
        0, 1,
    ))
    diagnostics = {
        "selected_strategy": selected_name,
        "candidate_scores": scores,
        "candidate_disagreement": disagreement,
        "background_stability": background_stability,
    }
    return mask, bg, confidence, candidates, diagnostics


def _composition(mask: np.ndarray) -> dict:
    h, w = mask.shape
    coverage = float(mask.mean())
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {
            "foreground_coverage": 0.0,
            "empty_space_ratio": 1.0,
            "bounding_box": None,
            "bounding_box_coverage": 0.0,
            "centroid_normalized": None,
            "placement": "unavailable",
            "margins_normalized": None,
        }
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cx, cy = float(xs.mean() / w), float(ys.mean() / h)
    horizontal = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    vertical = "top" if cy < 0.4 else "bottom" if cy > 0.6 else "middle"
    return {
        "foreground_coverage": coverage,
        "empty_space_ratio": 1.0 - coverage,
        "bounding_box": [x0, y0, x1, y1],
        "bounding_box_coverage": ((x1 - x0 + 1) * (y1 - y0 + 1)) / float(h * w),
        "centroid_normalized": [cx, cy],
        "placement": f"{vertical}_{horizontal}",
        "margins_normalized": [x0 / w, y0 / h, (w - 1 - x1) / w, (h - 1 - y1) / h],
    }


def _colour(rgb: np.ndarray, mask: np.ndarray, background: np.ndarray) -> dict:
    pixels = rgb[mask]
    if not len(pixels):
        return {
            "background_rgb": background.round().astype(int).tolist(),
            "dominant_colour": "none_or_neutral",
            "meaningful_colours": [],
            "colour_diversity": 0,
        }
    bins = {
        "red": (pixels[:, 0] > pixels[:, 1] * 1.35) & (pixels[:, 0] > pixels[:, 2] * 1.35),
        "green": (pixels[:, 1] > pixels[:, 0] * 1.25) & (pixels[:, 1] > pixels[:, 2] * 1.15),
        "blue": (pixels[:, 2] > pixels[:, 0] * 1.25) & (pixels[:, 2] > pixels[:, 1] * 1.15),
        "yellow": (pixels[:, 0] > 150) & (pixels[:, 1] > 130) & (pixels[:, 2] < 120),
        "dark": pixels.mean(axis=1) < 80,
    }
    ratios = {name: float(values.mean()) for name, values in bins.items()}
    meaningful = {name: value for name, value in ratios.items() if value >= 0.02}
    dominant = max(meaningful, key=meaningful.get) if meaningful else "none_or_neutral"
    return {
        "background_rgb": background.round().astype(int).tolist(),
        "dominant_colour": dominant,
        "colour_proportions": ratios,
        "meaningful_colours": sorted(meaningful),
        "colour_diversity": len(meaningful),
        "minimum_meaningful_ratio": 0.02,
    }


def _save_artifacts(
    image: Image.Image, mask: np.ndarray, comp: dict, candidates: dict[str, np.ndarray], output: Path
) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    normalized = output / "normalized.png"
    mask_path = output / "foreground_mask.png"
    overlay_path = output / "feature_overlay.png"
    foreground_path = output / "foreground_only.png"
    edge_path = output / "edge_map.png"
    page_path = output / "page_mask.png"
    density_path = output / "density_map.png"
    stroke_path = output / "stroke_map.png"
    image.save(normalized)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    Image.fromarray(np.full(mask.shape, 255, dtype=np.uint8)).save(page_path)
    rgb = np.asarray(image).copy()
    white = np.full_like(rgb, 255)
    white[mask] = rgb[mask]
    Image.fromarray(white).save(foreground_path)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    if comp["bounding_box"]:
        draw.rectangle(comp["bounding_box"], outline=(255, 0, 255), width=3)
        cx, cy = comp["centroid_normalized"]
        x, y = int(cx * image.width), int(cy * image.height)
        draw.line((x - 8, y, x + 8, y), fill=(255, 0, 0), width=2)
        draw.line((x, y - 8, x, y + 8), fill=(255, 0, 0), width=2)
    overlay.save(overlay_path)
    edge = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge.save(edge_path)
    Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10)).save(density_path)
    edge.point(lambda p: 255 if p > 20 else 0).save(stroke_path)
    candidate_paths = {}
    for name, value in candidates.items():
        path = output / f"candidate_{name}.png"
        Image.fromarray((value * 255).astype(np.uint8)).save(path)
        candidate_paths[name] = str(path)
    return {
        "normalized_image": str(normalized),
        "foreground_mask": str(mask_path),
        "foreground_only": str(foreground_path),
        "feature_overlay": str(overlay_path),
        "edge_map": str(edge_path),
        "page_mask": str(page_path),
        "density_map": str(density_path),
        "stroke_map": str(stroke_path),
        "candidate_masks": candidate_paths,
    }


# Quality-gate thresholds (D9). Justified, conservative defaults; below these a
# drawing lacks the detail/sharpness/contrast for reliable objective analysis.
MIN_DIMENSION_PX = 100          # below this, detail is insufficient
MIN_BLUR_VARIANCE = 15.0        # Laplacian variance; lower => blurred/defocused
MIN_CONTRAST_STD = 8.0          # near-flat scans carry little signal


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a discrete Laplacian — a standard sharpness/blur proxy.
    Pure-numpy (no OpenCV): 4-neighbour Laplacian on the interior."""
    g = gray.astype(np.float64)
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var()) if lap.size else 0.0


def _quality(image) -> dict:
    """Real quality gating (D9): resolution, blur (Laplacian variance), contrast.
    `supported` is now derived, not hardcoded."""
    gray = np.asarray(image.convert("L"))
    contrast_std = float(gray.std())
    blur_variance = _laplacian_variance(gray)
    min_dimension = int(min(image.width, image.height))

    resolution_ok = min_dimension >= MIN_DIMENSION_PX
    blur_ok = blur_variance >= MIN_BLUR_VARIANCE
    contrast_ok = contrast_std >= MIN_CONTRAST_STD

    reasons = []
    if not resolution_ok:
        reasons.append(f"resolution below {MIN_DIMENSION_PX}px (min_dimension={min_dimension})")
    if not blur_ok:
        reasons.append(f"low sharpness (blur_variance={blur_variance:.1f} < {MIN_BLUR_VARIANCE})")
    if not contrast_ok:
        reasons.append(f"low contrast (contrast_std={contrast_std:.1f} < {MIN_CONTRAST_STD})")

    # Three-state gate (Item 9). Resolution failure is a hard unsupported; a
    # single soft failure (blur OR contrast) is requires_review; both soft
    # failures is unsupported; all-pass is supported.
    soft_failures = int(not blur_ok) + int(not contrast_ok)
    if not resolution_ok or soft_failures >= 2:
        quality_status = "unsupported"
    elif soft_failures == 1:
        quality_status = "requires_review"
    else:
        quality_status = "supported"
    supported = quality_status == "supported"

    return {
        "width": image.width,
        "height": image.height,
        "min_dimension": min_dimension,
        "contrast_std": round(contrast_std, 3),
        "blur_variance": round(blur_variance, 3),
        "resolution_ok": resolution_ok,
        "blur_ok": blur_ok,
        "contrast_ok": contrast_ok,
        "supported": supported,
        "quality_status": quality_status,
        "unsupported_reasons": reasons,
        "thresholds_validated_on_real_dataset": False,
        "gate_thresholds": {
            "min_dimension_px": MIN_DIMENSION_PX,
            "min_blur_variance": MIN_BLUR_VARIANCE,
            "min_contrast_std": MIN_CONTRAST_STD,
        },
    }


def analyze_image(
    image_path: str | Path, output_dir: str | Path, emotion_checkpoint: str | Path | None = None
) -> Analysis:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    mask, background, seg_conf, candidates, seg_diagnostics = _segment(rgb)
    composition = _composition(mask)
    colour = _colour(rgb, mask, background)
    output = Path(output_dir)
    artifacts = _save_artifacts(image, mask, composition, candidates, output / "artifacts")
    quality = _quality(image)
    evidence = [
        Evidence("ev_seg_coverage", "objective_feature", composition["foreground_coverage"],
                 "border_colour_distance_v1", seg_conf),
        Evidence("ev_bbox_coverage", "objective_feature", composition["bounding_box_coverage"],
                 "selected_foreground_bounding_box_v1", seg_conf),
        Evidence("ev_centroid", "objective_feature", composition["centroid_normalized"],
                 "foreground_mask_moments_v1", seg_conf,
                 ["Unavailable on blank pages"]),
        Evidence("ev_dominant_colour", "objective_feature", colour["dominant_colour"],
                 "foreground_palette_v1", seg_conf,
                 ["Colour naming is coarse and background-dependent"]),
    ]
    analysis_context = {
        "artifacts": artifacts,
        "segmentation": {
            "confidence": seg_conf, "background_rgb": background.round().astype(int).tolist(),
            **seg_diagnostics,
        },
        "composition": composition,
        "colour": colour,
    }
    # Item 9 — enforce quality gating: when the image is UNSUPPORTED, suppress
    # emotion classification, psychologist rules and concern profiles, and record
    # which modules executed vs were suppressed and why.
    quality_status = quality.get("quality_status", "supported")
    executed_modules, suppressed_modules = ["quality", "segmentation", "composition", "colour"], []

    if quality_status == "unsupported":
        reason = "image quality unsupported: " + "; ".join(quality.get("unsupported_reasons", []))
        emotion = {
            "status": "suppressed", "reason": reason,
            "probabilities": {name: None for name in CLASSES},
            "model_family": "suppressed_low_quality",
        }
        rule_evaluations, concerns = [], []
        suppressed_modules = ["emotion_model", "psychologist_rules", "concern_profiles"]
    else:
        executed_modules.append("emotion_model")
        emotion = predict_emotion(image_path, emotion_checkpoint, analysis_context)
        if emotion["status"] == "available":
            evidence.append(Evidence(
                "ev_emotion_prediction", "model_prediction", emotion["probabilities"],
                emotion["model_name"], emotion["confidence"],
                ["Model probabilities are not psychological or diagnostic confidence."],
            ))
        rule_evaluations, concerns = evaluate_rules(composition, colour, evidence)
        executed_modules += ["psychologist_rules", "concern_profiles"]

    module_execution = {
        "quality_status": quality_status,
        "executed": executed_modules,
        "suppressed": suppressed_modules,
        "suppression_reason": (
            "; ".join(quality.get("unsupported_reasons", []))
            if quality_status == "unsupported" else None
        ),
    }
    result = Analysis(
        schema_version="3.0.0",
        image_path=str(Path(image_path).resolve()),
        quality=quality,
        segmentation={"status": "verified" if seg_conf >= 0.6 else "uncertain",
                      "confidence": seg_conf,
                      "background_rgb": background.round().astype(int).tolist(),
                      **seg_diagnostics},
        composition=composition,
        colour=colour,
        emotion=emotion,
        evidence=evidence,
        rule_evaluations=rule_evaluations,
        concerns=concerns,
        safety_disclaimer=DISCLAIMER,
        artifacts=artifacts,
        module_execution=module_execution,
    )
    output.mkdir(parents=True, exist_ok=True)
    portable = result.to_dict()
    portable["artifacts"] = {
        key: (
            {nested: Path(value).relative_to(output).as_posix() for nested, value in path.items()}
            if isinstance(path, dict) else Path(path).relative_to(output).as_posix()
        )
        for key, path in portable["artifacts"].items()
    }
    portable["image_path"] = Path(image_path).name
    (output / "analysis.json").write_text(
        json.dumps(portable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    finalize_case(portable, output)
    return result
