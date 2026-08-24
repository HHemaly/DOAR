"""Phase G0 -- objective, deterministic formal/graphic-element measurements
(line geometry, stroke texture, colour/fill, symmetry/regularity) proposed
in DOAR_Master_Rules_and_Features_Inventory_V3.pdf Appendix G.

STRICTLY OBJECTIVE_ONLY: every function here is a pure CV/numpy
measurement over the SAME foreground mask `features.objective_feature_row`
already uses -- no model, no learned weight, no network call. Reuses
`features.FeatureValue` (identical shape/contract), never redefines it.

CRITICAL CONSTRAINT (see tests/test_formal_features_no_concern_score_effect.py):
this module's output is NEVER passed to
case_interpretation.build_concern_domains/build_case_interpretation, and
nothing in this module reads or writes CONCERN_DOMAIN_MAP.json,
RULE_EVIDENCE_MATRIX.csv, the aggregator, or the rule engine. It is wired
into the app as a separate, additive, on-demand artifact
(`formal_features.json`) -- see doar_prototype_app.py's Technical/
Psychologist "Formal/graphic measurements (research, OBJECTIVE_ONLY)"
section -- computed only when explicitly requested, exactly like the
existing deep-visual-analysis button, never during ordinary Analyze.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .features import FeatureValue

FORMAL_FEATURES_VERSION = "formal_features_v1"

_ORIENTATION_BIN_COUNT = 18  # 10-degree bins over the 180-degree orientation range


def _load_gray_rgb_mask(image_path: str | Path, analysis: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = np.asarray(Image.open(analysis["artifacts"]["foreground_mask"]).convert("L")) > 0
    return gray, rgb, mask


# Phase G0.1, Step 4A: a filled region (e.g. a coloured-in shirt) is NOT a
# "line", however the plain whole-mask distance transform used before this
# pass cannot tell the two apart -- every foreground pixel in a large solid
# blob has a large distance-to-background too, and would be reported as an
# implausibly "thick line". Separated here via morphological granulometry:
# opening the mask with a kernel of radius `_STROKE_VS_FILL_RADIUS_FRACTION`
# (a fraction of the image's shorter side) REMOVES anything thinner than
# that kernel (any curvature -- straight lines, rings, curves alike) while
# PRESERVING anything at least that thick in every direction (filled
# blobs). This is a documented, uncalibrated constant (Step 4A explicitly
# permits this rather than overclaiming precision) -- a real stroke close
# to or above this width is a genuine, known edge case that can be
# misclassified as "filled" (see FORMAL_FEATURE_DEFINITION_TABLE.csv's
# known_confounders column); true, validated stroke/fill separation is an
# E-Graphic calibration task, not resolved here.
_STROKE_VS_FILL_RADIUS_FRACTION = 0.02  # kernel radius = 2% of min(height, width)


def _stroke_like_mask(mask: np.ndarray) -> np.ndarray:
    """Classifies each CONNECTED COMPONENT as a whole, not pixel-by-pixel:
    a component that survives the opening AT ALL (even just its interior)
    is a filled blob and is excluded IN ITS ENTIRETY -- a naive per-pixel
    `mask & ~opened` would still misclassify a filled blob's own outer
    boundary ring (itself thinner than the kernel) as "stroke-like",
    silently reintroducing the exact bug this function exists to fix. A
    component that the opening erases COMPLETELY (no surviving pixel
    anywhere in it) is genuinely thin everywhere along its own length and
    is kept in full (original, un-opened pixels) for the width
    measurement."""
    if not mask.any():
        return mask
    radius = max(2, round(min(mask.shape) * _STROKE_VS_FILL_RADIUS_FRACTION))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    stroke_like = np.zeros_like(mask)
    for label in range(1, n_labels):
        component = labels == label
        if not opened[component].any():
            stroke_like |= component
    return stroke_like


def _line_width_stats(mask: np.ndarray) -> tuple[float, float, bool]:
    """Distance-transform proxy over STROKE-LIKE pixels only (see
    `_stroke_like_mask`): at every such pixel, twice the distance to the
    nearest background pixel approximates the local stroke width at that
    point (exact at a stroke's medial axis, an overestimate elsewhere --
    an OBJECTIVE_ONLY proxy, never true physical pencil width). Returns
    (mean, variability, no_stroke_like_pixels_found) -- the third value
    lets the caller mark this honestly missing for a purely filled
    drawing (e.g. one solid silhouette), rather than reporting 0.0 as if
    it had been measured."""
    stroke_mask = _stroke_like_mask(mask)
    if not stroke_mask.any():
        return 0.0, 0.0, True
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    stroke_dist = dist[stroke_mask]
    return float(2 * stroke_dist.mean()), float(2 * stroke_dist.std()), False


def _darkness_stats(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Unlike line width, darkness is well-defined over the WHOLE
    foreground (a filled dark region is genuinely dark, not a measurement
    artifact) -- deliberately NOT restricted to stroke-like pixels."""
    if not mask.any():
        return 0.0, 0.0
    darkness = (255.0 - gray[mask].astype(np.float32)) / 255.0
    return float(darkness.mean()), float(darkness.std())


def _continuity_and_fragmentation(mask: np.ndarray) -> tuple[float, float, int]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    component_count = n_labels - 1  # exclude background label 0
    total = max(1, int(mask.sum()))
    if component_count == 0:
        return 0.0, 0.0, 0
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest_ratio = float(sizes.max() / total)
    fragmentation_rate = float(min(1.0, component_count / max(1, total / 50)))
    return largest_ratio, fragmentation_rate, component_count


def _orientation_stats(gray: np.ndarray, mask: np.ndarray) -> dict:
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    if not mask.any() or magnitude[mask].max() <= 0:
        return {"dominant_orientation": None, "entropy": 0.0, "coherence": 0.0, "cluster_count": 0}
    # A handful of high-curvature pixels (line endpoints/corners) can have
    # gradient magnitude comparable to or higher than the many pixels along
    # a long straight edge -- a hard top-percentile pixel COUNT would let
    # those few corner pixels dominate the histogram. Weighting every
    # sufficiently-strong pixel by its OWN gradient magnitude instead means
    # the long straight edges (far more pixels, each contributing weight)
    # correctly outweigh the few corner artifacts.
    weights_all = magnitude[mask]
    keep = weights_all > (0.05 * weights_all.max())
    if not keep.any():
        return {"dominant_orientation": None, "entropy": 0.0, "coherence": 0.0, "cluster_count": 0}
    gx_m, gy_m = gx[mask][keep], gy[mask][keep]
    weights = weights_all[keep]

    # Sobel gives the gradient (edge-normal) direction, which is
    # perpendicular to the STROKE's own direction -- rotate by 90 degrees
    # so "dominant orientation" reports the line/stroke's tangent
    # direction (a vertical stroke has a horizontal gradient, and must be
    # reported as "vertical", not "horizontal").
    angles = (np.degrees(np.arctan2(gy_m, gx_m)) + 90.0) % 180.0
    hist, _ = np.histogram(angles, bins=_ORIENTATION_BIN_COUNT, range=(0.0, 180.0), weights=weights)
    probs = hist.astype(np.float64) / hist.sum()
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / math.log2(_ORIENTATION_BIN_COUNT))  # normalized [0, 1]

    dominant_bin = int(np.argmax(hist))
    dominant_angle = (dominant_bin + 0.5) * (180.0 / _ORIENTATION_BIN_COUNT)
    if dominant_angle < 22.5 or dominant_angle >= 157.5:
        dominant_orientation = "horizontal"
    elif 67.5 <= dominant_angle < 112.5:
        dominant_orientation = "vertical"
    else:
        dominant_orientation = "diagonal"

    # Structure-tensor local coherence: (lambda1 - lambda2) / (lambda1 + lambda2),
    # 1.0 = perfectly locally aligned, 0.0 = locally isotropic/multidirectional.
    jxx = cv2.GaussianBlur(gx * gx, (5, 5), 0)
    jyy = cv2.GaussianBlur(gy * gy, (5, 5), 0)
    jxy = cv2.GaussianBlur(gx * gy, (5, 5), 0)
    trace_half = (jxx + jyy) / 2.0
    diff_term = np.sqrt(np.square((jxx - jyy) / 2.0) + np.square(jxy))
    lambda1 = trace_half + diff_term
    lambda2 = trace_half - diff_term
    denom = lambda1 + lambda2
    coherence_map = np.divide(lambda1 - lambda2, denom, out=np.zeros_like(denom), where=denom > 1e-9)
    strong_2d = np.zeros_like(mask)
    mask_indices = np.argwhere(mask)
    strong_2d[tuple(mask_indices[keep].T)] = True
    coherence = float(coherence_map[strong_2d].mean())

    peak_threshold = probs.max() * 0.3
    cluster_count = int((probs > peak_threshold).sum())

    return {"dominant_orientation": dominant_orientation, "entropy": entropy,
            "coherence": coherence, "cluster_count": cluster_count}


def _straightness(gray: np.ndarray, mask: np.ndarray) -> float:
    """Share of edge pixels that lie on a detected near-straight segment
    (Hough line transform) -- an OBJECTIVE_ONLY formal-geometry proxy, not
    a claim about drawing intent."""
    edges = cv2.Canny(gray, 50, 150)
    edges = np.where(mask, edges, 0)
    total_edge = int((edges > 0).sum())
    if total_edge == 0:
        return 0.0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=12, maxLineGap=3)
    if lines is None:
        return 0.0
    covered = np.zeros_like(edges)
    for line in lines:
        x1, y1, x2, y2 = np.asarray(line).ravel()[:4]
        cv2.line(covered, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=2)
    covered_edge_pixels = int(((covered > 0) & (edges > 0)).sum())
    return float(min(1.0, covered_edge_pixels / total_edge))


def _junction_corner_density_proxy(mask: np.ndarray) -> float:
    """Phase G0.1, Step 4D (renamed from `_crossing_density`/
    `stroke.crossing_density`): a corner/junction-RESPONSE density, not
    true skeleton-intersection crossing counting. cornerHarris responds to
    line crossings, T/L junctions, AND sharp bends/corners alike -- no true
    skeletonization tool is available in this environment's OpenCV build
    (no ximgproc/scikit-image) to isolate genuine crossings from corners.
    True crossing detection needs skeletonization/vectorization -- deferred
    to E-Graphic, see MASTER_RULE_FEATURE_REGISTRY_V3.json's
    junction_corner_density_proxy entry."""
    if not mask.any():
        return 0.0
    response = cv2.cornerHarris(mask.astype(np.float32), blockSize=5, ksize=3, k=0.04)
    if response.max() <= 0:
        return 0.0
    corners = response > (0.01 * response.max())
    return float(corners.sum() / max(1, int(mask.sum())))


def _fill_uniformity(mask: np.ndarray, grid: int = 8) -> float:
    h, w = mask.shape
    if h < grid or w < grid:
        return float("nan")
    cell_h, cell_w = h // grid, w // grid
    densities = []
    for i in range(grid):
        for j in range(grid):
            cell = mask[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            densities.append(float(cell.mean()))
    densities = np.asarray(densities)
    mean_density = densities.mean()
    if mean_density <= 1e-9:
        return float("nan")
    variance = densities.var()
    return float(1.0 / (1.0 + variance / (mean_density ** 2)))


# Phase G0.1, Step 4B: `composition.foreground_coverage` (how much of the
# page has ANY mark on it) is NOT "colour coverage" -- a full-page graphite
# drawing has high foreground coverage but almost no chromatic colour. This
# is a genuinely separate measurement: the fraction of FOREGROUND pixels
# that are chromatically saturated (neither near-black, near-white/paper,
# nor grey). Thresholds are documented, uncalibrated constants (Step 4B
# explicitly permits labeling this "exploratory" pending later calibration
# rather than pretending a validated cutoff exists).
_CHROMATIC_SATURATION_THRESHOLD = 0.15  # HSV saturation, 0..1 scale
_CHROMATIC_MIN_VALUE = 0.08             # excludes near-black marks (no visible hue)
_CHROMATIC_MAX_VALUE = 0.98             # excludes near-white/paper


def _chromatic_colour_coverage(rgb: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1][mask] / 255.0
    value = hsv[..., 2][mask] / 255.0
    chromatic = (saturation > _CHROMATIC_SATURATION_THRESHOLD) & \
                (value > _CHROMATIC_MIN_VALUE) & (value < _CHROMATIC_MAX_VALUE)
    return float(chromatic.mean())


def _bilateral_symmetry(mask: np.ndarray) -> float:
    return float(1.0 - np.logical_xor(mask, np.fliplr(mask)).mean())


def _spacing_regularity(mask: np.ndarray) -> float:
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    real_centroids = centroids[1:]  # drop background label 0
    if len(real_centroids) < 3:
        return float("nan")
    nearest_dists = []
    for i, c in enumerate(real_centroids):
        others = np.delete(real_centroids, i, axis=0)
        dists = np.linalg.norm(others - c, axis=1)
        nearest_dists.append(float(dists.min()))
    nearest_dists = np.asarray(nearest_dists)
    mean_d = nearest_dists.mean()
    if mean_d <= 1e-9:
        return float("nan")
    cv_ = nearest_dists.std() / mean_d
    return float(1.0 / (1.0 + cv_))


def compute_formal_features(image_path: str | Path, analysis: dict) -> dict[str, FeatureValue]:
    """Mirrors `features.objective_feature_row`'s exact contract (same
    `FeatureValue` shape, same never-fabricate-missing-as-zero discipline)
    but for the new formal/graphic dimensions from Appendix G. Reuses
    `analysis["colour"]["colour_diversity"]` (already computed by
    `analysis.py`) rather than recomputing it a second, subtly-different
    way. `colour.chromatic_coverage` is deliberately NOT a reuse of
    `analysis["composition"]["foreground_coverage"]` (Step 4B) -- the two
    measure genuinely different things (any mark vs. chromatically
    saturated marks) and conflating them was the exact bug this pass
    fixes; see `_chromatic_colour_coverage`."""
    conf = float(analysis["segmentation"]["confidence"])
    gray, rgb, mask = _load_gray_rgb_mask(image_path, analysis)
    has_foreground = bool(mask.any())

    mean_width, width_variability, no_stroke_like_pixels = _line_width_stats(mask)
    darkness_mean, darkness_variability = _darkness_stats(gray, mask)
    continuity, fragmentation_rate, component_count = _continuity_and_fragmentation(mask)
    orientation = _orientation_stats(gray, mask)
    straightness = _straightness(gray, mask)
    junction_corner_density = _junction_corner_density_proxy(mask)
    spatial_density_uniformity = _fill_uniformity(mask)
    bilateral_symmetry = _bilateral_symmetry(mask)
    spacing_regularity = _spacing_regularity(mask)
    edges = cv2.Canny(gray, 50, 150)
    detail_density = float(np.where(mask, edges > 0, False).mean())
    stroke_density_mean = float(mask.mean())
    # Step 4C: an EXPLORATORY composite, not a validated "scribble
    # detector" -- documented formula: whole-mask density * global
    # orientation entropy. A real scribble detector (per the PDF's own
    # Appendix G note) would need local coherence + short-stroke density +
    # curvature + intersections + spatial irregularity combined; no
    # weighting for that composite is invented here.
    scribble_candidate_score = float(stroke_density_mean * orientation["entropy"])

    chromatic_colour_coverage = _chromatic_colour_coverage(rgb, mask)
    colour_diversity = float(analysis["colour"]["colour_diversity"])
    brightness = float(gray.mean() / 255.0)
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = float(hsv[..., 1][mask].mean() / 255.0) if has_foreground else 0.0

    orientation_code = {"horizontal": 0.0, "vertical": 1.0, "diagonal": 2.0, None: float("nan")}[
        orientation["dominant_orientation"]]

    raw_values: dict[str, tuple[float, bool]] = {
        # name -> (value, missing_if_no_foreground)
        "line.mean_width": (mean_width, no_stroke_like_pixels),
        "line.width_variability": (width_variability, no_stroke_like_pixels),
        "line.darkness_mean": (darkness_mean, not has_foreground),
        "line.darkness_variability": (darkness_variability, not has_foreground),
        "line.continuity": (continuity, component_count == 0),
        "line.fragmentation_rate": (fragmentation_rate, component_count == 0),
        "line.straightness": (straightness, not has_foreground),
        "line.dominant_orientation_code": (orientation_code, orientation["dominant_orientation"] is None),
        "line.orientation_entropy": (orientation["entropy"], orientation["dominant_orientation"] is None),
        "line.orientation_coherence": (orientation["coherence"], orientation["dominant_orientation"] is None),
        "line.direction_cluster_count": (float(orientation["cluster_count"]), orientation["dominant_orientation"] is None),
        # Renamed (Step 4D): responds to crossings AND corners/junctions/
        # sharp bends (cornerHarris) -- never claim true skeleton-
        # intersection crossing counting.
        "stroke.junction_corner_density_proxy": (junction_corner_density, not has_foreground),
        "stroke.density_mean": (stroke_density_mean, not has_foreground),
        # Renamed (Step 4C).
        "stroke.scribble_candidate_score": (scribble_candidate_score, not has_foreground),
        # Renamed (Step 4B): genuine chromatic coverage, never equated
        # with composition.foreground_coverage.
        "colour.chromatic_coverage": (chromatic_colour_coverage, not has_foreground),
        "colour.diversity": (colour_diversity, False),
        "colour.brightness": (brightness, False),
        "colour.saturation": (saturation, not has_foreground),
        # Renamed (Step 4E): a WHOLE-DRAWING grid-occupancy measurement --
        # never the per-object region_fill_uniformity the PDF also
        # names (that stays EXPERIMENT_CANDIDATE, needs per-object masks).
        "fill.global_spatial_density_uniformity": (spatial_density_uniformity, math.isnan(spatial_density_uniformity)),
        "symmetry.bilateral": (bilateral_symmetry, not has_foreground),
        "symmetry.spacing_regularity": (spacing_regularity, math.isnan(spacing_regularity)),
        "scene.detail_density": (detail_density, not has_foreground),
    }

    result: dict[str, FeatureValue] = {}
    for name, (value, missing) in raw_values.items():
        numeric = float(value) if not (isinstance(value, float) and math.isnan(value)) else float("nan")
        result[name] = FeatureValue(
            value=numeric if math.isfinite(numeric) else 0.0,
            valid_min=None, valid_max=None,
            confidence=0.0 if missing else conf,
            method=FORMAL_FEATURES_VERSION,
            evidence_id=f"ev_formal_{name.replace('.', '_')}",
            missing=missing or not math.isfinite(numeric),
        )
    return result


def serialize_formal_features(row: dict[str, FeatureValue]) -> dict:
    from dataclasses import asdict
    return {name: asdict(value) for name, value in row.items()}


def run_and_persist_formal_features(case_dir: str | Path, image_path: str | Path, analysis: dict) -> dict:
    """On-demand orchestration -- mirrors visual_evidence.run_and_persist_
    initial_scan's own "compute once, write to disk, on request only"
    pattern. Writes formal_features.json; never touches detections.json,
    analysis.json, or any concern-domain/evidence artifact."""
    from .case_output import write_versioned

    case_dir = Path(case_dir)
    row = compute_formal_features(str(image_path), analysis)
    document = {
        "status": "available", "version": FORMAL_FEATURES_VERSION,
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "features": serialize_formal_features(row),
    }
    write_versioned(case_dir / "formal_features.json", document)
    return document


def render_formal_feature_overlay(image_path: str | Path, analysis: dict) -> Image.Image:
    """Step 6 (visual review output): draws the SAME detected straight-line
    segments `_straightness` uses (green) and the SAME corner/junction
    response points `_junction_corner_density_proxy` uses (red) directly over the
    original drawing, so a psychologist can SEE why DOAR reported a given
    straightness/crossing-density value -- audit-only, never fed back into
    any rule/aggregation."""
    gray, _rgb, mask = _load_gray_rgb_mask(image_path, analysis)
    base = Image.open(image_path).convert("RGB")
    overlay_bgr = cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2BGR).copy()

    edges = cv2.Canny(gray, 50, 150)
    edges_masked = np.where(mask, edges, 0)
    lines = cv2.HoughLinesP(edges_masked, 1, np.pi / 180, threshold=20, minLineLength=12, maxLineGap=3)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).ravel()[:4]
            cv2.line(overlay_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), thickness=2)

    if mask.any():
        response = cv2.cornerHarris(mask.astype(np.float32), blockSize=5, ksize=3, k=0.04)
        if response.max() > 0:
            corners = np.argwhere(response > (0.01 * response.max()))
            for y, x in corners:
                cv2.circle(overlay_bgr, (int(x), int(y)), 3, (0, 0, 220), thickness=-1)

    return Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
