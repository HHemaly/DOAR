"""Phase G0.1, Step 3 -- builds FORMAL_FEATURE_DEFINITION_TABLE.csv: one row
per measurement actually IMPLEMENTED in formal_features.py (never a
proposal-only row from MASTER_RULE_FEATURE_REGISTRY_V3.json's Appendix G
list -- that registry already tracks proposals; this table documents real
code with exact mathematical precision).

Every `psychological_interpretation_status` value below is OBJECTIVE_ONLY
-- this phase adds no new psychological interpretation, per the hard
constraint. Mirrors master_registry_v3_build.py's own build/write-function
pattern.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FORMAL_FEATURE_DEFINITION_TABLE_PATH = ROOT / "FORMAL_FEATURE_DEFINITION_TABLE.csv"

FIELDNAMES = [
    "feature_id", "display_name", "exact_definition", "formula_or_algorithm",
    "input_pixels_or_region", "image_scope", "unit", "expected_range",
    "higher_value_means", "lower_value_means", "implementation_status",
    "measurement_type", "proxy_or_direct", "known_confounders",
    "what_it_does_NOT_measure", "static_image_valid", "requires_segmentation_mask",
    "requires_process_data", "reliability_status", "psychological_interpretation_status", "notes",
]


def _row(**kwargs: Any) -> dict[str, Any]:
    row = {k: "" for k in FIELDNAMES}
    row["implementation_status"] = "IMPLEMENTED"
    row["static_image_valid"] = "yes"
    row["requires_segmentation_mask"] = "no"
    row["requires_process_data"] = "no"
    row["psychological_interpretation_status"] = "OBJECTIVE_ONLY"
    row.update(kwargs)
    return row


def build_formal_feature_definition_table() -> list[dict[str, Any]]:
    return [
        _row(
            feature_id="line.mean_width", display_name="Mean line width",
            exact_definition="Mean apparent stroke width, in pixels, over stroke-like connected components only.",
            formula_or_algorithm=(
                "1) Classify each connected component of the foreground mask as stroke-like or filled: open the "
                "mask with an elliptical kernel of radius = round(2% of min(H,W)); a component with ANY surviving "
                "pixel after opening is a filled blob (excluded in full); a component fully erased by the opening "
                "is stroke-like (kept in full). 2) Over stroke-like pixels only, compute the Euclidean distance "
                "transform of the (unopened) mask; mean_width = 2 * mean(distance at stroke-like pixels)."),
            input_pixels_or_region="Stroke-like connected components of the whole-image foreground mask (analysis.py's segmentation mask)",
            image_scope="WHOLE_DRAWING", unit="pixels", expected_range="0 to min(image_height, image_width)/2",
            higher_value_means="Thicker apparent strokes", lower_value_means="Thinner apparent strokes",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders=(
                "Image resolution/DPI (pixel width is not a physical unit); antialiasing; scan/photo blur; "
                "a real stroke near or above the ~2%-of-min-dimension opening radius can be misclassified as a "
                "filled blob and reported missing rather than measured (documented, uncalibrated threshold)."),
            what_it_does_NOT_measure="Physical pencil/pen pressure or force -- a flat scan/photo cannot recover applied pressure.",
            requires_segmentation_mask="no (uses the existing whole-image foreground/background mask only)",
            reliability_status="uncalibrated -- stroke/blob separation threshold requires E-Graphic calibration",
            notes="Renamed/fixed in Phase G0.1 Step 4A -- previously computed over the WHOLE mask, so a filled "
                  "region (e.g. a coloured-in shirt) was reported as an implausibly thick 'line'.",
        ),
        _row(
            feature_id="line.width_variability", display_name="Line-width variability",
            exact_definition="Standard deviation of local apparent stroke width over stroke-like pixels.",
            formula_or_algorithm="Same stroke-like classification as line.mean_width; width_variability = 2 * std(distance transform at stroke-like pixels).",
            input_pixels_or_region="Stroke-like connected components of the whole-image foreground mask",
            image_scope="WHOLE_DRAWING", unit="pixels", expected_range="0 to line.mean_width's own range",
            higher_value_means="Inconsistent/irregular stroke width", lower_value_means="Consistent, uniform stroke width",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="Same as line.mean_width (resolution, antialiasing, stroke/blob threshold).",
            what_it_does_NOT_measure="Physical pressure variability.",
            reliability_status="uncalibrated -- see line.mean_width",
        ),
        _row(
            feature_id="line.darkness_mean", display_name="Line darkness (mean)",
            exact_definition="Mean pixel darkness (255 - greyscale value, normalized to 0..1) over ALL foreground pixels.",
            formula_or_algorithm="darkness = (255 - grayscale_value) / 255 at every foreground-mask pixel; darkness_mean = mean(darkness).",
            input_pixels_or_region="Whole-image foreground mask (not restricted to stroke-like pixels -- see notes)",
            image_scope="WHOLE_DRAWING", unit="normalized darkness, 0..1", expected_range="0.0 (pure white) to 1.0 (pure black)",
            higher_value_means="Darker marks overall", lower_value_means="Fainter marks overall",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="Scan/photo exposure and white balance; paper colour/texture; marker vs. pencil ink density.",
            what_it_does_NOT_measure="Physical pencil pressure -- an appearance proxy only.",
            reliability_status="stable for its stated purpose (a plain intensity statistic)",
            notes="Deliberately computed over the WHOLE foreground (not stroke-like pixels only) -- a filled dark region "
                  "genuinely IS dark; excluding it would be wrong, unlike line.mean_width.",
        ),
        _row(
            feature_id="line.darkness_variability", display_name="Darkness variability",
            exact_definition="Standard deviation of the same per-pixel darkness value over all foreground pixels.",
            formula_or_algorithm="darkness_variability = std(darkness) over the whole foreground mask.",
            input_pixels_or_region="Whole-image foreground mask", image_scope="WHOLE_DRAWING",
            unit="normalized darkness, 0..1", expected_range="0.0 to ~0.5",
            higher_value_means="Inconsistent shading/ink density", lower_value_means="Consistent shading/ink density",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="Same as line.darkness_mean.", what_it_does_NOT_measure="Physical pressure variability.",
            reliability_status="stable for its stated purpose",
        ),
        _row(
            feature_id="line.continuity", display_name="Line continuity",
            exact_definition="Fraction of total foreground-mask area occupied by its single LARGEST 8-connected component.",
            formula_or_algorithm="continuity = max(connected_component_area) / total_foreground_area (cv2.connectedComponentsWithStats, connectivity=8).",
            input_pixels_or_region="Whole-image foreground mask", image_scope="WHOLE_DRAWING",
            unit="ratio, 0..1", expected_range="0.0 to 1.0",
            higher_value_means="One dominant continuous mark/structure", lower_value_means="Many separate, disconnected marks",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="Line thickness/gaps near the connectivity threshold; scan artifacts that break or merge strokes.",
            what_it_does_NOT_measure="Whether a SINGLE intended stroke was drawn continuously in one motion (process data, not recoverable from a static image).",
            reliability_status="stable for its stated purpose",
        ),
        _row(
            feature_id="line.fragmentation_rate", display_name="Fragmentation rate",
            exact_definition="Normalized count of separate 8-connected foreground components, capped at 1.0.",
            formula_or_algorithm="fragmentation_rate = min(1.0, component_count / max(1, total_foreground_area/50)).",
            input_pixels_or_region="Whole-image foreground mask", image_scope="WHOLE_DRAWING",
            unit="ratio, 0..1", expected_range="0.0 to 1.0",
            higher_value_means="Many small, disconnected marks", lower_value_means="Few, larger connected marks",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="The /50 normalization constant is a documented, uncalibrated scale choice, not a validated per-mark pixel budget.",
            what_it_does_NOT_measure="Intentional artistic fragmentation vs. accidental scan noise (cannot be distinguished).",
            reliability_status="uncalibrated normalization constant",
        ),
        _row(
            feature_id="line.straightness", display_name="Straightness",
            exact_definition="Fraction of Canny edge pixels (within the foreground mask) that lie on a Hough-detected near-straight line segment.",
            formula_or_algorithm=(
                "edges = Canny(gray, 50, 150) masked to foreground; lines = HoughLinesP(edges, rho=1, theta=pi/180, "
                "threshold=20, minLineLength=12, maxLineGap=3); straightness = |edge pixels covered by any drawn "
                "line segment (thickness 2)| / |total edge pixels|."),
            input_pixels_or_region="Canny edges within the whole-image foreground mask", image_scope="WHOLE_DRAWING",
            unit="ratio, 0..1", expected_range="0.0 to 1.0",
            higher_value_means="Predominantly straight-edged forms", lower_value_means="Predominantly curved/irregular forms",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="Canny thresholds (50/150) and Hough parameters are fixed, undocumented-elsewhere constants; short segments and heavy curvature reduce detected coverage even for intentionally straight strokes with jitter.",
            what_it_does_NOT_measure="Ruler-use or drawing tool.",
            reliability_status="fixed constants, not calibrated against a labelled dataset",
        ),
        _row(
            feature_id="line.dominant_orientation_code", display_name="Dominant orientation",
            exact_definition="The single most common STROKE (tangent) orientation bin, coded 0.0=horizontal, 1.0=vertical, 2.0=diagonal.",
            formula_or_algorithm=(
                "Sobel gradients gx,gy computed on greyscale; gradient magnitude-weighted pixels (>5% of max magnitude, within the "
                "foreground mask) are histogrammed into 18 ten-degree bins over angle=(atan2(gy,gx)+90 deg) mod 180 -- "
                "the +90 degree rotation converts the GRADIENT (edge-normal) direction Sobel actually measures into the "
                "STROKE's own tangent direction (a vertical stroke has a horizontal gradient and must read as 'vertical', "
                "not 'horizontal'). The bin with the largest total weight is the dominant orientation; its center angle "
                "is mapped to horizontal (<22.5 deg or >=157.5 deg), vertical (67.5-112.5 deg), or diagonal (else)."),
            input_pixels_or_region="Magnitude-weighted Sobel-gradient pixels within the whole-image foreground mask",
            image_scope="WHOLE_DRAWING", unit="categorical code {0.0, 1.0, 2.0}", expected_range="{0.0, 1.0, 2.0} or missing if no foreground",
            higher_value_means="(categorical -- not ordinal)", lower_value_means="(categorical -- not ordinal)",
            measurement_type="categorical", proxy_or_direct="direct",
            known_confounders="A handful of high-curvature corner pixels could bias the histogram if unweighted -- mitigated by magnitude-weighting (see notes).",
            what_it_does_NOT_measure="Intended drawing direction/motion (a static image has no time axis).",
            reliability_status="stable; verified by synthetic horizontal/vertical/diagonal tests",
            notes="The 90-degree gradient-to-stroke conversion is the single most important implementation detail of this "
                  "feature family -- documented explicitly in code (formal_features.py::_orientation_stats) and verified "
                  "by tests/test_formal_features.py::OrientationTests.",
        ),
        _row(
            feature_id="line.orientation_entropy", display_name="Orientation entropy",
            exact_definition="Shannon entropy of the (magnitude-weighted) stroke-orientation histogram, normalized to [0,1].",
            formula_or_algorithm="entropy = -sum(p_i * log2(p_i)) / log2(18), over the same 18-bin weighted histogram as line.dominant_orientation_code.",
            input_pixels_or_region="Same as line.dominant_orientation_code", image_scope="WHOLE_DRAWING",
            unit="normalized entropy, 0..1", expected_range="0.0 (single direction) to 1.0 (uniform across all 18 bins)",
            higher_value_means="Many competing stroke directions", lower_value_means="One dominant stroke direction",
            measurement_type="continuous", proxy_or_direct="direct",
            known_confounders="Same weighting/threshold choices as dominant orientation.",
            what_it_does_NOT_measure="Local (neighbourhood-level) direction agreement -- see orientation_coherence for that.",
            reliability_status="stable; verified by synthetic single-vs-multi-direction tests",
        ),
        _row(
            feature_id="line.orientation_coherence", display_name="Local orientation coherence",
            exact_definition="Mean structure-tensor coherence (lambda1-lambda2)/(lambda1+lambda2) over the same weighted pixel set.",
            formula_or_algorithm=(
                "Structure tensor J = [[gx*gx, gx*gy],[gx*gy, gy*gy]] Gaussian-blurred (5x5, sigma=0); "
                "lambda1,2 = trace/2 +/- sqrt(((Jxx-Jyy)/2)^2 + Jxy^2); coherence_map = (lambda1-lambda2)/(lambda1+lambda2); "
                "orientation_coherence = mean(coherence_map) over the same magnitude-weighted pixel set used for entropy."),
            input_pixels_or_region="Same as line.dominant_orientation_code", image_scope="WHOLE_DRAWING",
            unit="ratio, 0..1", expected_range="0.0 (locally isotropic/multidirectional) to 1.0 (perfectly locally aligned)",
            higher_value_means="Strokes locally agree in direction with their immediate neighbours", lower_value_means="Locally multidirectional/scribbled",
            measurement_type="continuous", proxy_or_direct="direct",
            known_confounders="A GLOBAL entropy diversity (many differently-oriented but individually straight strokes, "
                              "sparsely spaced) can coexist with HIGH local coherence -- entropy and coherence measure "
                              "genuinely different things and must not be conflated.",
            what_it_does_NOT_measure="Global direction diversity (that is orientation_entropy's job).",
            reliability_status="stable; verified by synthetic single-direction-vs-dense-crosshatch test",
        ),
        _row(
            feature_id="line.direction_cluster_count", display_name="Direction-cluster count",
            exact_definition="Count of orientation histogram bins whose weight exceeds 30% of the single largest bin's weight.",
            formula_or_algorithm="cluster_count = count(bin_probability > 0.3 * max(bin_probability)), over the same 18-bin histogram.",
            input_pixels_or_region="Same as line.dominant_orientation_code", image_scope="WHOLE_DRAWING",
            unit="count (integer-valued float)", expected_range="1 to 18",
            higher_value_means="Multiple distinct dominant directions present", lower_value_means="One (or few) dominant direction(s)",
            measurement_type="discrete", proxy_or_direct="proxy",
            known_confounders="The 30%-of-max threshold is a documented, uncalibrated constant.",
            what_it_does_NOT_measure="Spatial location of each direction cluster (a purely histogram-level count).",
            reliability_status="uncalibrated threshold constant",
        ),
        _row(
            feature_id="stroke.junction_corner_density_proxy", display_name="Junction/corner density (proxy)",
            exact_definition="Density of Harris corner-response peaks over the foreground mask, normalized by foreground area.",
            formula_or_algorithm="response = cv2.cornerHarris(mask, blockSize=5, ksize=3, k=0.04); corners = response > 0.01*max(response); density = count(corners)/foreground_area.",
            input_pixels_or_region="Whole-image foreground mask treated as a binary image", image_scope="WHOLE_DRAWING",
            unit="corner-responses per foreground pixel", expected_range="0.0 to ~1.0 (small drawings can have a high ratio)",
            higher_value_means="Many line crossings, corners, junctions, or sharp bends", lower_value_means="Few crossings/corners (smooth, mostly-parallel strokes)",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="cornerHarris cannot distinguish a TRUE line-crossing (X/T junction) from a sharp single-stroke bend or a small shape's own corner -- all three trigger the same response.",
            what_it_does_NOT_measure="True skeleton-graph line-crossing count (needs skeletonization/vectorization -- not available in this OpenCV build, no ximgproc/scikit-image).",
            reliability_status="proxy only -- renamed from 'crossing density' in Phase G0.1 Step 4D to avoid overclaiming",
            notes="Renamed from stroke.crossing_density. True crossing detection is deferred to E-Graphic.",
        ),
        _row(
            feature_id="stroke.density_mean", display_name="Stroke-density map (mean)",
            exact_definition="Fraction of the whole image occupied by the foreground mask.",
            formula_or_algorithm="stroke_density_mean = mean(foreground_mask) over the entire image (mask.mean()).",
            input_pixels_or_region="Whole image", image_scope="WHOLE_DRAWING", unit="ratio, 0..1", expected_range="0.0 to 1.0",
            higher_value_means="Denser/more crowded drawing", lower_value_means="Sparser drawing",
            measurement_type="continuous", proxy_or_direct="direct",
            known_confounders="Same as composition.foreground_coverage (analysis.py) -- this is that same measurement re-exposed under the formal-feature naming scheme, not a second independent computation.",
            what_it_does_NOT_measure="Semantic content density (number of distinct objects) -- purely a pixel-fraction measurement.",
            reliability_status="stable (direct pixel-fraction computation)",
        ),
        _row(
            feature_id="stroke.scribble_candidate_score", display_name="Scribble candidate score",
            exact_definition="Exploratory composite: stroke_density_mean * orientation_entropy.",
            formula_or_algorithm="scribble_candidate_score = stroke.density_mean * line.orientation_entropy (a simple product of two already-computed values -- no other weighting).",
            input_pixels_or_region="Derived from stroke.density_mean and line.orientation_entropy", image_scope="WHOLE_DRAWING",
            unit="unitless composite, 0..1", expected_range="0.0 to 1.0",
            higher_value_means="Dense AND multidirectional marks (candidate for 'scribble')", lower_value_means="Sparse or single-direction marks",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="A dense but perfectly single-direction hatch pattern scores LOW despite being visually "
                              "dense; this composite has NOT been validated against any labelled scribble dataset.",
            what_it_does_NOT_measure=(
                "A true, validated scribble detector -- the PDF's own Appendix G note lists local coherence, "
                "short-stroke density, curvature, and intersections as additional components a real detector would "
                "need; none of that additional weighting is invented here."),
            reliability_status="EXPLORATORY -- renamed from 'scribble density' in Phase G0.1 Step 4C to avoid implying a validated detector",
            notes="Renamed from stroke.scribble_density.",
        ),
        _row(
            feature_id="colour.chromatic_coverage", display_name="Chromatic colour coverage",
            exact_definition="Fraction of foreground pixels that are chromatically saturated (neither near-black, near-white, nor grey).",
            formula_or_algorithm=(
                "HSV-converted foreground pixels; chromatic = (saturation > 0.15) AND (0.08 < value < 0.98) "
                "[0..1-normalized HSV]; chromatic_coverage = mean(chromatic) over the foreground mask."),
            input_pixels_or_region="Whole-image foreground mask, HSV colour space", image_scope="WHOLE_DRAWING",
            unit="ratio, 0..1", expected_range="0.0 (fully achromatic) to 1.0 (fully saturated colour)",
            higher_value_means="More of the drawing uses saturated colour", lower_value_means="More of the drawing is black/grey/white",
            measurement_type="continuous", proxy_or_direct="direct",
            known_confounders="The 0.15 saturation threshold and 0.08/0.98 value bounds are documented, UNCALIBRATED constants (exploratory pending later calibration); scan colour-cast can shift measured saturation.",
            what_it_does_NOT_measure=(
                "composition.foreground_coverage (how much of the page has ANY mark) -- these are genuinely "
                "different measurements and must never be conflated; a full-page graphite drawing has high "
                "foreground_coverage but near-zero chromatic_coverage."),
            reliability_status="EXPLORATORY -- thresholds not calibrated against a labelled dataset",
            notes="Renamed/fixed in Phase G0.1 Step 4B -- previously this key literally reused composition.foreground_coverage, "
                  "which was the exact conflation this fix addresses.",
        ),
        _row(
            feature_id="colour.diversity", display_name="Colour diversity",
            exact_definition="Count of the fixed colour bins (red/green/blue/yellow/dark) whose pixel proportion exceeds 2% of the foreground -- reused directly from analysis.py, not recomputed.",
            formula_or_algorithm="Reused verbatim from analysis['colour']['colour_diversity'] (see analysis.py::_colour).",
            input_pixels_or_region="Whole-image foreground mask (computed by analysis.py, not this module)", image_scope="WHOLE_DRAWING",
            unit="count, 0..5", expected_range="0 to 5",
            higher_value_means="More distinct colour bins meaningfully present", lower_value_means="Monochrome/near-monochrome drawing",
            measurement_type="discrete", proxy_or_direct="direct",
            known_confounders="Only 5 coarse colour bins exist (red/green/blue/yellow/dark) -- cannot distinguish finer palette differences.",
            what_it_does_NOT_measure="Chromatic saturation strength (see colour.chromatic_coverage).",
            reliability_status="stable (deduplicated reuse of an existing, already-tested computation)",
        ),
        _row(
            feature_id="colour.brightness", display_name="Brightness/darkness palette",
            exact_definition="Mean greyscale brightness over the WHOLE image (including background/paper).",
            formula_or_algorithm="brightness = mean(grayscale_image) / 255.",
            input_pixels_or_region="Whole image (not restricted to foreground)", image_scope="WHOLE_DRAWING",
            unit="normalized brightness, 0..1", expected_range="0.0 (all black) to 1.0 (all white)",
            higher_value_means="Lighter overall palette (more white paper visible / lighter marks)", lower_value_means="Darker overall palette",
            measurement_type="continuous", proxy_or_direct="direct",
            known_confounders="Dominated by paper/background colour when foreground coverage is low; scan exposure/white balance.",
            what_it_does_NOT_measure="Foreground-only darkness (see line.darkness_mean for that).",
            reliability_status="stable (direct pixel statistic)",
        ),
        _row(
            feature_id="colour.saturation", display_name="Saturation",
            exact_definition="Mean HSV saturation over foreground pixels only.",
            formula_or_algorithm="saturation = mean(HSV_saturation_channel[foreground_mask]) / 255.",
            input_pixels_or_region="Whole-image foreground mask, HSV colour space", image_scope="WHOLE_DRAWING",
            unit="normalized saturation, 0..1", expected_range="0.0 (grey/achromatic) to 1.0 (fully saturated)",
            higher_value_means="More vivid colour in the marks", lower_value_means="More muted/grey marks",
            measurement_type="continuous", proxy_or_direct="direct",
            known_confounders="Scan/photo colour cast; JPEG compression can shift saturation slightly.",
            what_it_does_NOT_measure="Coverage/extent of colour (see colour.chromatic_coverage).",
            reliability_status="stable (direct pixel statistic)",
        ),
        _row(
            feature_id="fill.global_spatial_density_uniformity", display_name="Global spatial density uniformity",
            exact_definition="Inverse-variance uniformity of foreground density across an 8x8 grid of the WHOLE image.",
            formula_or_algorithm="Divide the image into an 8x8 grid; density_i = mean(mask) in cell i; uniformity = 1 / (1 + variance(density_i) / mean(density_i)^2).",
            input_pixels_or_region="Whole image, partitioned into a fixed 8x8 grid", image_scope="WHOLE_DRAWING",
            unit="unitless ratio, 0..1", expected_range="0.0 (very patchy) to 1.0 (perfectly even density across the grid)",
            higher_value_means="Marks spread evenly across the page", lower_value_means="Marks concentrated in a few regions",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="A fixed 8x8 grid is a documented, uncalibrated resolution choice; very small drawings may not fill even one grid cell.",
            what_it_does_NOT_measure=(
                "region_fill_uniformity (how evenly ONE specific object's interior is coloured) -- that needs a "
                "per-object segmentation mask DOAR does not yet have, and stays EXPERIMENT_CANDIDATE."),
            reliability_status="uncalibrated grid resolution",
            notes="Renamed from fill.uniformity in Phase G0.1 Step 4E to make the whole-drawing vs. per-object distinction explicit.",
        ),
        _row(
            feature_id="symmetry.bilateral", display_name="Bilateral symmetry",
            exact_definition="1 minus the fraction of foreground-mask pixels that disagree between the image and its horizontal mirror.",
            formula_or_algorithm="symmetry = 1 - mean(XOR(mask, fliplr(mask))).",
            input_pixels_or_region="Whole-image foreground mask", image_scope="WHOLE_DRAWING",
            unit="ratio, 0..1", expected_range="0.0 (fully asymmetric) to 1.0 (perfectly left-right symmetric)",
            higher_value_means="More left-right symmetric composition", lower_value_means="More asymmetric composition",
            measurement_type="continuous", proxy_or_direct="direct",
            known_confounders="Only tests LEFT-RIGHT (vertical-axis) symmetry, never top-bottom or rotational symmetry; a shape drawn off-center scores low even if intrinsically symmetric.",
            what_it_does_NOT_measure="Symmetry of one specific object independent of its position on the page (needs a per-object mask -- see 'Local symmetry', EXPERIMENT_CANDIDATE).",
            reliability_status="stable (direct pixel computation)",
        ),
        _row(
            feature_id="symmetry.spacing_regularity", display_name="Spacing regularity",
            exact_definition="Inverse coefficient-of-variation of nearest-neighbour centroid distances across all foreground connected components.",
            formula_or_algorithm=(
                "Requires >=3 connected components (else missing). For each component centroid, find its nearest "
                "OTHER centroid's distance; regularity = 1 / (1 + std(nearest_distances)/mean(nearest_distances))."),
            input_pixels_or_region="Centroids of all 8-connected foreground components", image_scope="WHOLE_DRAWING",
            unit="unitless ratio, 0..1", expected_range="0.0 (irregular spacing) to 1.0 (perfectly even spacing)",
            higher_value_means="Repeated elements are evenly spaced (e.g. a grid)", lower_value_means="Repeated elements are irregularly spaced",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="Requires at least 3 separate components; a single connected drawing (e.g. one continuous outline) cannot be measured (honestly reported missing).",
            what_it_does_NOT_measure="Spacing of parts WITHIN one object (only between separate components).",
            reliability_status="stable for its stated purpose",
        ),
        _row(
            feature_id="scene.detail_density", display_name="Detail density",
            exact_definition="Fraction of foreground-mask pixels that are also Canny edge pixels.",
            formula_or_algorithm="edges = Canny(gray, 50, 150); detail_density = mean(edges > 0 restricted to the foreground mask).",
            input_pixels_or_region="Whole-image foreground mask", image_scope="WHOLE_DRAWING",
            unit="ratio, 0..1", expected_range="0.0 to 1.0",
            higher_value_means="More fine detail/edges relative to the marked area", lower_value_means="Simpler, less-detailed marks",
            measurement_type="continuous", proxy_or_direct="proxy",
            known_confounders="Canny thresholds (50/150) are fixed, uncalibrated constants; heavy shading raises this even without added semantic detail.",
            what_it_does_NOT_measure="Semantic object/detail COUNT (a purely edge-density statistic, not an object detector).",
            reliability_status="fixed constants, not calibrated against a labelled dataset",
        ),
    ]


def write_formal_feature_definition_table() -> Path:
    rows = build_formal_feature_definition_table()
    with open(FORMAL_FEATURE_DEFINITION_TABLE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return FORMAL_FEATURE_DEFINITION_TABLE_PATH


if __name__ == "__main__":
    path = write_formal_feature_definition_table()
    print(f"wrote {path}")
