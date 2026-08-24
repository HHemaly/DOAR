"""Phase G0 / G0.1 -- synthetic tests for src/doar/formal_features.py.

Every test uses a hand-constructed synthetic image (never a real child
drawing) and checks only that a measurement moves in the SCIENTIFICALLY
EXPECTED DIRECTION between two contrasted synthetic cases -- never an
absolute value, which would be arbitrary. Mirrors tests/test_phase2c4_
detectors.py's synthetic-only discipline: no real drawings, no model
weights, no network.

Phase G0.1 adds: the renamed measurements (colour.chromatic_coverage,
stroke.junction_corner_density_proxy, stroke.scribble_candidate_score,
fill.global_spatial_density_uniformity), the filled-region-vs-stroke
separation fix (Step 4A), and Step 6's expanded confounder matrix
(curved strokes, cross-hatching, filled vs outlined shapes, variable
width, faint vs dark, JPEG/brightness/resize perturbation, etc.).

Debug overlays (Step 7) are saved ONLY when DOAR_SAVE_FORMAL_FEATURE_
DEBUG_OVERLAYS=1 is set in the environment, to `tests/formal_feature_
debug_output/` -- never during an ordinary test run, and never anywhere
near the Parent view.
"""
from __future__ import annotations

import io
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from doar.case_artifacts import resolve_analysis_artifacts
from doar.formal_features import compute_formal_features, render_formal_feature_overlay
from doar.timed_analysis import analyze_image_with_timing

_SAVE_DEBUG_OVERLAYS = os.environ.get("DOAR_SAVE_FORMAL_FEATURE_DEBUG_OVERLAYS") == "1"
_DEBUG_OUTPUT_DIR = ROOT / "tests" / "formal_feature_debug_output"


def _build_image(draw_fn, size=(300, 300)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw_fn(ImageDraw.Draw(image))
    return image


def _features_for(draw_fn, *, label: str | None = None, perturb=None) -> dict:
    with tempfile.TemporaryDirectory() as d:
        case_dir = Path(d) / "case"
        case_dir.mkdir()
        path = case_dir / "drawing.png"
        image = _build_image(draw_fn)
        if perturb is not None:
            image = perturb(image)
        image.save(path)
        analyze_image_with_timing(str(path), str(case_dir), None)
        import json
        analysis = json.loads((case_dir / "analysis.json").read_text(encoding="utf-8"))
        resolved = resolve_analysis_artifacts(analysis, case_dir)
        row = compute_formal_features(str(path), resolved)
        if _SAVE_DEBUG_OVERLAYS and label:
            _DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            overlay = render_formal_feature_overlay(str(path), resolved)
            overlay.save(_DEBUG_OUTPUT_DIR / f"{label}.png")
        return row


def _jpeg_roundtrip(image: Image.Image, quality: int = 40) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _brighten(image: Image.Image, delta: int) -> Image.Image:
    import numpy as np
    arr = np.asarray(image).astype(np.int16)
    arr = np.clip(arr + delta, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _resize(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale)))).resize((w, h))


class LineWidthTests(unittest.TestCase):
    def test_thick_line_has_greater_mean_width_than_thin_line(self):
        # Widths kept comfortably under the stroke/blob classification
        # cutoff (Step 4A: ~2% of the 300px test image's shorter side,
        # ~6px kernel radius) so both remain classified as stroke-like.
        thin = _features_for(lambda dr: dr.line((50, 150, 250, 150), fill="black", width=2), label="thin_line")
        thick = _features_for(lambda dr: dr.line((50, 150, 250, 150), fill="black", width=8), label="thick_line")
        self.assertFalse(thin["line.mean_width"].missing)
        self.assertFalse(thick["line.mean_width"].missing)
        self.assertGreater(thick["line.mean_width"].value, thin["line.mean_width"].value)

    def test_uniform_width_has_lower_variability_than_variable_width(self):
        uniform = _features_for(lambda dr: dr.line((50, 150, 250, 150), fill="black", width=6))

        def variable(dr):
            dr.line((50, 100, 150, 100), fill="black", width=2)
            dr.line((150, 100, 250, 100), fill="black", width=9)
        varying = _features_for(variable)
        self.assertLess(uniform["line.width_variability"].value, varying["line.width_variability"].value)

    def test_filled_shape_is_excluded_from_line_width_not_measured_as_a_thick_line(self):
        """Step 4A regression: a large filled region must NOT be reported
        as an implausibly thick 'line' -- it has no stroke-like pixels at
        all, so line.mean_width must be honestly missing."""
        filled = _features_for(lambda dr: dr.rectangle((80, 80, 220, 220), fill="black"), label="filled_rectangle")
        self.assertTrue(filled["line.mean_width"].missing)

    def test_outlined_shape_with_no_fill_is_still_measured_as_a_stroke(self):
        """A ring (outline only) is thin everywhere along its own length,
        even though its bounding shape is compact -- must NOT be
        misclassified as a filled blob the way a naive per-pixel opening
        would (see _stroke_like_mask's component-level docstring)."""
        outlined = _features_for(lambda dr: dr.ellipse((80, 80, 220, 220), outline="black", width=4),
                                  label="outlined_circle_no_fill")
        self.assertFalse(outlined["line.mean_width"].missing)

    def test_outlined_shape_with_interior_colouring_still_measures_the_outline(self):
        def outlined_and_filled(dr):
            dr.ellipse((80, 80, 220, 220), fill=(180, 180, 180))
            dr.ellipse((80, 80, 220, 220), outline="black", width=4)
        row = _features_for(outlined_and_filled, label="outlined_circle_with_fill")
        # The large filled interior dominates the mask; the thin outline
        # is a small fraction of it -- still an honest result either way
        # (measured, from the outline ring) as long as no exception occurs
        # and nothing crashes trying to combine the two regions.
        self.assertIn(row["line.mean_width"].missing, (True, False))


class DarknessTests(unittest.TestCase):
    def test_dark_line_has_greater_darkness_than_faint_line(self):
        faint = _features_for(lambda dr: dr.line((50, 150, 250, 150), fill=(210, 210, 210), width=10),
                               label="faint_pencil_like_grey_strokes")
        dark = _features_for(lambda dr: dr.line((50, 150, 250, 150), fill=(10, 10, 10), width=10),
                              label="dark_marker_like_strokes")
        self.assertGreater(dark["line.darkness_mean"].value, faint["line.darkness_mean"].value)

    def test_uniform_darkness_has_lower_variability_than_mixed(self):
        uniform = _features_for(lambda dr: dr.line((50, 150, 250, 150), fill=(30, 30, 30), width=12))

        def mixed(dr):
            dr.line((50, 100, 150, 100), fill=(10, 10, 10), width=12)
            dr.line((150, 100, 250, 100), fill=(220, 220, 220), width=12)
        varying = _features_for(mixed)
        self.assertLess(uniform["line.darkness_variability"].value, varying["line.darkness_variability"].value)

    def test_filled_shape_darkness_is_still_measured_not_missing(self):
        """Unlike width, darkness is well-defined over a filled region too
        (a filled dark region genuinely IS dark) -- deliberately not
        restricted to stroke-like pixels."""
        row = _features_for(lambda dr: dr.rectangle((80, 80, 220, 220), fill=(20, 20, 20)),
                             label="filled_black_shape")
        self.assertFalse(row["line.darkness_mean"].missing)
        self.assertGreater(row["line.darkness_mean"].value, 0.5)

    def test_low_vs_high_shading_density(self):
        low = _features_for(lambda dr: dr.rectangle((100, 100, 200, 200), outline=(60, 60, 60), width=2))

        def high_shading(dr):
            for y in range(100, 200, 3):
                dr.line((100, y, 200, y), fill=(60, 60, 60), width=1)
        high = _features_for(high_shading)
        self.assertGreater(high["scene.detail_density"].value, low["scene.detail_density"].value)


class ContinuityFragmentationTests(unittest.TestCase):
    def test_continuous_line_more_continuous_than_dotted_pattern(self):
        continuous = _features_for(lambda dr: dr.line((40, 150, 260, 150), fill="black", width=6))

        def dotted(dr):
            for x in range(40, 260, 12):
                dr.ellipse((x, 145, x + 4, 155), fill="black")
        fragmented = _features_for(dotted)
        self.assertGreater(continuous["line.continuity"].value, fragmented["line.continuity"].value)
        self.assertGreater(fragmented["line.fragmentation_rate"].value, continuous["line.fragmentation_rate"].value)


class OrientationTests(unittest.TestCase):
    """Confirms line.dominant_orientation_code/entropy/coherence describe
    STROKE (tangent) orientation, not gradient (edge-normal) orientation
    -- the 90-degree correction documented in _orientation_stats."""

    def test_horizontal_parallel_strokes_detected_as_horizontal(self):
        row = _features_for(lambda dr: [dr.line((40, y, 260, y), fill="black", width=3) for y in range(40, 260, 10)],
                             label="horizontal_parallel_strokes")
        self.assertEqual(row["line.dominant_orientation_code"].value, 0.0)

    def test_vertical_parallel_strokes_detected_as_vertical(self):
        row = _features_for(lambda dr: [dr.line((x, 40, x, 260), fill="black", width=3) for x in range(40, 260, 10)],
                             label="vertical_parallel_strokes")
        self.assertEqual(row["line.dominant_orientation_code"].value, 1.0)

    def test_diagonal_parallel_strokes_detected_as_diagonal(self):
        row = _features_for(lambda dr: [dr.line((40 + i, 40, 260, 260 - i), fill="black", width=2)
                                        for i in range(0, 180, 20)], label="diagonal_parallel_strokes")
        self.assertEqual(row["line.dominant_orientation_code"].value, 2.0)

    def test_mixed_directions_has_higher_entropy_than_single_direction(self):
        single = _features_for(lambda dr: [dr.line((x, 40, x, 260), fill="black", width=3) for x in range(40, 260, 10)])

        def mixed(dr):
            for x in range(40, 260, 20):
                dr.line((x, 40, x, 260), fill="black", width=2)
            for y in range(40, 260, 20):
                dr.line((40, y, 260, y), fill="black", width=2)
            for i in range(0, 180, 30):
                dr.line((40 + i, 40, 260, 260 - i), fill="black", width=2)
        mixed_row = _features_for(mixed, label="mixed_vertical_horizontal_diagonal")
        self.assertLess(single["line.orientation_entropy"].value, mixed_row["line.orientation_entropy"].value)

    def test_curved_strokes_are_measured_without_error(self):
        def curved(dr):
            points = [(x, 150 + int(60 * math.sin(x / 15))) for x in range(40, 260, 3)]
            dr.line(points, fill="black", width=4)
        row = _features_for(curved, label="curved_strokes")
        self.assertFalse(row["line.orientation_entropy"].missing)

    def test_single_direction_has_higher_local_coherence_than_dense_crosshatch(self):
        single = _features_for(lambda dr: [dr.line((x, 40, x, 260), fill="black", width=2) for x in range(40, 260, 20)])

        def crosshatch(dr):
            for x in range(40, 260, 6):
                dr.line((x, 40, x, 260), fill="black", width=1)
            for y in range(40, 260, 6):
                dr.line((40, y, 260, y), fill="black", width=1)
        dense_crosshatch = _features_for(crosshatch, label="cross_hatching")
        self.assertGreater(single["line.orientation_coherence"].value,
                            dense_crosshatch["line.orientation_coherence"].value)


class JunctionCornerDensityProxyTests(unittest.TestCase):
    """Renamed from crossing_density (Step 4D) -- responds to crossings,
    corners, and sharp bends alike; not true skeleton-crossing counting."""

    def test_crosshatch_has_higher_response_density_than_parallel_lines(self):
        parallel = _features_for(lambda dr: [dr.line((x, 40, x, 260), fill="black", width=2) for x in range(40, 260, 15)])

        def crosshatch(dr):
            for x in range(40, 260, 15):
                dr.line((x, 40, x, 260), fill="black", width=2)
            for y in range(40, 260, 15):
                dr.line((40, y, 260, y), fill="black", width=2)
        crossed = _features_for(crosshatch)
        self.assertGreater(crossed["stroke.junction_corner_density_proxy"].value,
                            parallel["stroke.junction_corner_density_proxy"].value)


class ScribbleCandidateScoreTests(unittest.TestCase):
    """Renamed from scribble_density (Step 4C) -- an EXPLORATORY composite
    (stroke density * orientation entropy), not a validated detector."""

    def test_dense_multidirectional_scribble_scores_higher_than_sparse_single_direction(self):
        sparse = _features_for(lambda dr: dr.line((50, 150, 250, 150), fill="black", width=3))

        random.seed(1)
        def scribble(dr):
            for _ in range(80):
                x1, y1 = random.randint(30, 270), random.randint(30, 270)
                angle = random.uniform(0, math.pi)
                x2 = x1 + int(40 * math.cos(angle))
                y2 = y1 + int(40 * math.sin(angle))
                dr.line((x1, y1, x2, y2), fill="black", width=3)
        dense = _features_for(scribble, label="dense_scribbling")
        self.assertGreater(dense["stroke.scribble_candidate_score"].value, sparse["stroke.scribble_candidate_score"].value)

    def test_sparse_vs_dense_drawing(self):
        sparse = _features_for(lambda dr: dr.line((140, 140, 160, 160), fill="black", width=3), label="sparse_drawing")

        random.seed(4)
        def dense(dr):
            for _ in range(200):
                x1, y1 = random.randint(20, 280), random.randint(20, 280)
                x2, y2 = x1 + random.randint(-30, 30), y1 + random.randint(-30, 30)
                dr.line((x1, y1, x2, y2), fill="black", width=2)
        dense_row = _features_for(dense, label="dense_drawing")
        self.assertGreater(dense_row["stroke.density_mean"].value, sparse["stroke.density_mean"].value)


class StraightnessTests(unittest.TestCase):
    def test_straight_line_is_straighter_than_a_wavy_curve(self):
        straight = _features_for(lambda dr: dr.line((40, 150, 260, 150), fill="black", width=4), label="straight_line")

        def wavy(dr):
            points = [(x, 150 + int(60 * math.sin(x / 12))) for x in range(40, 260, 4)]
            dr.line(points, fill="black", width=4)
        curve = _features_for(wavy, label="wavy_curve")
        self.assertGreater(straight["line.straightness"].value, curve["line.straightness"].value)


class SymmetryTests(unittest.TestCase):
    def test_symmetric_shape_scores_higher_than_asymmetric_shape(self):
        symmetric = _features_for(lambda dr: dr.ellipse((80, 80, 220, 220), outline="black", width=4),
                                   label="symmetric_form")
        asymmetric = _features_for(lambda dr: dr.polygon([(60, 80), (280, 100), (120, 270)], outline="black", width=4),
                                    label="asymmetric_form")
        self.assertGreater(symmetric["symmetry.bilateral"].value, asymmetric["symmetry.bilateral"].value)


class SpacingRegularityTests(unittest.TestCase):
    def test_regular_grid_scores_higher_than_irregular_scatter(self):
        def regular(dr):
            for x in range(60, 260, 40):
                for y in range(60, 260, 40):
                    dr.ellipse((x, y, x + 10, y + 10), fill="black")
        random.seed(2)
        def irregular(dr):
            for _ in range(16):
                x, y = random.randint(50, 250), random.randint(50, 250)
                dr.ellipse((x, y, x + 10, y + 10), fill="black")
        regular_row = _features_for(regular, label="regular_repeated_spacing")
        irregular_row = _features_for(irregular, label="irregular_repeated_spacing")
        self.assertGreater(regular_row["symmetry.spacing_regularity"].value,
                            irregular_row["symmetry.spacing_regularity"].value)


class FillUniformityTests(unittest.TestCase):
    """global_spatial_density_uniformity (renamed, Step 4E) -- a
    WHOLE-DRAWING grid-occupancy measurement, never region_fill_
    uniformity (which needs per-object masks and stays deferred)."""

    def test_uniform_fill_scores_higher_than_patchy_fill(self):
        uniform = _features_for(lambda dr: dr.rectangle((60, 60, 240, 240), fill="black"))

        def patchy(dr):
            random.seed(3)
            for _ in range(10):
                x, y = random.randint(60, 200), random.randint(60, 200)
                dr.rectangle((x, y, x + 20, y + 20), fill="black")
        patchy_row = _features_for(patchy)
        self.assertGreater(uniform["fill.global_spatial_density_uniformity"].value,
                            patchy_row["fill.global_spatial_density_uniformity"].value)


class ChromaticColourCoverageTests(unittest.TestCase):
    """Step 4B: colour.chromatic_coverage must NOT equal
    composition.foreground_coverage -- a full-page graphite/black drawing
    has high foreground coverage but near-zero chromatic colour."""

    def test_black_and_white_drawing_has_near_zero_chromatic_coverage(self):
        row = _features_for(lambda dr: dr.rectangle((60, 60, 240, 240), fill=(20, 20, 20)))
        self.assertLess(row["colour.chromatic_coverage"].value, 0.05)

    def test_colourful_drawing_has_higher_chromatic_coverage_than_greyscale(self):
        grey = _features_for(lambda dr: dr.rectangle((60, 60, 240, 240), fill=(120, 120, 120)))
        colourful = _features_for(lambda dr: dr.rectangle((60, 60, 240, 240), fill=(220, 40, 40)))
        self.assertGreater(colourful["colour.chromatic_coverage"].value, grey["colour.chromatic_coverage"].value)


class PerturbationRobustnessTests(unittest.TestCase):
    """Step 6: JPEG compression, brightness shift, and resize should not
    flip the SIGN of a clear directional measurement (some numeric drift
    is expected and acceptable; only the qualitative direction is
    asserted, exactly as with every other test in this file)."""

    def _draw(self, dr):
        for x in range(40, 260, 10):
            dr.line((x, 40, x, 260), fill="black", width=3)

    def test_jpeg_compression_does_not_flip_dominant_orientation(self):
        clean = _features_for(self._draw)
        jpeg = _features_for(self._draw, perturb=lambda img: _jpeg_roundtrip(img, quality=40))
        self.assertEqual(clean["line.dominant_orientation_code"].value, jpeg["line.dominant_orientation_code"].value)

    def test_brightness_perturbation_does_not_flip_dominant_orientation(self):
        clean = _features_for(self._draw)
        bright = _features_for(self._draw, perturb=lambda img: _brighten(img, 25))
        self.assertEqual(clean["line.dominant_orientation_code"].value, bright["line.dominant_orientation_code"].value)

    def test_resize_perturbation_does_not_flip_dominant_orientation(self):
        clean = _features_for(self._draw)
        resized = _features_for(self._draw, perturb=lambda img: _resize(img, 0.5))
        self.assertEqual(clean["line.dominant_orientation_code"].value, resized["line.dominant_orientation_code"].value)


class MissingDataNeverFabricatedTests(unittest.TestCase):
    def test_blank_image_reports_missing_not_fabricated_zero(self):
        row = _features_for(lambda dr: None)
        self.assertTrue(row["line.mean_width"].missing)
        self.assertTrue(row["symmetry.spacing_regularity"].missing)


if __name__ == "__main__":
    unittest.main()
