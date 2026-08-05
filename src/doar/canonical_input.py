"""Canonical input preparation (DOAR-TRACE Phase 2A.1, Section 6).

Phase 2A's transformation-robustness testing
(`docs/FEATURE_ROBUSTNESS_RESULTS.md`) found `segmentation.foreground_coverage`,
`composition.symmetry`, and `colour.dark_ratio` are measurably sensitive
to plain **resize** (both up and down) -- an artifact of resampling
changing which pixels get classified as foreground at mask boundaries,
not evidence about the child's drawing. Two images of the identical
physical drawing, captured at different camera resolutions, should not
score meaningfully differently on these features; today they can.

**What this module does NOT do**: it never adjusts brightness, contrast,
colour, shadow, or JPEG-compression characteristics. Per the task's
explicit instruction, those are themselves evidence sources (line
darkness/thickness, colour choice, contrast are exactly what several
rules and features measure) -- silently "cleaning" them before feature
extraction would destroy the evidence those features exist to capture.
Only geometry/resolution is normalized, and only for the specific
features Phase 2A's own real measurements showed were resize-sensitive.

**Design**: every existing feature in `objective_features` (60 features,
`features.py`) continues to be computed from the ORIGINAL, unmodified
upload -- nothing about the existing pipeline changes. This module adds
a SEPARATE, smaller set of "canonical" (resolution-normalized)
counterparts for exactly the features found resize-sensitive, computed
from a canonicalized copy of the image and stored under
`Analysis.canonical_features` -- never merged into or confused with
`objective_features`. See `docs/INPUT_NORMALIZATION_POLICY.md` for which
version each feature uses.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from PIL import Image

CANONICAL_INPUT_SCHEMA_VERSION = "canonical_input_v1"

# Longer-side target, in pixels. Chosen as a round, generous number well
# above this dataset's typical working resolution (verified against the
# quality gate's own MIN_DIMENSION_PX=100 floor) -- large enough that
# downscaling to it does not discard real drawing detail, small enough
# to bound resampling cost. NEVER upscales: a small source image is left
# at its native resolution rather than fabricating detail that was never
# captured.
CANONICAL_MAX_DIMENSION = 1600

# The features Phase 2A's real transformation-invariance testing
# (docs/FEATURE_ROBUSTNESS_RESULTS.md) found measurably resize-sensitive
# -- the only features this module recomputes on the canonical image.
# Not an arbitrary subset: every one of these appears in the 21 real
# fails table under a resize_* transform.
CANONICAL_GEOMETRY_FEATURE_IDS = frozenset({
    "segmentation.foreground_coverage",
    "segmentation.bounding_box_coverage",
    "composition.symmetry",
    "colour.dark_ratio",
})


def canonicalize_image(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Resizes ONLY if the image exceeds `CANONICAL_MAX_DIMENSION` on its
    longer side, preserving aspect ratio, using a high-quality resample
    filter. Never upscales. Never touches brightness/contrast/colour
    mode/compression. Returns the (possibly unchanged) image and a
    record of what was done."""
    longer_side = max(image.width, image.height)
    if longer_side <= CANONICAL_MAX_DIMENSION:
        return image, {
            "resized": False, "original_size": [image.width, image.height],
            "canonical_size": [image.width, image.height],
            "scale_factor": 1.0,
        }
    scale = CANONICAL_MAX_DIMENSION / longer_side
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.LANCZOS)
    return resized, {
        "resized": True, "original_size": [image.width, image.height],
        "canonical_size": list(new_size), "scale_factor": round(scale, 6),
    }


def compute_canonical_geometry_features(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """`rgb`/`mask` are the canonicalized image's own array and the
    foreground mask `_segment` produced FOR THAT canonicalized image
    (never the original-resolution mask reused at a different scale --
    that would silently mix two coordinate systems). Returns
    `FeatureValue`-shaped dicts (via `features.FeatureValue`) for exactly
    `CANONICAL_GEOMETRY_FEATURE_IDS`, reusing the same real formulas
    `features.py`/`analysis.py::_composition` use for the original-image
    versions -- this module never invents a different definition, only a
    different input resolution."""
    from .analysis import _composition
    from .features import FeatureValue

    comp = _composition(mask)
    fg = rgb[mask]
    dark_ratio = float((fg.mean(axis=1) < 80).mean()) if len(fg) else 0.0
    symmetry = float(1 - np.logical_xor(mask, np.fliplr(mask)).mean())

    values = {
        "segmentation.foreground_coverage": comp["foreground_coverage"],
        "segmentation.bounding_box_coverage": comp["bounding_box_coverage"],
        "composition.symmetry": symmetry,
        "colour.dark_ratio": dark_ratio,
    }
    result = {}
    for name, value in values.items():
        numeric = float(value)
        result[name] = asdict(FeatureValue(
            value=numeric, valid_min=None, valid_max=None, confidence=1.0,
            method="canonical_resized_v1", evidence_id=f"ev_feature_canonical_{name.replace('.', '_')}",
            missing=not np.isfinite(numeric),
        ))
    return result
