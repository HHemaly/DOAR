"""Pure functions behind the reference-mask annotation app (testable without Streamlit).

Reference-mask protocol (experiments/journal/annotation/ANNOTATION_PROTOCOL.md):
1. The annotator first decides whether a drawing/support mask is meaningful
   for the image (ASSESSABLE or one of the NOT_ASSESSABLE reasons).
2. For assessable images the annotator samples the support (paper) colour by
   clicking a few points; pixels whose CIE-Lab distance to the nearest sampled
   support colour exceeds a threshold the annotator sets by eye become the
   starting mask. This proposal comes from the annotator's own choices, not
   from any evaluated method.
3. The annotator corrects it with add/erase brush strokes until it covers
   every visible drawing mark (pencil, crayon, paint, ink; not paper texture,
   shadows, scanner noise, page edges or added digital noise).
"""

from __future__ import annotations

import numpy as np

ASSESSABILITY = (
    "ASSESSABLE",
    "NOT_ASSESSABLE_FULL_CANVAS",      # marks cover the page; no support to separate from
    "NOT_ASSESSABLE_PHOTO_SCENE",      # photograph with non-paper background objects
    "NOT_ASSESSABLE_NOT_A_CHILD_DRAWING",
    "NOT_ASSESSABLE_BLANK",
    "NOT_ASSESSABLE_OTHER",
)
MAX_SIDE = 1024  # images larger than this are annotated AND evaluated at this size


def working_size(h: int, w: int, max_side: int = MAX_SIDE) -> tuple[int, int]:
    s = min(1.0, max_side / max(h, w))
    return max(1, round(h * s)), max(1, round(w * s))


def to_working_resolution(image_rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    import cv2
    h, w = image_rgb.shape[:2]
    nh, nw = working_size(h, w, max_side)
    if (nh, nw) == (h, w):
        return image_rgb
    return cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def lab(image_rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def support_distance(image_rgb: np.ndarray, points_xy: list[tuple[int, int]], radius: int = 3) -> np.ndarray:
    """Lab distance of every pixel to the NEAREST sampled support colour.
    Each sample is the median Lab colour in a (2r+1)^2 window around a click."""
    if not points_xy:
        raise ValueError("sample at least one support point")
    L = lab(image_rgb)
    h, w = L.shape[:2]
    samples = []
    for x, y in points_xy:
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        samples.append(np.median(L[y0:y1, x0:x1].reshape(-1, 3), axis=0))
    d = np.stack([np.linalg.norm(L - s, axis=2) for s in samples], axis=0)
    return d.min(axis=0)


def proposal(image_rgb: np.ndarray, points_xy, threshold: float) -> np.ndarray:
    return support_distance(image_rgb, points_xy) > threshold


def combine(proposal_mask: np.ndarray, add: np.ndarray, erase: np.ndarray) -> np.ndarray:
    """Final mask = (proposal OR added strokes) AND NOT erased strokes."""
    return (np.asarray(proposal_mask, bool) | np.asarray(add, bool)) & ~np.asarray(erase, bool)


def strokes_from_canvas(rgba: np.ndarray, shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Split the canvas stroke layer into add (green) and erase (red) masks,
    rescaled with nearest neighbour to the working resolution."""
    import cv2
    if rgba is None:
        z = np.zeros(shape_hw, bool)
        return z, z
    a = rgba[..., 3] > 0
    add = a & (rgba[..., 1] > 150) & (rgba[..., 0] < 100)
    erase = a & (rgba[..., 0] > 150) & (rgba[..., 1] < 100)
    h, w = shape_hw
    rs = lambda m: cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
    return rs(add), rs(erase)


def overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    red = np.array([255, 0, 160], np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * red
    return out.clip(0, 255).astype(np.uint8)
