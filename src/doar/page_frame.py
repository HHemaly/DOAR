"""Page-frame assessability (DOAR-TRACE Phase 2A, Section 3).

A page-use or placement observation is only meaningful when the complete
sheet, or a reliable page boundary, is visible in the image. Nothing in
`analysis.py::_segment`/`_composition` ever checked this -- it silently
assumes the whole input image IS the page (`_estimate_background` samples
the image's own border pixels as "background/page"). This module adds
that missing check as an explicit, honest classical-CV assessment --
**not** a trained object/page detector (out of scope for this phase);
purely heuristic border/margin/content-bleed analysis, the same
dependency-light style as the rest of `analysis.py`.

Never fabricates a paper boundary: when the evidence is ambiguous, the
status is `uncertain`, never guessed as `full_page_detected`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

PAGE_FRAME_SCHEMA_VERSION = "page_frame_v1"

STATUSES = frozenset({
    "full_page_detected", "likely_full_page", "cropped_or_content_only", "uncertain", "failed",
})

# A status the caller may treat as "page-relative claims are safe to make".
ASSESSABLE_STATUSES = frozenset({"full_page_detected", "likely_full_page"})

_BORDER_BAND_PX_FRACTION = 0.03  # outermost ~3% of each dimension, used to sample border strips

# DOAR V1.1 Stage 6: this heuristic's own `limitations` text (below) admits
# it "cannot distinguish a genuinely full page with a very thin margin from
# a tightly cropped photo" -- yet the raw formulas below could reach 1.0
# exactly (a uniform white background is common and drives uniformity to
# ~1.0), representing false certainty for a method that structurally cannot
# resolve that ambiguity. This is a principled ceiling, not a cosmetic
# markdown: no AUTOMATIC classical-CV estimate from this module may ever be
# represented as more certain than this value. Genuine certainty (1.0) is
# reserved exclusively for a real human decision -- page_reference.py's
# `user_confirmed_full_frame`/`user_defined_page_corners` paths, which
# record an explicit user assertion, not a machine estimate, and are
# correctly untouched by this cap.
MAX_AUTOMATIC_CONFIDENCE = 0.90


def _cap(value: float) -> float:
    return round(min(value, MAX_AUTOMATIC_CONFIDENCE), 4)


@dataclass(frozen=True)
class PageFrameAssessment:
    page_frame_status: str
    confidence: float
    method: str
    detected_page_boundary: list[float] | None  # normalized [x0, y0, x1, y1] if found, else None
    border_evidence: dict[str, Any]
    cropping_evidence: dict[str, Any]
    limitations: list[str]
    evidence_id: str

    def __post_init__(self) -> None:
        if self.page_frame_status not in STATUSES:
            raise ValueError(f"page_frame_status {self.page_frame_status!r} not in {sorted(STATUSES)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence {self.confidence!r} out of [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def assessable(self) -> bool:
        return self.page_frame_status in ASSESSABLE_STATUSES


def _border_strip_uniformity(rgb: np.ndarray, band_px: int) -> float:
    """0..1: how visually uniform the outermost border band is (a real,
    unbroken page margin is close to uniform; content bleeding to the edge
    is not). Pure numpy, no learned model."""
    h, w = rgb.shape[:2]
    band_px = max(1, min(band_px, h // 3, w // 3))
    top, bottom = rgb[:band_px], rgb[-band_px:]
    left, right = rgb[:, :band_px], rgb[:, -band_px:]
    strip = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
    std = float(np.std(strip.astype(np.float32)))
    # std=0 -> perfectly uniform (score 1.0); std>=60 -> treat as fully non-uniform (score 0.0)
    return float(np.clip(1.0 - std / 60.0, 0.0, 1.0))


def _content_touches_edge(mask: np.ndarray) -> dict[str, bool]:
    return {
        "top": bool(mask[0].any()),
        "bottom": bool(mask[-1].any()),
        "left": bool(mask[:, 0].any()),
        "right": bool(mask[:, -1].any()),
    }


def _edge_touch_ratio(mask: np.ndarray) -> dict[str, float]:
    return {
        "top": float(mask[0].mean()),
        "bottom": float(mask[-1].mean()),
        "left": float(mask[:, 0].mean()),
        "right": float(mask[:, -1].mean()),
    }


def assess_page_frame(rgb: np.ndarray, mask: np.ndarray, background_stability: float) -> PageFrameAssessment:
    """`rgb`: the full image array. `mask`: the real foreground/content
    mask already computed by `analysis.py::_segment`. `background_stability`:
    the real value `_segment` already computes (border-colour consistency)
    -- reused, not recomputed, to avoid a second, possibly divergent
    background estimate."""
    h, w = mask.shape
    band_px = max(1, int(round(min(h, w) * _BORDER_BAND_PX_FRACTION)))
    uniformity = _border_strip_uniformity(rgb, band_px)
    touches = _content_touches_edge(mask)
    touch_ratios = _edge_touch_ratio(mask)
    n_edges_touched = sum(touches.values())
    max_touch_ratio = max(touch_ratios.values())

    border_evidence = {
        "border_strip_uniformity": round(uniformity, 4),
        "background_stability": round(float(background_stability), 4),
        "band_px": band_px,
    }
    cropping_evidence = {
        "edges_with_content_touching": touches,
        "edge_touch_ratios": {k: round(v, 4) for k, v in touch_ratios.items()},
        "n_edges_touched": n_edges_touched,
    }
    limitations = [
        "Heuristic classical-CV assessment only -- not a trained page/object detector.",
        "Cannot distinguish a genuinely full page with a very thin margin from a tightly cropped photo.",
    ]

    # Strong negative evidence: content clearly bleeds off multiple edges
    # with a high touch ratio -> almost certainly cropped, not the full sheet.
    if n_edges_touched >= 3 and max_touch_ratio > 0.15:
        return PageFrameAssessment(
            page_frame_status="cropped_or_content_only",
            confidence=_cap(0.55 + 0.15 * n_edges_touched),
            method="border_uniformity_and_content_edge_touch_v1",
            detected_page_boundary=None,
            border_evidence=border_evidence, cropping_evidence=cropping_evidence,
            limitations=limitations, evidence_id="ev_page_frame",
        )

    # Strong positive evidence: a wide, highly uniform border on all four
    # sides with essentially no content touching any edge.
    if n_edges_touched == 0 and uniformity >= 0.85 and background_stability >= 0.8:
        return PageFrameAssessment(
            page_frame_status="full_page_detected",
            confidence=_cap(0.6 + 0.4 * uniformity),
            method="border_uniformity_and_content_edge_touch_v1",
            detected_page_boundary=[0.0, 0.0, 1.0, 1.0],
            border_evidence=border_evidence, cropping_evidence=cropping_evidence,
            limitations=limitations, evidence_id="ev_page_frame",
        )

    # Moderate positive evidence: a reasonably uniform border, content
    # touches at most one edge lightly.
    if n_edges_touched <= 1 and uniformity >= 0.6 and max_touch_ratio < 0.10:
        return PageFrameAssessment(
            page_frame_status="likely_full_page",
            confidence=_cap(0.4 + 0.3 * uniformity),
            method="border_uniformity_and_content_edge_touch_v1",
            detected_page_boundary=[0.0, 0.0, 1.0, 1.0],
            border_evidence=border_evidence, cropping_evidence=cropping_evidence,
            limitations=limitations, evidence_id="ev_page_frame",
        )

    if n_edges_touched >= 2:
        return PageFrameAssessment(
            page_frame_status="cropped_or_content_only",
            confidence=_cap(0.4 + 0.1 * n_edges_touched),
            method="border_uniformity_and_content_edge_touch_v1",
            detected_page_boundary=None,
            border_evidence=border_evidence, cropping_evidence=cropping_evidence,
            limitations=limitations, evidence_id="ev_page_frame",
        )

    # Ambiguous: never guess.
    return PageFrameAssessment(
        page_frame_status="uncertain",
        confidence=round(0.5 * uniformity, 4),
        method="border_uniformity_and_content_edge_touch_v1",
        detected_page_boundary=None,
        border_evidence=border_evidence, cropping_evidence=cropping_evidence,
        limitations=limitations + ["Evidence did not clearly indicate either a full page or a crop; not guessed."],
        evidence_id="ev_page_frame",
    )


def failed_assessment(reason: str) -> PageFrameAssessment:
    return PageFrameAssessment(
        page_frame_status="failed", confidence=0.0, method="border_uniformity_and_content_edge_touch_v1",
        detected_page_boundary=None, border_evidence={}, cropping_evidence={},
        limitations=[f"Assessment failed: {reason}"], evidence_id="ev_page_frame",
    )
