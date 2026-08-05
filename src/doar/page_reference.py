"""Page-reference resolution (DOAR-TRACE Phase 2A.1, Section 3).

`page_frame.py` (Phase 2A) answers "is the whole page visible?" with a
status only -- it never produces an actual page polygon distinct from
the uploaded image. Every page-relative coverage/placement calculation
in `analysis.py`/`features.py` then silently divides by the whole image
frame's dimensions, treating "the image" and "the page" as identical by
default, never as an explicit decision. This module is the missing
link: it resolves an explicit `PageReference` -- a mode, a polygon (or
`None`), a confidence, and a provenance string -- so page-relative
features can be computed against the real page area when one is known,
and correctly withheld (`page_relative_features_assessable = False`)
when it is not.

**Does not infer missing parts of a cropped page.** A page reference is
either confirmed/detected as the full image, explicitly supplied by a
user as a polygon, or absent -- never guessed, extrapolated, or
manufactured from a partial view.

Six modes, exactly as specified:

  auto_detected_page       -- page_frame.py's classical-CV heuristic
                               judged the whole image is (likely) the
                               full page. Polygon = the whole image.
                               Never conflated with real confirmation.
  user_confirmed_full_frame -- an explicit user assertion that the
                               uploaded image shows the full page
                               edge-to-edge. Polygon = the whole image,
                               but the provenance is a human decision,
                               not an inference.
  user_defined_page_corners -- an explicit user-supplied quadrilateral
                               (4 pixel corners) marking the physical
                               page within the image. Polygon can be
                               smaller than the whole image.
  cropped_or_content_only  -- page_frame.py judged the page is not
                               fully visible. No polygon. Not assessable.
  uncertain                 -- evidence was ambiguous. No polygon. Not
                               assessable. Never guessed toward either
                               side.
  failed                    -- the assessment itself errored. No
                               polygon. Not assessable.

For the current dataset (batch processing, no interactive user), only
`auto_detected_page`/`cropped_or_content_only`/`uncertain`/`failed` are
ever produced -- `user_confirmed_full_frame`/`user_defined_page_corners`
require an explicit `user_page_declaration` argument that nothing in
the current batch pipeline supplies. They exist as a real, tested,
forward-looking schema/API for the interactive application (Section 3
of the task), not a decoration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PAGE_REFERENCE_SCHEMA_VERSION = "page_reference_v1"

PAGE_REFERENCE_MODES = frozenset({
    "auto_detected_page", "user_confirmed_full_frame", "user_defined_page_corners",
    "cropped_or_content_only", "uncertain", "failed",
})

# Modes under which page-relative coverage/placement may be computed
# against a real polygon. `cropped_or_content_only`/`uncertain`/`failed`
# are deliberately excluded -- see module docstring.
ASSESSABLE_PAGE_REFERENCE_MODES = frozenset({
    "auto_detected_page", "user_confirmed_full_frame", "user_defined_page_corners",
})

_FULL_FRAME_POLYGON = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

# Below this fraction of the image area, a user-supplied polygon is
# almost certainly a data-entry error (e.g. all 4 corners clicked near
# the same point), not a real page -- rejected outright rather than
# silently accepted and producing a wildly inflated page-relative
# coverage number.
_MIN_PLAUSIBLE_PAGE_AREA_FRACTION = 0.01


class InvalidPagePolygonError(ValueError):
    """Raised when a user-supplied page polygon fails basic geometric
    validation -- never silently coerced into something usable."""


@dataclass(frozen=True)
class PageReference:
    page_reference_mode: str
    # Normalized [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (0..1 of image width/
    # height) or None when no page region is known.
    page_polygon: list[list[float]] | None
    confidence: float
    obtained_via: str
    page_relative_features_assessable: bool
    limitations: list[str]
    evidence_id: str
    schema_version: str = PAGE_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.page_reference_mode not in PAGE_REFERENCE_MODES:
            raise ValueError(f"page_reference_mode {self.page_reference_mode!r} not in {sorted(PAGE_REFERENCE_MODES)}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence {self.confidence!r} out of [0, 1]")
        if self.page_relative_features_assessable and self.page_polygon is None:
            raise ValueError("page_relative_features_assessable=True requires a real page_polygon, never None")
        if self.page_reference_mode not in ASSESSABLE_PAGE_REFERENCE_MODES and self.page_relative_features_assessable:
            raise ValueError(f"mode {self.page_reference_mode!r} must never be marked assessable")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def page_area_fraction(self) -> float | None:
        """Fraction (0..1] of the whole image the confirmed/detected page
        polygon occupies -- 1.0 for the two whole-image modes, < 1.0 only
        possible for `user_defined_page_corners`."""
        if self.page_polygon is None:
            return None
        return _polygon_area(self.page_polygon)


def _polygon_area(points: list[list[float]]) -> float:
    """Shoelace formula on normalized [0, 1] coordinates."""
    n = len(points)
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _validate_and_normalize_corners(
    corners_px: list[list[float]], image_width: int, image_height: int,
) -> list[list[float]]:
    if len(corners_px) != 4:
        raise InvalidPagePolygonError(f"a page polygon must have exactly 4 corners, got {len(corners_px)}")
    if image_width <= 0 or image_height <= 0:
        raise InvalidPagePolygonError(f"invalid image dimensions {image_width}x{image_height}")
    normalized = []
    for corner in corners_px:
        if len(corner) != 2:
            raise InvalidPagePolygonError(f"each corner must be an (x, y) pair, got {corner!r}")
        x, y = corner
        if not (0 <= x <= image_width and 0 <= y <= image_height):
            raise InvalidPagePolygonError(
                f"corner ({x}, {y}) is outside the image bounds ({image_width}x{image_height})"
            )
        normalized.append([x / image_width, y / image_height])
    area = _polygon_area(normalized)
    if area < _MIN_PLAUSIBLE_PAGE_AREA_FRACTION:
        raise InvalidPagePolygonError(
            f"page polygon area ({area:.4f} of the image) is implausibly small "
            f"(< {_MIN_PLAUSIBLE_PAGE_AREA_FRACTION}) -- likely degenerate or mis-entered corners"
        )
    return normalized


def resolve_page_reference(
    page_frame_assessment: dict[str, Any],
    user_page_declaration: dict[str, Any] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> PageReference:
    """`page_frame_assessment` is `page_frame.assess_page_frame(...).to_dict()`
    -- always required, since it is the fallback automatic signal even
    when a user declaration is supplied (its `limitations` are folded
    into every non-user-declared PageReference).

    `user_page_declaration`, when supplied, takes precedence over the
    automatic assessment: it is `{"mode": "user_confirmed_full_frame"}`
    or `{"mode": "user_defined_page_corners", "corners": [[x, y], ...]}`
    (4 PIXEL corners; `image_width`/`image_height` are then required to
    normalize them). An explicit human decision is never overridden by
    an automatic heuristic."""
    if user_page_declaration is not None:
        mode = user_page_declaration.get("mode")
        if mode == "user_confirmed_full_frame":
            return PageReference(
                page_reference_mode="user_confirmed_full_frame",
                page_polygon=[list(p) for p in _FULL_FRAME_POLYGON],
                confidence=1.0,
                obtained_via="explicit_user_assertion_v1",
                page_relative_features_assessable=True,
                limitations=[
                    "The user asserted the full page is visible edge-to-edge; this is a recorded human "
                    "decision, not independently verified against pixel content.",
                ],
                evidence_id="ev_page_reference",
            )
        if mode == "user_defined_page_corners":
            corners_px = user_page_declaration.get("corners")
            if not corners_px:
                raise InvalidPagePolygonError("user_defined_page_corners requires a non-empty 'corners' list")
            if image_width is None or image_height is None:
                raise ValueError("image_width/image_height are required to normalize user-supplied pixel corners")
            polygon = _validate_and_normalize_corners(corners_px, image_width, image_height)
            return PageReference(
                page_reference_mode="user_defined_page_corners",
                page_polygon=polygon,
                confidence=1.0,
                obtained_via="user_supplied_corners_v1",
                page_relative_features_assessable=True,
                limitations=[
                    "The user manually marked the page corners; only basic geometric validity (4 in-bounds "
                    "corners, non-degenerate area) is checked -- the polygon's fit to the real page edges in "
                    "the pixels is not independently verified.",
                ],
                evidence_id="ev_page_reference",
            )
        if mode == "user_declared_cropped":
            # DOAR-TRACE Phase 2A.2, Section 6: the parent-facing "No, this
            # image is cropped" choice. Same page_reference_mode string
            # (`cropped_or_content_only`) and outcome
            # (page_relative_features_assessable=False) as the automatic
            # path -- the gating behaviour is identical either way -- but a
            # distinct `obtained_via` makes the explicit human statement
            # traceable and distinguishable from an automatic reading.
            return PageReference(
                page_reference_mode="cropped_or_content_only",
                page_polygon=None,
                confidence=1.0,
                obtained_via="explicit_user_assertion_v1",
                page_relative_features_assessable=False,
                limitations=[
                    "The user explicitly stated this image does not show the complete sheet of paper; a recorded "
                    "human decision, not an automatic inference.",
                ],
                evidence_id="ev_page_reference",
            )
        if mode == "user_declared_uncertain":
            # The parent-facing "I am not sure" choice.
            return PageReference(
                page_reference_mode="uncertain",
                page_polygon=None,
                confidence=0.5,
                obtained_via="explicit_user_assertion_v1",
                page_relative_features_assessable=False,
                limitations=[
                    "The user explicitly stated they are not sure whether the complete sheet is visible; a "
                    "recorded human decision, not an automatic inference.",
                ],
                evidence_id="ev_page_reference",
            )
        raise ValueError(f"unknown user_page_declaration mode {mode!r}")

    # No user declaration: fall back to the automatic classical-CV
    # assessment. Never manufactures a polygon for a partially-visible page.
    status = page_frame_assessment["page_frame_status"]
    confidence = float(page_frame_assessment["confidence"])
    base_limitations = list(page_frame_assessment.get("limitations", []))

    if status in ("full_page_detected", "likely_full_page"):
        return PageReference(
            page_reference_mode="auto_detected_page",
            page_polygon=[list(p) for p in _FULL_FRAME_POLYGON],
            confidence=confidence,
            obtained_via="classical_cv_border_uniformity_v1",
            page_relative_features_assessable=True,
            limitations=base_limitations + [
                "Automatically detected, not user-confirmed -- the page polygon is assumed to equal the full "
                "uploaded image frame; no smaller/partial page polygon is ever inferred automatically.",
            ],
            evidence_id="ev_page_reference",
        )
    if status == "cropped_or_content_only":
        return PageReference(
            page_reference_mode="cropped_or_content_only",
            page_polygon=None,
            confidence=confidence,
            obtained_via="classical_cv_border_uniformity_v1",
            page_relative_features_assessable=False,
            limitations=base_limitations + [
                "Content appears to extend beyond the visible frame; no page boundary is manufactured for a "
                "partially-visible page.",
            ],
            evidence_id="ev_page_reference",
        )
    if status == "uncertain":
        return PageReference(
            page_reference_mode="uncertain",
            page_polygon=None,
            confidence=confidence,
            obtained_via="classical_cv_border_uniformity_v1",
            page_relative_features_assessable=False,
            limitations=base_limitations,
            evidence_id="ev_page_reference",
        )
    if status == "failed":
        return PageReference(
            page_reference_mode="failed",
            page_polygon=None,
            confidence=0.0,
            obtained_via="classical_cv_border_uniformity_v1",
            page_relative_features_assessable=False,
            limitations=base_limitations,
            evidence_id="ev_page_reference",
        )
    raise ValueError(f"unrecognized page_frame_status {status!r}")


def page_relative_bounding_box_coverage(
    bounding_box_coverage: float | None, page_reference: PageReference,
) -> float | None:
    """Re-expresses an image-relative `bounding_box_coverage` (fraction of
    the WHOLE UPLOADED IMAGE) as a fraction of the CONFIRMED/DETECTED PAGE
    area instead. Returns `None` when no page reference is assessable --
    callers must treat `None` as `not_assessable`, never as 0.

    Mathematically: both values share the same bbox pixel-area numerator;
    `bounding_box_coverage = bbox_area / image_area` and
    `page_polygon`'s own area (in the same normalized units) is
    `page_area_fraction * image_area`, so
    `page_relative_coverage = bounding_box_coverage / page_area_fraction`.
    For the two whole-image modes (`auto_detected_page`,
    `user_confirmed_full_frame`), `page_area_fraction == 1.0`, so this is
    numerically identical to `bounding_box_coverage` -- the distinction
    only matters once a real, smaller `user_defined_page_corners` polygon
    exists."""
    if not page_reference.page_relative_features_assessable or bounding_box_coverage is None:
        return None
    area_fraction = page_reference.page_area_fraction
    if not area_fraction or area_fraction <= 0:
        return None
    return float(min(1.0, bounding_box_coverage / area_fraction))


# ---------------------------------------------------------------------------
# DOAR-TRACE Phase 2A.2, Section 6: the minimal, parent-facing page-
# reference control -- 4 plain choices, mapped to the API above. No corner
# editor is implemented in this phase (explicitly out of scope); the
# choices below are the only user-facing entry point into this module for
# the interactive prototype.
# ---------------------------------------------------------------------------

PARENT_PAGE_DECLARATION_CHOICES = ("auto", "yes", "no", "unsure")

PARENT_PAGE_DECLARATION_LABELS: dict[str, dict[str, str]] = {
    "auto": {"en": "Let the system decide", "ar": "دع النظام يقرر"},
    "yes": {"en": "Yes, the complete sheet is visible", "ar": "نعم، الورقة كاملة ظاهرة"},
    "no": {"en": "No, this image is cropped", "ar": "لا، هذه الصورة مقصوصة"},
    "unsure": {"en": "I am not sure", "ar": "لست متأكداً"},
}


def user_page_declaration_from_choice(choice: str) -> dict[str, str] | None:
    """Maps a parent-facing choice (Section 6's 4 options) to the real
    `user_page_declaration` argument `resolve_page_reference`/`analyze_image`
    already accept. `"auto"` maps to `None` (no declaration -- the
    automatic classical-CV assessment alone decides)."""
    if choice not in PARENT_PAGE_DECLARATION_CHOICES:
        raise ValueError(f"unknown page declaration choice {choice!r}, expected one of {PARENT_PAGE_DECLARATION_CHOICES}")
    return {
        "auto": None,
        "yes": {"mode": "user_confirmed_full_frame"},
        "no": {"mode": "user_declared_cropped"},
        "unsure": {"mode": "user_declared_uncertain"},
    }[choice]


def describe_declaration_choice(page_reference: dict) -> str:
    """Reconstructs which of the 4 parent-facing choices (`"auto"`,
    `"yes"`, `"no"`, `"unsure"`, or `"corners"` for the API-only
    `user_defined_page_corners` mode) produced a saved `PageReference`
    dict -- purely from its own persisted `obtained_via`/
    `page_reference_mode` fields, so no separate declaration file needs
    to be written for Technical-view traceability. Returns `"unknown"`
    only for `analysis.json` documents written before this function
    existed (missing/unrecognized fields)."""
    obtained_via = page_reference.get("obtained_via")
    mode = page_reference.get("page_reference_mode")
    if obtained_via == "classical_cv_border_uniformity_v1":
        return "auto"
    if obtained_via == "user_supplied_corners_v1":
        return "corners"
    if obtained_via == "explicit_user_assertion_v1":
        return {"user_confirmed_full_frame": "yes", "cropped_or_content_only": "no", "uncertain": "unsure"}.get(mode, "unknown")
    return "unknown"
