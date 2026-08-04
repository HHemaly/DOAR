"""Canonical evidence schema (Phase 1).

One structured representation for every piece of observable drawing
evidence the pipeline can produce, spanning image/page quality,
foreground/page segmentation, global composition, size/placement, colour,
strokes/shading/repetition/overwriting, primitive geometry, semantic
objects, object components/body parts, spatial relationships, relative
measurements, OCR/text, and corrections/erasures.

This module does not compute evidence itself -- see `evidence_adapter.py`
for the code that turns a real `Analysis` result into `EvidenceItem`
instances, reusing the existing extractors in `analysis.py`/`features.py`
rather than re-implementing them.

The one invariant this module exists to enforce, mechanically, everywhere
in the pipeline: **unavailable evidence must never look like a confirmed
negative measurement.** A status of `unavailable`, `failed`,
`insufficient_evidence`, or `requires_manual_review` must carry `value=None`
-- constructing an `EvidenceItem` that violates this raises immediately,
so the mistake is caught at the point it is made, not downstream in a
report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Statuses an evidence item may carry. A rule engine may only treat
# `available` or `experimental` evidence as usable for a positive or
# negative determination; every other status must propagate as
# insufficient/unavailable to any rule that depends on it.
EVIDENCE_STATUSES = frozenset({
    "available",          # measured, from a validated/trusted extractor
    "experimental",       # measured, from an extractor not yet validated
    "unavailable",        # no extractor exists for this evidence category
    "failed",              # an extractor exists and ran, but failed on this image
    "insufficient_evidence",  # extractor ran but could not produce a reliable result
    "requires_manual_review",  # extractor is not trusted enough to auto-decide
})

# Evidence with one of these statuses may be used by a rule to decide
# `triggered`/`not_triggered`. Every other status must propagate as
# insufficient/unavailable -- see `evidence_rule_engine.py`.
CONCLUSIVE_STATUSES = frozenset({"available", "experimental"})

# The evidence categories the Phase 1 canonical schema is required to
# cover. Not every category has an implemented extractor yet -- categories
# without one are represented by explicit `unavailable` EvidenceItems
# (see `evidence_adapter.py::UNIMPLEMENTED_CATEGORIES`), never omitted.
EVIDENCE_CATEGORIES = (
    "image_page_quality",
    "foreground_page_segmentation",
    "global_composition",
    "size_placement",
    "colour",
    "strokes_shading_repetition_overwriting",
    "primitive_geometry",
    "semantic_objects",
    "object_components_body_parts",
    "spatial_relationships",
    "relative_measurements",
    "ocr_text",
    "corrections_erasures",
)


@dataclass(frozen=True)
class EvidenceLocation:
    """Where in the image this evidence was found. All fields are
    optional -- not every evidence item is spatially localizable (e.g. a
    whole-page quality metric has no bounding box)."""

    region: str | None = None            # e.g. "bounding_box", "page", "component_3"
    bbox_xywh: tuple[float, float, float, float] | None = None  # normalized 0-1
    centroid_xy: tuple[float, float] | None = None              # normalized 0-1
    coordinate_space: str = "normalized_image"                  # vs "pixel"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    feature_id: str
    value: Any
    unit: str | None
    status: str
    confidence: float | None
    extractor: str
    extractor_version: str
    validation_status: str
    location: EvidenceLocation | None
    visualization_reference: str | None
    reason: str
    limitations: list[str] = field(default_factory=list)
    category: str | None = None

    def __post_init__(self) -> None:
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(
                f"EvidenceItem {self.evidence_id!r}: status {self.status!r} is not one of "
                f"{sorted(EVIDENCE_STATUSES)}"
            )
        if self.status not in CONCLUSIVE_STATUSES and self.value is not None:
            raise ValueError(
                f"EvidenceItem {self.evidence_id!r} has status {self.status!r} but a non-None "
                f"value ({self.value!r}). Unavailable evidence must never carry a value that "
                "could be mistaken for zero, false, or a confirmed measurement."
            )
        if self.category is not None and self.category not in EVIDENCE_CATEGORIES:
            raise ValueError(
                f"EvidenceItem {self.evidence_id!r}: category {self.category!r} is not one of "
                f"{EVIDENCE_CATEGORIES}"
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"EvidenceItem {self.evidence_id!r}: confidence {self.confidence!r} out of [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.location is not None:
            data["location"] = self.location.to_dict()
        return data


def unavailable_item(
    evidence_id: str,
    feature_id: str,
    *,
    category: str,
    extractor: str = "none",
    reason: str,
    limitations: list[str] | None = None,
) -> EvidenceItem:
    """Construct an explicit `unavailable` evidence item. Use this instead
    of simply omitting a category from the evidence set -- an omitted
    category is indistinguishable from "forgot to check"; an explicit
    `unavailable` item is a positive, auditable statement that the
    category was considered and no extractor exists for it."""
    return EvidenceItem(
        evidence_id=evidence_id,
        feature_id=feature_id,
        value=None,
        unit=None,
        status="unavailable",
        confidence=None,
        extractor=extractor,
        extractor_version="n/a",
        validation_status="not_applicable",
        location=None,
        visualization_reference=None,
        reason=reason,
        limitations=limitations or [],
        category=category,
    )


@dataclass
class EvidenceSet:
    """A validated, indexed collection of evidence for one image."""

    items: list[EvidenceItem]

    def __post_init__(self) -> None:
        ids = [item.evidence_id for item in self.items]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate evidence_id(s): {sorted(duplicates)}")

    def by_feature(self, feature_id: str) -> list[EvidenceItem]:
        return [item for item in self.items if item.feature_id == feature_id]

    def by_id(self, evidence_id: str) -> EvidenceItem | None:
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.items]
