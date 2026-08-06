"""Machine-readable mirror of docs/OBJECT_EVIDENCE_ONTOLOGY.md.

This module is the single source of truth for the candidate class list;
the markdown doc is the human-readable explanation of the same data. Keep
them in sync by hand -- there are few enough classes that a generated doc
would add more indirection than it saves.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectClass:
    name: str
    label_type: str  # "presence", "presence_and_count", or "presence_experimental"
    minimum_visible_evidence: str
    notes: str


# The 10 active pilot candidates. Order matches docs/OBJECT_EVIDENCE_ONTOLOGY.md.
CLASSES: tuple[ObjectClass, ...] = (
    ObjectClass("person", "presence_and_count",
                "head shape plus one connected body element",
                "no expression/state judgment in this phase"),
    ObjectClass("face", "presence",
                "one closed head-shape with >=1 internal mark",
                "presence only; expression is out of scope"),
    ObjectClass("hand", "presence_experimental",
                "a rounded or fingered terminal shape at a limb end",
                "experimental; omission (missing-hand) judgment is out of scope"),
    ObjectClass("animal", "presence_and_count",
                "one connected animal-like body shape",
                "general category only; no species-level distinction is made"),
    ObjectClass("house", "presence",
                "roof shape plus enclosed wall area", ""),
    ObjectClass("tree", "presence_and_count",
                "trunk shape plus canopy or branch mark", ""),
    ObjectClass("heart", "presence_and_count",
                "the lobed-top, pointed-bottom silhouette", ""),
    ObjectClass("star", "presence_and_count",
                ">=4 radiating points from a shared center", ""),
    ObjectClass("circle", "presence_and_count",
                "a closed or near-closed round contour",
                "also the target of the classical-CV circularity baseline"),
    ObjectClass("vehicle", "presence",
                "body shape plus at least one class-typical feature "
                "(wheel, wing, sail)", ""),
)

CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in CLASSES)

# Explicitly postponed from this pilot -- recorded, not silently dropped.
# See docs/OBJECT_EVIDENCE_ONTOLOGY.md "Explicitly postponed" section for why.
POSTPONED_CLASSES: tuple[str, ...] = ("eye", "mouth")

# Annotation statuses an annotator may record for a (image, class) pair.
# Mirrors the task's required vocabulary -- "uncertain" must never be
# silently converted to a negative.
ANNOTATION_STATUSES = frozenset({"present", "absent", "uncertain", "not_assessable"})


def class_by_name(name: str) -> ObjectClass:
    for c in CLASSES:
        if c.name == name:
            return c
    raise KeyError(f"{name!r} is not a Phase 2B ontology class (postponed: {POSTPONED_CLASSES})")
