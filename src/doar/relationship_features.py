"""Deterministic, descriptive spatial-relationship features (DOAR
multimodal-evidence milestone).

Computed purely from bboxes already present on VERIFIED `VisualEntity`
records (the Gemini Observer/Verifier pipeline's own output) -- no new
detector, no new model, no network call. Only VERIFIED entities with a
real bbox are considered, matching the "visually confirmed" bar every
other part of this pipeline uses: an unverified/guessed bbox should not
anchor a relationship claim.

**These features are DESCRIPTIVE ONLY.** Nothing in this module assigns
a psychological meaning to spatial separation/centrality/repetition/
relative size -- `spatially_separated_entity_ids` is a neutral distance
ranking, never a claim about social/emotional isolation. As of this
milestone, no row in `RULE_EVIDENCE_MATRIX.csv` has an
`observable_feature` naming a relationship concept, so nothing here is
wired into `drawing_synthesis.py`'s rule bridge either -- these values
appear only in the objective/descriptive profile. If a future frozen
rule is added whose observable is a relationship measurement, wiring it
in is a small, explicit addition to the rule bridge -- never an implicit
one made by this module.
"""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .visual_entity import VisualEntity

IMAGE_CENTER_BBOX = (0.5, 0.5, 0.0, 0.0)  # a zero-size "point" bbox at the image center


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def _distance(bbox_a: tuple[float, float, float, float], bbox_b: tuple[float, float, float, float]) -> float:
    ax, ay = _center(bbox_a)
    bx, by = _center(bbox_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _verified_entities_with_bbox(entities: list["VisualEntity"]) -> list["VisualEntity"]:
    return [e for e in entities if e.case_verification_status == "verified" and e.bbox]


def compute_relationship_features(entities: list["VisualEntity"]) -> dict:
    """Returns a plain dict (JSON-serializable) -- never a claim about
    what the relationships MEAN, only what is measurable:

        entity_count            -- how many verified, bbox-bearing entities exist
        figures                 -- per-entity: label, relative_size (0-1, largest=1.0), centrality (0-1)
        pairwise_distances      -- normalized center-to-center distance for every pair
        spatially_separated_entity_ids -- entities whose nearest neighbour is farther than
                                    the median pairwise distance (purely a distance-ranking
                                    fact -- neutral spatial description, not a claim about
                                    social/emotional isolation, which is a distinct,
                                    unmeasured psychological construct)
        central_entity_ids      -- the entity/entities closest to the image center
        repeated_labels         -- canonical_label -> count, for labels appearing more than once
    """
    verified = _verified_entities_with_bbox(entities)
    count = len(verified)
    if count == 0:
        return {
            "entity_count": 0, "figures": [], "pairwise_distances": [],
            "spatially_separated_entity_ids": [], "central_entity_ids": [], "repeated_labels": {},
        }

    sizes = {e.entity_id: e.bbox[2] * e.bbox[3] for e in verified}
    max_size = max(sizes.values())
    relative_sizes = {eid: (s / max_size if max_size else 0.0) for eid, s in sizes.items()}
    centrality = {e.entity_id: round(1 - min(1.0, _distance(e.bbox, IMAGE_CENTER_BBOX)), 4) for e in verified}

    pairwise = []
    for i in range(count):
        for j in range(i + 1, count):
            a, b = verified[i], verified[j]
            pairwise.append({"a": a.entity_id, "b": b.entity_id, "distance": round(_distance(a.bbox, b.bbox), 4)})

    spatially_separated_ids: list[str] = []
    if count >= 2:
        distances_all = sorted(d["distance"] for d in pairwise)
        median = distances_all[len(distances_all) // 2]
        nearest: dict[str, float] = {}
        for e in verified:
            own = [d["distance"] for d in pairwise if e.entity_id in (d["a"], d["b"])]
            if own:
                nearest[e.entity_id] = min(own)
        spatially_separated_ids = sorted(eid for eid, d in nearest.items() if d > median)

    central_ids = sorted(centrality, key=lambda eid: centrality[eid], reverse=True)[:1]

    label_counts = Counter(e.canonical_label for e in verified)
    repeated_labels = {label: n for label, n in label_counts.items() if n > 1}

    return {
        "entity_count": count,
        "figures": [
            {"entity_id": e.entity_id, "label": e.canonical_label,
             "relative_size": round(relative_sizes[e.entity_id], 4), "centrality": centrality[e.entity_id]}
            for e in verified
        ],
        "pairwise_distances": pairwise,
        "spatially_separated_entity_ids": spatially_separated_ids,
        "central_entity_ids": central_ids,
        "repeated_labels": repeated_labels,
    }
