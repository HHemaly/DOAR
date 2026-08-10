"""DOAR Visual Knowledge V2: rich, searchable visual entities.

`VisualFinding` (`visual_evidence.py`) stores a detection as
`label/bbox/confidence/status` -- correct, but too thin to support future
open-world search, alias/synonym matching, or per-entity expert review.
`VisualEntity` is an ADDITIVE superset: every existing `VisualFinding`
field is preserved verbatim (same meaning, mostly same name), plus new
attributes populated ONLY from information DOAR already computes --
nothing here invents a capability the detector pipeline doesn't have.

Backward compatibility is structural, not incidental: `detections.json`
keeps writing its existing `"findings"` list byte-for-byte unchanged
(`visual_evidence.py` is untouched); `VisualEntity` records are added
under a NEW `"entities"` key. Every existing consumer (the rule engine,
`visual_qa.py`, `find_matching`, case reload) keeps reading `"findings"`
and needs zero changes -- confirmed by test, not just claimed.

Scientific safety, unchanged from `visual_evidence.py`:
- `model_validation_status` (was `VisualFinding.validation_status`) is a
  DETECTOR-level property; `case_verification_status` is a NEW, orthogonal
  field for what an expert reviewer has said about THIS SPECIFIC
  detection in THIS SPECIFIC case -- never conflated, never blended into
  a single number.
- `candidate_labels` records what the detector considered, WITH
  confidence; `canonical_label` is the (single) label that finding
  actually reports. Neither is auto-promoted into the other, and neither
  ever changes `evidence_status`/`rule_mapping_status` (still computed
  exactly as `visual_evidence.py` already computes them).
- `visual_similarities` is a deliberately EMPTY placeholder for every
  entity today -- no embedding/similarity computation exists yet in this
  codebase; populating it now would mean fabricating data. Structurally
  present so a future phase can fill it in without another schema
  migration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .visual_evidence import VisualFinding

# ---------------------------------------------------------------------------
# Vocabulary: canonical_label -> aliases/broader categories/possible
# subtypes. Reuses visual_qa.py's OWN existing English/Arabic synonym
# table (the one already used for Q&A target extraction) as the alias
# source -- one vocabulary, not two independently-maintained ones.
# broader_categories are a small, conservative, manually-curated map over
# the SAME real labels our detectors/broad-scan vocabulary already use
# (phase2c7.detector_policy.OBJECT_CLASS_POLICY + eye +
# broad_vocabulary.UNVALIDATED_EXTRA_TARGETS) -- not a general ontology.
# possible_subtypes is empty everywhere today (nothing in this codebase
# distinguishes subtypes yet) -- present for future expansion, per
# instruction "build the structure so it can expand later."
# ---------------------------------------------------------------------------

_BROADER_CATEGORIES: dict[str, tuple[str, ...]] = {
    "person": ("figure",), "face": ("figure", "body_part"), "hand": ("body_part",),
    "eye": ("body_part",), "mouth": ("body_part",),
    "tree": ("plant", "nature"), "flower": ("plant", "nature"),
    "house": ("building",), "door": ("building_element",), "window": ("building_element",),
    "heart": ("symbol",), "star": ("symbol", "sky_element"), "circle": ("geometric_shape",),
    "animal": ("living_thing",), "cat": ("animal", "living_thing"), "dog": ("animal", "living_thing"),
    "bird": ("animal", "living_thing"),
    "vehicle": ("transport",), "road": ("infrastructure",),
    "sun": ("sky_element", "nature"), "moon": ("sky_element", "nature"), "cloud": ("sky_element", "nature"),
    "weapon_or_weapon_like_object": ("object",),
}

# entity_type: one of object/symbol/geometric_shape/decorative_mark/
# text_like/abstract_mark/scribble/unknown -- deterministic, by label.
# Any label NOT in this table (on-demand queries / unrecognized
# broad-scan hits) defaults to "unknown", never force-classified as
# "object" -- per instruction, an unresolved region is preserved as
# unknown rather than guessed into a category.
_ENTITY_TYPE_BY_LABEL: dict[str, str] = {
    "person": "object", "face": "object", "hand": "object", "eye": "object", "mouth": "object",
    "tree": "object", "house": "object", "animal": "object", "vehicle": "object",
    "cat": "object", "dog": "object", "bird": "object", "flower": "object", "door": "object",
    "window": "object", "sun": "object", "moon": "object", "cloud": "object", "road": "object",
    "weapon_or_weapon_like_object": "object",
    "heart": "symbol", "star": "symbol",
    "circle": "geometric_shape",
}

ENTITY_TYPES = frozenset({
    "object", "symbol", "geometric_shape", "decorative_mark", "text_like", "abstract_mark",
    "scribble", "unknown",
})

CASE_VERIFICATION_STATUSES = frozenset({"unverified", "verified", "rejected"})

# Same thirds-bucket convention analysis.py::_composition already uses
# for the whole-drawing centroid -- reused here per-entity so
# `page_position` means the same thing everywhere in DOAR.
_H_LEFT, _H_RIGHT = 0.4, 0.6
_V_TOP, _V_BOTTOM = 0.4, 0.6


def classify_entity_type(canonical_label: str) -> str:
    return _ENTITY_TYPE_BY_LABEL.get(canonical_label, "unknown")


def _aliases_for(canonical_label: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(aliases_en, aliases_ar) from visual_qa.py's existing synonym
    table -- Arabic aliases are the entries containing non-ASCII
    characters, English the rest; the source table doesn't separate them
    itself, so this is a light, deterministic split, not a translation."""
    from .visual_qa import _VISUAL_SYNONYMS
    synonyms = _VISUAL_SYNONYMS.get(canonical_label, ())
    en = tuple(s for s in synonyms if s.isascii())
    ar = tuple(s for s in synonyms if not s.isascii())
    return en, ar


def compute_relative_size(bbox: tuple[float, float, float, float] | None) -> float | None:
    """bbox is already normalized xywh (0..1 of image) -- area fraction
    is a direct product, no extra computation or assumption needed."""
    if bbox is None:
        return None
    _x, _y, w, h = bbox
    return round(max(0.0, w) * max(0.0, h), 6)


def compute_page_position(bbox: tuple[float, float, float, float] | None) -> str | None:
    """Same thirds-bucket convention/thresholds as
    analysis.py::_composition's whole-drawing placement -- applied to
    this entity's own bbox center instead of the drawing's foreground
    centroid."""
    if bbox is None:
        return None
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    horizontal = "left" if cx < _H_LEFT else "right" if cx > _H_RIGHT else "center"
    vertical = "top" if cy < _V_TOP else "bottom" if cy > _V_BOTTOM else "middle"
    return f"{vertical}_{horizontal}"


# Same 5-bin colour vocabulary as analysis.py::_colour (red/green/blue/
# yellow/dark) -- applied to an entity's own cropped pixels instead of
# the whole foreground mask, since within a tight bbox crop there is no
# separate foreground/background distinction to make.
def compute_dominant_colors(image_path: str, bbox: tuple[float, float, float, float] | None
                             ) -> tuple[str, ...] | None:
    if bbox is None:
        return None
    try:
        from PIL import Image
        import numpy as np
    except ImportError:  # pragma: no cover -- Pillow/numpy are hard project dependencies
        return None
    if not Path(image_path).exists():
        return None
    x, y, w, h = bbox
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        iw, ih = img.size
        left, top = max(0, int(x * iw)), max(0, int(y * ih))
        right, bottom = min(iw, int((x + w) * iw)), min(ih, int((y + h) * ih))
        if right <= left or bottom <= top:
            return None
        crop = img.crop((left, top, right, bottom))
    pixels = np.asarray(crop, dtype=np.float32).reshape(-1, 3)
    if pixels.size == 0:
        return None
    bins = {
        "red": (pixels[:, 0] > pixels[:, 1] * 1.35) & (pixels[:, 0] > pixels[:, 2] * 1.35),
        "green": (pixels[:, 1] > pixels[:, 0] * 1.25) & (pixels[:, 1] > pixels[:, 2] * 1.15),
        "blue": (pixels[:, 2] > pixels[:, 0] * 1.25) & (pixels[:, 2] > pixels[:, 1] * 1.15),
        "yellow": (pixels[:, 0] > 150) & (pixels[:, 1] > 130) & (pixels[:, 2] < 120),
        "dark": pixels.mean(axis=1) < 80,
    }
    ratios = {name: float(values.mean()) for name, values in bins.items()}
    meaningful = tuple(sorted(name for name, value in ratios.items() if value >= 0.15))
    return meaningful or None


@dataclass(frozen=True)
class VisualEntity:
    entity_id: str
    entity_type: str                            # one of ENTITY_TYPES
    canonical_label: str
    candidate_labels: tuple[tuple[str, float], ...]   # [(label, confidence), ...]
    aliases_en: tuple[str, ...]
    aliases_ar: tuple[str, ...]
    broader_categories: tuple[str, ...]
    possible_subtypes: tuple[str, ...]           # empty today -- see module docstring
    visual_similarities: tuple[str, ...]         # empty today -- see module docstring
    bbox: tuple[float, float, float, float] | None
    crop_ref: str | None                         # not populated this phase (no crop-saving step yet)
    dominant_colors: tuple[str, ...] | None
    relative_size: float | None
    page_position: str | None
    shape_features: dict | None                  # None today -- no shape detector exists
    line_features: dict | None                   # None today -- no per-entity line detector exists
    detector: str
    checkpoint: str
    prompt: str
    confidence: float
    model_validation_status: str                 # VALIDATED | EXPERIMENTAL | UNKNOWN | DISABLED_FOR_RULES
    case_verification_status: str                 # one of CASE_VERIFICATION_STATUSES
    evidence_status: str
    rule_mapping_status: str
    related_rule_ids: tuple[str, ...]
    source: str
    query: str | None
    timestamp: str
    memory_status: str = "not_indexed"           # placeholder -- Visual Memory does not exist yet

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id, "entity_type": self.entity_type,
            "canonical_label": self.canonical_label,
            "candidate_labels": [list(c) for c in self.candidate_labels],
            "aliases_en": list(self.aliases_en), "aliases_ar": list(self.aliases_ar),
            "broader_categories": list(self.broader_categories),
            "possible_subtypes": list(self.possible_subtypes),
            "visual_similarities": list(self.visual_similarities),
            "bbox": self.bbox, "crop_ref": self.crop_ref,
            "dominant_colors": list(self.dominant_colors) if self.dominant_colors else None,
            "relative_size": self.relative_size, "page_position": self.page_position,
            "shape_features": self.shape_features, "line_features": self.line_features,
            "detector": self.detector, "checkpoint": self.checkpoint, "prompt": self.prompt,
            "confidence": self.confidence, "model_validation_status": self.model_validation_status,
            "case_verification_status": self.case_verification_status,
            "evidence_status": self.evidence_status, "rule_mapping_status": self.rule_mapping_status,
            "related_rule_ids": list(self.related_rule_ids), "source": self.source, "query": self.query,
            "timestamp": self.timestamp, "memory_status": self.memory_status,
        }

    @staticmethod
    def from_dict(d: dict) -> "VisualEntity":
        return VisualEntity(
            entity_id=d["entity_id"], entity_type=d.get("entity_type", "unknown"),
            canonical_label=d["canonical_label"],
            candidate_labels=tuple(tuple(c) for c in d.get("candidate_labels") or ()),
            aliases_en=tuple(d.get("aliases_en") or ()), aliases_ar=tuple(d.get("aliases_ar") or ()),
            broader_categories=tuple(d.get("broader_categories") or ()),
            possible_subtypes=tuple(d.get("possible_subtypes") or ()),
            visual_similarities=tuple(d.get("visual_similarities") or ()),
            bbox=tuple(d["bbox"]) if d.get("bbox") else None, crop_ref=d.get("crop_ref"),
            dominant_colors=tuple(d["dominant_colors"]) if d.get("dominant_colors") else None,
            relative_size=d.get("relative_size"), page_position=d.get("page_position"),
            shape_features=d.get("shape_features"), line_features=d.get("line_features"),
            detector=d["detector"], checkpoint=d["checkpoint"], prompt=d["prompt"],
            confidence=d["confidence"], model_validation_status=d["model_validation_status"],
            case_verification_status=d.get("case_verification_status", "unverified"),
            evidence_status=d["evidence_status"], rule_mapping_status=d["rule_mapping_status"],
            related_rule_ids=tuple(d.get("related_rule_ids") or ()), source=d.get("source", "initial_scan"),
            query=d.get("query"), timestamp=d.get("timestamp", ""),
            memory_status=d.get("memory_status", "not_indexed"),
        )

    def matches_search_term(self, term: str) -> bool:
        """Case-insensitive match against canonical label OR any alias --
        the richer counterpart to visual_evidence.find_matching, adding
        alias/vocabulary search without changing that function's own
        (still-used-elsewhere) behavior."""
        q = term.strip().lower()
        if not q:
            return False
        haystack = {self.canonical_label.lower(), *[a.lower() for a in self.aliases_en],
                    *[a.lower() for a in self.aliases_ar]}
        return any(q == h or q in h or h in q for h in haystack)


def visual_finding_to_entity(finding: VisualFinding, *, image_path: str | None = None) -> VisualEntity:
    """The conversion this phase exists to add: every VisualFinding
    becomes exactly one VisualEntity, preserving every existing field
    and populating only the NEW attributes that are reliably derivable
    from information DOAR already has (bbox-derived geometry/colour if a
    real bbox exists; vocabulary/type from the label). Nothing is
    invented: a finding with no bbox (most current object-class
    detections -- Phase 2C.7 never tracked one) correctly gets
    `dominant_colors=None`/`relative_size=None`/`page_position=None`/
    `crop_ref=None`, never a fabricated value."""
    aliases_en, aliases_ar = _aliases_for(finding.label)
    dominant_colors = compute_dominant_colors(image_path, finding.bbox) if image_path else None
    return VisualEntity(
        entity_id=finding.finding_id, entity_type=classify_entity_type(finding.label),
        canonical_label=finding.label, candidate_labels=((finding.label, finding.confidence),),
        aliases_en=aliases_en, aliases_ar=aliases_ar,
        broader_categories=_BROADER_CATEGORIES.get(finding.label, ()),
        possible_subtypes=(), visual_similarities=(),
        bbox=finding.bbox, crop_ref=None,
        dominant_colors=dominant_colors, relative_size=compute_relative_size(finding.bbox),
        page_position=compute_page_position(finding.bbox),
        shape_features=None, line_features=None,
        detector=finding.detector, checkpoint=finding.checkpoint, prompt=finding.prompt,
        confidence=finding.confidence, model_validation_status=finding.validation_status,
        case_verification_status="unverified", evidence_status=finding.evidence_status,
        rule_mapping_status=finding.rule_mapping_status, related_rule_ids=finding.related_rule_ids,
        source=finding.source, query=finding.query, timestamp=finding.timestamp,
    )


def build_entities_from_findings(findings: list[VisualFinding], *, image_path: str | None = None
                                  ) -> list[VisualEntity]:
    return [visual_finding_to_entity(f, image_path=image_path) for f in findings]


def apply_expert_review_to_entities(entities: list[VisualEntity], review_history: list[dict]
                                     ) -> list[VisualEntity]:
    """Derives `case_verification_status` from the EXISTING
    `clinician_review.json` history (expert_review.py is untouched --
    this only READS its output) -- the latest `confirm`/`reject` action
    whose `target_label` matches an entity's `canonical_label` (or one of
    its aliases) sets that entity's status; `rename` and `note` actions
    do not change verification status (they are recorded in the review
    history itself, already preserved there). Entities with no matching
    review action stay "unverified". Never mutates formal ground
    truth -- this is a per-case, display-only projection."""
    latest_action_by_label: dict[str, str] = {}
    for entry in review_history:
        label = (entry.get("target_label") or "").strip().lower()
        action = entry.get("action")
        if label and action in ("confirm", "reject"):
            latest_action_by_label[label] = action
    status_for_action = {"confirm": "verified", "reject": "rejected"}

    updated = []
    for entity in entities:
        candidates = {entity.canonical_label.lower(), *[a.lower() for a in entity.aliases_en]}
        action = next((latest_action_by_label[c] for c in candidates if c in latest_action_by_label), None)
        new_status = status_for_action.get(action, "unverified") if action else "unverified"
        if new_status != entity.case_verification_status:
            entity = _with_verification_status(entity, new_status)
        updated.append(entity)
    return updated


def _with_verification_status(entity: VisualEntity, status: str) -> VisualEntity:
    d = entity.to_dict()
    d["case_verification_status"] = status
    return VisualEntity.from_dict(d)
