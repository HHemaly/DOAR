"""Machine-readable rule registry v2 (Phase 1).

This module does **not** replace `resources/psychology_sources/rules_registry.json`
or its loader in `rules.py` -- that registry and the tier-1 dispatcher
`rules.py::evaluate_rules` remain the live, tested path for the existing
pipeline (`analyze_image`) and are left untouched.

What this module adds is a **provenance and threshold-source layer** on
top of the same 19 rules: for each rule, exactly where in the sole
supplied source PDF it comes from (document, page, informal section
heading), and -- the specific gap `RULE_FEATURE_COVERAGE.md` identified --
whether its numerical threshold (if any) is directly stated by the
source, a tolerance band around a stated center, or has no numeric
support in the source at all. This provenance was derived by reading the
PDF text directly this session; it is not invented, and it does not
change any rule's `confidence_ceiling`, wording, or evaluation outcome.

`RuleV2` is what `evidence_rule_engine.py` consumes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REGISTRY = Path(__file__).resolve().parents[2] / "resources" / "psychology_sources" / "rules_registry.json"
SOURCE_PDF = "التحليل النفسي للصور.pdf"

# threshold_source values:
#   "directly_sourced"       -- the PDF states this exact number
#   "sourced_center_invented_band"  -- PDF states a center value; the tolerance
#                                       band around it is an implementation choice
#   "invented_numeric_stand_in"     -- PDF gives a qualitative phrase only; a
#                                       number was chosen with no source anchor
#   "invented_no_anchor"     -- PDF gives no number or qualitative magnitude at
#                               all for this boundary (e.g. "top" with no split point)
#   "not_applicable"         -- rule has no numeric threshold (tier-2, content-presence only)
RULE_PROVENANCE: dict[str, dict[str, Any]] = {
    "PSY_AR_EYES_WIDE_001": {"page": 1, "section": "رسم العيون (Eye drawing)", "threshold_source": "not_applicable"},
    "PSY_AR_EYES_STERN_002": {"page": 1, "section": "رسم العيون", "threshold_source": "not_applicable"},
    "PSY_AR_EYES_CLOSED_003": {"page": 1, "section": "رسم العيون", "threshold_source": "not_applicable"},
    "PSY_AR_ANIMAL_TIGER_WOLF_004": {"page": 1, "section": "رسم الحيوانات (Animal drawing)", "threshold_source": "not_applicable"},
    "PSY_AR_ANIMAL_FOX_005": {"page": 1, "section": "رسم الحيوانات", "threshold_source": "not_applicable"},
    "PSY_AR_ANIMAL_SQUIRREL_006": {"page": 1, "section": "رسم الحيوانات", "threshold_source": "not_applicable"},
    "PSY_AR_ANIMAL_LION_007": {"page": 1, "section": "رسم الحيوانات", "threshold_source": "not_applicable"},
    "PSY_AR_GEOMETRY_008": {"page": 1, "section": "الأشكال الهندسية (Geometric shapes)", "threshold_source": "not_applicable"},
    "PSY_AR_STARS_009": {"page": 1, "section": "الأشكال الهندسية — النجوم (Stars)", "threshold_source": "not_applicable"},
    "PSY_AR_FLOWERS_CLOUDS_SUN_010": {"page": 1, "section": "الأشكال الهندسية — الزهور والسحب والشمس (Flowers/clouds/sun)", "threshold_source": "not_applicable",
                                       "note": "Precondition ('while distracted') is not observable from a static image; see CURRENT_SYSTEM_AUDIT.md Section 10."},
    "PSY_AR_CIRCLES_011": {"page": 1, "section": "الأشكال الهندسية — الدوائر (Circles)", "threshold_source": "not_applicable"},
    "PSY_AR_TRANSPORT_012": {"page": 1, "section": "رسم وسائل النقل (Transportation)", "threshold_source": "not_applicable"},
    "PSY_AR_HEARTS_013": {"page": 2, "section": "رسم القلوب (Heart drawing)", "threshold_source": "not_applicable"},
    "PSY_AR_SIZE_HALF_014": {"page": 2, "section": "حجم الرسم (Drawing size)", "threshold_source": "sourced_center_invented_band",
                              "source_quote": "تغطي حوالي 50% من حجم الورقة", "operational_threshold": "0.40 <= coverage <= 0.60",
                              "provenance_note": "PDF states the center ('about 50%'); the +/-10 percentage-point tolerance band is an implementation choice, not stated in the source."},
    "PSY_AR_SIZE_FULL_015": {"page": 2, "section": "حجم الرسم", "threshold_source": "invented_numeric_stand_in",
                              "source_quote": "تغطي كل الورقة", "operational_threshold": "coverage >= 0.90",
                              "provenance_note": "PDF says only 'covers the whole page' (no percentage); 0.90 is a practical stand-in chosen by the implementation."},
    "PSY_AR_SIZE_SMALL_016": {"page": 2, "section": "حجم الرسم", "threshold_source": "directly_sourced",
                               "source_quote": "لا تتجاوز 20%", "operational_threshold": "0 < coverage <= 0.20",
                               "provenance_note": "PDF states this exact percentage; the implementation's threshold matches it directly."},
    "PSY_AR_PLACE_TOP_017": {"page": 2, "section": "موقع الرسم (Drawing location)", "threshold_source": "invented_no_anchor",
                              "source_quote": "في أعلى الصفحة", "operational_threshold": "centroid_y < 0.4",
                              "provenance_note": "PDF gives no numeric boundary for 'top' at all; the 0.4 vertical split is entirely an implementation choice."},
    "PSY_AR_PLACE_LEFT_018": {"page": 2, "section": "موقع الرسم", "threshold_source": "invented_no_anchor",
                               "source_quote": "الجانب الأيسر", "operational_threshold": "centroid_x < 0.4",
                               "provenance_note": "PDF gives no numeric boundary for 'left' at all; the 0.4 horizontal split is entirely an implementation choice."},
    "PSY_AR_PLACE_RIGHT_019": {"page": 2, "section": "موقع الرسم", "threshold_source": "invented_no_anchor",
                                "source_quote": "الجانب الأيمن", "operational_threshold": "centroid_x > 0.6",
                                "provenance_note": "PDF gives no numeric boundary for 'right' at all; the 0.6 horizontal split is entirely an implementation choice."},
}

# Which canonical evidence feature_id(s) each rule's observable requires.
# Tier-1 rules map to real, implemented features; tier-2 rules map to
# feature_ids that currently have no extractor (see evidence_adapter.py
# UNIMPLEMENTED_CATEGORIES) -- looking one up must resolve to an explicit
# `unavailable` evidence item, never a KeyError and never a silent pass.
REQUIRED_FEATURE_IDS: dict[str, list[str]] = {
    "wide_eyes": ["semantic.face.eyes"],
    "stern_eyes": ["semantic.face.eyes"],
    "closed_eyes": ["semantic.face.eyes"],
    "tiger_or_wolf": ["semantic.object.animal_species"],
    "fox": ["semantic.object.animal_species"],
    "squirrel": ["semantic.object.animal_species"],
    "lion": ["semantic.object.animal_species"],
    "repeated_geometric_shapes": ["geometry.shape_repetition"],
    "stars": ["semantic.object.symbol"],
    "flowers_clouds_sun": ["semantic.object.symbol"],
    "circles": ["geometry.primitive_shape"],
    "vehicles": ["semantic.object.category"],
    "hearts": ["semantic.object.symbol"],
    "coverage_about_half": ["composition.bounding_box_coverage"],
    "coverage_full": ["composition.bounding_box_coverage"],
    "coverage_small": ["composition.bounding_box_coverage"],
    "placement_top": ["composition.centroid_normalized"],
    "placement_left": ["composition.centroid_normalized"],
    "placement_right": ["composition.centroid_normalized"],
}


@dataclass(frozen=True)
class RuleV2:
    rule_id: str
    title_english: str
    arabic: str
    observable: str
    tier: str
    activation_status: str
    source_document: str
    source_page: int
    source_section: str
    faithful_source_quote: str | None
    required_feature_ids: list[str]
    threshold_source: str
    operational_threshold: str | None
    provenance_note: str | None
    scientific_support: str
    confidence_ceiling: float
    professional_reasoning: str
    parent_safe_wording: str
    references: list[str]
    limitations: list[str]
    requires_clinician_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rules_v2() -> list[RuleV2]:
    """Load the 19 registry rules and attach the provenance layer. Raises
    if any registry rule is missing provenance or a feature-id mapping --
    a new rule added to the registry without updating this module must
    fail loudly, not silently lose its citation."""
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    out: list[RuleV2] = []
    for rule in raw["rules"]:
        rid = rule["rule_id"]
        if rid not in RULE_PROVENANCE:
            raise ValueError(f"Rule {rid!r} has no entry in RULE_PROVENANCE — cannot cite its source.")
        if rule["observable"] not in REQUIRED_FEATURE_IDS:
            raise ValueError(f"Rule {rid!r} observable {rule['observable']!r} has no REQUIRED_FEATURE_IDS mapping.")
        prov = RULE_PROVENANCE[rid]
        out.append(RuleV2(
            rule_id=rid,
            title_english=rule["english"],
            arabic=rule["arabic"],
            observable=rule["observable"],
            tier=rule["tier"],
            activation_status=rule["activation_status"],
            source_document=SOURCE_PDF,
            source_page=prov["page"],
            source_section=prov["section"],
            faithful_source_quote=prov.get("source_quote"),
            required_feature_ids=list(REQUIRED_FEATURE_IDS[rule["observable"]]),
            threshold_source=prov["threshold_source"],
            operational_threshold=prov.get("operational_threshold"),
            provenance_note=prov.get("provenance_note") or prov.get("note"),
            scientific_support=rule["scientific_support"],
            confidence_ceiling=float(rule["confidence_ceiling"]),
            professional_reasoning=rule["professional_reasoning"],
            parent_safe_wording=rule["parent_safe_wording"],
            references=list(rule["references"]),
            limitations=list(rule["limitations"]),
        ))
    return out
