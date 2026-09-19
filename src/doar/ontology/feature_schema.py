"""Feature ontology schema (DOAR V2).

A canonical, machine-readable description of what a feature *is*,
independent of any one extractor's implementation of it. This is the
missing layer identified by the V2 audit: `features.py` and
`evidence_adapter.py` both compute feature values, but neither declares,
in one place, a feature's value type, what must be true of the image
before the feature may be assessed at all, or which confounds threaten
its validity.

The one invariant this schema exists to make structurally impossible to
violate downstream: a feature whose `requirements` are not satisfied must
never be treated as "confirmed absent" just because no detector fired.
`FeatureDefinition.requirements` names the prerequisite feature IDs (each
itself a feature in the ontology, evaluated first) that must hold before
this feature may be scored TRUE/FALSE at all; if they do not hold, the
correct state is UNKNOWN / not-assessable, never FALSE. This is the same
rule the task calls out by name for omission rules (missing hands, feet,
eyes): "not detected" must not silently become "confirmed absent."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "feature_ontology.yaml"
)


class FeatureType:
    """Supported value types for a canonical feature. Not an enum so a
    fifth type can be added by editing this tuple alone."""

    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    CONTINUOUS = "continuous"
    SET = "set"
    GEOMETRY = "geometry"

    ALL = (BOOLEAN, CATEGORICAL, ORDINAL, CONTINUOUS, SET, GEOMETRY)


@dataclass(frozen=True)
class FeatureDefinition:
    feature_id: str
    name: str
    category: str
    value_type: str
    definition: str
    unit: str | None = None
    # Other feature_ids (in this same ontology) that must be assessable
    # and TRUE/non-null before this feature may be scored at all, e.g.
    # "eyes_missing" requires "human_figure_present" and
    # "face_region_visible". An empty tuple means the feature has no
    # prerequisite beyond the image existing.
    requirements: tuple[str, ...] = ()
    assessability_notes: str = ""
    candidate_methods: tuple[str, ...] = ()
    confounds: tuple[str, ...] = ()
    allowed_values: tuple[Any, ...] | None = None  # for categorical/ordinal
    rules_using_feature: tuple[str, ...] = field(default_factory=tuple)
    source_module: str | None = None  # where an existing extractor (if any) lives
    status: str = "PLANNED"  # KEEP / ADAPT / REPLACE / LEGACY / PLANNED

    def __post_init__(self) -> None:
        if self.value_type not in FeatureType.ALL:
            raise ValueError(
                f"Feature {self.feature_id!r}: value_type {self.value_type!r} "
                f"not in {FeatureType.ALL}"
            )
        if self.value_type in (FeatureType.CATEGORICAL, FeatureType.ORDINAL) and not self.allowed_values:
            raise ValueError(
                f"Feature {self.feature_id!r}: {self.value_type} features must "
                "declare allowed_values."
            )


class FeatureOntology:
    """A validated, indexed collection of `FeatureDefinition`s."""

    def __init__(self, definitions: list[FeatureDefinition]):
        by_id: dict[str, FeatureDefinition] = {}
        for d in definitions:
            if d.feature_id in by_id:
                raise ValueError(f"Duplicate feature_id in ontology: {d.feature_id!r}")
            by_id[d.feature_id] = d
        # Every declared requirement must itself be a known feature -- an
        # ontology that references a prerequisite it never defines can
        # never be safely evaluated (the engine would have no way to know
        # whether that prerequisite holds).
        for d in definitions:
            for req in d.requirements:
                if req not in by_id:
                    raise ValueError(
                        f"Feature {d.feature_id!r} requires undefined feature {req!r}"
                    )
        self._by_id = by_id

    def __contains__(self, feature_id: str) -> bool:
        return feature_id in self._by_id

    def __getitem__(self, feature_id: str) -> FeatureDefinition:
        return self._by_id[feature_id]

    def get(self, feature_id: str) -> FeatureDefinition | None:
        return self._by_id.get(feature_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def requirements_of(self, feature_id: str) -> tuple[str, ...]:
        return self[feature_id].requirements


def _definition_from_dict(feature_id: str, raw: dict[str, Any]) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=feature_id,
        name=raw["name"],
        category=raw["category"],
        value_type=raw["type"],
        definition=raw["definition"],
        unit=raw.get("unit"),
        requirements=tuple(raw.get("requirements", []) or []),
        assessability_notes=raw.get("assessability", {}).get("description", "") if isinstance(raw.get("assessability"), dict) else str(raw.get("assessability", "")),
        candidate_methods=tuple(raw.get("candidate_methods", []) or []),
        confounds=tuple(raw.get("confounds", []) or []),
        allowed_values=tuple(raw["allowed_values"]) if raw.get("allowed_values") else None,
        rules_using_feature=tuple(raw.get("rules_using_feature", []) or []),
        source_module=raw.get("source_module"),
        status=raw.get("status", "PLANNED"),
    )


def load_feature_ontology(path: str | Path | None = None) -> FeatureOntology:
    """Load and validate the canonical feature ontology from YAML (or
    JSON -- the loader accepts either since YAML is a superset)."""
    p = Path(path) if path is not None else DEFAULT_ONTOLOGY_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    features = raw.get("features", raw)
    definitions = [_definition_from_dict(fid, body) for fid, body in features.items()]
    return FeatureOntology(definitions)


def dump_feature_ontology_json(ontology: FeatureOntology) -> str:
    """Serialize an ontology back to JSON -- used by audit tooling that
    wants a machine-readable snapshot without a YAML dependency."""
    payload = {
        d.feature_id: {
            "name": d.name,
            "category": d.category,
            "type": d.value_type,
            "unit": d.unit,
            "definition": d.definition,
            "requirements": list(d.requirements),
            "assessability": d.assessability_notes,
            "candidate_methods": list(d.candidate_methods),
            "confounds": list(d.confounds),
            "allowed_values": list(d.allowed_values) if d.allowed_values else None,
            "rules_using_feature": list(d.rules_using_feature),
            "source_module": d.source_module,
            "status": d.status,
        }
        for d in ontology
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
