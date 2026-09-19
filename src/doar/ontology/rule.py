"""Declarative rule schema (DOAR V2).

A rule is data, not a Python `if` statement: its premise is a composable
tree of feature conditions, evaluated by `ontology.engine`, never
hand-coded per rule. This is what lets the engine be generic (one
evaluator for every rule) instead of accumulating one bespoke function
per rule the way an ad hoc implementation would.

`RuleFramework` is the distinction the task requires kept visible and
never blurred: `CLASSICAL_PROJECTIVE` rules encode a symbol/interpretation
folk-psychology claim (the kind found in projective-drawing literature,
including this repository's own Arabic source notes -- see
`rule_schema.py`/`resources/psychology_sources/rules_registry.json`);
`EMPIRICAL_EXPRESSIVE` is reserved for a rule whose link between an
observable drawing feature and an expressive/affective label is
supported by a cited, checkable empirical study. **No rule may claim
`EMPIRICAL_EXPRESSIVE` without a real citation in `citation_full`** --
`legacy_adapters.py` tags every rule sourced from the existing 19-rule
registry `CLASSICAL_PROJECTIVE`, because that registry's own scientific
support fields document that none of its claims have empirical backing
(most explicitly `not_found_for_specific_claim`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class RuleFramework:
    EMPIRICAL_EXPRESSIVE = "EMPIRICAL_EXPRESSIVE"
    CLASSICAL_PROJECTIVE = "CLASSICAL_PROJECTIVE"
    # Reserved for a rule whose framework has not yet been determined by a
    # literature check -- a rule must never default here silently; loaders
    # must set one of the two real values above or raise.
    UNCLASSIFIED = "UNCLASSIFIED"

    ALL = (EMPIRICAL_EXPRESSIVE, CLASSICAL_PROJECTIVE, UNCLASSIFIED)


class EvidenceGrade:
    """How strong the framework's own literature support is for this
    specific rule -- independent of `framework`, since even an
    EMPIRICAL_EXPRESSIVE-framework rule can be weakly supported."""

    NOT_FOUND = "NOT_FOUND_FOR_CLAIM"
    INDIRECT = "INDIRECT_SUPPORT_ONLY"
    CONFLICTING = "CONFLICTING_EVIDENCE_IN_LITERATURE"
    DIRECT = "DIRECTLY_SUPPORTED"
    TODO = "TODO_NOT_YET_ASSESSED"

    ALL = (NOT_FOUND, INDIRECT, CONFLICTING, DIRECT, TODO)


# --------------------------------------------------------------------- #
# Premise tree
# --------------------------------------------------------------------- #

_OPERATORS = ("equals", "not_equals", "gte", "lte", "gt", "lt", "in", "contains")


@dataclass(frozen=True)
class FeatureCondition:
    """A leaf premise: a single claim about one feature's effective
    value. `operator`/`expected` are not evaluated here -- see
    `engine.evaluate_condition` -- so this class stays pure data."""

    feature_id: str
    operator: str = "equals"
    expected: Any = True

    def __post_init__(self) -> None:
        if self.operator not in _OPERATORS:
            raise ValueError(f"FeatureCondition: operator {self.operator!r} not in {_OPERATORS}")

    def referenced_feature_ids(self) -> tuple[str, ...]:
        return (self.feature_id,)


@dataclass(frozen=True)
class AllOf:
    """AND over its sub-premises (three-valued: see `tristate.tri_and`)."""

    conditions: tuple["Premise", ...]

    def referenced_feature_ids(self) -> tuple[str, ...]:
        return tuple(fid for c in self.conditions for fid in c.referenced_feature_ids())


@dataclass(frozen=True)
class AnyOf:
    """OR over its sub-premises (three-valued: see `tristate.tri_or`)."""

    conditions: tuple["Premise", ...]

    def referenced_feature_ids(self) -> tuple[str, ...]:
        return tuple(fid for c in self.conditions for fid in c.referenced_feature_ids())


@dataclass(frozen=True)
class NotOf:
    condition: "Premise"

    def referenced_feature_ids(self) -> tuple[str, ...]:
        return self.condition.referenced_feature_ids()


Premise = FeatureCondition | AllOf | AnyOf | NotOf


# --------------------------------------------------------------------- #
# Citation + requirements
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Citation:
    document: str
    author: str | None = None
    page: int | str | None = None
    section: str | None = None
    doi: str | None = None
    quote: str | None = None
    status: str = "documented"  # "documented" | "TODO"

    def __post_init__(self) -> None:
        if self.status == "TODO" and self.document not in ("TODO", ""):
            # A citation marked TODO must not also smuggle in a fabricated
            # document name -- either it is filled in and documented, or it
            # is honestly empty and TODO.
            raise ValueError("Citation.status == 'TODO' requires document == 'TODO' or ''.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextRequirements:
    age: bool = False
    age_range: tuple[float, float] | None = None
    task: bool = False
    allowed_tasks: tuple[str, ...] | None = None
    narrative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------- #
# Rule
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    framework: str
    premise: Premise
    domain: str  # claimed interpretation/domain, e.g. "sad_expression", "introversion_hypothesis"
    citation: Citation
    evidence_grade: str
    requirements: ContextRequirements = field(default_factory=ContextRequirements)
    confidence_ceiling: float = 0.0
    professional_reasoning: str = ""
    parent_safe_wording: str = ""
    limitations: tuple[str, ...] = ()
    contradicting_rules: tuple[str, ...] = ()
    requires_clinician_review: bool = True

    def __post_init__(self) -> None:
        if self.framework not in RuleFramework.ALL:
            raise ValueError(f"Rule {self.rule_id!r}: framework {self.framework!r} not in {RuleFramework.ALL}")
        if self.evidence_grade not in EvidenceGrade.ALL:
            raise ValueError(f"Rule {self.rule_id!r}: evidence_grade {self.evidence_grade!r} not in {EvidenceGrade.ALL}")
        if self.framework == RuleFramework.EMPIRICAL_EXPRESSIVE and self.citation.status == "TODO":
            raise ValueError(
                f"Rule {self.rule_id!r} claims EMPIRICAL_EXPRESSIVE but has no real citation "
                "(citation.status == 'TODO'). Fabricating empirical support is not permitted -- "
                "either supply a real citation or tag this rule CLASSICAL_PROJECTIVE/UNCLASSIFIED."
            )
        if not (0.0 <= self.confidence_ceiling <= 1.0):
            raise ValueError(f"Rule {self.rule_id!r}: confidence_ceiling {self.confidence_ceiling!r} out of [0, 1]")

    def required_feature_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.premise.referenced_feature_ids()))
