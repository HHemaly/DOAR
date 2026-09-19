"""Case context: everything about the child/task that is not the image.

Every field defaults to `None` -- unknown -- and nothing in this codebase
may infer a value for one of these fields from the drawing itself. A
rule that requires `age` when `context.age is None` must resolve to a
named missing-context outcome (see `ontology.engine`), never silently
skip the check or guess a plausible value.

This is the one context object for the whole pipeline. There is no
separate "image-only" vs. "with-context" code path: setting
`context.age = 9` and re-running `evaluate_rule`/`evaluate_all_rules`
with the same `FeatureStore` re-evaluates every age-gated rule with no
other code change required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CaseContext:
    age: float | None = None
    sex: str | None = None
    drawing_task: str | None = None  # e.g. "free_drawing", "draw_a_person", "kinetic_family_drawing"
    narrative: str | None = None  # what the child said about the drawing, verbatim
    other_clinical_history: str | None = None

    def has(self, field_name: str) -> bool:
        return getattr(self, field_name) is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EMPTY_CONTEXT = CaseContext()
