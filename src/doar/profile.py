"""
profile.py -- optional child/context information the parent provides.

This is stored ALONGSIDE a case, never fed into the emotion model,
objective features, or rule evaluation -- those must stay image-only, so
the model can never learn to key off demographic text instead of the
drawing. Its only consumers are report/UI display (clearly tagged
`source: user_provided`, never confused with measured evidence) and the
chat's EvidenceRetriever, which surfaces it as context, not as evidence
for a claim about the drawing itself.

If a future tier_3_prompt_or_age_dependent rule is added to
rules_registry.json (none exist today -- see RULE_AND_FEATURE_COVERAGE.md),
`drawing_instruction` is what would let rules.py resolve it out of
`not_assessable_context_unknown`. Until such a rule exists and is
reviewed, this module must never be imported by rules.py, concerns.py, or
emotion.py -- enforced by a regression test (tests/test_profile.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .case_output import write_versioned

ALLOWED_AGE_RANGES = ("2-3", "4-5", "6-7", "8-9", "10-12", "unspecified")


@dataclass(frozen=True)
class ChildProfile:
    age_range: str = "unspecified"
    gender: str | None = None
    drawing_instruction: str | None = None
    date: str | None = None
    parent_concern: str | None = None
    language: str = "en"

    def __post_init__(self):
        if self.age_range not in ALLOWED_AGE_RANGES:
            raise ValueError(f"age_range must be one of {ALLOWED_AGE_RANGES}, got {self.age_range!r}")
        if self.language not in ("en", "ar"):
            raise ValueError(f"language must be 'en' or 'ar', got {self.language!r}")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["source"] = "user_provided"
        return data


def save_profile(case_dir: str | Path, profile: ChildProfile) -> Path:
    path = Path(case_dir) / "profile.json"
    write_versioned(path, profile.to_dict())
    return path


def load_profile(case_dir: str | Path) -> ChildProfile | None:
    import json
    path = Path(case_dir) / "profile.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("source", None)
    return ChildProfile(**data)
