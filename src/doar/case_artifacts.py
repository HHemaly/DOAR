"""DOAR V1.1: case-relative artifact path resolution.

`analysis.py::analyze_image` deliberately stores every `artifacts.*` path
in `analysis.json` RELATIVE to the case directory (e.g.
`"artifacts/foreground_mask.png"`, not an absolute machine path) --
portable across machines and safe to move a case directory. Any consumer
that re-opens a SAVED case (as opposed to the single `analyze_image` call
that wrote it, which never changes working directory in between) must
resolve those relative paths against the case directory explicitly,
never rely on the process's current working directory.

Found via a real bug (DOAR V1.1 stabilization, Problem A): Technical
View's live re-display of objective features called
`features.objective_feature_row(image_path, analysis)`, and
`objective_feature_row` opens `analysis["artifacts"]["foreground_mask"]`
directly with `Image.open(...)` -- with the app's process cwd (repo
root, wherever `streamlit run` was launched from) instead of the case
directory, `Image.open("artifacts/foreground_mask.png")` raised
`[Errno 2] No such file or directory`, even though the file existed
exactly where `analyze_image` had written it.
"""
from __future__ import annotations

from pathlib import Path


def resolve_artifact_path(case_dir: str | Path, value: str) -> Path:
    """A single artifact path value, absolute or case-relative, resolved
    to an absolute `Path` anchored at `case_dir`. Already-absolute values
    (e.g. from an analysis.json written by older code, or a caller that
    passes machine-absolute paths directly) are returned unchanged --
    never silently rejoined onto `case_dir`."""
    p = Path(value)
    return p if p.is_absolute() else Path(case_dir) / p


def resolve_analysis_artifacts(analysis: dict, case_dir: str | Path) -> dict:
    """Returns a shallow copy of `analysis` with every `artifacts.*` path
    (including nested dicts, e.g. `artifacts.candidate_masks`) resolved to
    an absolute path anchored at `case_dir` as a string -- safe to pass to
    any function that does `Image.open(analysis["artifacts"][...])`
    regardless of the caller's current working directory. Missing/absent
    `artifacts` key is a no-op (returns `analysis` unchanged, not an
    error -- an older case may not have every path)."""
    artifacts = analysis.get("artifacts")
    if not artifacts:
        return analysis
    case_dir = Path(case_dir)
    resolved = {}
    for key, value in artifacts.items():
        if isinstance(value, dict):
            resolved[key] = {nested: str(resolve_artifact_path(case_dir, v)) for nested, v in value.items()}
        else:
            resolved[key] = str(resolve_artifact_path(case_dir, value))
    return {**analysis, "artifacts": resolved}
