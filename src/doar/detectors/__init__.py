"""
detectors/ -- Phase 3 evaluation scaffolding ONLY.

No detector is implemented, adopted, or enabled here. This package exists so
that whichever detector is eventually piloted (per
PHASE3_DETECTOR_EVALUATION_PLAN.md) has a stable, testable contract to
implement (schema.py) and a way to be scored against the held-out annotation
sample (metrics.py, docs/PHASE3_ANNOTATION_SCHEMA.md) before it can ever be
wired into rules.py.

Nothing in this package is imported by analysis.py, rules.py, or any other
user-facing code path -- it has no effect on current output.
"""
