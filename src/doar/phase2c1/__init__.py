"""Phase 2C.1 -- human annotation expansion and detector-readiness scaffolding.

Builds on Phase 2B (src/doar/phase2b/) but never modifies it:
artifacts/phase2b/annotation_manifest.csv stays untouched; this package's own
schema/store are versioned separately (schema.py). No module here imports the
rule-evaluation engine (see tests/test_phase2c1_safety.py) and no supervised
detector training happens in this phase.
"""
