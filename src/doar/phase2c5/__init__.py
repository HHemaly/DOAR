"""Phase 2C.5 -- rule-critical model-assisted annotation.

Creates trusted, human-reviewed visual evidence (eye/mouth/hand/face/person
bounding boxes, plus a narrow eye state/detail attribute) for the specific
DOAR rules in resources/psychology_sources/rules_registry_v2.json that
actually need it (see rule_traceability.py) -- not a generic object
ontology expansion. Detector output (Grounding DINO primary,
OWLv2 comparison, both reused from Phase 2C.4's frozen, unmodified loaders)
is only ever a PROPOSAL; a proposal becomes trusted annotation only after
explicit human review (schema.BBOX_SOURCES). Never touches
src/doar/phase2c1/{schema,store}.py or any Phase 2C.4/2C.4A frozen
artifact -- this is a new, additive schema/store, versioned separately
(schema.SCHEMA_VERSION), backward-compatible in spirit (same status
vocabulary, same annotator_type/review_status conventions, same
atomic-write CSV store pattern) but not the same file.

No rule is activated by this package.

No emotion label (Angry/Fear/Happy/Sad) is referenced anywhere in it.
"""
