"""Phase 2C.5 part-target ontology -- deliberately small and rule-derived
(see rule_traceability.py), never a general part ontology. `eye` and
`mouth` are genuinely new part classes (absent from Phase 2B/2C.1's
10-class object ontology, src/doar/phase2b/ontology.py::CLASS_NAMES,
unchanged by this module); `hand`/`face`/`person` already exist as
presence-annotated object classes there -- Phase 2C.5 only adds bounding
boxes on top of their existing Phase 2C.1 presence judgments, it does not
re-litigate presence for them.
"""
from __future__ import annotations

PART_ONTOLOGY_VERSION = "phase2c5_part_ontology_v1"

# Every entry here must trace to at least one rule_id in
# rule_traceability.RULE_ANNOTATION_TRACEABILITY -- enforced by a test.
PART_TARGETS = ("eye", "mouth", "hand", "face", "person")

# hand/face/person reuse their existing Phase 2C.1 object-class presence
# judgment (src/doar/phase2b/ontology.py::CLASS_NAMES) as a starting point;
# eye/mouth have no prior presence judgment and are assessed from scratch.
TARGETS_WITH_EXISTING_PRESENCE = frozenset({"hand", "face", "person"})
TARGETS_NEW_THIS_PHASE = frozenset({"eye", "mouth"})

OBJECT_STATUSES = frozenset({"present", "absent", "uncertain", "not_assessable"})

BBOX_SOURCES = frozenset({"model_proposed", "human_accepted", "human_edited", "human_drawn"})

EYE_STATE_VALUES = frozenset({"open", "closed", "uncertain", "not_assessable"})
EYE_DETAIL_VALUES = frozenset({"detailed", "undetailed", "missing", "uncertain", "not_assessable"})

# Only `eye` carries attributes this phase (PSY_AR_EYES_CLOSED_003 needs
# eye_state; EN_COMPILED_EYES_MISSING_DETAIL_020 needs eye_detail). No
# other target has a defensible, operationally-defined attribute yet --
# see rule_traceability.py for what was considered and postponed
# (PSY_AR_EYES_STERN_002, EN_COMPILED_FACE_EXPRESSION_021).
TARGET_ALLOWED_ATTRIBUTE_KEYS: dict[str, frozenset[str]] = {
    "eye": frozenset({"eye_state", "eye_detail"}),
    "mouth": frozenset(),
    "hand": frozenset(),
    "face": frozenset(),
    "person": frozenset(),
}

ATTRIBUTE_ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "eye_state": EYE_STATE_VALUES,
    "eye_detail": EYE_DETAIL_VALUES,
}
