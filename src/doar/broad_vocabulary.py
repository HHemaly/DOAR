"""DOAR MVP: the broad open-vocabulary detection vocabulary for the
initial visual scan -- combines (1) every target already evaluated in
Phase 2C.4/2C.4A/2C.7 (`phase2c7.detector_policy`), which carries a real
`validation_status`, and (2) a small, deliberately modest set of
additional common child-drawing nouns that have NEVER been individually
measured against human ground truth -- these always get `validation_
status="UNKNOWN"` downstream (visual_evidence.py), never fabricated as
validated.

Per this phase's explicit instruction: detection coverage and
rule-validation status are different concepts. This module only decides
WHAT to look for; it says nothing about whether a hit should be trusted.
"""
from __future__ import annotations

from .phase2c7.detector_policy import OBJECT_CLASS_POLICY
from .phase2c7.visual_detector import NON_DISPATCHABLE_TARGETS

# Common child-drawing content beyond the 12 already-benchmarked targets
# (10 Phase 2C.4 object classes + eye/mouth). Never individually
# evaluated against human ground truth -- always UNKNOWN validation
# status. Deliberately modest, not exhaustive, per instruction ("a
# practical broad visual vocabulary", not an unbounded one).
UNVALIDATED_EXTRA_TARGETS: tuple[str, ...] = (
    "sun", "moon", "cloud", "flower", "bird", "cat", "dog",
    "door", "window", "road", "weapon_or_weapon_like_object",
)


def frozen_policy_targets(eye_entry) -> tuple[str, ...]:
    """The dispatchable targets from the frozen Phase 2C.7 policy (every
    OBJECT_CLASS_POLICY entry except documentation-only rows) plus eye."""
    from .phase2c7.detector_policy import full_policy

    policy = full_policy(eye_entry)
    return tuple(t for t in policy if t not in NON_DISPATCHABLE_TARGETS)


def broad_scan_vocabulary(eye_entry) -> tuple[str, ...]:
    """Every target the initial scan attempts: frozen-policy targets
    (whatever their validation status -- DISABLED targets are excluded
    downstream by visual_detector.analyze_image itself, not here) plus
    the unvalidated extras. Deduplicated, order-preserving."""
    seen: dict[str, None] = {}
    for t in frozen_policy_targets(eye_entry):
        seen.setdefault(t, None)
    for t in UNVALIDATED_EXTRA_TARGETS:
        seen.setdefault(t, None)
    return tuple(seen)


def is_frozen_policy_target(target: str) -> bool:
    return target in OBJECT_CLASS_POLICY or target == "eye"
