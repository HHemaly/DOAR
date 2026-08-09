"""Phase 2C.7 Stage 3: deterministic development/holdout split over the
real, human-reviewed eye images -- prevents any threshold/model-selection
decision from touching the same images used to report a final number.

"Group-disjoint" here is inherited for free, not re-derived: every
pilot_id in the Phase 2C.6 expansion manifest already corresponds to
exactly one Phase 7B duplicate-group (Phase 2B's own `select_pilot_sample`
/ Phase 2C.5's `select_expansion_sample` both select at most one image per
group before blinding) -- so a plain pilot_id-level split cannot put two
images from the same duplicate-group on opposite sides of the split. This
module does not need to re-check group membership; it is a structural
guarantee already tested at the point pilot_ids were selected
(tests/test_phase2c6_expansion_manifest.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

DEV_HOLDOUT_SEED = 20260809  # reuses Phase 2C.5's own EXPANSION_SEED_STAGE_B value for traceability
DEV_FRACTION = 0.8


@dataclass(frozen=True)
class DevHoldoutSplit:
    dev_pilot_ids: tuple[str, ...]
    holdout_pilot_ids: tuple[str, ...]
    seed: int
    dev_fraction: float


def split_dev_holdout(pilot_ids: list[str], *, seed: int = DEV_HOLDOUT_SEED,
                       dev_fraction: float = DEV_FRACTION) -> DevHoldoutSplit:
    """Deterministic: same `pilot_ids` + `seed` always produces the same
    split, order-independent (sorts before shuffling)."""
    if not (0.0 < dev_fraction < 1.0):
        raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction}")
    ordered = sorted(pilot_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_dev = round(len(ordered) * dev_fraction)
    dev_ids = tuple(sorted(ordered[:n_dev]))
    holdout_ids = tuple(sorted(ordered[n_dev:]))
    return DevHoldoutSplit(dev_pilot_ids=dev_ids, holdout_pilot_ids=holdout_ids,
                            seed=seed, dev_fraction=dev_fraction)


def assert_no_overlap(split: DevHoldoutSplit) -> None:
    overlap = set(split.dev_pilot_ids) & set(split.holdout_pilot_ids)
    if overlap:
        raise ValueError(f"dev/holdout overlap detected: {sorted(overlap)}")


def assert_holdout_untouched_by(used_pilot_ids: set[str], split: DevHoldoutSplit) -> None:
    """Call this before reporting ANY holdout number -- raises if a
    threshold/model-selection step's own used-pilot-id set intersects the
    holdout. Mirrors Phase 2C.1's `assert_pilot_ids_exclude_locked_test`
    guard pattern exactly, applied to this phase's own protected split."""
    leaked = used_pilot_ids & set(split.holdout_pilot_ids)
    if leaked:
        raise ValueError(f"holdout pilot_ids used before freezing: {sorted(leaked)}")
