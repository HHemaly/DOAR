"""Phase 2C.5 Stage 7: staged expansion sampling protocol -- DESIGN ONLY.

`select_expansion_sample` is implemented and unit-tested against synthetic
fixtures in this phase, but is NEVER invoked against the real
duplicate-group manifest this phase -- no 300-500-image (or any-size)
expansion sample has actually been drawn. That is the explicit next step
pending your review of this protocol (Stage 7's own stop condition).

Wraps (never modifies) Phase 2B's own frozen `select_pilot_sample`
selection logic (src/doar/phase2b/dataset.py) -- same group-disjoint,
non-conflicted, deterministic-seed selection Phase 2C.1's 80-image pilot
itself used -- adding only an `exclude_image_ids` filter so an expansion
round can never re-select an image already used by an earlier round.
Never reads or filters on emotion-class label; `load_group_lookup`
(Phase 2B, frozen) never exposed one to begin with.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Distinct from Phase 2B/2C.1's own DEFAULT_SEED=2026 (src/doar/phase2b/dataset.py)
# specifically so an expansion round can never silently reproduce the exact
# same shuffle and re-select the original 80-pilot's groups even without
# the exclusion filter below acting as a second, independent safeguard.
EXPANSION_SEED_STAGE_B = 20260809


@dataclass(frozen=True)
class StageTarget:
    stage: str
    n_min: int | None
    n_max: int | None
    status: str
    gate: str
    note: str


STAGED_EXPANSION_PLAN: list[StageTarget] = [
    StageTarget(
        stage="A_current_dev_pilot", n_min=80, n_max=80, status="already_exists",
        gate="none -- reused as-is",
        note="The existing Phase 2C.1 73-image dev-eligible cohort, used this phase to "
             "validate the schema/UI/proposal pipeline. Not resampled.",
    ),
    StageTarget(
        stage="B_first_expansion", n_min=300, n_max=500, status="protocol_designed_not_executed",
        gate="(1) 10-20 image feasibility pilot judged acceptable per target; "
             "(2) annotation UI/schema verified against the pilot; "
             "(3) this protocol reviewed and approved by the user.",
        note="Drawn via select_expansion_sample() below from train/valid-eligible images "
             "only, excluding every image_id already used by Stage A, with a fixed, "
             "recorded seed. No proposal run or annotation begins until all three gates "
             "above are satisfied.",
    ),
    StageTarget(
        stage="C_larger_expansion", n_min=1000, n_max=None, status="conditional_not_designed_in_detail",
        gate="Stage B's learning-curve (does per-class support keep changing rule-coverage "
             "conclusions as n grows?) and annotator-burden numbers justify continuing.",
        note="Sizing/seed to be finalized after Stage B's readiness metrics exist -- not "
             "invented ahead of that evidence.",
    ),
    StageTarget(
        stage="D_remaining_dataset", n_min=None, n_max=None, status="conditional_not_designed",
        gate="Only if Stage C's results show continued scientific or practical value "
             "(e.g. rule-coverage or fine-tuning-readiness still improving with more data).",
        note="Full 3,688-drawing corpus. No plan committed to this stage happening at all.",
    ),
]


def select_expansion_sample(n: int, *, seed: int, exclude_image_ids: set[str],
                             manifest_path=None) -> list[dict]:
    """Group-disjoint, non-conflicted, deterministic selection -- identical
    algorithm to phase2b.dataset.select_pilot_sample, plus an
    `exclude_image_ids` filter applied before shuffling. Never modifies
    phase2b/dataset.py or phase2b/duplicate_groups.py."""
    from ..phase2b.duplicate_groups import load_group_lookup

    lookup = load_group_lookup(manifest_path)
    clean = {iid: info for iid, info in lookup.items()
             if info["conflict_status"] == "clean" and iid not in exclude_image_ids}
    by_group: dict[str, list[str]] = {}
    for iid, info in clean.items():
        by_group.setdefault(info["group_id"], []).append(iid)

    group_ids = sorted(by_group.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    if len(group_ids) < n:
        raise ValueError(
            f"Only {len(group_ids)} distinct clean, non-excluded duplicate-groups are "
            f"available, fewer than the requested sample size {n}.")
    selected = []
    for gid in group_ids[:n]:
        members = sorted(by_group[gid])
        image_id = members[0]
        selected.append({"image_id": image_id, "path": clean[image_id]["path"], "group_id": gid})
    selected.sort(key=lambda r: r["image_id"])
    return selected


def to_protocol_dict() -> dict:
    return {
        "expansion_seed_stage_b": EXPANSION_SEED_STAGE_B,
        "selection_method": "select_expansion_sample (group-disjoint, non-conflicted, "
                             "excludes Stage A image_ids, no emotion-label filtering)",
        "stages": [
            {"stage": s.stage, "n_min": s.n_min, "n_max": s.n_max, "status": s.status,
             "gate": s.gate, "note": s.note}
            for s in STAGED_EXPANSION_PLAN
        ],
        "privacy": "Blinded pilot IDs assigned via phase2b.dataset.blind_copy_sample "
                    "(frozen, unmodified); original source mapping stored only under "
                    "outputs/ (gitignored); no drawing, mapping, or emotion-folder path is "
                    "ever committed.",
        "executed_this_phase": False,
    }
