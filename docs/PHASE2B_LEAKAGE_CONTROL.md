# Phase 2B Leakage Control

## Phase 2B does not lock or approve a dataset partition

That is Phase 7B's own, separate, still-open task
(`PHASE7B_DUPLICATE_POLICY.md`) — its `outputs/phase7b/APPROVED_POLICY.json`
and `outputs/phase7b/final_partition/FROZEN.json` do not exist, and
`dataset_gate.check_clean_split_gate()` currently reports
`gate_passed = False` (checked directly this session:
`duplicate_policy_approved` and `manifest_frozen` both fail; the other 6
checks pass). Phase 2B does not depend on this gate passing — its own
20-image pilot sample is small enough to control leakage directly and
independently, described below.

## What Phase 2B reuses (not reinvents)

`src/doar/phase2b/duplicate_groups.py` reads the existing, real,
already-computed union-find duplicate-group assignment from
`outputs/phase7b/final_partition/partition_manifest.csv`
(`partition.py::compute_duplicate_groups`, dHash threshold 6) rather than
recomputing perceptual hashes from raw images. This file is local-only
(gitignored, derived from the private dataset) — `load_group_lookup()`
raises a clear `FileNotFoundError` if it is absent, so any code depending
on it fails loudly rather than silently falling back to an unsound
default.

## How this pilot avoids leakage

`src/doar/phase2b/dataset.py::select_pilot_sample()` selects **at most one
image per duplicate group**. This is verified, not assumed: the real
output at `artifacts/phase2b/duplicate_groups.csv` shows all 20 selected
images landing in 20 distinct `source_image_group` values — there is no
image in the pilot sample that shares a near-duplicate group with any
other image in the pilot sample. Because this pilot only ever *evaluates*
zero-shot/classical baselines (no supervised training, no threshold
tuning against a held-out split — `docs/PHASE2B_MODEL_SELECTION.md`), the
train/valid leakage concern that motivates Phase 7A/7B's much larger
partition effort does not directly apply to Phase 2B's own evaluation;
one-image-per-group selection is nonetheless applied as the more
conservative, defensible choice, and is the property any future
supervised Phase 2C effort will need regardless.

## What Phase 2B does NOT claim

- Does not claim the Phase 7B partition is approved or final.
- Does not claim its own 20-image sample is representative of the full
  dataset's object-class distribution — with `n=20`, no representativeness
  claim would be defensible (`PHASE2B_ANNOTATION_PROTOCOL.md`).
- Does not exclude the 674 conflict-flagged images from *existing* by
  claiming they are erroneous — they are simply not drawn from for this
  pilot's sample, consistent with keeping this small pilot on the
  cleanest, least-ambiguous subset available.
- Does not compute or claim subject-level (child-level) independence —
  `PHASE7A_DATASET_READINESS.md` already established this dataset has no
  subject identifier at all; Phase 2B inherits that same limitation and
  does not attempt to work around it.

## Roboflow augmentation awareness

Phase 7A/7B's perceptual-hash duplicate-group computation is the
established mechanism for catching Roboflow-style near-duplicate
augmentations (rotation/crop/color-jitter variants of the same source
image) in this dataset — Phase 2B relies on that existing mechanism
rather than building a second one. `PHASE2B_AUDIT_AND_PLAN.md` Section 5
records the real, current state of that mechanism (dHash-6, not yet
approved, precision concerns documented in `PHASE7B_DUPLICATE_POLICY.md`
Section 3/14) — a real, inherited limitation of the underlying grouping
Phase 2B did not attempt to re-verify or improve.
