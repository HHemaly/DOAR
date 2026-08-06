# Phase 2B Annotation Protocol

## Sample size: a real, stated constraint

The task's own stated reasonable initial range is 200-500 images, "but
determine final number from real constraints." The real constraint hit in
this pilot: annotation was performed by a single human-equivalent
annotator directly visually inspecting each blinded image within one
working session's compute/time budget, not by a crowd or a paid panel.
**80 images were selected and blind-copied** (`scripts/phase2b_audit_dataset.py
--n 80`, matching this pilot's real run), but only the **first 20 by
pilot_id (`p2b_0000`-`p2b_0019`) were actually hand-annotated**, each
judged against all 10 ontology classes (200 total (image, class)
judgments), real and individually inspected, none templated or guessed.
The remaining 60 blinded images exist on disk (locally, not committed)
but carry no annotation — they are available for a follow-up annotation
pass without needing to reselect or re-blind. This is below the 200-500
range and is recorded here honestly rather than padded with unexamined
labels. Section 4 of `PHASE2B_AUDIT_AND_PLAN.md` already anticipated
this: "which candidate classes occur often enough" is a Phase 2B
*finding*, not a precondition — this pilot's own finding
(`artifacts/phase2b/class_frequency_audit.csv`) is that only 2 of 10
classes reach even 5 positive examples at n=20, meaning a larger
annotation pass (starting with the already-blinded remaining 60 images)
is the single highest-value next step for Phase 2C.

**Reproducibility note**: `blind_copy_sample`'s re-identification shuffle
consumes a number of random draws equal to the *full* selected-list
length, so it is not invariant to `n` — reselecting with `--n 20` does
NOT reproduce the first 20 pilot_ids of the real `--n 80` run. Always
pass `--n 80` to reproduce this pilot's exact `p2b_0000`-`p2b_0019`
assignment (verified directly this session: re-running `select_pilot_sample(80)`
+ `blind_copy_sample` reproduces the identical 20 image_ids in the
identical order).

## Selection procedure

1. Start from the existing Phase 7B partition manifest
   (`outputs/phase7b/final_partition/partition_manifest.csv` — real,
   on-disk, not the *approved* partition, see
   `docs/PHASE2B_LEAKAGE_CONTROL.md`).
2. Restrict to `conflict_status == "clean"` (3,350 of 3,688 images).
3. Group by `group_id` (Phase 7A/7B's existing duplicate-group
   computation) — 2,200 distinct groups in the clean pool.
4. Deterministically shuffle group order with a fixed seed (2026, see
   `src/doar/phase2b/dataset.py::DEFAULT_SEED`), take the first N groups,
   and select the lowest `image_id` within each — **exactly one image
   per duplicate group**, guaranteeing zero within-pilot leakage by
   construction (verified: `artifacts/phase2b/duplicate_groups.csv`
   shows all 20 selected images in 20 distinct groups).
5. Copy each selected image into a flat working directory under an
   opaque `p2b_XXXX` filename (`src/doar/phase2b/dataset.py::blind_copy_sample`),
   in an independently-shuffled order — no folder name, filename, or
   listing order carries any trace of the original emotion-class label.
6. The pilot_id-to-real-image_id mapping is kept in a separate file the
   annotator does not consult while labeling.

## What the annotator saw and recorded

For each of the 10 ontology classes (`src/doar/phase2b/ontology.py`), one
status from `{present, absent, uncertain, not_assessable}`
(`ontology.ANNOTATION_STATUSES`) plus, when `present`, an instance count.
**`uncertain` was used whenever genuinely unclear (never silently
converted to a negative), matching the ontology's own documented
ambiguous cases** (e.g. sun vs. star, a wheel drawn alone). One image
(`p2b_0018`) was entirely `not_assessable` — a heavily cropped/zoomed
fragment with no recognizable content, an honest, real example of the
"unsuitable image" case `PHASE2B_AUDIT_AND_PLAN.md` Section 7
anticipated.

## Single annotator — stated plainly

**Only one annotator (this session) labeled this pilot.** No
inter-annotator agreement statistic is reported or invented — there is
nothing to compute it from. `artifacts/phase2b/annotation_manifest.csv`'s
`annotator` column records this explicitly
(`single_annotator_session_2026-08-06`) rather than implying a panel.
Any future annotation campaign (Phase 2C) should add a second independent
annotator and reuse `human_review.py`'s existing agreement machinery
(`review.py::compute_agreement`), per `PHASE3_DETECTOR_EVALUATION_PLAN.md`
Section 5's own requirement.

## Sufficient-support threshold

A class needs **>=5 real positive (`present`) examples** in the sample
before any precision/recall/F1 is computed for it
(`src/doar/phase2b/evaluation.py::MIN_POSITIVE_SUPPORT`). Below that,
`sufficient_support = False` is recorded and no metric is reported as if
it were reliable — 8 of the 10 candidate classes fall below this bar at
n=20 (`artifacts/phase2b/class_frequency_audit.csv`).

## What annotation never used

No emotion-class folder label (Angry/Fear/Happy/Sad) was visible to the
annotator at any point during labeling (blind copy, Section "Selection
procedure" step 5) — object presence/absence was judged purely from
pixels. No detector or model output was consulted while annotating (the
annotation pass in this session was completed and written to
`artifacts/phase2b/annotation_manifest.csv` before any real CLIP/classical
-CV inference was run against the same images).
