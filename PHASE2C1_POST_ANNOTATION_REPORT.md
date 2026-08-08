# Phase 2C.1 Post-Annotation Quality and Detector-Readiness Report

**Status: post-annotation analysis only. No training, fine-tuning, threshold tuning,
rule activation, Parent View change, or Phase 2C.2 work was performed.** Branch
`feature/doar-phase2c-annotation-expansion`. Source data: the real, private
`outputs/phase2c1/annotation_store.csv` and `outputs/phase2c1/private_pilot_mapping.csv`
(gitignored, never committed) — every number below was computed directly against
them, then re-derivable from the committed artifacts under `artifacts/phase2c1/`.

## 0. A data-hygiene bug found and fixed before analysis

Before any statistics were computed, the raw store was inspected directly. The 200
migrated Phase 2B rows still carried `annotator_type="human"` — the exact conflation
the pre-annotation audit's code fix (commit `bffaa4b`) was meant to prevent. Root
cause: the real `outputs/phase2c1/annotation_store.csv` was seeded by
`migrate_phase2b_provisional()` **before** that fix landed in code, and was never
re-migrated afterward. The fix only changed future migrations, not this already-
serialized file.

**Corrected, verified, and re-exported this session**: all 200 rows where
`source_manifest_version == "phase2b_annotation_manifest_v1_migrated"` were updated
to `annotator_type="legacy_provisional_human"`. A full before/after diff confirmed
**exactly 200 rows changed, and in every one of them the only field that changed was
`annotator_type`** — no status, instance_count, bbox, or note was touched. A backup
of the pre-fix file was kept at
`outputs/phase2c1/annotation_store_pre_annotator_type_fix_2026-08-08.csv` (private,
gitignored). `outputs/phase2c1/exports/` was regenerated from the corrected store.
Had this not been caught, every statistic in this report computed as "genuine human"
would have silently included the single-annotator Phase 2B pilot as if it were real
new annotation.

## A. Annotation integrity — all checks passed after the fix above

| Check | Result |
|---|---|
| Unique pilot_ids with genuine human annotation | **80** (`p2b_0000`–`p2b_0079`, exact match) |
| Each genuine-human image has all 10 ontology classes | **Yes**, 0 incomplete |
| Genuine-human rows | **800** (80 × 10) |
| Legacy provisional rows | **200** (20 × 10) |
| Total rows | **1000** |
| Duplicate annotation keys | **0** |
| Missing classes | **0** |
| Invalid status/count/bbox combinations | **0** (structurally impossible — `AnnotationRecord.__post_init__` raises before an invalid row can ever be loaded; the store loading without exception is itself the proof) |
| Unknown annotator types | **0** (only `human` and `legacy_provisional_human` present, post-fix) |
| Blank `image_id` / `source_image_group` | **0** |

Machine-readable: `artifacts/phase2c1/annotation_integrity_report.json`.

## B. Real annotation statistics (genuine human annotation only, n=80 images)

| Class | Present | Absent | Uncertain | Not assessable | Prevalence | Assessable | Uncertainty rate |
|---|---|---|---|---|---|---|---|
| person | 59 | 21 | 0 | 0 | 0.738 | 80 | 0.000 |
| face | 71 | 9 | 0 | 0 | 0.888 | 80 | 0.000 |
| hand | 43 | 37 | 0 | 0 | 0.538 | 80 | 0.000 |
| animal | 13 | 67 | 0 | 0 | 0.163 | 80 | 0.000 |
| house | 14 | 66 | 0 | 0 | 0.175 | 80 | 0.000 |
| tree | 15 | 64 | 1 | 0 | 0.188 | 79 | 0.013 |
| heart | 5 | 75 | 0 | 0 | 0.063 | 80 | 0.000 |
| star | 6 | 74 | 0 | 0 | 0.075 | 80 | 0.000 |
| circle | 4 | 76 | 0 | 0 | 0.050 | 80 | 0.000 |
| vehicle | 5 | 75 | 0 | 0 | 0.063 | 80 | 0.000 |

Completion rate: **100%** (80/80 images, 10/10 classes each). Overall uncertainty
rate across all 800 human rows: **1/800 = 0.13%** (a single `tree` row). Full table:
`artifacts/phase2c1/human_class_frequency.csv`.

**Classes marked present per image** (how many of the 10 classes a given image has
at least one hit for):

| # classes present | # images |
|---|---|
| 1 | 11 |
| 2 | 21 |
| 3 | 26 |
| 4 | 12 |
| 5 | 6 |
| 6 | 3 |
| 8 | 1 |

Median is 3 classes present per image; no image has 0 or 7/9/10. Full table:
`artifacts/phase2c1/annotation_status_distribution.csv`.

**Readiness-criteria methodology note (per instruction, stated before results)**:
the minimum-positive-support floor (5) reuses Phase 2B's own existing, documented
convention (`docs/PHASE2B_ANNOTATION_PROTOCOL.md`, `phase2b/evaluation.py::MIN_POSITIVE_SUPPORT`)
rather than being invented here. The additional "evaluation-supported" bar (>=20
positive examples) is a new, Phase 2C.1-specific threshold, defined in code
(`quality.py::EVALUATION_SUPPORTED_MIN_POSITIVE`) and applied mechanically to this
round's counts — it was not chosen after looking at any detector output, because no
detector was run this session.

## C. Human vs. legacy provisional comparison (first 20 images) — NOT inter-rater reliability

This is explicitly a **human vs legacy provisional comparison**, never labeled or
computed as inter-rater reliability, and **no Cohen's kappa is reported** — the
provisional side is a single, unreviewed Phase 2B annotator, not confirmed
independent ground truth (`docs/PHASE2B_ANNOTATION_PROTOCOL.md`).

- **n compared**: 200 (20 images × 10 classes)
- **Overall percent agreement**: **75.5%** (151/200)
- **Disagreements**: 49

| Class | n compared | % agreement |
|---|---|---|
| person | 20 | 75.0% |
| face | 20 | 85.0% |
| hand | 20 | 40.0% |
| animal | 20 | 60.0% |
| house | 20 | 85.0% |
| tree | 20 | 75.0% |
| heart | 20 | 90.0% |
| star | 20 | 80.0% |
| circle | 20 | 70.0% |
| vehicle | 20 | 95.0% |

`hand` and `animal` show the weakest agreement (40% and 60%) — worth a closer look
before trusting either class's provisional labels for anything.

**Present/absent transitions** (both sides had a firm present/absent judgment):
- Provisional=present → human=absent: **1**
- Provisional=absent → human=present: **14**

The genuine human annotator found substantially more positives than the original
Phase 2B single-pass pilot missed (14 vs. 1) — directionally consistent with a more
careful, complete annotation pass catching real instances a faster pilot pass missed,
rather than the two annotators disagreeing symmetrically.

**Disagreements involving uncertain/not_assessable**: 34 of the 49. A large share of
these — 10 — come from a single image, `p2b_0018`, which Phase 2B's pilot marked
**entirely `not_assessable`** (a heavily cropped/zoomed fragment, per
`docs/PHASE2B_ANNOTATION_PROTOCOL.md`). The genuine human annotator judged it
assessable after all: `person`/`face` present, the other 8 classes absent. This is a
real, single-image resolution, not spread across many images.

Full disagreement table (49 rows: pilot_id, class_name, human_status,
provisional_status, both instance counts):
`artifacts/phase2c1/human_vs_provisional_disagreements.csv`. Per-class summary:
`artifacts/phase2c1/human_vs_provisional_comparison.csv`.

**On Cohen's kappa**: only one genuine human annotator exists in this store
(`quality.compute_agreement_report` confirms `sufficient_annotators: False`,
`human_annotators: ["1"]`). No human-human kappa is reported anywhere in this round,
consistent with the pre-annotation audit's fix.

## D. Locked-test accounting

| Original split | Count |
|---|---|
| train | 65 |
| valid | 8 |
| test | **7** |

**Matches the pre-annotation audit's prediction exactly (65/8/7).** All 80 images —
including the 7 from the original locked test split — were annotated and are
included in the descriptive statistics in Sections B and C above; this is explicitly
permitted for an annotation-expansion phase.

**The 7 locked-test pilot_ids are marked unavailable for any future detector
training, threshold-tuning, model-selection, or calibration use.**
`workspace.assert_pilot_ids_exclude_locked_test()` (added in the pre-annotation
audit, unchanged this session) is the enforcement point any future Phase 2C.2+ code
must call before using a set of pilot images for those purposes. It was not invoked
by anything in this session — no detector split was created, nothing was trained.

## E. Detector readiness by class (from genuine human annotation only)

| Class | Readiness tier | n present | Rationale |
|---|---|---|---|
| person | **evaluation_supported** | 59 | Enough for a descriptive precision/recall/ranking-separation pass with a reasonably informative 95% CI at n=80. |
| face | **evaluation_supported** | 71 | Same. |
| hand | **evaluation_supported** | 43 | Same. |
| animal | exploratory_only | 13 | Above the floor, below the confident-evaluation bar (20). |
| house | exploratory_only | 14 | Same. |
| tree | exploratory_only | 15 | Same (1 uncertain row, 1.3% — well under the high-uncertainty bar). |
| heart | exploratory_only | 5 | At the minimum-support floor exactly. |
| star | exploratory_only | 6 | Same category. |
| circle | **insufficient_positive_support** | 4 | Below the minimum-support floor (5). |
| vehicle | exploratory_only | 5 | At the minimum-support floor exactly. |

**No class triggered `high_uncertainty`** (>15% of images uncertain/not_assessable
for that class) — the genuine human annotation pass used `uncertain`/`not_assessable`
essentially never (1/800 rows total), which is itself a notable, positive finding
about this round's annotation confidence.

**What each tier permits, scientifically**:
- **evaluation_supported**: a descriptive precision/recall/F1/ranking-separation
  pass against existing or freshly-run baseline predictions is defensible and
  informative. Still not sufficient for supervised training (Phase 2B's own finding:
  even much larger effective-sample-size classes needed hundreds of independent
  groups before a CNN was judged viable) or for threshold selection using the locked
  test split.
- **exploratory_only**: any computed metric should be reported as directional only,
  never as a settled result — consistent with `docs/PHASE2B_EVALUATION_PROTOCOL.md`'s
  own existing discipline for small-n classes.
- **insufficient_positive_support**: no precision/recall/F1 should be computed or
  reported for this class at all at this sample size.
- **high_uncertainty** (not triggered this round): the class definition itself may be
  ambiguous or hard to judge from these drawings; more annotation would not resolve
  this without first revisiting the ontology's inclusion/exclusion criteria.

Full table: `artifacts/phase2c1/detector_readiness_by_class.csv`.

## F. Phase 2B baseline re-evaluation plan (planning only — nothing executed)

`artifacts/phase2b/raw_predictions.csv` contains real CLIP zero-shot similarity
scores and classical-CV circularity outputs, but **only for the original 20
annotated images** — confirmed by inspection this session (20 rows, not 80). The
broader "80-image smoke test" mentioned in `PHASE2B_IMPLEMENTATION_REPORT.md` was a
failure check only; its raw scores for the other 60 were never persisted.

| | Descriptive evaluation | Threshold development | Future detector training |
|---|---|---|---|
| **What it means** | Re-score already-existing predictions against corrected ground truth at the existing fixed threshold (0.28, `artifacts/phase2b/model_manifest.json`) | Search for a better operating threshold from ranking/precision curves | Supervised fine-tuning or a new model |
| **20 originally-annotated images** | **Legitimate now** (in a future, explicitly-scoped step) — zero new inference, just re-scoring existing `raw_predictions.csv` against the corrected human labels | Not permitted against these 20 alone (too few, and this pilot's own prior finding was that threshold 0.28 is miscalibrated for at least `person`) | Not permitted — no supervised split exists |
| **60 newly-annotated images** | **Requires rerunning CLIP inference and the classical-CV circularity pass first** (inference only, not training — no predictions exist yet for these images at all) | Not permitted | Not permitted |
| **7 locked-test-split images (subset of the 80)** | May appear in descriptive-only statistics | **Forbidden** — must be excluded via `assert_pilot_ids_exclude_locked_test()` before any threshold search | **Forbidden** |

**Recommendation, not executed this session**: a future, explicitly-scoped step
should (1) re-score the existing 20-image predictions against the corrected human
labels (free, no new inference) as a sanity check on Phase 2B's original findings,
then (2) rerun CLIP + classical-CV inference on the 60 newly-annotated images (cheap,
deterministic, no training), before any threshold or model-selection work — which
itself must wait for a properly designed, leakage-safe, locked-test-excluded split.

## G. Artifacts written this session

All under `artifacts/phase2c1/` (committed; no source paths, no emotion labels, no
drawings):

- `annotation_integrity_report.json`
- `human_class_frequency.csv`
- `annotation_status_distribution.csv`
- `human_vs_provisional_comparison.csv`
- `human_vs_provisional_disagreements.csv`
- `detector_readiness_by_class.csv`

Generated by `scripts/phase2c1_generate_report.py` (committed, reusable for a future
annotation round — reads only the private, gitignored store/mapping and never writes
anything except these derived, non-private tables).

## What was NOT done this session

No training, fine-tuning, threshold tuning, rule activation, Parent View
modification, or Phase 2C.2 work of any kind. No new CLIP/CV inference was run. No
detector split was created. Nothing was pushed.
