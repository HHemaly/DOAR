# DOAR-TRACE Phase 2C.1 Implementation Report

**Status: real, executed, tested. Annotation complete, detector readiness assessed.
No detector trained, no threshold tuned, no rule activated.** Branch
`feature/doar-phase2c-annotation-expansion`, starting commit `a0ea8e8`, checkpoint
tag `checkpoint/pre-doar-trace-phase2c1` (verified, unmoved, matches on origin).
Phase 2C.1 is a human-annotation-expansion and detector-readiness-assessment phase
for the 10-class object-evidence ontology Phase 2B piloted — not a psychological
validation, not a production detector deployment, and no rule in
`rules_registry_v2.json` was activated by anything in this phase.

## 1. What Phase 2B left open

Per `CURRENT_TO_TARGET_GAP_V7.md`: 60 of the 80 blind-selected pilot images remained
unannotated, only `person`/`face` had sufficient positive support at n=20, and no
second annotator existed for any inter-rater comparison. Phase 2C.1's job was to
close the annotation gap and assess, honestly, which classes (if any) are now ready
for further detector work — not to build a detector.

## 2. Provenance audit (before any annotation)

`PHASE2C1_ANNOTATION_PROVENANCE_AUDIT.md`: the exact original 80-image Phase 2B
pilot was independently reconstructed from the real dataset and verified three ways
— 20/20 known `image_id`↔relative-path hashes, 20/20 known `image_id`→`group_id`
pairs, and (the decisive test) re-running the real `select_pilot_sample(80, seed=2026)`
production code reproduced the identical `p2b_0000`–`p2b_0019`→`image_id` mapping
already committed in `artifacts/phase2b/annotation_manifest.csv`, 0 mismatches.
Reproducibility confirmed before the private blinded workspace was built.

## 3. Infrastructure built

`src/doar/phase2c1/{schema,store,workspace,quality}.py` and
`phase2c_annotation_app.py` (Streamlit):

- **schema.py**: versioned `AnnotationRecord` (separate from
  `artifacts/phase2b/annotation_manifest.csv`, never written to). `status` has no
  default — a row can never be silently "absent". `uncertain`/`not_assessable` can
  never carry a positive `instance_count` or bbox. `annotator_type` distinguishes a
  genuine Phase 2C.1 annotation (`human`) from a migrated Phase 2B legacy label
  (`legacy_provisional_human`) — added during the pre-annotation audit specifically
  so the two could never be conflated (see §5).
- **store.py**: CSV-backed, atomic-write store keyed by `annotation_id`
  (`pilot_id`+`class_name`+`annotator_id`) — structural duplicate-row prevention,
  safe autosave/resume.
- **workspace.py**: wraps Phase 2B's own `select_pilot_sample`/`blind_copy_sample`
  (never reimplements them) to build the private, gitignored 80-image blinded
  workspace; migrates the 20 already-annotated Phase 2B images as provisional review
  targets; retains `original_split` provenance per image (added during the
  pre-annotation audit, see §5) with `assert_pilot_ids_exclude_locked_test()` as a
  future-readiness guard for detector work — not wired into anything in this phase.
- **quality.py**: class support, completion rate, review coverage, human-human
  Cohen's kappa/percent agreement (restricted to genuine `human` annotators only), a
  separately-labeled human-vs-provisional reference comparison (percent agreement
  only, never kappa), a structural integrity report, and a detector-readiness
  classifier — all computed from whatever is actually in the store, nothing
  fabricated.
- **phase2c_annotation_app.py**: one blinded drawing at a time, all 10 ontology
  classes, autosave, Previous/Next/Jump/First-unannotated navigation, bbox entry,
  CSV/JSON export. Review mode requires the reviewer's own independent judgment to
  be saved before the primary label is revealed (anti-anchoring). Never displays the
  emotion label, source folder, a detector prediction, or a psychological
  interpretation.

## 4. Real annotation pass

**80 of 80 images annotated by one genuine human annotator** (800 rows, all 10
classes per image, 100% completion). The 20 already-annotated Phase 2B images were
additionally available as provisional review targets. Real per-class prevalence
ranges from 5.0% (`circle`) to 88.8% (`face`); full detail in
`PHASE2C1_POST_ANNOTATION_REPORT.md` §B.

## 5. A real bug found and fixed twice — once in code, once in already-serialized data

The pre-annotation audit found that `migrate_phase2b_provisional()` tagged migrated
Phase 2B rows as `annotator_type="human"`, indistinguishable from a genuine new
annotator — which would have let `quality.py`'s agreement tooling silently count a
single-annotator, unreviewed Phase 2B label as a second real rater for human-human
Cohen's kappa. Fixed in code (commit `bffaa4b`, `legacy_provisional_human` type,
`_distinct_human_annotators` restricts the kappa candidate pool, a separate
`compute_provisional_reference_comparison` for the human-vs-provisional case).

**The post-annotation review found the fix had not reached the already-seeded real
store** — `outputs/phase2c1/annotation_store.csv` was built before the fix landed
and was never re-migrated, so its 200 legacy rows still carried the stale
`annotator_type="human"`. Corrected this session: exactly 200 rows updated, verified
by full before/after diff that `annotator_type` was the *only* field that changed on
any row (no annotation judgment touched), store and exports regenerated. Full detail:
`PHASE2C1_POST_ANNOTATION_REPORT.md` §0.

## 6. Locked-test-split accounting

The reconstructed 80-image pilot spans all three original dataset splits — verified
both before annotation (predicted 65/8/7) and after (confirmed exactly 65/8/7).
Original-split provenance is retained in the private `private_pilot_mapping.csv`
(never in the annotator-facing app) and enforced by
`workspace.assert_pilot_ids_exclude_locked_test()` for any future detector
training/tuning/model-selection/calibration code — not invoked by anything in this
phase; no detector split exists yet.

## 7. Detector readiness (from genuine human annotation only, never from any
   detector/model output)

| Tier | Classes |
|---|---|
| `evaluation_supported` (n present >= 20) | person (59), face (71), hand (43) |
| `exploratory_only` (5 <= n present < 20) | animal (13), house (14), tree (15), heart (5), star (6), vehicle (5) |
| `insufficient_positive_support` (n present < 5) | circle (4) |
| `high_uncertainty` (> 15% uncertain/not_assessable) | none this round |

Full rationale per class: `PHASE2C1_POST_ANNOTATION_REPORT.md` §E,
`artifacts/phase2c1/detector_readiness_by_class.csv`.

## 8. Human vs. legacy provisional comparison (not inter-rater reliability)

75.5% overall percent agreement across the 200 (pilot_id, class) pairs both sides
judged (first 20 images only) — weakest on `hand` (40%) and `animal` (60%),
strongest on `vehicle` (95%). The genuine human annotator resolved one image
(`p2b_0018`) that Phase 2B's pilot had marked entirely `not_assessable`, accounting
for 10 of the 49 disagreements by itself. No Cohen's kappa is reported anywhere —
only one genuine human annotator exists in this round
(`compute_agreement_report`'s own `sufficient_annotators: False`). Full detail:
`PHASE2C1_POST_ANNOTATION_REPORT.md` §C.

## 9. Tests and safety

`tests/test_phase2c1_{schema,store,workspace,quality,safety,app}.py` — schema
invariants, store persistence/duplicate-prevention/corrupt-manifest handling,
workspace migration and locked-test-split guards, agreement math (Cohen's kappa
verified against hand-computed values), the new integrity-report/readiness-
classifier/disagreement-table functions (44 quality tests alone, synthetic fixtures
only), and Streamlit AppTest coverage for both annotation modes. Structural safety
tests confirm no `src/doar/phase2c1/*` module imports the rule-evaluation engine, no
diagnostic language appears in any static string, no emotion-class word appears in
the app's source, and `doar_prototype_app.py` (Parent View) stays unmodified. Full
suite: 964+/964+ passing (see final verification run this session), ruff clean,
compileall clean.

## 10. Remaining scientific risks and honest limitations

1. Single annotator throughout — the human-vs-provisional comparison is a reference
   check, not a validated ground-truth confirmation; no genuine second human
   annotator has yet independently re-annotated any image.
2. `hand` and `animal` show comparatively weak agreement against the provisional
   labels (40%/60%) — worth investigating before trusting either class further, even
   though both currently sit in `exploratory_only`/`evaluation_supported`.
3. `circle` remains below minimum positive support (4) even after the full 80-image
   pass — Phase 2B's own classical-CV circularity baseline already showed severe
   over-firing on this class; no amount of additional annotation of *this* 80-image
   set will raise its support further.
4. Existing Phase 2B CLIP/CV predictions only cover the original 20 images; the 60
   newly-annotated images have no baseline predictions at all yet (§F re-evaluation
   plan, not executed).
5. The 7 locked-test-split images are annotated and included in descriptive
   statistics, but must stay excluded from any future training/tuning/selection —
   enforced by a guard function, not yet exercised by any real training code because
   none exists yet.

## 11. Exact recommendation for Phase 2C.2

**Do not begin automatically — this report is the stopping point**, per the same
discipline Phase 2B's own report established. If authorized: (a) rerun CLIP
zero-shot + classical-CV inference (not training) on the 60 newly-annotated images;
(b) re-score all existing predictions (old and new) against the corrected human
reference at the existing fixed threshold, as a purely descriptive pass; (c) only
then consider a properly-designed threshold-development step that explicitly
excludes the 7 locked-test images; (d) treat `person`/`face`/`hand` as the only
classes with enough support for that descriptive pass to be informative; (e)
investigate the `hand`/`animal` human-vs-provisional disagreement before relying on
either. None of this was attempted here, and none is claimed to be.
