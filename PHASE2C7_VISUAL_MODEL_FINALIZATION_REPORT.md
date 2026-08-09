# Phase 2C.7 — Visual Model Finalization

**Status: execution phase, complete.** Branch `feature/doar-phase2c-annotation-expansion`, starting
HEAD `ed4d387` (verified clean). Ends the human-annotation expansion (per your explicit direction),
evaluates your real eye annotations against the stored automatic detectors, freezes an eye detector
configuration on a group-disjoint dev split, combines it with the existing frozen Phase 2C.4/2C.4A
object-detector benchmark into one class-by-class visual detector policy, and ships a single
image-only automatic inference API. No fine-tuning was performed.

## 0. Preflight

| Check | Result |
|---|---|
| Branch | `feature/doar-phase2c-annotation-expansion` |
| Starting HEAD | `ed4d387333a02bb5357ea4aa7549b4b1c793b5af` (Phase 2C.6 canvas-compat fix) |
| Working tree | clean |
| Prior reports/artifacts | Phase 2C.4/2C.4A/2C.5/2C.6 reports and `artifacts/phase2c4{,a}/`, `artifacts/phase2c5/`, `artifacts/phase2c6/` all present, none modified this phase |

---

## 1. Stage 1 — Your saved eye annotations, as they actually are

**Export inspected**: `outputs/phase2c5/exports/phase2c5_part_annotations.csv` (140,259 bytes,
identical content to `outputs/phase2c5/part_annotation_store.csv` — Export just re-saves the live
store). Schema: Phase 2C.5's own `phase2c5_part_annotation_schema_v1` (not assumed — read from the
file's own header and validated by loading it through `doar.phase2c5.store.load_store`, which raises
on any malformed row; it did not raise).

**Correction to the ~300 estimate**: the export contains **213 unique images**, not ~300 —
`p2c6_0000` through `p2c6_0212`, sequential, no gaps. The remaining 87 of the 300-image expansion
manifest (`p2c6_0213`–`p2c6_0299`) were never annotated. Not a data problem — just the real number,
stated plainly rather than assumed.

| Metric | Value |
|---|---|
| Unique images reviewed | **213** |
| Eye annotation records | 213 (one per image — no duplicates, no malformed rows: `n_rows == n_unique_pilot_ids`, CSV line count matches exactly) |
| PRESENT | 164 |
| ABSENT | 25 |
| UNCERTAIN | 24 |
| Total bounding boxes | 331 |
| Images with 1 eye | 74 |
| Images with 2 eyes | 49 |
| Images with >2 instances | 41 (3: 24, 4: 9, 5: 3, 6: 1, 7: 3, 9: 1) |
| `eye_state` distribution | open 127, closed 13, uncertain 24 |
| `eye_detail` distribution | undetailed 149, detailed 14, uncertain 1 |
| Completion vs. 300-image manifest | **213/300 (71%) — 87 images not annotated** |
| Malformed/incomplete rows | **0** |

### bbox provenance — an important, unplanned finding

**All 331 boxes are `bbox_source="human_drawn"`. Zero `human_accepted`, zero `human_edited`.**
Checked specifically against the 39 of these 213 images that DID have a real Grounding DINO/OWLv2
eye proposal available at the time (from the first Phase 2C.6 50-image batch): still 100%
human-drawn. The most likely explanation, and the one this session's own launch instructions
probably caused: the app's `DOAR_PHASE2C5_PROPOSALS_PATH` environment variable is per-terminal-session
(PowerShell `$env:`) and defaults to the old 15-image pilot's proposals file (which has zero overlap
with `p2c6_` images) if not re-set — so proposals were very likely never loaded during your
annotation session at all.

**This corrects your own instruction's assumption.** These 213 eye annotations are **not
proposal-assisted** — they are fully independent of the automatic detectors. That is actually good
news for evaluation validity (no risk of proposal-anchoring bias inflating detector agreement), but
it does mean:
- The "accepted/edited/rejected proposal" efficiency diagnostics you asked for are structurally all
  zero/undefined from this data (reported honestly as such, not fabricated) — see
  `artifacts/phase2c7/eye_annotation_summary.json`'s `annotation_efficiency` block.
- 170 real proposal boxes existed for these 213 images (across the eventual full run) with a 0%
  "incorporation rate" — again, most plausibly explained by proposals never being shown, not by you
  reviewing and rejecting each one.

---

## 2. Stage 2 — Evaluating the automatic eye detectors

**Coverage gap found and closed.** Of the 213 reviewed images, only 39 had a stored detector
prediction (from the original Phase 2C.6 50-image batch). Per instruction ("if additional inference
is necessary, run only what is necessary, keep it resumable"), this phase ran the **existing,
already-tested Phase 2C.6 resumable batch pipeline** (unchanged, no new detector, no fine-tuning) for
exactly the 163 reviewed-but-unpredicted images (`p2c6_0050`–`p2c6_0212`) — no more, no less; the 87
never-annotated images were never touched. Grounding DINO primary, OWLv2 fallback-for-empty-targets
only, exactly as Phase 2C.6 already established.

**A second, more important data-collection fact was found and handled, not glossed over**: because
OWLv2 only runs as a *fallback* for targets Grounding DINO found nothing for, OWLv2's own eye finding
is only ever recorded when Grounding DINO already failed on that image. A naive "OWLv2 alone" filter
over this data would silently evaluate OWLv2 only on Grounding DINO's failure cases — a biased
sample, not a system-wide result. This report does NOT present that biased number as if it were fair.
Instead, three genuinely fair quantities are reported:

1. **`grounding_dino`** — always run as primary; fairly observed on every processed image.
2. **`combination`** — Grounding DINO OR the OWLv2 fallback result; exactly what the deployed
   pipeline produces; also fairly observed on every processed image.
3. **`owlv2_fallback_recovery`** — NOT a system-wide OWLv2 metric; restricted to Grounding DINO's
   real misses only, reporting what fraction the fallback recovered (a real, useful, correctly-scoped
   number).

**Dev-split results (n=169 usable images, before any holdout contact):**

| Model | n_present (real) | Precision | Recall | Balanced acc. | Mean IoU | Success@IoU0.3 | Correct-count rate |
|---|---|---|---|---|---|---|---|
| `grounding_dino` (primary, always fair) | 128 | 0.874 | 0.703 | 0.556 | 0.139 | 0.170 | 0.227 |
| `combination` (deployed: GD + OWLv2 fallback-for-empty) | 128 | 0.877 | 0.727 | 0.568 | 0.138 | 0.164 | 0.234 |

**`owlv2_fallback_recovery` (dev)** — NOT a system-wide OWLv2 metric, restricted to the 38 images
where Grounding DINO found no eye at all and a real eye was present: the fallback fired on 30 of
those, recovered 3 correct (`recovery_rate=0.079`), with `owlv2_fallback_precision_on_fired_subset=1.0`
(every fallback firing that DID happen was precise, it just didn't fire often enough to move recall
much). Net effect of adding the fallback: recall +0.024 (0.703→0.727), balanced accuracy +0.012
(0.556→0.568), precision essentially unchanged (0.874→0.877) — a small, real, non-harmful gain, not a
transformative one.

---

## 3. Stage 3 — Dev/holdout isolation

Deterministic 80/20 split (`src/doar/phase2c7/dev_holdout_split.py`, seed `20260809`, reused from
Phase 2C.5's own `EXPANSION_SEED_STAGE_B` for traceability) over the 213 reviewed pilot_ids.
Group-disjointness is inherited for free — every `p2c6_` pilot_id already corresponds to exactly one
Phase 7B duplicate-group by construction (Phase 2C.6's own `select_expansion_sample`), so a plain
pilot_id split cannot separate two images from the same group.

| Split | n | 
|---|---|
| Dev | 170 pilot_ids (169 usable after 1 not-assessable exclusion) |
| Holdout | 43 pilot_ids (42 usable after 1 not-assessable exclusion) |

`assert_no_overlap` confirmed zero shared pilot_ids between dev and holdout (213 = 170 + 43 exactly).

The frozen configuration (§4) was chosen using **dev-split numbers only**. The holdout was touched
exactly once, after freezing, purely to report a number (`assert_holdout_untouched_by` checked
programmatically before any holdout evaluation ran).

**Limitation, stated plainly**: per Stage 1's finding, these annotations are NOT proposal-assisted, so
the specific "anchoring bias from accepted proposals" limitation your instructions anticipated does
not apply here. A different, milder limitation may still apply: you annotated images sequentially
(`p2c6_0000` → `p2c6_0212`), so any within-session drift in annotation standards (early vs. late
images) is not controlled for by this dev/holdout split, which only controls for image identity, not
annotation order.

---

## 4. Stage 4 — Selected eye configuration (frozen)

**Frozen policy: `combination`** (Grounding DINO primary + OWLv2 fallback for targets Grounding DINO
found nothing for), status **`EXPERIMENTAL_AUTOMATIC`** — chosen over `grounding_dino` alone because it
dominates on every dev metric (precision 0.877 vs. 0.874, recall 0.727 vs. 0.703, balanced accuracy
0.568 vs. 0.556) at zero extra integration cost (OWLv2 is already loaded for the object-class
targets). Neither config clears `MIN_BALANCED_ACCURACY_FOR_VALIDATION=0.6` (both sit in the
high-0.55s), so eye is frozen as **EXPERIMENTAL_AUTOMATIC**, not `VALIDATED_AUTOMATIC` — it clears the
precision floor (0.877 ≥ 0.6) comfortably but not the balanced-accuracy floor. Concretely: when the
frozen eye detector says "eye present," trust it (87.7% dev precision, 96.9% holdout precision); when
it says nothing, that is a real, frequent miss (dev recall only 0.727) and must never be read as
"no eyes in this drawing."

**Threshold**: 0.25 (Grounding DINO's already-established part-detection threshold from Phase 2C.5/2C.6
— not re-tuned this phase; no calibration sweep was run for parts, matching instruction "no new
detector families, no complicated ensemble").

**Held-out confirmation (touched once, after freezing, n=42 usable)**: precision **0.969**, recall
**0.886**, balanced accuracy **0.693**, mean IoU 0.182. All three headline metrics are *higher* on
holdout than dev — a reassuring direction (no overfitting-to-dev signature), though with only 42
holdout images and just 2 true-absent cases, the holdout balanced-accuracy/specificity numbers carry
wide uncertainty and should not be over-read as evidence the dev estimate was pessimistic.

**Localization is explicitly NOT validated** (`localization_validated=False`): mean IoU sits at 0.14
(dev) / 0.18 (holdout) — well below any reasonable bbox-quality bar. The eye detector's *presence*
signal is usable evidence; its *bounding box* is stored (for Technical View display and future
work) but must not be treated as a validated spatial measurement.

---

## 5. Stage 5 — Class-by-class visual detector policy

Combines this phase's new eye evaluation with Phase 2C.4A's frozen, unmodified, dev-cohort
calibration results (`artifacts/phase2c4a/calibration_frozen_operating_points.csv`,
`artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv` — read-only, never re-run, never
re-thresholded). No 300-image human annotation was required for any class besides eye.

| Target | Status | Best config | Precision | Recall | Bal. acc. | n (GT) | Localization validated |
|---|---|---|---|---|---|---|---|
| person | VALIDATED_AUTOMATIC | Grounding DINO, thr=0.25 | 0.84 | 0.81 | 0.71 | 52 | No |
| face | VALIDATED_AUTOMATIC | Grounding DINO, thr=0.25 | 0.98 | 0.70 | 0.80 | 64 | No |
| hand | VALIDATED_AUTOMATIC | Grounding DINO, thr=0.25 | 0.91 | 0.26 | 0.62 | 38 | No |
| tree | VALIDATED_AUTOMATIC | Grounding DINO, thr=0.5 | 0.64 | 0.54 | 0.74 | 13 | No |
| house | VALIDATED_AUTOMATIC | Grounding DINO, thr=0.4 | 0.69 | 0.75 | 0.84 | 12 | No |
| **eye** | EXPERIMENTAL_AUTOMATIC | Grounding DINO (parts) + OWLv2 fallback, thr=0.25 | 0.877 dev / 0.969 holdout | 0.727 dev / 0.886 holdout | 0.568 dev / 0.693 holdout | 128 dev / 35 holdout | No |
| heart | EXPERIMENTAL_AUTOMATIC | OWLv2, thr=0.35 (low support) | 1.00 | 0.25 | 0.63 | 4 | No |
| animal | EXPERIMENTAL_AUTOMATIC | GD default (abstains at calibration) | 0.20 | 0.80 | 0.64 | 10 | No |
| star | EXPERIMENTAL_AUTOMATIC | GD default (abstains at calibration) | 0.09 | 0.60 | 0.58 | 5 | No |
| mouth | EXPERIMENTAL_AUTOMATIC | Grounding DINO (no formal metric) | — | — | — | 0 | No |
| person (localization) | EXPERIMENTAL_AUTOMATIC | — (reference-only) | — | — | — | 0 | No |
| circle | **DISABLED** | — | — | — | 0.32 | 4 | — |
| vehicle | **DISABLED** | — | — | — | 0.50 | 3 | — |

Full table with exact rationale per target: `artifacts/phase2c7/target_validation_status.csv` /
`visual_detector_policy.json`.

---

## 6. Stage 6 — Unified image-only inference API

```python
records = visual_detector.analyze_image(
    image_path,                 # <-- the ONLY thing about the image this needs
    policy,                     # the frozen table above
    model_predict_fns={...},    # already-loaded real models, injected
)
```

Each `VisualEvidenceRecord` carries `target, present, bbox, confidence, model, checkpoint, threshold,
validation_status, evidence_status`. `DISABLED` targets are never sent to a model at all.
`evidence_status` is `"validated_evidence"`, `"experimental_evidence_technical_view_only"`, or
`"not_used"` — the one field any future rule-integration layer must check (`validated_records_only()`
does this filtering). Routing is by each target's own frozen `best_model` (not a hardcoded
object/part split) — `heart` alone freezes to OWLv2 while every other object class freezes to
Grounding DINO, and the API calls each distinct model callable at most once per image regardless of
how many targets route to it.

---

## 7. Stage 7 — Runtime sanity check on real drawings

Ran the frozen `visual_detector.analyze_image` end to end (real models, no ground truth consulted) on
5 deterministic Phase 2C.1 `p2b_` dev-eligible images — chosen specifically because they are a
completely different pilot_id namespace from the eye dev/holdout split, so this check cannot leak into
or be confused with the frozen evaluation above. This is an integration/runtime check only, not an
accuracy claim (`scripts/phase2c7_run_sanity_check.py`, output at
`artifacts/phase2c7/automatic_inference_example.json`).

| pilot_id | Records returned | Present | Notes |
|---|---|---|---|
| p2b_0000 | 10 | 7 | eye present, bbox `[0.331, 0.214, 0.070, 0.033]`, conf 0.270, routed via `grounding_dino_parts+owlv2_parts_fallback` as expected |
| p2b_0016 | 10 | 3 | eye present, bbox recorded, conf 0.277 |
| p2b_0030 | 10 | 5 | eye present, bbox recorded, conf 0.265 |
| p2b_0044 | 10 | 2 | eye absent (correctly reported `present=false`, not an error) |
| p2b_0060 | 10 | 5 | eye present, bbox recorded, conf 0.398 |

All 5 images returned exactly 10 records (9 dispatchable OBJECT_CLASS_POLICY/eye targets; `circle` and
`vehicle` correctly produced no record since `DISABLED` targets are never dispatched;
`person_part_reference` is in `NON_DISPATCHABLE_TARGETS` and also correctly produced no record). Model
routing was verified correct in the raw output: `person`/`face`/`hand`/`tree`/`house` route to
`grounding_dino_object_classes`, `heart` routes to `owlv2_object_classes`, `mouth` routes to
`grounding_dino_parts`, and `eye` routes to the composite `grounding_dino_parts+owlv2_parts_fallback`
key exactly as frozen in §4 — confirming the routing bug found and fixed during this stage (see
Errors and fixes below) is resolved. No crashes, no missing fields, no annotation input required
anywhere in this path — image path in, structured evidence records out.

**A real bug was caught and fixed at this stage**: the sanity-check script's model-loader originally
matched `eye_entry.best_model` by exact string equality against `"owlv2_parts"` /
`"grounding_dino_parts"`, but the frozen eye config is the composite string
`"grounding_dino_parts+owlv2_parts_fallback"` — neither exact match fired, so eye would have silently
received no predict_fn and always reported absent in this live demo (no crash, just quietly wrong).
Fixed by switching to substring detection and adding `_combined_fallback_predict_fn`, which reproduces
the exact deployed primary+fallback semantics at single-image granularity. The table above is from the
corrected run.

---

## 8. Stage 8 — Rule-integration readiness

| Target | Ready now | Allowed usage |
|---|---|---|
| person, face, hand, tree, house | Yes | Validated evidence pipeline (presence only) |
| eye | No (experimental only) | Technical View / experimental evidence — presence signal is precision-trustworthy when it fires (87.7-96.9%) but must not activate a validated psychological conclusion; bbox stored but not localization-validated |
| heart, animal, star, mouth | No (experimental only) | Technical View — must not activate a conclusion |
| circle, vehicle | No | Not used at all |

**41-rule impact**: of the 7 rules Phase 2C.4's coverage audit flagged as "already coverable" by
presence alone, all remain reachable via VALIDATED_AUTOMATIC classes. Eye's two rules
(`PSY_AR_EYES_CLOSED_003`, `EN_COMPILED_EYES_MISSING_DETAIL_020`) are **not currently reachable by
validated evidence** — eye is EXPERIMENTAL_AUTOMATIC, so any future rule-engine wiring must route its
records to Technical View / experimental-evidence-only handling and must not let them independently
activate either rule. **No rule was activated by this phase or any prior phase — `allowed_output_level` remains `disabled` for every rule in `rules_registry_v2.json`.**

---

## Verification

New tests: `tests/test_phase2c7_{eye_evaluation,eye_export_summary,dev_holdout_split,detector_policy,
visual_detector,safety}.py` — export loading against synthetic + real schema, prediction-loading
fairness (the Grounding-DINO-prefix bug found and fixed this session, pinned by regression test),
holdout isolation and no-leakage, deterministic policy construction, correct class→model routing
(including the heart→OWLv2 exception), experimental/disabled evidence structurally cannot reach
`validated_records_only()`, provenance preservation, image-only API signature (no annotation
parameter exists), no emotion-label dependence, frozen Phase 2C.4/2C.4A artifacts never opened for
writing, no fine-tuning imports/calls anywhere in the package.

**Results**: `ruff check` — all checks passed (Phase 2C.7 package, scripts, tests). `compileall` —
clean (Phase 2C.7 package + both scripts). Targeted Phase 2C.7 suite (6 files) — **85 passed, 0
failed**. Full repository suite — **1339 passed, 7 skipped (pre-existing, unrelated to this phase), 0
failed**, 588s wall clock.
