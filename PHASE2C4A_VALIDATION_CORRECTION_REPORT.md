# Phase 2C.4A — Validation / Correction Pass

**Status: analysis-only correction, no new inference, no new downloads, no new annotation,
no rule activated.** Operates exclusively on Phase 2C.4's already-committed predictions
(`artifacts/phase2c4/raw_predictions_{model}.csv`, `artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv`,
commit `39e65b8`, frozen and never overwritten). New code:
`src/doar/phase2c4/{macro_metrics,uncertainty,calibration,model_status}.py`,
`scripts/phase2c4a_run_corrections.py`, outputs under `artifacts/phase2c4a/`.

This pass exists because the original Phase 2C.4 report used `full_80` for its
model-ranking table (a cohort that includes the 7 locked-test images) and computed
macro-F1 by silently dropping classes where F1 was mathematically undefined — both
methodological problems that needed fixing before anything is built on top of Phase 2C.4.

---

## 1. Model selection cohort correction

All model-selection conclusions below use **`dev_eligible_excl_test` (73 images) only**.
The `full_80` table from the original report is retained in
`artifacts/phase2c4a/full_80_summary_for_reference_only.csv` for reference/reproducibility,
never for selection. `locked_test_descriptive_only` (7 images) appears in its own table
(§ below) and was not used to choose a model, threshold, prompt, class, or ensemble
anywhere in this correction pass — enforced at runtime by
`workspace.assert_pilot_ids_exclude_locked_test`, called before any calibration threshold
is chosen (`scripts/phase2c4a_run_corrections.py`), and by a dedicated regression test
(§ Verification).

### Dev-only ranking (`dev_eligible_excl_test`, 73 images, fixed 10-class macro)

| Rank | Model | Macro balanced accuracy | Macro F1 (corrected) | Low-support classes (n_present < 5) |
|---|---|---|---|---|
| 1 | **OWLv2** | **0.631** | 0.355 | heart, circle, vehicle |
| 2 | Grounding DINO | 0.596 | 0.340 | heart, circle, vehicle |
| 3 | YOLO-World | 0.520 | 0.092 | heart, circle, vehicle |
| 4 | Florence-2 | 0.501 | 0.370 | heart, circle, vehicle |

**OWLv2 remains the leading model on the development-eligible cohort** — 0.631 vs.
Grounding DINO's 0.596 macro balanced accuracy. This is the same ranking the (flawed)
`full_80` table produced (0.642 vs. 0.604), so the cohort correction does not change
which model leads, though the margin narrows slightly and (§4) the two models' bootstrap
confidence intervals overlap substantially — the ranking is directionally consistent, not
a large-margin result.

---

## 2. Macro metric correction

**Bug found and fixed**: `phase2b.evaluation.compute_class_metrics` correctly returns
`f1=None` when F1 is mathematically undefined — either because the detector made no
usable positive prediction at all for a class (`tp+fp == 0`, so precision is undefined),
or because precision and recall are both exactly 0.0 (the harmonic-mean 0/0 case). The
original report's macro-F1 table was computed by averaging only the classes with a
defined `f1`, which **silently dropped** every class a detector never fired on. A
detector that never predicts a class positive got *zero credit and zero penalty* for
that class under that scheme — inflating its macro F1 relative to a detector that fires
but is simply wrong. This is most visible in YOLO-World: on `full_80`, only 4 of 10
classes had a defined F1 (person/face/hand/tree); the other 6
(animal/house/heart/star/vehicle/circle) never fired at all and were dropped, giving a
macro F1 of 0.218 computed over 4 classes, not 10.

**Fix**: `src/doar/phase2c4/macro_metrics.py::f1_for_macro` — if `f1` is already defined,
use it; if `f1` is undefined and genuine positive ground truth exists in the cohort
(`n_present > 0`), score it **0.0** (a real miss, not a class that "doesn't count"); only
exclude a class from the macro average when `n_present == 0` (no detection task exists
for that class in that cohort at all — documented separately, not conflated with a miss).
Every model's macro F1 below is computed over the **same fixed 10-class denominator**
(`macro_metrics.macro_average_for_model_cohort` raises if any class is missing).

### Corrected macro F1 (`dev_eligible_excl_test`) — before vs. after

| Model | Original report macro F1 (full_80, classes silently dropped) | Corrected macro F1 (dev cohort, fixed 10-class denominator) | Classes excluded from F1 (n_present=0) |
|---|---|---|---|
| OWLv2 | 0.467 | **0.355** | none |
| Grounding DINO | 0.399 | **0.340** | none |
| Florence-2 | 0.386 | **0.370** | none |
| YOLO-World | 0.218 | **0.092** | none |

Every model's corrected macro F1 is lower than originally reported — YOLO-World's most
severely (0.218 → 0.092, more than halved), because 6 of its 10 classes never fired at
all and are now correctly scored as misses instead of being dropped. **Balanced accuracy
was not affected by this bug** (it was already computed per-class with its own,
independent None-propagation rule and macro-averaged without a dropping issue on this
data — verified by recomputing it against the original `full_80` figures: 0.642/0.604/
0.518/0.501, identical to the original report). Balanced accuracy remains the primary
class-imbalance-aware ranking metric, per instruction; corrected macro F1 is now reported
alongside it, not in place of it.

Full per-class detail: `artifacts/phase2c4a/dev_only_model_selection_summary.csv`
(`excluded_f1_classes` / `excluded_bal_acc_classes` columns record exactly which classes,
if any, were excluded and are empty for every model on the dev cohort — no class in this
cohort had zero positive ground truth).

---

## 3. Florence-2 interpretation — corrected

The original report's own text already avoided calling Florence-2 "poor" outright and
computed balanced accuracy specifically to avoid the misleading F1/count read — but its
section heading and surrounding language still risked being read as a verdict on the
model. Restated precisely, and encoded in
`src/doar/phase2c4/model_status.py::MODEL_CONFIGURATION_STATUS` (with a regression test
that greps for forbidden generalizing phrases, § Verification):

> The tested configuration — **Florence-2-base + the `<OPEN_VOCABULARY_DETECTION>` task +
> this benchmark's bare-noun-phrase prompt** — did not produce discriminative
> presence/absence evidence on this benchmark (macro balanced accuracy 0.501 on the dev
> cohort, 95% bootstrap CI [0.500, 0.502] — indistinguishable from chance, and a tight
> interval, not noise). This is a finding about **this configuration**, not a claim that
> Florence-2 (the architecture) is incapable of open-vocabulary detection.

Status recorded: `failed_inconclusive_for_current_use` for this exact configuration.
**Florence-2 was not rerun over all 80 drawings in this correction pass.** Recorded as
optional future work, not scheduled: a small (5–10 development-image, no locked-test)
sanity experiment comparing documented Florence-2 task formulations (e.g.
`<CAPTION_TO_PHRASE_GROUNDING>` instead of `<OPEN_VOCABULARY_DETECTION>`) if scientifically
justified — requires your separate approval before any execution, same as every other
"future work" item in this project.

---

## 4. Uncertainty

80 drawings total (73 dev-eligible), and several classes have very few positive
instances in the dev cohort: **heart (4), circle (4), vehicle (3)** all fall below
`MIN_POSITIVE_SUPPORT=5` (the same threshold Phase 2B's own evaluation module already
uses) and are flagged `low_support_classes` in every summary table. `star` (5) and
`animal`/`house`/`tree` (10/12/13) clear the floor but are still modest. **No strong claim
in this report rests on heart, circle, or vehicle alone.**

### 95% bootstrap CIs, macro metrics, `dev_eligible_excl_test` (2000 resamples, image-level, seed=20260807)

| Model | Macro balanced accuracy [95% CI] | Macro F1 [95% CI] |
|---|---|---|
| OWLv2 | 0.631 [0.601, 0.670] | 0.355 [0.287, 0.414] |
| Grounding DINO | 0.596 [0.540, 0.659] | 0.340 [0.292, 0.393] |
| YOLO-World | 0.520 [0.505, 0.536] | 0.092 [0.048, 0.132] |
| Florence-2 | 0.501 [0.500, 0.502] | 0.370 [0.333, 0.414] |

**Reading this honestly**: OWLv2's and Grounding DINO's balanced-accuracy intervals
overlap (OWLv2's lower bound 0.601 sits above Grounding DINO's point estimate 0.596 but
well inside Grounding DINO's own interval) — the ranking is directionally consistent
across 2000 resamples but the gap between the top two models is **not large relative to
sampling uncertainty** at n=73. Florence-2's interval is tight and centered almost exactly
on 0.500, which is itself informative: its chance-level performance is stable across
resamples, not an artifact of a few unlucky images. Full detail:
`artifacts/phase2c4a/dev_cohort_bootstrap_ci.csv`.

---

## 5. Calibration — design + execution

**The original zero-shot/default-threshold results remain frozen**
(`artifacts/phase2c4/phase2c4_model_class_cohort_metrics.csv`, unmodified by this pass).
This section is a **separate, secondary** experiment using only the already-recorded raw
scores in `artifacts/phase2c4/raw_predictions_{owlv2,grounding_dino}.csv` — no new
inference.

**A real constraint on what this data can validly recover, stated before any result
below**: `post_process_grounded_object_detection` discards below-threshold detections
before a score is ever written to those CSVs, so a row's score is exactly `0.0` whenever
a class was *not* detected at the model's original operating threshold (OWLv2: 0.1,
Grounding DINO: 0.25). An undetected class's true confidence is therefore only known to
be *below* the original threshold, never the exact value. A threshold sweep is only
faithfully re-derivable from this data for candidate thresholds **≥ the original
threshold** — raising the bar is valid (a row scored 0.0 stays correctly "not detected"
at any higher t); lowering it is not (the real sub-threshold score is unknown). This
constraint was not worked around and conveniently matches the requested
precision-oriented (not recall-maximizing) calibration goal: **this pass can only ever
raise precision, never manufacture more recall.**

**Predeclared operating-point criterion** (fixed in `calibration.py` before any per-class
dev-cohort result was computed, never adjusted afterward): among candidate thresholds
whose dev-cohort precision is ≥ **0.6**, pick the smallest such threshold (the most
recall-preserving choice that still clears the bar); ties broken by balanced accuracy,
then recall. If no candidate threshold reaches 0.6 precision, the class **abstains** — no
calibrated operating point is recommended for it, rather than forcing a low-precision
choice into a system that will feed a psychological rule. 0.6 was chosen because a
detector feeding false "class present" evidence into a rule is not a neutral error in
this system; it was set once, before results, and not revisited.

Candidate grids swept (dev cohort only): OWLv2 {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40};
Grounding DINO {0.25, 0.30, 0.35, 0.40, 0.45, 0.50}.

### Frozen operating points (dev-cohort selection, before any locked-test contact)

| Model | Class | Original threshold | Decision | Frozen threshold | Dev precision | Dev recall |
|---|---|---|---|---|---|---|
| OWLv2 | person | 0.10 | calibrated | 0.10 (unchanged) | 1.000 | 0.308 |
| OWLv2 | face | 0.10 | calibrated | 0.10 (unchanged) | 0.964 | 0.422 |
| OWLv2 | hand | 0.10 | calibrated | 0.10 (unchanged) | 0.800 | 0.211 |
| OWLv2 | house | 0.10 | calibrated | 0.10 (unchanged) | 1.000 | 0.667 |
| OWLv2 | tree | 0.10 | calibrated | 0.10 (unchanged) | 1.000 | 0.462 |
| OWLv2 | heart | 0.10 | calibrated | **0.35** | 1.000 | 0.250 | 
| OWLv2 | animal / star / circle / vehicle | 0.10 | **abstain** | — | no threshold reached 0.6 precision | — |
| Grounding DINO | person | 0.25 | calibrated | 0.25 (unchanged) | 0.840 | 0.808 |
| Grounding DINO | face | 0.25 | calibrated | 0.25 (unchanged) | 0.978 | 0.703 |
| Grounding DINO | hand | 0.25 | calibrated | 0.25 (unchanged) | 0.909 | 0.263 |
| Grounding DINO | house | 0.25 | calibrated | **0.40** | 0.692 | 0.500 |
| Grounding DINO | tree | 0.25 | calibrated | **0.50** | 0.636 | 0.538 |
| Grounding DINO | animal / heart / star / circle / vehicle | 0.25 | **abstain** | — | no threshold reached 0.6 precision | — |

**Reading this**: OWLv2's default threshold was already the best available choice on 5 of
6 non-abstaining classes (raising it further only lost recall without gaining anything);
`heart` needed raising to 0.35 to reach 0.6 precision, at a real recall cost (1.00 → 0.25)
— and `heart` is a low-support class (n_present=4), so this specific calibration should be
treated as provisional. Grounding DINO's `house` and `tree` needed real threshold
increases (0.25 → 0.40 / 0.50) to reach the precision bar, consistent with § 8's
dominant-failure-mode finding (Grounding DINO over-fires on small-symbol/background
classes at its default threshold). **6 of 10 classes abstain for both models** —
`animal`/`star`/`circle`/`vehicle` (plus `heart` for Grounding DINO) never reach 0.6
precision at any threshold this data can validate, meaning no amount of raising the bar
alone fixes these classes; the failure mode is not simply "threshold too low." Full sweep
table: `artifacts/phase2c4a/calibration_dev_threshold_sweep.csv`; frozen decisions:
`artifacts/phase2c4a/calibration_frozen_operating_points.csv`.

### Locked-test cohort — descriptive only, reported once, after freezing

Applied the already-frozen thresholds above to the 7 locked-test images purely for
description; no threshold was adjusted afterward.

| Model | Class | Frozen threshold | Locked precision | Locked recall |
|---|---|---|---|---|
| OWLv2 | house | 0.10 | 1.000 | 1.000 |
| OWLv2 | tree | 0.10 | 1.000 | 0.500 |
| OWLv2 | heart | 0.35 | 0.000 | 0.000 |
| Grounding DINO | person | 0.25 | 1.000 | 0.857 |
| Grounding DINO | house | 0.40 | 1.000 | 0.500 |
| Grounding DINO | tree | 0.50 | 1.000 | 0.500 |

(Locked cohort is 7 images; several classes have 0 negatives or 0 positives there, so
precision/recall are frequently undefined or based on 1–2 examples — see
`artifacts/phase2c4a/calibration_locked_test_descriptive.csv` for the complete,
undownsampled table. **Not used to revise any threshold or conclusion above.**)

---

## 6. Keep two finalists — supported by the dev-cohort results

The dev-cohort class-wise pattern supports **not** collapsing to one universal detector:

- **OWLv2 = conservative primary evidence candidate.** High precision where it fires
  (0.80–1.00 on 5 of 6 non-abstaining classes at its default threshold), lower recall.
  Best fit for evidence that will feed directly into a psychological rule, where a false
  "present" is more costly than a missed one.
- **Grounding DINO = high-recall candidate-generation / annotation-assistance
  candidate.** Substantially higher recall on person/face/animal (0.70–0.93+) at a real
  precision cost (collapses below 0.25 on several classes at its default threshold) —
  not trustworthy unreviewed, but a strong *proposal generator* for a human review pass
  (§9 of the original `PHASE2C4_DETECTOR_BENCHMARK_REPORT.md`).

**No ensemble was implemented.** Recorded as a future experiment design only:

> **FT-E1 (design only, not executed)**: compare (a) OWLv2 alone, (b) Grounding DINO
> alone, (c) a class-aware combination — e.g. OWLv2's calibrated operating point for
> person/face/hand/house/tree/heart, Grounding DINO's proposals routed to human review
> for animal/star/circle/vehicle/heart (its abstaining classes) — against genuine human
> ground truth on a cohort disjoint from whatever data trained/calibrated any component.
> Would need its own frozen dev/test split discipline, not reuse of this pass's dev
> cohort for both calibration and ensemble evaluation (that would leak). Not scheduled.

---

## 7. Rule safety — reaffirmed, distinctions made explicit

**No detector-driven psychological rule was activated by this correction pass** —
verified by grep (no `rules_registry_v2.json` write, no `allowed_output_level` mutation,
same check as every prior phase). Two distinctions kept explicit throughout this document
and its code:

- **A model produces a bounding box** (OWLv2, Grounding DINO, YOLO-World all return real
  box coordinates; not used for evaluation in Phase 2C.4 or this pass, per instruction) —
  this is a raw model output, nothing more.
- **Bounding-box localization has been validated** — this has NOT happened. Genuine
  human bounding-box ground-truth coverage remains **0 / 235 = 0.0%**
  (`PHASE2C3_AUDIT_AND_DESIGN.md` §2, unchanged since Phase 2C.3; not re-measured here
  since no new annotation occurred). A model box's *presence* was used only as a
  presence/absence signal (does *any* box exist for this class); its *coordinates* have
  never been checked against a human-drawn box on this dataset, and no code in this
  repository claims otherwise.

---

## 8. Recommended next phase

**Phase 2C.5 should proceed with model-assisted human annotation focused on rule-critical
evidence**, using the corrected findings above as its basis:

- **Eye presence/bbox + eye state/detail** — highest-value addition by rule count
  (unlocks 4 rules, unchanged from the original report's §9 analysis).
- **Mouth presence/bbox** — unlocks 1 rule directly.
- **Hand bbox/count** — unlocks the relative-size and omission halves of 2 rules, no new
  class needed (`hand` is already benchmarked; OWLv2 is precision-usable on it at its
  default threshold, recall 0.21 — annotation should not assume the detector alone finds
  most hands).
- **Face/body-part relative size** — justified once part bboxes exist (needs a reference
  scale, per the original report's Stage-F design).
- **Face-expression appearance taxonomy** — only if a defensible annotation protocol can
  be defined; flagged as the hardest of the five, unchanged from the original assessment.

**Grounding DINO is recommended as the proposal generator** for this annotation pass —
its high recall on person/face/animal (and generally, its behavior of over-proposing
rather than under-proposing) is exactly the profile a human-reviewed candidate-box
workflow wants, supported by §6's dev-cohort finding, not asserted from the original
`full_80` numbers alone.

**OWLv2 remains the primary automatic detector candidate** for whichever of the 7
already-coverable rules (§4 of the original report) get exercised without waiting for new
annotation — restricted to its calibrated, precision-≥0.6 classes only
(person/face/hand/house/tree, plus heart with the caveat that it is low-support). The 6
classes that abstained under calibration (animal/star/circle/vehicle for both models, plus
heart for Grounding DINO) should **not** feed a rule automatically at any threshold this
data supports — annotation extension or a different model, not further threshold-tuning
on the same detector, is the right next step for them.

**Fine-tuning remains not recommended yet**, unchanged from the original report — this
correction pass did not change the underlying bbox-coverage or annotation-volume facts
that finding rested on.

---

## Verification

New/updated tests (all synthetic-fixture-only, no real model weights, no private data):

- `tests/test_phase2c4a_macro_metrics.py` — `f1_for_macro`/`balanced_accuracy_for_macro`
  zero-division policy (defined value passes through; undefined + n_present>0 → 0.0;
  undefined + n_present==0 → excluded); `macro_average_for_model_cohort` raises on a
  missing class (fixed 10-class denominator enforced); regression test reproducing the
  exact YOLO-World bug (6 dropped classes) and asserting the corrected macro F1 is lower
  than the naive (drop-undefined) average would have been.
- `tests/test_phase2c4a_calibration.py` — threshold sweep only ever evaluates candidates
  ≥ the model's original threshold (asserted against `CANDIDATE_THRESHOLDS`);
  `select_operating_point` picks the smallest threshold clearing the precision bar and
  abstains when none does (both branches exercised); `freeze_operating_points` /
  `apply_frozen_thresholds` are separate functions — a structural test asserts
  `apply_frozen_thresholds` never calls `sweep_thresholds`/`select_operating_point`
  (AST-level, mirrors the existing no-threshold-tuning check style), i.e. the locked-test
  application path cannot re-derive a threshold.
- `tests/test_phase2c4a_uncertainty.py` — `bootstrap_macro_ci` is deterministic given a
  fixed seed (two calls with the same inputs/seed produce identical output); CI bounds
  bracket the point estimate on a synthetic fixture with known variance.
- `tests/test_phase2c4a_safety.py` — model-selection code path only ever touches
  `DEV_ELIGIBLE`-cohort rows before producing a ranking (grep/AST check that
  `dev_only_model_selection_summary` construction never reads `LOCKED_TEST` rows);
  `assert_pilot_ids_exclude_locked_test` is actually called before calibration freezing
  (not just documented); `model_status.py` contains none of `FORBIDDEN_GENERALIZING_PHRASES`
  anywhere in this repository's tracked `.md`/`.py` files; no rule-registry file is
  imported or written by any `phase2c4a` module.

Run: `pytest tests/test_phase2c4a_*.py` plus full regression
(`pytest tests -q`), `ruff check`, `python -m compileall` — results in the final report to
the user.
