# Phase 2C.2 Descriptive Object-Baseline Evaluation Report

**Status: descriptive evaluation only. No training, fine-tuning, threshold tuning,
calibration, rule activation, Parent View change, or ontology expansion was
performed.** Branch `feature/doar-phase2c-annotation-expansion`. Ground truth:
genuine Phase 2C.1 human annotation only (`annotator_type == "human"`), never the
legacy Phase 2B provisional labels — enforced structurally
(`doar.phase2c2.ground_truth.genuine_human_ground_truth`, tested).

## 0. What was run, and confirmation nothing was changed

Phase 2B's own, unmodified `ZeroShotClipDetector.load_real()` (ViT-B-32, `openai`
weights, CPU, `detect_threshold=0.28`, `uncertain_margin=0.03`, prompt template `"a
child's drawing containing a {cls}"`) and `detect_circles_classical()`
(`circularity_threshold=0.75`, `min_contour_area=30`) were run against all 80
private blinded images via the existing, unmodified
`scripts/phase2b_run_baseline.py`. **Verified byte-identical** to Phase 2B's
original values for the two overlapping images (`p2b_0000`, `p2b_0001`) — same
model, same weights, same preprocessing, same threshold, same output, confirming
nothing drifted. `artifacts/phase2c2/model_manifest.json` records this explicitly
(`"reused_unchanged_from_phase2b": true`).

**No threshold was tuned.** `predicted_positive_from_clip()` reads only the
already-computed `{class}__status == "detected"` field the fixed threshold produced
— structurally verified (`tests/test_phase2c2_safety.py::NoThresholdTuningTests`).

## 1. Cohorts (original_split provenance preserved)

| Cohort | n images | Definition |
|---|---|---|
| `full_80` | 80 | All annotated pilot images — descriptive only |
| `dev_eligible_excl_test` | 73 | Excludes the 7 original-test-split images |
| `locked_test_descriptive_only` | 7 | Original test split — reported, never used for selection |

The 7 locked-test images are included in `full_80`'s descriptive numbers below but
**must never inform model selection, threshold choice, or calibration** —
unchanged from the Phase 2C.1 audit's guard (`assert_pilot_ids_exclude_locked_test`,
not invoked by anything in this evaluation).

## 2. Full metrics — `full_80` cohort (all 80 images, descriptive)

| Class | Baseline | n present | TP | FP | FN | TN | Precision | Recall | F1 | Specificity | Ranking sep. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| person | CLIP | 59 | 2 | 0 | 57 | 21 | 1.000 | 0.034 | 0.066 | 1.000 | **0.531** |
| face | CLIP | 71 | 4 | 0 | 67 | 9 | 1.000 | 0.056 | 0.107 | 1.000 | **0.264** |
| hand | CLIP | 43 | 3 | 2 | 40 | 35 | 0.600 | 0.070 | 0.125 | 0.946 | **0.645** |
| animal | CLIP | 13 | 0 | 1 | 13 | 66 | 0.000 | 0.000 | — | 0.985 | 0.488 |
| house | CLIP | 14 | 4 | 0 | 10 | 66 | 1.000 | 0.286 | 0.444 | 1.000 | **0.740** |
| tree | CLIP | 15 | 4 | 1 | 11 | 63 | 0.800 | 0.267 | 0.400 | 0.984 | **0.738** |
| heart | CLIP | 5 | 1 | 0 | 4 | 75 | 1.000 | 0.200 | 0.333 | 1.000 | 0.584 |
| star | CLIP | 6 | 0 | 2 | 6 | 72 | 0.000 | 0.000 | — | 0.973 | 0.554 |
| circle | CLIP | 4 | 0 | 2 | 4 | 74 | 0.000 | 0.000 | — | 0.974 | 0.457 |
| **circle** | **classical CV** | 4 | 3 | **73** | 1 | 3 | **0.039** | **0.750** | 0.075 | 0.039 | 0.362 |
| vehicle | CLIP | 5 | 0 | 0 | 5 | 75 | — | 0.000 | — | 1.000 | 0.347 |

Full per-cohort table (all 3 cohorts x all baselines): `artifacts/phase2c2/class_metrics_by_cohort.csv`.
False-positive/false-negative pilot_ids per class/baseline/cohort:
`artifacts/phase2c2/fp_fn_examples.csv`.

**Reading this table**: precision looks artificially perfect for several classes
(`person`, `face`, `house`, `heart`) — this is an artifact of the fixed threshold
almost never firing "detected" at all (near-zero recall), not evidence the model is
reliable. **Ranking separation is the more honest signal**: it does not depend on
the miscalibrated threshold at all.

## 3. Ranking-separation reading (threshold-independent, the honest signal)

| Tier | Classes |
|---|---|
| Real, moderate-to-good signal (>0.65) | house (0.740), tree (0.738) |
| Weak-to-moderate signal (0.5–0.65) | hand (0.645), heart (0.584), star (0.554), person (0.531) |
| Essentially random (~0.45–0.5) | animal (0.488), circle/CLIP (0.457) |
| **Worse than random** (representation problem, not just miscalibration) | **face (0.264)**, **vehicle (0.347)** |

`face` and `vehicle` ranking below 0.5 means CLIP's raw similarity score is
*anti-correlated* with true presence on this dataset — a genuine representation
failure for these two classes, not something a different threshold could fix.

## 4. Failure modes

1. **Threshold miscalibration dominates recall everywhere.** The fixed 0.28
   threshold (chosen from Phase 2B's original n=20 pilot) almost never fires
   "detected" at n=80 either — recall is <10% for 8 of 10 classes even where the
   underlying ranking signal is real (e.g. `house`, `tree`). This is the single
   biggest, most fixable issue, but fixing it requires a genuinely held-out
   calibration set — not attempted here.
2. **Representation failure for `face` and `vehicle`.** Worse-than-random ranking
   separation means the CLIP embedding itself doesn't separate these classes well
   on children's line drawings, regardless of threshold.
3. **Classical-CV `circle` massively over-fires.** 73 of 76 true negatives are
   false positives (precision 0.039) — confirms and sharpens Phase 2B's original
   n=20 finding (0.059 precision) with a 4x larger, more reliable sample. High
   recall (0.75) shows it does catch real circles, just among a flood of
   false-positive contours (curved lines, loops, letters) common in line drawings.
4. **`animal` and `circle`/CLIP sit at essentially random ranking** — no usable
   signal from either baseline for these classes as currently configured.

## 5. Comparison with Phase 2B's original 20-image findings

| Class | n=20 ranking sep. | n=80 ranking sep. | What changed |
|---|---|---|---|
| person | 0.704 | 0.531 | **Weaker** — the n=20 estimate (9 positives) was optimistic |
| face | 0.178 | 0.264 | Still worse-than-random both times — **confirmed, not an n=20 fluke** |
| house | 0.938 | 0.740 | Still the strongest class, but **weaker** than the n=20 estimate suggested |
| tree | 0.933 | 0.738 | Same pattern — still good, **weaker** than n=20 suggested |
| star | **1.000** | 0.554 | **Dropped drastically** — the "perfect" n=20 score was 1 positive example, pure noise |
| heart | 0.219 | 0.584 | **Improved substantially** — n=20's 2 positives gave a misleadingly bad estimate |
| circle | 0.529 | 0.457 | Stable — near-random both times |
| animal | 0.417 | 0.488 | Stable — near-random both times |
| vehicle | n/a (0 positives at n=20) | 0.347 | New finding — worse than random |
| hand | n/a (0 positives at n=20) | 0.645 | New finding — weak-to-moderate signal |

**Which Phase 2B conclusions hold up**: `face`'s anti-correlated ranking (the
pilot's headline negative finding) is **confirmed**, not weakened, at 4x the sample
size. Classical-CV `circle` over-firing is **confirmed and sharpened**.

**Which conclusions do not hold up**: `star`'s apparent "perfect" separation was
sampling noise from a single positive example — this is direct, quantitative
evidence for the methodology docs' own repeated warning
(`docs/PHASE2B_EVALUATION_PROTOCOL.md`) that n=20 (and especially 1–3-positive)
estimates are not generalizable. `person`, `house`, and `tree` were directionally
right but overstated at n=20. Full table: `artifacts/phase2c2/comparison_to_phase2b_20.csv`.

## 6. Locked-test cohort (descriptive only, n=7)

Denominators are small enough that most metrics are undefined (e.g. `person`/`face`
have 0 true negatives in this 7-image cohort — every locked-test image the human
annotator marked `present` for both). Full per-class numbers:
`artifacts/phase2c2/class_metrics_by_cohort.csv` (`cohort ==
"locked_test_descriptive_only"`). **These numbers are reported for completeness
only and were not used anywhere in this report's conclusions or comparisons.**

## 7. Artifacts written this session

All under `artifacts/phase2c2/` (committed; opaque pilot_ids and numeric outputs
only — no source paths, no emotion labels, no drawings, matching Phase 2B's own
precedent for `raw_predictions.csv`):

- `raw_predictions_80.csv` — real CLIP + classical-CV predictions, all 80 images
- `class_metrics_by_cohort.csv` — full metric table, 33 rows (10 classes x 3
  cohorts CLIP + `circle` x 3 cohorts classical CV)
- `fp_fn_examples.csv` — every false-positive/false-negative pilot_id, by class/baseline/cohort
- `comparison_to_phase2b_20.csv` — old vs. new, per class
- `model_manifest.json` — explicit record that model/threshold/prompt/preprocessing are unchanged

Generated by `scripts/phase2c2_evaluate.py` (reusable, reads only the private
store/mapping/predictions, writes only these derived tables).

## What was NOT done this session

No threshold tuning, no calibration, no supervised training or fine-tuning, no rule
activation, no Parent View change, no ontology expansion, no stronger detector
experiment. Nothing pushed.
