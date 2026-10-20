# Pre-registration — DOAR journal validation (MVA submission)

Committed on branch `journal/mva-2026` **before** any of the experiments below
was run. Changing a threshold or rule after seeing its result requires a new,
dated entry in the change log at the end of this file, and the paper reports
both the original and the changed analysis.

## Research questions

- **RQ1** How much do augmentation copies of the same source drawing that sit
  in different splits inflate validation performance and change encoder
  ranking and adaptation conclusions?
- **RQ2** Do automatic QC flags and confidence/assessability states predict
  whether a drawing-mark mask, and the features computed from it, are correct?
- **RQ3** Under a leakage-controlled split and human-referenced measurements,
  which representation and which measurements remain defensible?

## Claims, tests and decision rules

| ID | Claim | Experiment / script | Pre-set rule |
|---|---|---|---|
| C1 | Same-Roboflow-stem files are copies of one drawing | A5 blinded pair review (40 same-stem train–valid pairs + 20 same-class controls; same for plain-name pairs) | Confirmed if the lower Wilson 95% bound of "judged same" among same-stem pairs is ≥ 0.80 and controls are judged same ≤ 10% |
| C1b | Image-evidence candidates are duplicates | A4 dihedral dHash ≤ 6; CLIP ViT-B/32 (OpenAI weights) cosine ≥ 0.95; A5 reviews 40 cross-split candidates per evidence type | Use an evidence type for T1 grouping if ≥ 90% of reviewed candidates are judged same; otherwise tighten the threshold (dHash ≤ 4, cosine ≥ 0.97) and record it |
| C2 | Leakage inflates validation scores | A2: re-score every saved T0 validation prediction file on shared-source vs independent images | Report all runs; inflation is supported if the bootstrap CI of (shared − independent) macro-F1 excludes 0 for the selected configuration and the sign is positive in ≥ 80% of runs. Class mix and source family are reported as alternative explanations |
| C3 | T1 removes cross-split copies | A3 (T0 groups + rf-stem + confirmed image-evidence edges) | Zero cross-split edges of any accepted evidence type; T1 frozen only after author approval |
| C4 | Leakage changes model selection | B1–B3 on T1 with unchanged code, hyper-parameters and seeds | No directional hypothesis. Reported: T0 vs T1 tables, Kendall τ of probe rankings, adaptation gains. Shortlist = top-3 10-seed probe means; finalists = top-2 adaptation configs on seeds 19/42/73; selection = highest 5-seed mean (tie → lower SD) |
| C5 | Selected model generalises to unseen test drawings | B4, once, after `SELECTION_FROZEN.json` | Report macro-F1 with image-level bootstrap CI, per-class F1; paired bootstrap + exact McNemar vs comparator and vs linear probe. No change of anything after B4 |
| C6 | Automatic QC acceptance overstates mask validity | C1 (80-image holdout visual review, frozen rubric) and C2/C3 (≈120 reference masks; 40 double-annotated) | Wilson CIs. Supported if the proportion of V2 `MASK_USABLE` masks judged GOOD has an upper 95% bound < 0.90, or A0-"OK" masks have median F1 < 0.7 against references |
| C7 | Support-aware V2 beats standard methods | C4 paired comparisons on references | Primary metric pixel F1; paired bootstrap of the mean per-image difference, Wilcoxon signed-rank, Holm across 7 comparisons; reported overall and per V2 support regime |
| C8 | Mask error propagates into named features | C4 feature agreement (reference-mask features vs each method's) | Per feature ICC(2,1) and MAE; a feature is "reliable under method M" if ICC ≥ 0.75 (Koo & Li 2016 "good") |
| C9 | Assessability states identify bad measurements | C4 risk–coverage | Bad = F1 < 0.5 on an assessable image, or any image a human marked not assessable. Gate supported if AUROC ≥ 0.75 and AURC below the no-gate risk with a bootstrap CI excluding equality; otherwise reported as a negative result |

## Fixed parameters

- Bootstrap: 10,000 image resamples, percentile intervals, seed 20261020.
- Classical baselines: Otsu (global, luminance), Sauvola and Niblack
  (window 25, k = 0.2), adaptive Gaussian (block 35, C = 10), Otsu on Lab
  distance from the median colour. Not tuned on any reference mask.
- Boundary F1 tolerance: 2 px at working resolution (longest side ≤ 1024 px).
- Reference sample: pool = T0 train + valid minus the 84 development and 80
  holdout images and their Roboflow-stem siblings; strata = V2 support regime
  × A0 QC state; largest-remainder allocation, ≥ 4 per non-empty stratum;
  seed string `doar-journal-2026-c2`.
- E1 hyper-parameters: exactly the archived E1-A2 / E1-B values
  (probe: head LR 3e-4, WD 1e-4, 30 epochs, patience 5; adaptation: head LR
  3e-4, encoder LR 3e-5, WD 1e-4, batch 16, label smoothing 0.05, partial =
  final ~25% of visual units, 30 epochs, patience 5).

## What is not claimed

No psychological, clinical or diagnostic meaning of any label, feature or
prediction. The four labels are benchmark categories of unknown origin.

## Change log

- 2026-10-20 — initial version.
