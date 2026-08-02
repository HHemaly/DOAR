# Phase 3A — Preliminary End-to-End Training and Functional Validation

**Status: FUNCTIONAL PRELIMINARY EXPERIMENT. Not a leakage-safe scientific
evaluation.** Duplicates, near-duplicates, and label-conflict images (already
quantified in `CURRENT_STATE_AUDIT.md` §2 and the interrupted dataset-audit
request) were deliberately **not** removed for this run, per instruction. The
metrics below almost certainly overstate real generalization — see §6.

Goal: verify that training, evaluation, inference, feature extraction, rules,
report generation (EN/AR), and versioned case output all work together
end-to-end on the full, currently-readable dataset, and surface any wiring
bugs. Detector work and psychological-rule activation were explicitly out of
scope.

---

## 1. Dataset (no exclusion)

Manifest built directly from the original dataset directory
(`C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing`), no leakage
quarantine applied:

| Split | Angry | Fear | Happy | Sad | Total |
|---|---|---|---|---|---|
| train | 641 | 502 | 921 | 757 | 2,821 |
| valid | 77 | 45 | 110 | 78 | 310 |
| test | 123 | 130 | 168 | 136 | 557 |
| **Total** | | | | | **3,688** |

**Unreadable files: 0** (all 3,688 manifest rows have `readable=True`).

**Split protocol**: the dataset's existing `train/valid/test` folder
structure was used directly — it is already fixed and deterministic (the
same image always lands in the same split on every run), which satisfies
reproducibility without needing a new pooled-and-resplit step for this
functional-verification pass. This is a deliberate scoping choice, not an
oversight: a from-scratch stratified re-split is separate work, orthogonal to
what Phase 3A needed to prove.

The leakage gate (run automatically by `train-image-model`) still detected
the same known issue and was overridden with an explicit, audit-logged
justification rather than bypassed silently — see
`outputs/phase3a/model/leakage_gate/leakage_override_audit.jsonl`.

---

## 2. Exact commands

```powershell
python main.py build-manifest --dataset "$DATA" --output "outputs\phase3a\manifest.csv"

python main.py train-image-model --dataset "$DATA" --model mobilenet_v3_small `
  --output "outputs\phase3a\model" --seed 42 --epochs 10 --batch-size 4 `
  --grad-accum-steps 4 --image-size 224 --device auto `
  --allow-leakage-override --override-justification "Phase 3A: preliminary end-to-end functional verification..."

python main.py export-probabilities --model "outputs\phase3a\model\best.pt" `
  --manifest "outputs\phase3a\manifest.csv" --device auto `
  --output "outputs\phase3a\probability_export.json" --splits valid

python main.py evaluate-predictions --export "outputs\phase3a\probability_export.json" `
  --split valid --output "outputs\phase3a\eval"

python main.py analyze-image --image "<one real image per class>" `
  --emotion-checkpoint "outputs\phase3a\model\best.pt" --output "outputs\phase3a\case_<Class>"
```

Evaluation deliberately used the **validation** split, not test — this keeps
the locked-test guard untouched and avoids any appearance of a "final"
evaluation, consistent with the preliminary framing.

---

## 3. Model / configuration

`mobilenet_v3_small`, seed 42, 10 epochs, batch size 4 with 4 gradient-
accumulation steps (effective batch 16), image size 224, mixed precision on
CUDA (automatic), device `cuda` (confirmed in `training_result.json`).

## 4. Training time

**520.6 seconds (~8.7 minutes)** for 10 epochs on the full 2,821-image train
split, GPU (`cuda`), confirmed from `training_result.json`.

## 5. Actual metrics (validation split, n=310, PRELIMINARY)

| Metric | Value |
|---|---|
| Accuracy | 0.729 |
| Macro-F1 | 0.704 |
| Weighted F1 | 0.730 |
| Balanced accuracy | 0.713 |
| Log loss | 0.733 |
| Multiclass Brier | 0.386 |
| ECE | 0.045 |
| Macro OvR ROC-AUC | 0.905 |
| Macro PR-AUC | 0.778 |

Per-class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Angry | 0.734 | 0.610 | 0.667 | 77 |
| Fear | 0.542 | 0.711 | 0.615 | 45 |
| Happy | 0.856 | 0.864 | 0.860 | 110 |
| Sad | 0.684 | 0.667 | 0.675 | 78 |

Confusion matrix (rows = true, columns = predicted, order Angry/Fear/Happy/Sad):
`[[47,11,8,11],[4,32,2,7],[6,3,95,6],[7,13,6,52]]`

Loss/F1 curve: `outputs/phase3a/loss_curve.png` (train loss fell 1.217→0.700
monotonically; valid macro-F1 rose 0.578→peak 0.704 at epoch 7, with the
usual epoch-to-epoch noise of a single-seed run on ~300 validation images).

## 6. Why this run scores higher than the earlier leakage-cleaned run — and why that's a warning, not an improvement

This run's macro-F1 (0.704) is *higher* than the equivalent leakage-cleaned
run from an earlier session (0.656, same architecture family). That is not
evidence this model is better — it is the expected signature of **validation
leakage**: with duplicates and near-duplicates retained across splits, the
model can partly "recognize" validation images it effectively already saw
(as a near-duplicate) during training. This is precisely why the result is
labeled preliminary and must not be cited as a thesis number. It is, however,
a useful **data point for prioritizing the dataset audit**: it gives a rough
sense of the inflation leakage causes on this specific dataset/architecture
(~0.05 macro-F1 in this one run) — not a precise estimate, but a real signal
that the audit matters.

## 7. Real images through the complete pipeline

One real image per class (`test` split) run through `analyze-image` with the
Phase 3A checkpoint — every stage exercised: quality gate, segmentation,
composition, colour, deep emotion inference, tier-aware rule evaluation
(19 rules: 6 Tier-1 evaluated, 13 Tier-2 correctly `missing_detector`),
concern-convergence engine (correctly `[]`, disabled in production),
label-provenance audit, bilingual (EN/AR) report generation, and versioned
case output.

| Class | Predicted | Confidence | Label-audit status |
|---|---|---|---|
| Angry | Angry | 0.814 | CONSISTENT |
| Fear | Angry | 0.744 | POSSIBLE_CONFLICT |
| Happy | Happy | 0.553 | CONSISTENT |
| Sad | Sad | 0.803 | CONSISTENT |

3/4 correct; the one miss (Fear→Angry) is a plausible confusion also visible
in the confusion matrix above (Fear has the weakest precision of the 4
classes). All 4 cases produced `bilingual.html`, `parent_en.html`,
`parent_ar.html`, `professional_en.html`, `professional_ar.html` —
confirmed present for every case. `rules.json`: 19/19 rules present with
correct `tier`/`activation_status` fields for all 4 cases. `concerns.json`:
`[]` for all 4, as required. No `versions/` directory created (correct —
first analysis of each image, nothing to version yet).

## 8. What worked / what failed

**Worked**: leakage-gate override + audit logging; GPU training end-to-end;
probability export and evaluation on the deep checkpoint; the full
`analyze-image` pipeline including Arabic report generation, tier-aware rule
dispatch, disabled concern engine, and label-provenance auditing, across all
4 test images with zero errors, exceptions, or warnings beyond a benign
sklearn imputer notice unrelated to this path.

**Failed**: nothing. `grep`-scanned the full run log for
errors/exceptions/tracebacks — none found.

## 9. Fixes made this phase

**None required.** No bug was found in this run. This is itself a
meaningful (if unexciting) result: it confirms the Phase 1/2 fixes hold up
under a fresh end-to-end run against a materially different dataset
configuration (full uncleaned data, different checkpoint, different sample
images) than any previously tested this session.

## 10. Recommended next step

Resume the dataset/label-quality audit that was in progress before this
priority change — Phase 3A's own result (§6) is itself evidence for why that
audit matters: leakage measurably inflates validation metrics on this exact
dataset/architecture pairing. A leakage-safe re-run of this same experiment,
once the audit's cleaned/versioned manifests exist, would be the natural
comparison point.
