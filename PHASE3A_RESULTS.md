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

## 6. This run's score vs. the earlier leakage-cleaned run — corrected framing (2026-08-02)

> **Correction**: an earlier draft of this section asserted that validation
> leakage specifically *caused* the gap between this run's macro-F1 (0.704)
> and the earlier leakage-cleaned run's (0.656). That overstated what a
> single uncontrolled comparison can support. Corrected below.

The two runs are **not a controlled comparison**: this run used
`train-image-model` directly for 10 epochs on the full uncleaned dataset;
the earlier run used `compare-deep-models` (a different invocation path,
though the same underlying trainer) for 12 epochs as part of a 2-model
sweep on the leakage-cleaned dataset. Epoch count, exact code path, and
dataset composition all differ simultaneously — leakage is not isolated as
the sole variable.

The correct, defensible statement: this run's higher score is **consistent
with possible validation-leakage inflation** (duplicates/near-duplicates
retained across splits could let the model partly "recognize" validation
images already seen in training) **and with other experimental differences**
(fewer epochs here, a different training invocation, a different — larger —
train set). Which factor(s) actually account for the ~0.05 macro-F1 gap
cannot be determined from this one pair of runs. A real answer requires a
controlled comparison: same architecture, same seed, same epoch count, same
code path, cleaned vs. uncleaned dataset as the only difference — not yet
run. The result stays labeled preliminary regardless of cause.

## 7. Real images through the complete pipeline — a pipeline smoke test, not a performance validation

**These 4 images test whether the pipeline's stages connect and run
correctly end-to-end — they are not a representative sample and must not be
read as evidence of real-world accuracy.** Four images (one per class) is
far too small to estimate per-class performance; that is what the §5
validation-split metrics (n=310) are for, and even those are preliminary per
§6.

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

## 7a. Version-history check (completed 2026-08-02, was missing from the original pass)

Re-analyzed `case_Angry` twice more, independently verified with matched
SHA-256 hashing (an earlier informal check mixed MD5 and SHA-256, which
cannot be compared — redone correctly here):

1. **Same image, same checkpoint, re-analyzed twice** (2nd and 3rd runs):
   output was byte-identical both times (SHA-256 match confirmed
   independently), and correctly **no** `versions/` directory was created —
   `write_versioned()` only archives when content actually changes, and a
   fully deterministic pipeline given identical inputs produces identical
   output. This is correct behavior, not a failure to version.
2. **Same image, re-analyzed a 4th time WITHOUT the checkpoint** (forcing a
   genuine content change: `emotion.status` available→unavailable): this
   correctly triggered archiving. `versions/history.jsonl` recorded exactly
   the 4 files whose content actually changed (`analysis.json`,
   `evidence.json`, `judges.json`, `emotion.json`) — not `rules.json` or
   `concerns.json`, whose content did not change, correctly not versioned. The
   archived `analysis.v1....json` was independently opened and confirmed to
   still contain the **original** prediction (`Angry`, available) — the
   prior case output was not lost. The case was then re-analyzed a 5th time
   with the checkpoint restored, confirmed to reproduce the original
   `Angry`/0.814 result exactly, completing a full round-trip.

**Result: version history preservation confirmed correct**, both for the
"nothing changed" case (Phase 1's original test coverage) and, now
additionally, for a real multi-write round-trip on an actual Phase 3A case
(not previously exercised).

## 7b. Report inspection (completed 2026-08-02, was missing from the original pass)

Structural/technical inspection of all 5 report files for `case_Angry`
(`bilingual.html`, `parent_en.html`, `parent_ar.html`, `professional_en.html`,
`professional_ar.html`). **Scope disclaimer, stated plainly per instruction:
this checks structure, encoding, and technical consistency only. Arabic
linguistic quality (fluency, grammar, natural phrasing, nuance-for-nuance
equivalence with the English text) has NOT been validated and requires
review by a qualified Arabic speaker — that has not happened.**

Checked and confirmed for all 5 files:
- Valid UTF-8 decode; **zero mojibake/replacement characters** in any file.
- `<meta charset="utf-8">` correctly declared in all 5.
- Arabic files contain substantial real Arabic-script content (680–2,777
  characters) with `dir="rtl"` correctly set; English files contain zero
  Arabic characters and no `dir="rtl"`, as expected.
- **Zero unresolved template placeholders** in any file (checked for
  `{{...}}`, bare `{var}`, literal `None`/`undefined`/`NaN`/`TODO`/`FIXME`/
  `[object Object]`) — nothing found.
- All 5 files non-empty (5,185–25,237 bytes).
- **Required sections present**: `professional_*` reports contain all 7
  expected sections (Image quality, Segmentation, Composition, Colours,
  Emotion model, Rule evaluations, Deterministic judges) in both languages;
  `parent_*` reports correctly *omit* the Rule-evaluations section in both
  languages (by design — parent reports are simplified, not a bug) while
  keeping the other 6.
- **No contradiction found**: the predicted class ("Angry") appears
  consistently across all 4 language/audience variants; the non-diagnostic
  disclaimer is present in both languages (`"not a diagnosis"` in English,
  `تشخيص`-rooted phrasing in Arabic); quality metrics (e.g. image
  width/height) appear as matching numeric values in both language versions
  of the same report.
- One structural asymmetry noted (not a bug): professional reports include
  numeric confidence/percentage figures; parent reports do not — consistent
  with parent reports being the simplified, less statistical variant by
  design.

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

## 9a. Artifact inventory — git status, exact paths, sizes, hashes (completed 2026-08-02)

**All of `outputs/phase3a/` is git-ignored** — confirmed via
`git check-ignore -v`, matched by the pre-existing `.gitignore:5:outputs/`
rule (consistent with the project's existing convention of never tracking
run outputs, checkpoints, or datasets). Nothing in this phase changed that
convention.

| Artifact | Size (bytes) | SHA-256 |
|---|---|---|
| `outputs/phase3a/manifest.csv` | 1,165,611 | `a95c5c72413f9a30e7ee3b91b4571fbd5c25a6fe70f8f33445734a6643fc8049` |
| `outputs/phase3a/model/best.pt` | 18,491,607 | `96c177faff89db7d9b79a6237dd66e08da77ce4891b92d9d54d3751bddb7bd40` |
| `outputs/phase3a/model/last.pt` | 18,491,735 | `356bcae9c2aafb081502c07784bede9096aa91d0da9669fd74b2c67b3f70d90b` |
| `outputs/phase3a/model/training_result.json` | 2,593 | `d60e3551cae6b40e0f1b041801d4f2f67d3f81bfdb58088d98527b9ee5883ff7` |
| `outputs/phase3a/model/resolved_config.json` | 805 | `f7f0950fdd7bddb9aa8ccff8e7b8eeb42a55ff053ef8e0dc52606214a6e3c6c4` |
| `outputs/phase3a/model/leakage_gate/leakage_override_audit.jsonl` | 513 | `25998220004c76df1359316431d5075c9ae837e1c32a69aa8d184eb2230e1e5e` |
| `outputs/phase3a/loss_curve.png` | 52,053 | `d83a237fb0e54a6ed56442eb38005b69e358732ab9291975a5356fb734970250` |
| `outputs/phase3a/probability_export.json` | 182,582 | `98ea1d5d93912613260c49a13a86f11b3f71f3ec1e7e7c9ddc47106c767a2971` |
| `outputs/phase3a/eval/metrics.json` | 1,863 | `4f002d846deb9d414602c3e6677ecea3b9172739e6528cc3ffdc04b10affd129` |

**Internal consistency cross-check**: `probability_export.json`'s own
recorded `checkpoint_hash` field (`96c177fa...bd40`) exactly matches the
independently-computed SHA-256 of `best.pt` above — confirms the evaluation
in §5 provably used the exact checkpoint this phase trained, not a stale or
substituted file.

**Recoverability, without retraining (none was done for this check)**:
every artifact above currently exists on disk and is individually
re-derivable from `best.pt` + `manifest.csv` (the export/evaluate commands in
§2 are deterministic given those two files). Full retraining reproducibility
(bit-for-bit identical `best.pt` from a fresh run) is **not guaranteed**
even with the same seed (42) and command — standard CUDA/cuDNN
non-determinism applies unless deterministic-algorithm flags are explicitly
set, which this run did not do. What *is* guaranteed reproducible: the same
manifest from the same dataset directory (file-content-addressed, not
order-dependent), and the same evaluation metrics from the same checkpoint
file (deterministic inference). If bit-for-bit training reproducibility is
ever required, that needs a separate, explicit deterministic-mode run — not
done here, and not necessary for Phase 3A's functional-verification purpose.

## 10. Recommended next step

Resume the dataset/label-quality audit that was in progress before this
priority change — Phase 3A's own result (§6) is itself evidence for why that
audit matters: leakage measurably inflates validation metrics on this exact
dataset/architecture pairing. A leakage-safe re-run of this same experiment,
once the audit's cleaned/versioned manifests exist, would be the natural
comparison point.
