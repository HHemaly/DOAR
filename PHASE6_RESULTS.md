# Phase 6 — Selected-Model End-to-End Integration Validation

**Status:** Preliminary. No test-split evaluation occurred (`analyze-image`
is a single-image inference command, distinct from the guarded
`--split test` aggregate-evaluation path used by `evaluate`/
`evaluate-predictions`, which was never invoked). No dataset cleaning, no
detector work, no psychological-rule activation, and no production/default
model change occurred in this phase.

**Goal:** Verify that the preliminary `efficientnet_b0` candidate from
Phase 5 works correctly through the complete existing DOAR pipeline, using
the same 4 smoke-test drawings Phase 3A already used, compared side-by-side
against the existing Phase 3A `mobilenet_v3_small` checkpoint as a reference.
**This is an integration test, not a performance or psychological
validation** — 4 drawings must not be read as representative of either
model's real-world accuracy.

---

## 1. Checkpoint selection

**Selected: `outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt`**
(SHA-256 `5c83a7fe6251fe5580f77ff61523442bbb7aaeaac0ac1514ce32a7754a7187c8`).

Selection was made strictly by validation macro-F1 among the three Phase 5
`efficientnet_b0` checkpoints:

| Seed | Validation macro-F1 |
|---|---|
| **42** | **0.7364** ← highest, selected |
| 123 | 0.7322 |
| 2026 | 0.7186 |

This checkpoint is the Phase 5 seed-42 **reference copy**, which was
temperature-scale calibrated in place during Phase 5
(`calibration_status: temperature_scaled`, `temperature: 1.1436`) — it is
therefore the calibrated, best-macro-F1 `efficientnet_b0` artifact Phase 5
actually produced. Using it is a deliberate choice: it reflects what a real
downstream deployment of this checkpoint would look like (calibrated
probabilities), and per §5's inconclusive-superiority finding, seed 42 was
not shown to be a worse choice than seeds 123/2026 among the three
`efficientnet_b0` seeds — it is simply the seed with the highest recorded
validation macro-F1 among them, per the primary selection criterion this
phase was instructed to use.

**Reference checkpoint:** `outputs/phase3a/model/best.pt` (`mobilenet_v3_small`,
seed 42, Phase 3A's own independent training run — SHA-256
`96c177faff89db7d9b79a6237dd66e08da77ce4891b92d9d54d3751bddb7bd40`). This is
**uncalibrated** (`calibration_status: uncalibrated`, `temperature: 1.0`) —
Phase 3A never ran calibration. This is a real, expected difference between
the two checkpoints under test, not a bug (see §4).

---

## 2. Preservation of prior-phase artifacts

Before and after all Phase 6 work, every artifact from Phase 3A, Phase 4,
and Phase 5 was re-hashed and compared against its own phase's recorded
manifest:

| Manifest | Entries | Pre-Phase-6 | Post-Phase-6 |
|---|---|---|---|
| `outputs/phase3a/artifact_manifest.txt` | 9 | ALL MATCH | ALL MATCH |
| `outputs/phase4/ARTIFACT_MANIFEST.tsv` | 25 | ALL MATCH | ALL MATCH |
| `outputs/phase5/ARTIFACT_MANIFEST.tsv` | 120 | ALL MATCH | ALL MATCH |

No prior-phase artifact was modified. `analyze-image` only reads checkpoints
(no in-place mutation, unlike `calibrate`), so no copy-before-use step was
needed here — verified directly rather than assumed.

---

## 3. Pipeline runs

The exact same 4 test-split images Phase 3A used (one per class, filenames
matched from `outputs/phase3a/case_*/analysis.json`) were each run through
`main.py analyze-image` twice — once with the selected `efficientnet_b0`
checkpoint, once with the Phase 3A `mobilenet_v3_small` checkpoint — for 8
total runs, plus 2 additional runs exercising the versioning path (§5).

| Class | Image file |
|---|---|
| Angry | `2-1_jpg.rf.f967ea55654e6fa4badbe2906373b7f7.jpg` |
| Fear | `Fear_1_10_jpg.rf.eb405d0904de942eabf3841d65ea4d01.jpg` |
| Happy | `Happy_1_19_jpg.rf.87ed22895e6a85ffbf4d1c43c2420ac1.jpg` |
| Sad | `3-4_jpg.rf.f9a20de96b74b4e2ace5fb04c364791d.jpg` |

**All 8 runs exited 0.** The full combined log
(`outputs/phase6/analyze_all.log`) was scanned for `error`/`exception`/
`traceback`/`warning` — **none found**. Every stage ran: quality gate,
segmentation, composition, colour, deep emotion inference, tier-aware rule
evaluation, disabled concern-convergence engine, label-provenance context,
bilingual (EN/AR) report generation, and case persistence.

---

## 4. Per-drawing comparison: `efficientnet_b0` vs. `mobilenet_v3_small`

| Class (true label) | EfficientNet pred / conf | MobileNet pred / conf | Agree? |
|---|---|---|---|
| Angry | Angry / 0.8759 | Angry / 0.8140 | Yes |
| Fear | Angry / 0.7599 | Angry / 0.7438 | Yes |
| Happy | Happy / 0.5561 | Happy / 0.5533 | Yes |
| Sad | Sad / 0.6335 | Sad / 0.8033 | Yes |

**Both checkpoints agree on the predicted class for all 4 drawings** (3/4
correct against the true label; both miss the same case, Fear→Angry — the
same confusion Phase 3A's single-checkpoint run and Phase 4/5's aggregate
confusion matrices already show as the weakest class for every architecture
tested so far). Confidence values differ per drawing and are not
consistently higher or lower for either checkpoint (EfficientNet is more
confident on Angry/Fear, less confident on Sad; both are close on Happy).
`efficientnet_b0`'s probabilities reflect its calibration
(`temperature_scaled`, T=1.1436); `mobilenet_v3_small`'s are raw
(`uncalibrated`, T=1.0) — a real, expected difference in reported
probability shape between the two checkpoints, not a defect in either.

### Objective features (should be, and are, checkpoint-independent)

For all 4 drawings, the `quality`, `segmentation`, `composition`, and
`colour` sections of `analysis.json` were compared field-by-field between
the EfficientNet and MobileNet runs: **identical in all 4 cases, all 4
sections.** This is the expected, correct result — these features are
computed from the image alone and must not depend on which emotion
checkpoint is loaded. No coupling bug was found.

### Evidence and rule propagation

`evidence.json`'s `ev_emotion_prediction` entry correctly differs between
checkpoints in all 4 cases, matching each checkpoint's own
`analysis.json.emotion.probabilities` exactly (cross-checked numerically,
not just visually). The 4 other evidence items (segmentation coverage,
bounding-box coverage, centroid, dominant colour) are identical between
checkpoints in all 4 cases, as expected.

`rules.json` was **identical between checkpoints for all 4 drawings.** This
is correct, not a bug: the 6 rules that can currently evaluate
(`PSY_AR_SIZE_*`, `PSY_AR_PLACE_*`) are all composition/placement-based —
none of the 19 registered rules currently consume emotion-model evidence, so
no rule was expected to change based on which emotion checkpoint produced
`ev_emotion_prediction`. The remaining 13 rules correctly report
`missing_detector`, unchanged from Phase 3A. `concerns.json` was `[]` for
all 8 runs, as required (`CONCERNS_ENABLED = False` in production).
`judges.json` was identical between checkpoints for all 4 drawings.

### Reports (English and Arabic)

All 40 generated report files (4 drawings × 2 checkpoints × 5 report types:
`bilingual.html`, `parent_en.html`, `parent_ar.html`, `professional_en.html`,
`professional_ar.html`) were scanned for unresolved template placeholders
(`{{...}}`, `None`, `undefined`, `NaN`, `[object Object]`) and mojibake —
**zero issues found across all 40 files.** Spot-checked one Arabic report
(`case_Angry_efficientnet/reports/professional_ar.html`): 2,777 Arabic-script
characters present, `dir="rtl"` correctly set (same structural pattern
Phase 3A already validated for report generation in general; Arabic
linguistic fluency is still not validated by a qualified speaker — same open
item as Phase 3A/4/5).

Each checkpoint's own numeric confidence value was confirmed to appear
correctly in its own `professional_en.html` (e.g. `case_Angry_efficientnet`
shows `0.8758658766746521`; `case_Angry_mobilenet` shows
`0.8139899969100952` — both exact matches to each run's own
`analysis.json`). No cross-contamination between the two checkpoints'
report outputs was found.

---

## 5. Version-history / repeated-case behavior (with the new checkpoint)

Repeated the Phase 3A §7a round-trip using `case_Angry_efficientnet`:

1. **Same image, same checkpoint, re-analyzed** (2nd run): `analysis.json`
   SHA-256 identical before and after
   (`1b133435d21fdbe5ebcd2e52fd22ebb455cff19a4f82a14aad12ef6855797798`), and
   correctly **no** `versions/` directory was created — deterministic given
   identical inputs.
2. **Same image, re-analyzed without `--emotion-checkpoint`** (forcing a
   genuine content change, `emotion.status` available→unavailable): this
   correctly triggered archiving. `versions/history.jsonl` recorded exactly
   the 4 files whose content actually changed (`analysis.json`,
   `evidence.json`, `judges.json`, `emotion.json`) — not `rules.json` or
   `concerns.json`, whose content did not change. The archived
   `analysis.v1....json` was independently opened and confirmed to still
   contain the **original** EfficientNet prediction (`Angry`, 0.8758658766746521,
   `available`) — nothing was lost.
3. **Checkpoint restored, re-analyzed a 3rd time**: reproduced the original
   result exactly (`Angry`, 0.8758658766746521, `available`), completing the
   round-trip. `versions/` now contains 9 files (2 archived batches of 4 +
   `history.jsonl`), consistent with the two genuine content changes that
   occurred (checkpoint removed, then restored).

**Result: version-history preservation confirmed correct** with the new
checkpoint, replicating Phase 3A's result under a materially different
architecture/calibration configuration.

---

## 6. Fixes made this phase

**None required.** No bug, error, exception, warning, or structural defect
was found anywhere in this integration run — the full log was scanned and
came back clean, objective features were confirmed checkpoint-independent,
evidence/report propagation was confirmed correct, and versioning behavior
was confirmed correct. Consistent with Phase 3A's own "no bug found" result
for the same integration surface. No regression tests were added, since no
code changed.

---

## 7. Verification (tests, Ruff, compileall)

- **Test suite:** `pytest tests/` — **226 passed**, 9 pre-existing warnings, 0 failures (same count as Phase 5 — no tests added, since no code changed).
- **compileall:** `python -m compileall src main.py` — **exit 0**.
- **Ruff:** `ruff check .` — **792 findings, exit code 1, a failing run.** Identical count to Phase 5's run. `git status` confirms zero tracked files were modified in Phase 6 (this document and `SESSION_HANDOFF.md` are the only additions), so all 792 findings are pre-existing and none are attributable to this phase.

---

## 8. Artifacts

- `outputs/phase6/case_{class}_efficientnet/` — 4 cases, EfficientNet-B0 (Phase 5 seed-42 calibrated checkpoint).
- `outputs/phase6/case_{class}_mobilenet/` — 4 cases, MobileNet (Phase 3A checkpoint, reference).
- `outputs/phase6/case_Angry_efficientnet/versions/` — version-history round-trip artifacts (§5).
- `outputs/phase6/analyze_all.log` — combined stdout/stderr for all 10 `analyze-image` invocations.
- `outputs/phase6/ARTIFACT_MANIFEST.tsv` — 202 files, paths + sizes + SHA-256, ≈8.3 MB total.

All of `outputs/` is gitignored; none of the above were committed to git.

---

## 9. Recommendation

Integration validation passed with no defects found. Both `efficientnet_b0`
(Phase 5's preliminary pick) and `mobilenet_v3_small` (Phase 3A's reference)
run correctly through every pipeline stage, agree on all 4 smoke-test
predictions, and correctly propagate their respective outputs into evidence
and bilingual reports without cross-contamination or silent data loss.

This supports, but does not by itself justify, moving `efficientnet_b0`
forward: 4 drawings confirm the pipeline **works** with this checkpoint, not
that the checkpoint is the right one to standardize on (that remains the
inconclusive `efficientnet_b0` vs. `resnet18` question from
`PHASE5_RESULTS.md` §5, unresolved by this phase and not attempted here).
**Recommended next decision is the user's**, per the same options already
raised in `SESSION_HANDOFF.md` §9: run a formal statistical comparison on
the existing 3-seed data, pick on secondary criteria alone, gather more
seeds, or defer until after the dataset audit produces a leakage-safe
evaluation.

---

## 10. Explicitly out of scope for this phase (per user instruction)

The following were **not** done and should not be inferred from this
document: changing the default or production model; test-split evaluation
(the guarded `--split test` aggregate path was never invoked); dataset
cleaning; detector implementation; psychological-rule activation; starting
a new numbered phase. The 4 drawings used here are the same 4 Phase 3A used
as a pipeline smoke test — they are not, and must not be read as, a
performance or psychological validation of either model.
