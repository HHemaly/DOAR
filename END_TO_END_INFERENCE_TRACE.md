# End-to-End Inference Trace

**Status: evidence document only. No model was trained. No test-split image
was accessed or evaluated.** This traces the *actual* `analyze-image`
pipeline against one real, unmodified, non-test-split image, with the
literal CLI commands and the literal output files it produced, so that
absent outputs (no detected objects, only one triggered rule) can be
attributed to a specific, cited cause instead of assumed.

Repository state: commit `2ae5796` (HEAD at the time of this trace).
Environment: Python 3.11.9, PyTorch 2.7.1+cu118, CUDA available (not
required — `predict_image` auto-selects CPU/GPU and this trace's numbers
do not depend on which one ran).

---

## 1. Image selection (development split only)

- **Source manifest**: `outputs/phase5/manifest.csv` — the original,
  pre-Phase-7A/7B manifest (3,688 rows: train=2,821, valid=310, test=557).
  This is the same manifest every existing model checkpoint was trained
  against (see `CURRENT_CAPABILITY_AUDIT.md` §1).
- **Selected row**: `image_id=d4e6800b359215a2`, `split=train`, `class=Fear`,
  `path=C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing\train\Fear\f52.jpg`.
  Selected programmatically (first resolvable `train`-split row after
  skipping the 4 already-disclosed Phase 6 smoke-test images and the
  17-image Phase 7B duplicate-analysis component, purely so this trace
  doesn't entangle with prior disclosures) — **not** cherry-picked for a
  favorable result.
- **Confirmed NOT test-split**: `split == "train"` in the manifest, and
  this image is not part of `outputs/phase7b/final_partition/`'s `test`
  rows either (that partition is provisional/unlocked anyway, see
  `CURRENT_CAPABILITY_AUDIT.md` §4). `test_guard.py`'s
  `--unlock-test`/`--confirm-final-evaluation` flags were never passed to
  any command in this trace, and no test-split command (`evaluate`,
  `export-probabilities`, `apply-late-fusion`) was run at all.
- **Original data untouched**: the CLI only *reads* this file; nothing in
  `analyze-image` writes to the source dataset.

## 2. Checkpoint selection

`outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt` —
chosen because it is the checkpoint most consistently referenced across
`PHASE5_RESULTS.md`, `PHASE6_RESULTS.md`, and `SESSION_HANDOFF.md` as the
closest thing to a "production candidate," and because Phase 6 already
used this exact file for real `analyze-image` integration smoke tests.
**Per `CURRENT_CAPABILITY_AUDIT.md` §1/§3, this checkpoint was trained on
the original, duplicate-contaminated split with the leakage gate
explicitly overridden — its `best_valid_macro_f1=0.7364` and this
inference's output must be read as preliminary/development-only, not a
leakage-safe generalization estimate.** Loaded and inspected directly
(`torch.load(..., weights_only=False)`): `model_name=efficientnet_b0`,
`seed=42`, `calibration_status=temperature_scaled`, `temperature=1.1436`.

## 3. Command run (exact, reproducible)

```powershell
.\.venv\Scripts\python.exe main.py analyze-image `
  --image "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing\train\Fear\f52.jpg" `
  --output outputs\traced_inference_case001 `
  --emotion-checkpoint outputs\phase5\seed42_reference\efficientnet_b0_seed_42\best.pt
```

Exit code `0`. No stderr output. Full output tree written to
`outputs/traced_inference_case001/`:
`analysis.json`, `evidence.json`, `rules.json`, `concerns.json`,
`judges.json`, `detections.json`, `emotion.json`, `clinician_review.json`,
`reports/{professional,parent}_{en,ar}.html`, `reports/bilingual.html`,
`artifacts/*.png` (9 images: normalized, foreground mask, foreground-only,
feature overlay, edge map, page mask, density map, stroke map, 3 candidate
masks).

## 4. Stage-by-stage trace

### 4.1 Quality gate (`analysis.py::_quality`)
```json
{"width": 225, "height": 225, "min_dimension": 225, "contrast_std": 73.898,
 "blur_variance": 14720.175, "resolution_ok": true, "blur_ok": true,
 "contrast_ok": true, "quality_status": "supported", "unsupported_reasons": []}
```
**Result: `supported`.** All three thresholds (resolution ≥100px, blur
variance ≥15.0, contrast std ≥8.0) passed, so every downstream module ran.
Note field `thresholds_validated_on_real_dataset: false` — these
thresholds are engineering defaults, not empirically calibrated on this
dataset (a real, disclosed limitation, not this trace's problem).

### 4.2 Segmentation (`analysis.py::_segment`)
Selected strategy `colour_distance` (of 3 candidate heuristics),
confidence `0.707`, background estimated as `rgb(238,216,191)`. This is a
**foreground/background split of the whole drawing**, not object
recognition — it produces one mask covering all ink/color on the page,
with no notion of "this is a person" or "this is a tree."

### 4.3 Composition (`analysis.py::_composition`)
```json
{"foreground_coverage": 0.535, "bounding_box_coverage": 0.991,
 "centroid_normalized": [0.552, 0.549], "placement": "middle_center"}
```
Real, measured numbers from the actual mask — not fabricated.

### 4.4 Colour (`analysis.py::_colour`)
`dominant_colour: "dark"`, meaningful colours `[dark, green, red]` — coarse
bucket-based colour naming from the same mask.

### 4.5 Emotion model (`emotion.py::predict` → `deep/inference.py::predict_image`)
```json
{"status": "available", "top_class": "Fear", "confidence": 0.8395,
 "probabilities": {"Angry": 0.010, "Fear": 0.839, "Happy": 0.133, "Sad": 0.017},
 "calibration_status": "temperature_scaled", "temperature": 1.144,
 "model_name": "efficientnet_b0", "checkpoint": "best.pt",
 "checkpoint_sha256": "5c83a7fe...c8"}
```
A **real PyTorch forward pass ran** (not a stub, not a mock) and predicted
`Fear` with 0.839 calibrated confidence. `label_provenance.json`
independently notes the dataset folder's own label is also `Fear`
(`audit_status: "CONSISTENT"`) — a post-hoc audit signal only, computed
*after* and never fed back into the prediction.

### 4.6 Rules (`rules.py::evaluate_rules`, all 19 real registry rules evaluated)
| Status | Count | Rule IDs |
|---|---|---|
| `weak_support` (triggered) | **1** | `PSY_AR_SIZE_FULL_015` (bounding-box coverage 0.991 ≥ 0.90 threshold) |
| `not_matched` (evaluated, condition false) | 5 | `SIZE_HALF`, `SIZE_SMALL`, `PLACE_TOP`, `PLACE_LEFT`, `PLACE_RIGHT` |
| `missing_detector` (never evaluable) | 13 | all 13 tier-2 eyes/animal/shape/symbol rules |

Only 1 of 19 rules fired, at `confidence_ceiling=0.15` (capped, per the
registry's own "unvalidated" framing). **This is the correct, honest
result for this specific image** — the drawing genuinely fills the
bounding box (0.991 coverage), which is exactly what `coverage_full`
checks for; nothing was suppressed or hidden.

### 4.7 Concerns (`concerns.py::derive_concerns`)
`[]` (empty). `CONCERNS_ENABLED = False` is hardcoded in `concerns.py` —
this returns `[]` unconditionally regardless of how many rules fire or
what the model predicts. Confirmed: this is not this image's result, it
is every image's result until a human explicitly flips that flag with a
real concern taxonomy in place.

### 4.8 Object detection (`case_output.py::finalize_case`)
```json
{"status": "unavailable", "detections": []}
```
This exact literal dict is written **unconditionally, for every case,
regardless of image content** — `case_output.py:61`:
`_write(output / "detections.json", {"status": "unavailable", "detections": []})`.
There is no object-detector code path to trace; nothing was attempted and
nothing failed. See `CURRENT_CAPABILITY_AUDIT.md` §7 for why.

### 4.9 Judges (`judges.py::run_judges`) — deterministic, non-LLM
All 6 judges (`quality`, `segmentation`, `feature`, `emotion`, `rule`,
`safety`) returned `"pass"`; `overall_status: "pass"`;
`clinical_output_suppressed: false`. `safety_judge` scanned both English
and Arabic rule/report text for diagnostic-language patterns and found
none; the non-diagnostic disclaimer was confirmed present.

### 4.10 Reports (`reports.py::save_reports`)
5 HTML files written (`professional_en/ar.html`, `parent_en/ar.html`,
`bilingual.html`). **Confirmed by direct grep**: `professional_en.html`
contains a full "Rule evaluations" table; **`parent_en.html` contains zero
occurrences of "Rule evaluations" or any `rule_id`** — the current parent
report omits the rules section entirely
(`reports.py:90`: `rule_section = "" if parent else f"""...`). This is a
concrete, code-confirmed gap against what you specified for the Parent
view (which must show evaluated/triggered/unevaluable rules) — addressed
in `TARGET_APPLICATION_ARCHITECTURE.md` and the prototype.

### 4.11 Deterministic Q&A (`qa.py::answer`, exercised via `main.py qa`)
```powershell
.\.venv\Scripts\python.exe main.py qa --analysis outputs\traced_inference_case001\analysis.json --question "What objects were detected?" --language en
```
→ `"Unavailable: a verified object/person detector was not run in this
release."` (`availability: "missing_detector"`, grounded, no fabrication).
Also exercised "What rules triggered?" (English) and the Arabic equivalent
of the same question — both returned the real rule-status summary above,
correctly localized. This is a real, working, deterministic,
evidence-grounded Q&A layer — see `CURRENT_CAPABILITY_AUDIT.md` §16.

### 4.12 Case persistence (`case_output.py::write_versioned`)
`clinician_review.json` initialized to
`{"status": "not_submitted", "history": [], "ai_output_preserved": true}`.
No prior version existed for this fresh case directory, so no
`versions/` archive was created this run (archiving-on-change is tested
separately in `tests/test_case_output_versioning.py`, not exercised here
since this was a first write).

## 5. Two supplementary minimal traces — isolating why an output can be absent

To give a complete, evidence-backed answer to "why might output show no
objects or rules," two more real runs were made (same real pipeline, no
mocking):

**Trace B — same image, no `--emotion-checkpoint`**:
```json
"emotion": {"status": "unavailable", "reason": "No emotion checkpoint was supplied."}
"module_execution": {"executed": ["quality","segmentation","composition","colour","emotion_model","psychologist_rules","concern_profiles"], "suppressed": []}
```
Composition/colour/rules still ran normally (they don't depend on the
emotion model) — only the emotion probabilities were absent, with an
explicit machine-readable reason. This is a **configuration** cause, not a
missing capability.

**Trace C — a tiny (40×40px), blank, synthetically-generated white PNG
(not real dataset data), with the same checkpoint**:
```json
"quality": {"quality_status": "unsupported", "unsupported_reasons": [
  "resolution below 100px (min_dimension=40)",
  "low sharpness (blur_variance=0.0 < 15.0)",
  "low contrast (contrast_std=0.0 < 8.0)"]}
"emotion": {"status": "suppressed", "reason": "image quality unsupported: ..."}
"module_execution": {"executed": ["quality","segmentation","composition","colour"],
  "suppressed": ["emotion_model","psychologist_rules","concern_profiles"]}
```
`rule_evaluations: []`, `concerns: []` — the quality gate suppressed
emotion, rules, and concerns entirely, with the exact reason recorded.
This is the **quality-gate suppression** cause.

## 6. Direct answers to the diagnostic questions

- **Missing implementation?** Yes, for object detection specifically —
  `detections.json` is a hardcoded stub for every case (§4.8). 13 of 19
  rules can never fire for the same reason (no eye/animal/shape/symbol
  detector exists).
- **Failed model loading?** No — the checkpoint loaded and ran a real
  forward pass in this trace (§4.5).
- **Incorrect configuration?** Only reproducible if `--emotion-checkpoint`
  is omitted (Trace B) — not the case in the main trace.
- **Unsupported feature?** N/A for this image; would apply to the two
  `shape.*` features in `features.py` which are hardcoded `NaN`/missing
  (no shape detector), unrelated to this trace's rule/emotion output.
- **Low confidence?** Not the cause here — the model's confidence (0.839)
  was well above any suppression threshold; low emotion confidence does
  not suppress rules or concerns in the current code (only the *quality*
  gate does, via `analysis.py`'s `quality_status`).
- **Rule prerequisites not satisfied?** Yes, for 13/19 rules structurally
  (missing_detector, §4.6) and for 5/19 on this specific image (evaluated,
  condition false).
- **UI suppression?** Not applicable to this CLI trace — no UI was
  involved. (The Phase 7B review interface the user separately opened is
  addressed in `CURRENT_CAPABILITY_AUDIT.md` §0/§15 — it is a different
  application that never calls any of this code.)
- **Runtime error?** None — exit code 0, empty stderr, all 6 judges
  passed.

## 7. What this trace genuinely demonstrates

A real, unmodified drawing from the training split, run through the real
`analyze-image` pipeline with a real trained-and-loaded PyTorch checkpoint,
produces: a real (if coarse) foreground/background segmentation, real
measured composition/colour statistics, a real calibrated emotion
prediction with full probability distribution, a real 19-rule evaluation
(1 triggered, 5 evaluated-and-not-matched, 13 structurally unevaluable),
an explicitly-disabled concern layer, a hardcoded "no object detector"
stub, six deterministic pass/fail judges, bilingual HTML reports (with a
confirmed parent/professional asymmetry in rule visibility), and a
working deterministic grounded Q&A layer in both languages — end to end,
with no fabricated values anywhere in the trace.

**Processing time was not recorded** — no timing field exists anywhere in
`schemas.py::Analysis` or the case output today. This is a real gap
against your technical-view requirement, addressed in the prototype
(§Phase 4) by adding wall-clock instrumentation around the pipeline call.
