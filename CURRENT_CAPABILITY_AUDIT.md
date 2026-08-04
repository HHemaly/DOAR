# Current Capability Audit — DOAR v3

**Status: evidence-based repository audit only. No model was trained. No
test-split data was accessed. Nothing was fixed or changed by writing this
document** (code changes, if any, happen only in the later, clearly-labeled
prototype work and are listed separately). Every claim below is anchored to
an exact file, function, or command actually read or run — see
`END_TO_END_INFERENCE_TRACE.md` for the executed proof underlying §§6–11.
Documentation (`*_RESULTS.md`, `SESSION_HANDOFF.md`, etc.) was **cross-checked
against code and the filesystem, not trusted at face value** — every place
that check surfaced a discrepancy is flagged explicitly.

---

## 0. Direct answer to the main concern

**Confirmed from code: the interface you opened (`phase7b_review_app.py`)
is a separate, temporary dataset-engineering tool, not the DOAR
application.** Evidence:

- `phase7b_review_app.py` imports only `doar.human_review` (grep-confirmed,
  zero other `doar.*` imports).
- `src/doar/human_review.py`'s own module docstring: *"Deliberately kept
  free of any UI framework import... `phase7b_review_app.py` is a thin
  presentation layer over these functions."* It operates entirely on
  `outputs/phase7b/human_review/app_data/items_registry.json` — anonymized
  image-pair composites for duplicate-detection review — and has no
  concept of emotion classes, rules, features, or reports.
- `src/doar/analysis.py`, `src/doar/case_output.py`, and `streamlit_app.py`
  were grepped for any import of `partition` or `human_review`: **zero
  matches in all three.** The main analysis pipeline cannot reach, and is
  not reachable from, the Phase 7B review tool.
- That is exactly why the interface showed only duplicate image pairs and
  nothing about objects, features, rules, interpretation, or
  recommendations — it was never designed to show any of that. It is a
  scientific-dataset-quality tool (deduplication threshold review), not a
  preview of the parent- or clinician-facing application.

The actual single-image analysis pipeline is `main.py analyze-image` →
`src/doar/analysis.py::analyze_image`, exposed today only through
`streamlit_app.py` — a **raw JSON-dump, English-only, mixed
researcher/clinician-review tool** titled "DOAR v3 - Psychologist Review"
(`streamlit_app.py:10`). It is closer to your "Technical/Research View"
requirement than to a parent-facing view, but even it is unstyled and has
no plain-language layer, no child-context intake, and no chat. **There is
currently no parent-facing view of any kind in this repository.**

---

## 1. Capability matrix

Legend: **Verified** = implemented and exercised by a real, non-mocked
execution path (test or this session's own run) · **Not evaluated** =
implemented, plausible, but never run against real data/checkpoints in
tests · **Partial** = real for part of the claim, stubbed/absent for the
rest · **Placeholder** = a data contract or hardcoded stub, no real logic
· **Missing** = does not exist · **Disabled** = implemented, intentionally
turned off · **Blocked** = correct behavior, but blocked by missing
detector/validation/data.

| # | Component | Status | Key evidence |
|---|---|---|---|
| 1 | Emotion-classification models (code) | **Verified** | `deep/trainers.py`, `deep/inference.py`; real forward pass in this session's trace (§6) |
| 2 | Checkpoints load | **Verified** | 8/8 sampled `.pt`/`.joblib` files loaded successfully this session (§2) |
| 3 | Training/evaluation results | **Partial — preliminary** | Real metrics exist (Phase 3A–5), but every checkpoint trained on a duplicate-contaminated, leakage-overridden split (§2, §5) |
| 4 | Dataset manifests/splits | **Verified, but unlocked** | 3,688-row manifest resolves; no partition is currently approved (§5) |
| 5 | Duplicate control (7A/7B) | **Partial — provisional** | Real analysis + tooling exist; `decisions.json` absent — zero human review done (§5) |
| 6 | Objective feature extraction | **Verified, with 2 explicit gaps** | `features.py` real, 53 features; 2 `shape.*` features hardcoded NaN (§8) |
| 7 | Object detection | **Missing (schema-only)** | `detectors/` is a data contract, never imported by any user-facing path (§7) |
| 8 | Missing-object detection | **Missing** | No detector exists to report anything is "missing" either (§7) |
| 9 | Rule engine (dispatcher) | **Verified** | `rules.py::evaluate_rules`, real dispatch, tested against live registry (§9) |
| 10 | Rule executable coverage | **Blocked, mostly** | 6/19 executable (tier-1), 13/19 permanently `missing_detector` (§9) |
| 11 | Arabic/English reporting | **Verified, with a real gap** | `reports.py`+`localization.py` real; parent report omits rules entirely (§11, code-confirmed) |
| 12 | Case persistence/versioning | **Verified** | `case_output.py::write_versioned`, archives on change, tested (§12) |
| 13 | Psychological interpretation layer | **Disabled by design** | `concerns.py`: `CONCERNS_ENABLED = False` hardcoded (§9) |
| 14 | Existing main application | **Partial — researcher-only** | `streamlit_app.py`: real data, no parent view, English-only UI chrome (§13) |
| 15 | Phase 7B review interface | **Verified, but separate** | Fully functional; confirmed isolated from the main pipeline (§0) |
| 16 | Chat/LLM integration | **Missing (deterministic QA exists)** | No LLM/chat SDK anywhere in the repo; `qa.py` is real, deterministic, grounded (§16) |
| 17 | Tests / validation evidence | **Mixed, mapped in full** | ~1/3 of tests drive the real pipeline on real generated images; rest are synthetic-data unit tests (§17) |

---

## 2. Models genuinely trained, and on what

Verified by loading checkpoints directly (`torch.load(..., weights_only=False)`
/ `joblib.load`) — **8 of 8 sampled artifacts loaded without error**:

| Checkpoint | Loads | Metadata found |
|---|---|---|
| `outputs/phase3a/model/best.pt` | ✅ | mobilenet_v3_small, seed 42, macro-F1 0.7043, uncalibrated |
| `outputs/phase4/deep/runs/resnet18_seed_42/best.pt` | ✅ | macro-F1 0.7146, uncalibrated |
| `outputs/phase5/deep/runs/{resnet18,efficientnet_b0,mobilenet_v3_small}_seed_123/best.pt` | ✅ (all 3) | macro-F1 0.712/0.732/0.671, all `temperature_scaled` |
| `outputs/phase5/seed42_reference/efficientnet_b0_seed_42/best.pt` | ✅ (used in this session's own trace) | macro-F1 0.7364, `temperature_scaled`, T=1.144 |
| `outputs/full_run/feature_models/runs/random_forest_seed_42/model.joblib` | ✅ | sklearn Pipeline, classes match |
| `outputs/full_run/fusion/runs/mlp_early_fusion_seed_42/fusion.joblib` | ✅ | `doar_fusion_bundle_v1`, uncalibrated |
| `outputs/full_run/fusion_calibrated/fusion_calibrated.joblib` | ✅ | same bundle, calibrated, T=1.908 |

**No `.pt`/`.joblib` file exists anywhere under `outputs/phase6/`,
`outputs/phase7a/`, or `outputs/phase7b/`** — matching those phases' own
claims that no training occurred there (Phase 6 is integration-only; 7A/7B
are dataset-partition work; Phase 7 itself is proposal-only,
`PHASE7_RESULTS.md:3`).

**Discrepancy found**: `outputs/full_run/final_test/` contains
`metrics.json`, `probability_export.json`, and
`final_test_unlock_log.jsonl` — the locked test split **was** unlocked and
evaluated once, in an earlier session predating the numbered Phase
sequence. This is disclosed in `CURRENT_STATE_AUDIT.md` but is not
mentioned in any `PHASE3A`–`PHASE7B` document. It was **not** re-opened or
re-evaluated during this audit — its existence is reported here only
because the task explicitly asked whether the test set had ever been
accessed. Separately, Phase 6 disclosed 4 individual test-folder images
used for non-aggregate integration smoke tests (`PHASE6_RESULTS.md` §2).

**Classical and fusion checkpoints exist and load**, but only under
`outputs/full_run/` and `outputs/exploratory_full/` — sessions that predate
and sit outside the numbered Phase 3A–7B sequence entirely. No numbered
Phase document references them.

## 3. Which results remain preliminary because of duplicate leakage

**All of them, without exception.** Every checkpoint in §2 was trained on
`outputs/phase5/manifest.csv`'s original split. Direct evidence:
`outputs/phase5/deep/leakage_gate/leakage_report.json` reports
`"status": "FAIL_LEAKAGE_DETECTED"` — 324 exact cross-split duplicates,
1,442 near-duplicate cross-split pairs, 48 conflicting-label groups — and
`leakage_override_audit.jsonl` records the explicit override used to train
anyway, with justification text quoting "explicitly not leakage-safe...
dataset cleaning deferred per user instruction." **Since neither Phase
7A's nor Phase 7B's leakage-safe partitions have been approved (§5),
there is currently no leakage-safe-trained checkpoint anywhere in this
repository.** Every macro-F1 number in `PHASE3A`–`PHASE5` results must be
read as development-time signal only, not a generalization estimate — this
is also each document's own stated position, independently confirmed here
against the underlying leakage-gate JSON, not just re-quoted from the docs.

## 4. Were any rules trained? How do they work?

**No rule was trained.** All 19 rules in
`resources/psychology_sources/rules_registry.json` are static,
human-authored entries (`source_type: psychologist_supplied_notes`,
`scientifically_validated: false`, self-declared in the registry itself).
`rules.py::evaluate_rules()` is a **pure dispatcher**: for each rule it
reads a `tier` field and either (a) computes a real measurement from
`composition` (tier-1, 6 rules), (b) returns `missing_detector`
unconditionally (tier-2, 13 rules, no detector exists), or (c) would return
`not_assessable_context_unknown` for a tier-3 prompt-dependent rule (none
currently exist in the registry). No machine learning, no statistical
fitting, no threshold tuning against outcome data — thresholds like
"bounding-box coverage ≥ 0.90" are hard-coded engineering choices in
`rules.py::_status_for`, not learned. Full 19-rule table with tiers,
activation status, and confidence ceilings is in
`RULE_AND_FEATURE_COVERAGE.md`.

## 5. Dataset splits — exact current status

**No duplicate-detection threshold or clustering policy is locked or
approved anywhere in this repository right now.** Two provisional
candidates exist on disk, both explicitly marked not-final:

- `outputs/phase7a/` — aHash threshold 5, single-linkage. Superseded for
  development use per `PHASE7B_DUPLICATE_POLICY.md:42`.
- `outputs/phase7b/final_partition/` — dHash threshold 6, single-linkage
  (`partition_config.json`, verified directly:
  `near_dup_threshold=6, hash_field="dhash"`). `PHASE7B_DUPLICATE_POLICY.md`'s
  most recent correction text (top-of-file notice, §17, closing lines)
  states unambiguously this must **not** be treated as final. Filesystem
  confirms this is real, not stale documentation: **`decisions.json` does
  not exist anywhere under `outputs/phase7b/human_review/`** — zero human
  review decisions have been recorded as of this audit.

Two independent, code-enforced mechanisms guard the *original* manifest's
`test` split against accidental access: `src/doar/test_guard.py::require_test_access`
(requires `--unlock-test`, `--confirm-final-evaluation`, and a non-empty
`--initiated-by`, logging every unlock to `final_test_unlock_log.jsonl`),
wired into `evaluate`, `export-probabilities`, and `apply-late-fusion` in
`main.py`, plus a second independent check inside
`models.py::evaluate_model` itself. Neither guard was invoked in this
audit or in `END_TO_END_INFERENCE_TRACE.md`.

## 6–11. Core single-image pipeline — see `END_TO_END_INFERENCE_TRACE.md` for the executed proof

Rather than duplicate that document, this section states the standing
capability each stage represents; the trace document is the evidence.

- **§6 Segmentation/composition/colour** (`analysis.py`): real, dependency-light
  computer vision (background-distance thresholding, connected-component
  cleanup, coarse colour binning) — not machine-learned, not object-aware.
  It answers "how much of the page has ink and where" — never "what is
  drawn."
- **§7 Object detection**: `src/doar/detectors/` is **explicitly scaffolding
  only** — its own package docstring: *"No detector is implemented,
  adopted, or enabled here... Nothing in this package is imported by
  analysis.py, rules.py, or any other user-facing code path."*
  `case_output.py:61` writes `{"status": "unavailable", "detections": []}`
  **unconditionally, for every case, regardless of image content.** There
  is no code path that could produce a bounding box, mask, or object label
  today, for any image. **Missing-object detection (item 8) does not exist
  either** — a detector would be a prerequisite for even defining "missing."
- **§8 Objective feature extraction** (`features.py::objective_feature_row`):
  53 real numeric features (quality/segmentation/composition/colour/stroke/shape
  families), each carrying its own `FeatureValue.confidence` and
  `missing` flag. Two features (`shape.enclosed_shape_count`,
  `shape.repetition_score`) are **honestly hardcoded to `NaN`/`missing=True`**
  rather than fabricated — a documented, intentional gap (no shape
  detector exists). This module is wired into fusion-checkpoint inference
  (`emotion.py`'s fusion branch) but is **not** called during a plain
  deep-model or classical-checkpoint `analyze-image` run — the per-case
  `evidence.json`/`analysis.json` in a typical run does not include these
  53 features unless a fusion bundle checkpoint is used.
- **§9 Rule engine + concern layer**: `rules.py` dispatch is real and
  tested against the live registry (`tests/test_rule_coverage_baseline.py`,
  `tests/test_phase2_rule_tiers_and_aggregation.py`). `concerns.py::derive_concerns`
  has real convergence logic (requires ≥2 evidence IDs from ≥2 independent
  source types) but **`CONCERNS_ENABLED = False` is hardcoded** — it
  returns `[]` unconditionally in production regardless of input,
  documented in-file as intentional pending a real concern-specific
  taxonomy. Verified live in this session's trace (§6 of the trace doc):
  concerns were `[]` even with a rule triggered and a confident model
  prediction.
- **§10 Judges** (`judges.py::run_judges`): 6 deterministic, non-LLM,
  regex/logic-based judges (quality, segmentation, feature, emotion, rule,
  safety) plus an `overall_status` roll-up. The safety judge scans **both**
  English and Arabic rule/report text for diagnostic-language patterns
  (`_ARABIC_DIAGNOSTIC` regex + `DIAGNOSTIC_PATTERNS`) — this is a real,
  working, deterministic safety gate already in production, not a design
  proposal.
- **§11 Bilingual reporting** (`reports.py`, `localization.py`): real HTML
  generation with genuine Arabic field/value translation (not just RTL
  styling). **Code-confirmed gap**: `reports.py:90`
  (`rule_section = "" if parent else ...`) means the current "parent"
  report variant contains **zero** rule content — directly contradicting
  what you specified for the Parent view. This is fixed in the prototype,
  not in this audit document.

## 12. Case persistence and versioning

`case_output.py::write_versioned` archives the previous version of any
changed JSON artifact under `<case>/versions/` with a `history.jsonl`
audit trail before overwriting, specifically to fix a real bug found in an
earlier audit (silent overwrite on re-analysis). Tested in
`tests/test_case_output_versioning.py` with real file I/O. `clinician_review.json`
(per-case) and `review.py`'s separate append-only `review_master.csv`
(cross-case, Cohen's/Fleiss' kappa agreement) are two distinct,
non-conflicting review-tracking mechanisms — both real, neither related to
the Phase 7B duplicate-pair review (§15).

## 13. Existing main application (`streamlit_app.py`)

Real: reads actual `analysis.json`/`judges.json`/`detections.json` per
case, offers 18 tabs (Summary, Original Image, Quality, Segmentation,
Composition, Colours, Lines/Strokes, Shapes, Objects, OCR, Emotion, Rules,
Concerns, Judges, Psychologist Review, Reports, Q&A, Experiments), and can
trigger a fresh `analyze_image()` run from an uploaded file. **Explicitly
not a parent-facing view**: nearly every tab is a raw `st.json(...)` dump,
English-only chrome (`st.title`, `st.sidebar.header`, etc.), no child
context intake, no plain-language explanation layer, no follow-up chat
(only the deterministic Q&A tab, which itself is fine but is presented as
a raw JSON envelope, not conversational text). Several tabs explicitly
self-disclose limitations already: `"Shape features are preliminary..."`,
`"OCR module unavailable in this release."`

## 14. Chat / LLM integration

**None exists.** Full-repository grep (`src/`, `main.py`, both Streamlit
apps, `tests/`, `pyproject.toml`) for `openai|gemini|anthropic|chatgpt|gpt-|
chat_provider` returns zero matches in any active code path. The only
`pyproject.toml` optional-dependency groups are `ml, cv, dev, ingest, ui,
deep, embeddings` — no LLM SDK group. One adjacent artifact:
`legacy/src/parent_ai_helper.py` (outside the active `src/doar` package)
has a docstring claiming an *optional* LLM hook, but its own
`generation_method` is hardcoded to `"template"` and no API/HTTP code
exists in the file — aspirational documentation, not a working
integration. **What does exist and works today**: `src/doar/qa.py` — a
real, deterministic, evidence-grounded, bilingual question-answering
module (no LLM, no network call) that answers strictly from a saved
case's `analysis.json`/`judges.json`, always cites evidence IDs, and
returns a safe "unavailable" envelope for unknown questions rather than
guessing. This is directly reusable as the "deterministic evidence-based
fallback" required by `LLM_GROUNDING_AND_SAFETY_DESIGN.md`.

## 15. Phase 7B review interface

Fully functional (225 blind pairs across 4 categories, incremental save,
5-file export — built and tested in this project's immediately preceding
session) and **structurally isolated** from the main pipeline (§0). It
must remain a separate tool per your instruction; the prototype in this
task does not touch it.

## 16. Rule source register and psychology-ingest governance

(Full detail from a dedicated sub-audit.) The registry cannot be modified
by any automated process: `ingest-psychology-pdf` → `psychology_ingest.py::build_draft_registry`
produces an inert draft (`confidence_ceiling: 0.0`,
`visibility: "blocked_pending_review"`, `review_status: "pending"` for
every extracted candidate) written to a user-specified output path that is
**never read back or merged** into the live registry by any code. No LLM,
web search, or auto-merge script touches `rules_registry.json` anywhere in
the repository. The one real gap: **the CSV registers
(`RULE_COVERAGE_MATRIX.csv`, `RULE_SOURCE_REGISTER.csv`) agree with the
JSON registry today, but no automated test enforces that agreement** — a
future manual edit to one without the other could silently drift
undetected (`test_csv_register_integrity.py` checks CSV well-formedness
only, never cross-references `rules_registry.json`).

## 17. Tests and validation evidence — full map

39 test files were mapped individually (excluding the 6 already covered by
other audit sections: `test_partition*.py`, `test_human_review.py`,
`test_leakage_gate.py`, `test_label_provenance.py`,
`test_trainer_regression.py`). **Roughly a third drive the real
`analyze_image` pipeline against real, small, programmatically-generated
image fixtures** (not mocks): `test_objective.py`, `test_quality_gate.py`,
`test_quality_suppression_judges.py`, `test_integrity_fixes.py`,
`test_safety_and_leakage.py`, `test_resolve_leakage.py`,
`test_smoke_experiment.py`, `test_case_output_versioning.py`,
`test_gradcam_and_preprocessing.py`, `test_preprocessing_consistency.py`.
Config/CLI-level tests exercise real shipped registry/config files and, in
several cases, real `main.py` subprocess invocation
(`test_gate_resolved_output.py`, `test_fusion_config_and_provenance.py`,
`test_export_and_guard.py`, `test_rule_coverage_baseline.py`,
`test_phase2_rule_tiers_and_aggregation.py`). The remainder
(`test_ablation.py`, `test_calibration_wiring.py`, `test_embedding_comparison.py`,
`test_evaluation.py`, `test_explainability.py`, `test_fusion_calibration.py`,
`test_late_fusion.py`, `test_localization.py`, `test_provenance.py`,
`test_qa_expanded.py`, `test_review_form.py`, `test_concerns_and_ingest.py`,
`test_phase3_detector_scaffolding.py`) are deliberately synthetic-data unit
tests of pure numeric/logic functions — several docstrings say so
explicitly, to keep them fast and torch-independent.
**No test in the suite loads or runs inference against any of the actual
on-disk Phase checkpoints** — the checkpoint-loading verification in this
audit (§2) was done independently, outside the test suite, in this
session.

---

## 18. Direct answers to your explicit questions

- **Which models have genuinely been trained?** Phase 3A (mobilenet_v3_small,
  1 seed), Phase 4 (4 architectures, 1 seed), Phase 5 (3 architectures × 3
  seeds) — all real, all checkpoints load. Also classical/fusion models
  under `outputs/full_run/`, outside the numbered phases.
- **On which dataset and split?** `outputs/phase5/manifest.csv`'s original
  train/valid/test split (train=2,821) — never Phase 7A's or 7B's
  partitions (neither has a trained checkpoint at all).
- **Which results remain preliminary because of duplicate leakage?** All of
  them (§3) — no exceptions exist in this repository today.
- **Were any rules trained? How do they work?** No. Static, human-authored
  hypotheses; a pure dispatcher evaluates 6 against real measurements and
  returns `missing_detector` for the other 13 (§4, §9).
- **How many rules are actually executable?** 6 of 19 (31.6%) can ever
  produce anything other than `missing_detector`.
- **Which features does each rule require, and which detectors are
  missing?** Full table in `RULE_AND_FEATURE_COVERAGE.md`.
- **Is object detection genuinely implemented?** No — schema/contract only,
  never wired into any executable path (§7).
- **Why might the existing inference output show no objects or rules?**
  Objects: because no detector has ever been implemented, for any image,
  under any configuration — not a bug, a genuinely missing capability.
  Rules: 13/19 always show `missing_detector` for the same reason; the
  other 6 only "show nothing" when their specific composition condition
  isn't met on a given image (a real, correct evaluation, not a failure).
  See `END_TO_END_INFERENCE_TRACE.md` §6 for a concrete image where exactly
  1 of 19 fired.
- **Is this an interface problem, missing capability, configuration
  problem, or combination?** For the specific interface you opened
  (Phase 7B review app): **interface problem** in the sense that it's the
  wrong app entirely (§0), not a bug in it. For "no objects detected" in
  the real pipeline: **missing capability**, not a bug. For "no emotion
  prediction" in general: can be a **configuration** issue (no checkpoint
  passed) — demonstrated directly in the trace document.
- **What can currently be demonstrated honestly from one uploaded image?**
  A real foreground/background segmentation and composition/colour
  measurement; a real calibrated emotion-model prediction (if a checkpoint
  is supplied) with full probability distribution and honest
  "unavailable"/"suppressed" states when it isn't; a real 19-rule
  evaluation with 3 honestly-distinguished outcomes
  (`weak_support`/`not_matched`/`missing_detector`); a deterministic,
  evidence-grounded bilingual Q&A layer; bilingual HTML reports (with the
  parent/professional rule-visibility asymmetry noted above); and an
  explicit, correctly-empty "no objects detected, no detector exists" and
  "no concerns, this feature is disabled" state. It cannot honestly
  demonstrate any object/symbol detection, any psychological
  interpretation beyond the 6 coverage/placement rules, or a diagnosis of
  any kind (and does not attempt to).
