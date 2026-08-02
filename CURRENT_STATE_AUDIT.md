# DOAR — Current State Audit

Date: 2026-08-02
Auditor: Claude (session-based), working directly in `DOAR-main`
Method: direct source reading, `grep`/structural search across `src/doar/`, execution of the CLI, the test suite, and a full pipeline run against the real local dataset (`C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing`). Nothing below is asserted from documentation alone unless explicitly marked "per docs, unverified."

This document supersedes prior audits (`AUDIT_REPORT.md`, `AUDIT_V3.md`, `docs/REMAINING_IMPLEMENTATION_AUDIT.md`) as the current source of truth where they disagree with observed code/execution. Those files are left in place for history but should not be trusted over this one.

---

## 0. Top-line summary

| Question | Answer |
|---|---|
| Does the ML pipeline (features → deep model → fusion → calibration → locked test) run end-to-end on real data? | **Yes, verified** — full run completed 2026-08-02, see §2. |
| Is the "psychologist rules" layer producing psychologically meaningful output? | **No.** 13 of 19 rules can structurally never fire (no detector exists). Concern profiles are hardcoded off and, independently, cannot fire even if re-enabled (source-type bug, see §5). |
| Is the source psychology PDF faithfully represented? | **Yes**, now directly verified (prior audits claimed the PDF was unreadable; it is not — see §5.1). The PDF itself is a short non-clinical Arabic pop-psychology article, not a validated instrument. |
| Is there a real database? | **No.** Everything is flat JSON/CSV files on disk. No SQL, no ORM, no migrations. |
| Is there a REST/HTTP API? | **No.** Only a CLI (`main.py`, 32 subcommands) and a Streamlit app (`streamlit_app.py`). |
| Is the project under version control? | **No.** No `.git` directory anywhere under `DOAR-main` or its parent. No commit history exists to inspect. |
| Does folder-label information leak into single-image inference? | **No leakage found** in the `analyze-image`/`predict-image` path — it takes only an image path and an optional checkpoint, never a label. But the *isolation/audit mechanism* the new spec requires (recording `original_source_label` and comparing it post-hoc) **does not exist yet** — this is a gap to build, not a leak to fix. |
| Are case outputs immutable/versioned? | **No.** Re-running `analyze-image` on the same output directory silently overwrites `analysis.json`, `evidence.json`, `rules.json`, `concerns.json`, `judges.json`, `emotion.json`, `detections.json`. Only `clinician_review.json` (preserved if present) and the review-master CSV (genuinely append-only) follow the required pattern. |
| Test suite status | **168/168 passing**, `python -m unittest discover -s tests -p "test_*.py"`, 34.6s, 2026-08-02. |

---

## 1. Repository layout and what actually exists

```
DOAR-main/
  main.py                 32-command CLI (argparse), 619 lines
  streamlit_app.py         Single-page multi-tab review UI (now with upload+analyze, added this session)
  src/doar/                 ~40 modules, ~4,700 lines (see module table below)
  tests/                    35 files, 2,999 lines, 168 tests, all passing
  resources/psychology_sources/
    rules_registry.json     19 ACTIVE rules (curated, low-confidence, cited)
    (no draft registry file was present before this audit — one now exists at
     outputs/full_run/psych_ingest from a fresh ingestion run, see §5.1)
  التحليل النفسي للصور.pdf   The source PDF. Present, readable (see §5.1).
  docs/                     14 markdown files — architecture notes, prior audits, guides
  configs/                  TOML model/training configs
  outputs/                  Run artifacts (gitignored-looking, not actually git-tracked since there is no git)
  .venv/                    Working Python 3.11.9 virtualenv, already provisioned
```

No `.git` in `DOAR-main` or its parent `DOAR/`. `git status`/`git log` both fail with "not a git repository." This contradicts the harness's session-start metadata (which reported "is a git repository: true" for a different working-directory context) — the actual project folder itself has never been initialized as a git repo. **There is no git history to inspect, despite instructions assuming one exists.** See `DECISION_LOG.md` for the recommendation.

### 1.1 Module inventory (`src/doar/`), by responsibility

| Area | Files | Lines | Status |
|---|---|---|---|
| Core analysis | `analysis.py` (377), `features.py` (162), `dataset.py` (138) | 677 | Implemented, exercised |
| Rules / concerns | `rules.py` (82), `concerns.py` (113) | 195 | Implemented but **mostly inert** — see §5 |
| Emotion inference | `emotion.py` (135), `deep/inference.py` (73), `models.py` (129) | 337 | Implemented, exercised this session |
| Deep training | `deep/trainers.py` (284), `deep/compare.py` (162), `deep/registry.py` (99), `deep/embeddings.py` (218), `deep/calibration.py` (126), `deep/preprocessing.py` (182), `deep/augmentations.py`, `deep/datasets.py`, `deep/checkpoints.py`, `deep/selection.py` | ~1,150 | Implemented, exercised this session (trained 2 real models) |
| Fusion | `fusion/trainer.py` (218), `fusion/calibrate.py` (149), `fusion/late.py` (146), `fusion/probability.py` (149), `fusion/oof.py` (109), `fusion/embedding_comparison.py` (144) | ~915 | Implemented, exercised this session |
| Leakage / provenance | `leakage.py` (267), `provenance.py` (136), `test_guard.py` (74) | 477 | Implemented, exercised — genuinely blocks (see §4) |
| Explainability | `explain/feature_importance.py` (96), `explain/gradcam.py` (143) | 239 | Implemented, exercised this session |
| Reporting / Q&A | `reports.py` (131), `qa.py` (212), `judges.py` (157), `localization.py` (103) | 603 | Implemented, exercised |
| Persistence | `case_output.py` (25), `review.py` (181) | 206 | Implemented but **not immutable** (see §6) |
| Ingestion | `psychology_ingest.py` (107) | 107 | Implemented, **verified working this session** (was previously believed blocked) |
| Experiment/eval | `experiments.py` (174), `evaluation.py` (251), `ablation.py` (128), `thesis.py` (172), `readiness.py` (210), `smoke.py` (191), `probability_export.py` (202) | 1,328 | Implemented, exercised this session |
| Misc | `config.py`, `schemas.py`, `uncertainty.py`, `gpu_smoke.py` | ~262 | Implemented |

No file in `src/doar/` is empty or a stub except `explain/__init__.py` (0 lines, a package marker). No dead/orphaned top-level module was found — everything under `src/doar/` is reachable from `main.py`'s command dispatch.

---

## 2. Data flow, input to output — verified by actual execution

This session ran the **entire** pipeline end-to-end against the real dataset, not synthetic data:

```
raw dataset (3,688 images, 4 classes: Angry/Fear/Happy/Sad)
  → resolve-leakage        1,513 images quarantined (cross-split dup/near-dup); 4 more removed
                            by hand for contradictory labels (see §4) → 2,171 clean images
  → build-manifest          leakage_status: PASS (post-cleanup)
  → extract-features        2,171 rows, 0 failures, 59 objective features/image
  → train-feature-model     5 classical models, winner random_forest, valid macro-F1 0.571
  → compare-deep-models      2 CNNs × 1 seed × 12 epochs (GPU), winner mobilenet_v3_small, 0.656
  → extract-embeddings       generic (ResNet18) + fine-tuned (winning checkpoint)
  → train-fusion-model       3 fusion strategies, winner mlp_early_fusion, valid macro-F1 0.691
  → calibrate-fusion         temperature scaling, NLL 0.885→0.747
  → run-ablation             feature-family ablation on the classical baseline
  → compare-embeddings       generic vs fine-tuned embedding comparison
  → explain-features/-gradcam  tabular importance + Grad-CAM, both produced real output
  → export-probabilities → evaluate-predictions (LOCKED TEST, run once)
                            macro-F1 0.637, accuracy 70.4%, n=98
  → generate-thesis-outputs   5 real figures + matching source-data JSON
  → analyze-image             single-case bilingual report, correct emotion prediction
```

Every number above is real, produced this session, stored under `outputs/full_run/`. Full detail was written up in a separate results report during this conversation; this audit does not repeat the figures beyond what's needed for context.

**Two engineering defects were found and worked around during that run (not yet fixed in code):**

1. `resolve_leakage()` in `leakage.py` computes `conflicting_labels` (pixel-identical images filed under two different emotion classes) but never adds them to `leaked_image_ids`, so `materialize_clean_dataset()` does not remove them. The downstream leakage *gate* (`enforce_leakage_gate`, called by every training command) correctly still blocks on them — so nothing incorrect gets trained — but `resolve-leakage`'s own output silently ships a dataset that will fail the very next gate it hits. Fixed by hand this session (4 files deleted); **not fixed in code**.
2. `main.py`'s `evaluate` command, when given `--deep-comparison`, resolves to a PyTorch `.pt` checkpoint and passes it to `evaluate_model()` in `models.py`, which unconditionally calls `joblib.load()`. This crashes (`UnpicklingError`) for any deep or fusion checkpoint — `evaluate --deep-comparison --split test` **cannot work as documented** in `docs/LOCAL_WORKFLOW_WINDOWS.md`/`RUN_GUIDE_WINDOWS.md`. The working path is `export-probabilities` → `evaluate-predictions`, which correctly dispatches on file suffix (`.pt`/`.pth` vs `.joblib`) and does support all three model families. Not fixed in code.

Both are itemized in `IMPLEMENTATION_PLAN.md` as safe, local, behavior-preserving bug fixes.

---

## 3. Models and checkpoints — what is real vs. what merely exists as code

| Model family | Code path | Real checkpoint from this session? | Notes |
|---|---|---|---|
| Classical feature models (LR, linear-SVM, RF, ExtraTrees, HistGB) | `models.py`, `experiments.py` | Yes — `outputs/full_run/feature_models/runs/*/model.joblib` | Winner: `random_forest`. |
| Deep CNNs (`small_cnn`, `mobilenet_v3_small`) | `deep/trainers.py`, `deep/registry.py` | Yes — `outputs/full_run/deep/runs/*/best.pt` | Only 2 of the 4 registry-supported architectures (`resnet18`, `efficientnet_b0` also exist in `deep/registry.py` but were not trained this session — time budget, not a code limitation). |
| Fusion (`early_scaled_concat`, `pca_early_fusion`, `mlp_early_fusion`) | `fusion/trainer.py` | Yes — `outputs/full_run/fusion/runs/*/fusion.joblib` | Winner: `mlp_early_fusion`. |
| Calibrated fusion bundle | `fusion/calibrate.py` | Yes — `outputs/full_run/fusion_calibrated/fusion_calibrated.joblib` | Temperature scaling, valid-only fit, `raw_preserved: true`. |
| CLIP / DINOv2 embeddings | `deep/embeddings.py` | Partially — `resnet18` backbone embeddings were extracted (generic + fine-tuned); `open-clip-torch` is an optional dependency (`pyproject.toml` extra `embeddings`) and was **not exercised** this session. | `embeddings` extra installs `open-clip-torch`; not confirmed working end-to-end. |

Every checkpoint file above was produced by commands actually run this session and can be re-inspected at the paths listed. No checkpoint in this table is a placeholder or example file.

Pre-existing checkpoints from before this session also exist under `outputs/smoke/small_cnn/best.pt` and `outputs/quick_demo/deep/...` — these appear to be from an earlier smoke-test run (not this session), unverified provenance beyond what their own `training_result.json`/`executed_config.json` record.

---

## 4. Labels: how they enter training, and whether they leak into inference

- **Training path** (`train`, `train-feature-model`, `compare-deep-models`, `train-fusion-model`): labels come from the `class` column of `manifest.csv`, itself derived from the ImageFolder directory structure (`train/<Class>/*.jpg`). This is the only source of ground truth in the codebase — there is no separate expert-annotated label file.
- **Inference path** (`analyze-image`, `predict-image`): verified by reading `analysis.py::analyze_image()`, `emotion.py::predict()`, `deep/inference.py::predict_image()`. None of these functions accept or reference a class label. `predict()`'s only use of `CLASSES` is to validate that a *loaded checkpoint's* stored class-order matches the fixed 4-class schema (`Angry, Fear, Happy, Sad`) — a schema-consistency check, not a use of the specific image's true label. **No folder-label leakage into single-image inference or reports was found.**
- **What does not exist yet**: the spec's requested `original_source_label` provenance field, and the `CONSISTENT / POSSIBLE_CONFLICT / UNCERTAIN / REVIEWED_CONFIRMED / REVIEWED_CORRECTED / ADJUDICATION_REQUIRED` audit-status mechanism. Nothing in the current schema (`schemas.py`) records a per-case original label at all — dataset labels are used only in aggregate for training/evaluation, never attached to an individual case's `analysis.json`. This needs to be built (see `IMPLEMENTATION_PLAN.md` Phase 2); it is a **missing feature**, not a leak to remediate.
- **Leakage across splits** (a different, already-solved problem): `leakage.py` correctly detects and — when using `resolve-leakage` — removes cross-split exact/near-duplicate images before any split-respecting command runs. Verified this session: 1,513/3,688 raw images were cross-split duplicates. The one gap (label-conflict rows not auto-quarantined) is noted in §2.

---

## 5. The rules/concerns layer — detailed audit

This is the area most likely to be over-claimed if not checked carefully, so it was checked line by line.

### 5.1 Source PDF — now verified, not merely assumed

Prior audits (`docs/PSYCHOLOGIST_SOURCE_AUDIT.md`, dated 2026-07-19) state the PDF `التحليل النفسي للصور.pdf` was unreadable in their sandbox and that the registry was built from "pasted text." This session, `main.py ingest-psychology-pdf --pdf "التحليل النفسي للصور.pdf" --output ... --source-id doar_source_v1` was run directly and **succeeded** (`pypdf` is installed and works), extracting 43 candidate statements across the PDF's 2 pages.

Content of the PDF (now directly read, not inferred): a short, informal Arabic article covering — eye style (wide/stern/closed), 4 named animals (tiger/wolf, fox, squirrel, lion), geometric shapes, stars, flowers/clouds/sun, circles, vehicles, hearts, drawing coverage (~50% / full page / <20%), and placement (top / left / right). It cites no study, no scoring system, no normative sample, no author credentials visible in the extracted text.

**Finding: the 19-rule `rules_registry.json` is thematically faithful to this PDF** — every rule maps to a statement actually present in it. Nothing in the registry was fabricated relative to the source. The registry's own honesty is a genuine strength: every rule carries `scientific_support: not_found_for_specific_claim` (or similar), a confidence ceiling of 0.05–0.25, and real citations (DOIs/PMIDs) to the actual skeptical academic literature about drawing-based assessment — e.g. `REF_GENERAL_LIMITS_1998` (Cherney et al., "Drawing conclusions: a re-examination of empirical and conceptual bases for psychological evaluation of children from their drawings," DOI 10.1111/j.2044-8260.1998.tb01289.x) and a 2025 systematic review of projective techniques. These citations were spot-checked and are real, correctly characterized (they report weak/no support, and the registry says exactly that rather than overselling).

### 5.2 Rule → feature wiring — the core finding

`rules.py::_status_for()` is the only function that turns a rule's `observable` field into an actual evaluated status. It has explicit branches for exactly 6 observables:

- `coverage_about_half`, `coverage_full`, `coverage_small` → read `composition["bounding_box_coverage"]`
- `placement_top`, `placement_left`, `placement_right` → read `composition["placement"]`

**Every other observable falls through to the final `return "missing_detector", [], [...]` branch — unconditionally, for any image.** That covers 13 of the registry's 19 rules: `wide_eyes`, `stern_eyes`, `closed_eyes`, `tiger_or_wolf`, `fox`, `squirrel`, `lion`, `repeated_geometric_shapes`, `stars`, `flowers_clouds_sun`, `circles`, `vehicles`, `hearts`. There is no eye detector, no animal classifier, no shape/symbol detector anywhere in `src/doar/` — confirmed by the earlier finding that `shape.enclosed_shape_count` and `shape.repetition_score` are explicitly `NaN` (`features.py` lines 136-143) with a comment stating no true shape detector exists.

**68% of the active rule registry (13/19 rules) can never produce anything but `missing_detector`, on any input, in the current codebase.** This is not a bug in the sense of incorrect output — `missing_detector` is the textually correct status, and the code is careful never to treat it as negative evidence — but it means the registry substantially overstates what the system currently evaluates. Full per-rule detail is in `RULE_COVERAGE_MATRIX.csv`.

### 5.3 Concern profiles — hardcoded off, and structurally unreachable even if re-enabled

`concerns.py` has `CONCERNS_ENABLED = False` at module scope — `derive_concerns()` returns `[]` immediately unless called with `enabled=True` (only done in tests). This alone means **no concern profile has ever been emitted by the running system.**

More importantly, even if that flag were flipped: `derive_concerns()` requires supporting evidence from ≥2 distinct `source_type`s, and its `_source_type()` helper checks `rule.get("source_type") == "psychologist_supplied_hypothesis"` first — but `rules.py::evaluate_rules()` sets `"source_type": "psychologist_supplied_hypothesis"` on literally every rule evaluation it emits, including the 6 that are backed by real composition measurements. So `_source_type()` always returns `"clinician_symbolic"` for every piece of evidence the rules engine can ever produce, the resulting `source_types` set is always exactly `{"clinician_symbolic"}`, and the diversity check `non_clinical_sources = source_types - {"clinician_symbolic"}` is always empty. **The multi-source convergence requirement is unsatisfiable by construction from the current rules engine's output — not just switched off.** This was verified by reading both files together, not assumed from either alone.

This is arguably *good* safety engineering (nothing can ever over-claim a "concern"), but it should be described accurately: the concern-convergence system is fully built and tested in isolation (`tests/test_concerns_and_ingest.py`), but is disconnected from ever actually firing in the live pipeline by two independent mechanisms.

### 5.4 Test coverage of the rules/concerns layer

`tests/test_concerns_and_ingest.py` (86 lines) and `tests/test_confidence_ceiling.py` (43 lines) exercise `derive_concerns()` directly with `enabled=True` and check ceiling enforcement — good unit coverage of the *logic*. There is no test that exercises `evaluate_rules()` end-to-end and asserts on the `missing_detector` proportion, so the 68%-inert finding above was not previously documented anywhere in the repo, including the prior audits.

---

## 6. Persistence — file-based, mostly mutable, one genuinely append-only path

- No database engine of any kind (`sqlite3`, `sqlalchemy`, `psycopg`, etc. — all absent from both `pyproject.toml` and the source tree).
- Per-case outputs (`analysis.json`, `evidence.json`, `rules.json`, `concerns.json`, `judges.json`, `emotion.json`, `detections.json`) are written by `case_output.py::_write()`, which is an unconditional `path.write_text(...)` — **re-running analysis on the same output directory silently overwrites all of these with no history retained.**
- `clinician_review.json` is the one exception with a soft guard (`if not review.exists(): _write(...)`) — an existing structured review is not clobbered by a re-analysis.
- The one genuinely append-only mechanism is `review.py::append_reviews()`, which opens the shared `review_master.csv` in `"a"` mode — structured psychologist reviews accumulate correctly and are never overwritten.
- **No soft-delete, no revision/version numbering, no immutable case history exists for the AI-generated analysis itself.** This directly conflicts with the spec's Section 14 requirement ("append-only or immutable-versioned storage... do not remove existing records") for the analysis artifacts (though not for the review log, which already complies). This is a real architecture gap, not a bug — see `IMPLEMENTATION_PLAN.md`.

---

## 7. UI and "API" — current real behavior

- **No REST/HTTP API exists.** Confirmed by grep across `src/`, `main.py`, and `pyproject.toml` — no Flask, FastAPI, uvicorn-as-a-server, or route decorators anywhere. The only two interfaces are the CLI and Streamlit.
- **CLI** (`main.py`): 32 subcommands, all were enumerated via `--help` and cross-referenced against their handler code in this and the previous session; every one dispatches to a real function in `src/doar/` (no stub handlers).
- **Streamlit app** (`streamlit_app.py`): originally read-only (browse a pre-existing case folder). This session added an upload-and-analyze flow (file uploader → `analyze_image()` → auto-load the resulting case) directly wired to the real `analyze_image()` code path — verified by running the exact same call chain outside Streamlit and inspecting the produced case folder and a Q&A response against it. Every existing tab (Summary, artifacts, Emotion, Rules, Concerns, Reports, Q&A) reads real analysis output; none of them render mock/placeholder data — confirmed by reading each tab's source (`streamlit_app.py` lines 36–127) against the schema `analyze_image()` actually writes.
- **External LLM dependency**: none. No `.env`/credentials file exists in the repo, and no code path calls an external LLM API (OpenAI/Anthropic/Gemini SDKs are absent from `pyproject.toml`). `qa.py` is a fully deterministic keyword router over saved evidence (verified by full read, §5-adjacent). This already satisfies the spec's "must work without an external LLM API" requirement — trivially, since none is wired in at all yet. The optional "external AI audit" layer described in the spec's Section 13 (explain/translate/audit, with deterministic post-hoc validation) does **not exist yet** and would be new work, not a modification of an existing broken thing.

---

## 8. Dependencies, environment, compute

- Python 3.11.9 in `.venv`, already provisioned and working (confirmed via `python --version`, `import numpy, PIL, sklearn, torch, torchvision`).
- `torch 2.7.1+cu118`, CUDA available and verified working (`torch.cuda.is_available() == True`; GPU actually used for deep training and embedding extraction this session).
- Optional extras (`pyproject.toml`): `ml`, `cv`, `dev`, `ingest`, `ui`, `deep`, `embeddings` — all appear installed in `.venv` except `open-clip-torch` usage was not exercised (installed per `pip list`, not confirmed functionally end-to-end this session).
- No missing credentials — none are required by anything currently implemented.
- No missing core data — the dataset referenced throughout (`Combined_Drawing`) is present and was used for the full run in §2.

---

## 9. Scientific/clinical overclaim scan

Grepped `reports.py`, `judges.py`, `qa.py`, `rules_registry.json`, and the HTML report templates for diagnostic language. Findings:

- `judges.py::safety_judge` actively scans rule-derived narrative text (English + Arabic) for diagnostic patterns (`has/diagnosed with/suffers from + condition`) and fails the case if found or if the non-diagnostic disclaimer is missing. This is a real, executed guard, not decorative — confirmed via `tests/test_quality_suppression_judges.py` and `tests/test_safety_and_leakage.py`.
- No occurrence of "diagnos-", "abuse probability", "anxiety percentage," or similar prohibited phrasing was found being *generated* by any report/QA code path.
- The rules registry's `parent_safe_wording` fields were spot-checked and consistently use hedged, non-diagnostic language ("may reflect," "is not evidence that").
- **No violation of the Section 6 scientific boundaries was found in current output.** The main risk is not overclaiming in existing text — it's that 68% of the registry that *could* eventually be activated concerns literal psychological-trait/animal-symbol claims (e.g., "drawing a fox means the child was thinking of doing something sly") that would need to stay permanently disabled or be replaced with literature-grounded alternatives before ever being turned on, per the new spec's Tier system.

---

## 10. What is proven to work vs. what merely exists in code

**Proven this session (executed, output inspected):**
leakage detection + quarantine; manifest building; objective feature extraction (59 features/image); classical model training + selection; CNN training on GPU; embedding extraction (ResNet18 only); fusion training + calibration; ablation; embedding comparison; feature-importance explainability; Grad-CAM; locked test evaluation; thesis figure generation; single-image `analyze-image` end-to-end including HTML report generation; `qa.answer()` against a real case; Streamlit upload→analyze→browse→Q&A flow; PDF ingestion.

**Exists in code, implemented, but not exercised this session:** `resnet18`/`efficientnet_b0` deep architectures (registry supports them; not trained due to time budget); CLIP/DINOv2 embeddings; late-fusion/stacking (`fusion/late.py`); multi-seed runs (only seed 42 used this session; the code supports `[42,123,2026]` and was used this way in earlier smoke/quick_demo runs per their own metadata, not independently reverified here); the structured psychologist-review Streamlit tab (`review.py` append path was read and is structurally sound, but no real reviewer has used it in this environment).

**Exists in code but is functionally inert / cannot currently produce user-facing output no matter the input:** 13/19 rules (`missing_detector` always); the entire concern-profile mechanism (§5.3); shape features (`NaN` by design); object/person/animal/face detection (`detections.json` explicitly reports `unavailable` — confirmed, not a placeholder claiming success); OCR (same — explicitly unavailable).

**Does not exist at all yet (not "broken," simply not built):** any database; any REST API; per-case original-label provenance/audit-status tracking; immutable/versioned case history; the free-drawing-appropriate rule tiering (Tier 1/2/3) requested in the new spec — the current registry has no tier concept, just a flat list; any detector for eyes, faces, people, houses, trees, animals, or symbols; multi-model/multi-seed disagreement-based label auditing; Confident-Learning-style label quality ranking; longitudinal tracking across multiple drawings of the same child.

---

## 11. Immediate corrections this audit makes to prior documentation

- `docs/PSYCHOLOGIST_SOURCE_AUDIT.md` incorrectly states the PDF is unreadable — it is readable, and was read this session (§5.1). That file should be marked superseded, not deleted (append-only principle applies to project history too).
- `AUDIT_V3.md` §2.2 lists "D6 concern-convergence engine" as "✅ Fixed" — technically true in that the stub `return []` was replaced with real logic, but the practical effect (concerns still never fire, and now cannot even if re-enabled) is not stated there. This audit's §5.3 is the accurate, complete picture.
- No prior document flagged the `resolve-leakage`/`enforce_leakage_gate` inconsistency (§2) or the `evaluate --deep-comparison` crash (§2) — both found by actually running the commands this session, not by code reading alone.

See `DECISION_LOG.md` for how these findings translate into planned actions, and `IMPLEMENTATION_PLAN.md` for sequencing.
