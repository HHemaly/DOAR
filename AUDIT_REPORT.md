# DOAR — Audit Report (Phase 1)

**Repository:** HHemaly/DOAR
**Audited by:** senior ML/CV review pass
**Python target:** 3.11 · **Runtime targets:** VS Code (Windows) + Google Colab
**Date:** 2026-07-13

> This report is an **inspection**, not a rewrite. It traces the real execution
> paths, states what actually runs versus what only exists as code, and proposes
> the smallest scientifically defensible architecture that reaches the thesis goal.

---

## 0. TL;DR

DOAR today is a **rule-based visual-analysis + safety pipeline**, not a
supervised machine-learning system. It extracts objective image features,
applies literature-referenced *non-diagnostic* psychological rules, builds and
validates structured claims, and produces a safe parent-facing answer gated by a
10-point judge. That interpretive/safety spine is genuinely good and worth
preserving.

**The gap between the current repo and the thesis brief is one whole level:**
there is **no trained image classifier, no train/val/test split, no evaluation,
no confusion matrix** anywhere in the codebase. The "emotion" output is an
honestly-labelled heuristic, not a model. Section 5/6 of the brief (supervised
training + evaluation) must be **built from scratch**; they are not "broken" —
they are **absent**.

The five-level scientific separation the brief demands maps cleanly onto the
existing claim-type system, so we extend rather than replace.

---

## A. What currently works

| Component | File(s) | Status |
|---|---|---|
| Objective feature extraction (colour, composition, strokes, quality) | `pipeline.py::_extract_features`, notebook Part A | ✅ Runs, reproducible, pure OpenCV/NumPy |
| Local single-image pipeline | `pipeline.py::run_full_pipeline_v2` | ✅ Verified end-to-end on a real image (Judge PASS, 10/10) |
| Local batch runner | `analyze_dataset.py`, `pipeline.py::run_dataset` | ✅ Walks class folders, continues on error |
| V1 psychological rules (7) | `pipeline.py::_evaluate_v1_rules` | ✅ Literature-referenced, multi-feature, non-diagnostic |
| V2 extended rules (21) + weighted themes | `src/psychological_rules_v2.py` | ✅ Tiered, min_cluster≥2, evidence_strength tagged |
| Heuristic emotion estimate | `src/emotion_heuristic.py` | ✅ Honestly labelled `method=heuristic, confidence=low` |
| Claim builder (6 types) | `src/claim_builder.py` | ✅ Structured, IDs, evidence, sensitive flags |
| Numeric grounding validator (±5%) | `src/numeric_validator.py` | ✅ `find_ungrounded_numbers` fixes the "99% empty" bug |
| OCR / psych-safety validators | `src/ocr_validator.py`, `src/psych_safety_validator.py` | ✅ Threshold-gated status assignment |
| Safety policy (BLOCK/FLAG/sanitise) | `src/safety_policy.py` | ✅ Negative-lookbehind fix for "not diagnostic" |
| 10-point final response judge | `src/final_response_judge.py` | ✅ Sanitises then falls back to safe answer |
| Template parent answer (EN) + Arabic | `src/parent_ai_helper.py`, `src/arabic_translator.py` | ✅ Verified-claims-only; cannot invent objects |
| Output manager (report cards, thesis figs) | `src/output_manager.py` | ✅ Per-image JSON + report_card.png + 4 figures |
| Centralised thresholds | `config/thresholds.json` | ✅ Every value has a justification key |

## B. What partially works

- **OCR** (`pipeline.py::_run_ocr`): PaddleOCR→EasyOCR→empty fallback chain is
  sound, but **untested against the real dataset** and EasyOCR first-run downloads
  models (slow, needs network). No Arabic OCR path despite Arabic output.
- **Object detection** (notebook Part B, SAM+CLIP): defined but **heavy and
  optional**. In the local `pipeline.py` path, `detected_objects` is **always an
  empty list** — object/symbol claims therefore never fire locally. So the
  "annotated image with genuine detections" (brief §15B) currently has nothing to
  annotate.
- **Arabic translation**: works via `deep-translator` but needs network; offline
  path only covers a small hardcoded label set, not full sentences.

## C. What is experimental / heuristic (must stay labelled as such)

- The entire **emotion output** — feature heuristic, not a classifier.
- **Tier-2 symbolic rules** (animals, shapes, eyes, hearts) — `evidence_strength`
  ranges down to `symbolic_speculative`; fox/squirrel are
  `requires_further_validation` with empty `sources`.
- **CLIP "confidence"** — raw cosine similarity, **not calibrated probability**
  (brief §7 flags this explicitly; see D3).

## D. What is broken / defective

1. **No ML level exists (Level 3).** No training loop, no `DataLoader`, no
   split, no checkpoint, no metrics. The brief's §5/§6/§18 depend entirely on
   this and it must be built.
2. **Object detection is disconnected locally.** `_extract_features` never
   populates `detected_objects`, so visual-object/symbol claims and the annotated
   image are dead paths in the VS Code run.
3. **Bounding boxes not threaded to the visual validator.** `visual_claim_validator`
   receives a claim but the crop/bbox is not reliably passed from detection →
   claim → validator; validation can silently fall back to the **full image**
   (brief §7's named bug). Even where a crop exists, cosine similarity is treated
   like a probability.
4. **Notebook ≠ modules.** Core logic (Parts A–F) lives **only in notebook cells**
   and is partially re-implemented in `pipeline.py`. Two sources of truth drift
   apart. `run_full_pipeline()` (v1) exists only in the notebook and relies on
   `IPython.get_ipython().user_ns` — it cannot run in VS Code.
5. **Colab-only cells** (`google.colab.drive`, `/content/drive/...` paths,
   `!pip install`) sit in the notebook and break a plain `jupyter`/VS Code run.
6. **No leakage control.** No duplicate/near-duplicate detection; nothing prevents
   the same drawing landing in train and test.

## E. What is missing (relative to the brief)

Dataset inspection module & CSVs · deterministic leak-safe split · training
module + checkpoints + history · evaluation module + confusion matrix + per-class
metrics · Grad-CAM explainability · HTML technical report · HTML psychologist
review form + master CSV + kappa · annotated-image renderer · `main.py` unified
CLI · clean `notebooks/DOAR_Colab.ipynb` that only *calls* modules · `tests/`
suite · reproducibility manifest (versions, seed, git hash) · full README.

## F. What is duplicated

- Feature extraction exists **3×**: notebook Part A cell, notebook
  `run_full_pipeline` inline copy, and `pipeline.py::_extract_features`.
- Safety patterns exist in both `src/safety_policy.py` and notebook Part F
  (`BLOCK_PATTERNS`) with slightly different regexes.
- `OUTPUT_DIR`/path constants redeclared in nearly every notebook cell.

## G. What is unused

- `append_v2_cells.py`, `append_quickstart.py` — one-off notebook mutators, no
  runtime role now that modules exist.
- Notebook Part D QA router + Part C attribute-retrieval (sentence-transformers)
  are **not called** by `run_full_pipeline_v2`.
- `visual_claim_validator.py` is imported defensively but effectively inert
  locally (no detections to validate, no CLIP loaded).

## H. Colab dependencies

`from google.colab import drive` · `drive.mount('/content/drive')` ·
hardcoded `/content/drive/MyDrive/Masters/...` dataset + output paths ·
`!pip install` cells · `IPython.get_ipython().user_ns` scope lookups in
`run_full_pipeline`.

## I. What prevents VS Code execution

- `run_full_pipeline()` needs the IPython namespace (fixed for v2 by
  `pipeline.py`, but v1 still notebook-bound).
- Colab paths and `!pip` magics in notebook cells.
- Heavy optional deps (torch, segment-anything, paddle) presented as required in
  Part B/C headers. *(Mitigated: `pipeline.py` degrades gracefully, but the
  notebook does not.)*

## J. What prevents dataset-scale training/testing

Everything in §D1/§E: there is no model, no split, no loaders, no eval. Also no
class-imbalance handling, no seeding, no checkpoint format. This is greenfield.

---

## Reconstructed pipeline (image → output), as it runs today (VS Code, v2)

```
image
 └─ _extract_features            → colour / composition / stroke / quality   [Level 1]
 └─ _run_ocr (Paddle→Easy→∅)     → ocr_results                                [Level 1]
 └─ _evaluate_v1_rules (7)       → psychological_rule_activations             [Level 4]
 └─ evaluate_v2_rules (21)       → ..._v2 + compute_theme_scores              [Level 4]
 └─ estimate_emotional_tendency  → feature_based_emotional_tendency (heuristic)[≈Level 4, NOT Level 3]
 └─ build_all_claims + themes    → claims[] (6 types)
 └─ validators (numeric/ocr/psych)→ validator_status, show_to_user
 └─ generate_parent_answer (EN)  → parent_answer + gentle_questions
 └─ translate_output (AR)        → parent_answer_ar
 └─ judge_final_response (10-pt)  → PASS / REWRITE_REQUIRED / BLOCK
 └─ output_manager               → analysis_en.json, analysis_ar.json, report_card.png
```

**Object detection (Part B) and QA router (Part D) are NOT in this path locally.**
The dataset ground-truth label (Level 2) is captured
(`metadata.label_from_dataset`) but never *used* because there is no classifier
to compare it against.

---

## Mapping the brief's 5 levels onto the code

| Brief level | Exists? | Where / gap |
|---|---|---|
| L1 Objective observation | ✅ | `_extract_features`, OCR |
| L2 Dataset ground-truth label | ⚠️ captured, unused | `metadata.label_from_dataset` |
| **L3 ML prediction** | ❌ **absent** | **build: `src/models/`, training, eval** |
| L4 Rule-based indicators | ✅ | v1+v2 rules, themes, claims |
| L5 Psychologist validation | ⚠️ data model only | claims have status; **no review UI/CSV/kappa** |

---

## Recommendations

**Preserve (reuse as-is or lightly):** all of §A — feature extraction, rules
v1/v2, claim architecture, validators, safety policy, judge, parent helper,
Arabic translator, output manager, `thresholds.json`.

**Modify:** unify feature extraction into one `src/features/` module (kill the 3
copies); thread bbox/crop through detection→claim→validator and stop calling
cosine similarity a probability; make object detection optional-but-connected;
split notebook logic out so the notebook only *calls* modules.

**Delete / archive:** `append_v2_cells.py`, `append_quickstart.py`; the Colab-only
cells (replace with a clean `notebooks/DOAR_Colab.ipynb`); duplicated path
constants and duplicated safety regexes.

**Build (missing):** dataset inspector + leak-safe split, training + evaluation,
`main.py` CLI, HTML technical/psychologist/parent reports, annotated-image
renderer, Grad-CAM, tests, reproducibility manifest, README.

---

## Proposed final architecture (simplest defensible)

```
DOAR/
  config/            thresholds.json, model.yaml, split.yaml
  src/
    data/            discover.py, inspect.py, dedupe.py, split.py, loaders.py
    features/        extract.py            (single source of truth — merges 3 copies)
    detection/       objects.py, clip_validate.py   (bbox-threaded, optional)
    models/          classifier.py, train.py, evaluate.py, gradcam.py
    rules/           rules_v1.py, rules_v2.py, themes.py
    claims/          builder.py, validators/…
    safety/          policy.py, judge.py
    reports/         technical_html.py, parent_html.py, psychologist_html.py, annotate.py
    utils/           seeding.py, repro.py, io.py
  notebooks/DOAR_Colab.ipynb          (calls modules only)
  scripts/                            (thin wrappers)
  tests/
  main.py            analyze-image | analyze-dataset | inspect | train | evaluate | thesis
  requirements.txt / requirements-dev.txt
  README.md
```

Design rule: **the classifier prediction, the emotion heuristic, and the
psychological interpretation each keep their own separate confidence and are
never merged into one number** (brief §12, §3).

---

## Proposed phases (and where each can actually run)

| Phase | Deliverable | Runs where |
|---|---|---|
| 1 Audit | **this file** | ✅ done here |
| 2 Stabilize | one feature module, path/seed/config cleanup, `main.py` skeleton | ✅ here |
| 3 Modularize | move notebook logic into `src/`, clean Colab notebook | ✅ here |
| 4 Dataset | inspector + dedupe + leak-safe split + loaders + CSVs/figures | ⚠️ **needs the dataset** → your machine/Colab |
| 5 Train/Eval | baseline + transfer model, checkpoints, metrics, confusion matrix | ⚠️ **needs the dataset + compute** → Colab GPU |
| 6 Interpretation | thread bbox, connect detection, wire L3 into claims | ✅ code here, ⚠️ full run needs data |
| 7 Reports | technical/parent/AR/psychologist HTML + annotated images | ✅ here |
| 8 Thesis outputs | figures + tables from real eval | ⚠️ needs Phase 5 outputs |
| 9 Validation | tests + honest example cases (incl. failures) | ⚠️ real examples need data |

---

## The one hard blocker (must be explicit)

I **cannot access your dataset from this environment.** The Google Drive folder
(`1751R2MY8umA707S16YalIQQ3K8fUc48V`) returns empty to my Drive tools, and the
Windows path `C:\Users\Ahmed\...` is on your machine, not here. There is also no
GPU here for training.

**Consequence:** Phases 4, 5, and 8 (dataset-inspection CSVs, model training,
confusion matrices, real per-class metrics, real example cases) **cannot be
executed by me** — I can only *write and unit-test the code*, and you run it in
Colab (GPU) or locally. I will **not** fabricate dataset statistics, accuracy
numbers, confusion matrices, or psychologist agreement (brief §28 — no fake
results). Any such artifact in the repo will be produced by *your* run, not
invented by me.

**What I can fully build and verify here:** Phases 1, 2, 3, 6 (code), 7, and 9
(tests + a synthetic smoke image), plus all the training/eval **code** ready to
run on your data.
