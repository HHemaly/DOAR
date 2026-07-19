# DOAR — Independent Architecture Review & Recommendation (v3)

**Reviewer role:** CV researcher / multimodal architect / XAI / clinical-DSS engineer / MSc supervisor
**Inputs reviewed:** `doar_v3` source ZIP (authoritative latest), repo `HHemaly/DOAR`
(`main`, `HHemaly-patch-1`, `claude/nice-allen-IgPnj`), psychologist rules registry.
**Date:** 2026-07-19

> **Access note (honest):** The branch `feature/integrated-doar-pipeline` and the
> "psychologist PDF" you referenced are **not present** in the accessible
> `HHemaly/DOAR` remote (confirmed via git refs *and* the GitHub API — only
> `main`, `HHemaly-patch-1`, `claude/nice-allen-IgPnj` exist). The Google Drive
> dataset folder also returns empty to my tools. **However**, the v3 ZIP already
> contains `resources/psychology_sources/rules_registry.json` +
> `docs/PSYCHOLOGIST_RULES_TRANSLATION.md`, which appear to be the transcribed,
> structured form of that PDF. My review therefore treats the ZIP's registry as
> the psychologist-rules source of truth. If the PDF differs, drop it into
> `resources/psychology_sources/` and re-run ingestion (the README already
> documents this path).

---

## 1. ZIP vs repository — which is the real trunk?

| Dimension | `main` (repo) | `claude/nice-allen-IgPnj` (my earlier branch) | **`doar_v3` ZIP** |
|---|---|---|---|
| Structure | notebook + flat `src/` | flat `src/` + `main.py` CLI | **proper `src/doar/` package** |
| Primary ML classifier | none | ResNet/MobileNet transfer + eval | **model registry + deep trainers + fusion** |
| Objective features | rule inputs only | basic colour/comp/stroke | **`features.py` (157 lines) feature families** |
| Deep embeddings | — | — | **`deep/embeddings.py` (CLIP/backbone)** |
| Fusion | — | — | **early_scaled_concat / PCA / MLP fusion** |
| Calibration | — | — | **temperature scaling (valid-only)** |
| Uncertainty | — | — | **entropy / margin / category** |
| Config system | scattered constants | `thresholds.json` | **TOML + hashing + resolved-config save** |
| Experiment framework | — | single train/eval | **multi-seed [42,123,2026], val-selection, test-locked** |
| Psychologist rules | 21 cautious rules (code) | same, ported | **structured registry w/ evidence IDs + confidence ceilings + refs** |
| Clinical separation | mixed into pipeline | separated in reports | **enforced in config (`psychologist_rules_used=false`)** |
| Q&A / judges / Streamlit | notebook QA | — | **`qa.py`, `judges.py`, `streamlit_app.py`** |
| Docs | README | AUDIT/README | **14 docs incl. thesis architecture** |
| Compiles | n/a (notebook) | yes | **yes (`compileall` exit 0)** |

**Verdict:** the **v3 ZIP is decisively the strongest trunk** and is materially
more advanced than either repo branch, including my own. Per your instruction
("do not force the current architecture if a better one is found; use what is
good"), my recommendation is to **adopt v3 as the base** and push it to the repo,
rather than continue `claude/nice-allen-IgPnj`. My branch's genuinely useful,
non-overlapping pieces (leak-safe split with perceptual-hash dedup, HTML
technical/parent/psychologist reports, Cohen's/Fleiss' kappa agreement,
per-detection crop + annotation, Grad-CAM figure disclaimer) are **grafted in**,
not discarded.

---

## 2. Audit of the v3 architecture

### 2.1 What is already strong (retain)
- **Scientific separation is enforced, not just described.** The thesis diagram,
  and crucially the fusion config (`psychologist_rules_used=false`,
  `concern_profiles_used=false`) + the code (`test_used:false`,
  `selection_split:"valid"`) prevent the clinical layer from touching the
  classifier and prevent test-set leakage. This is the single most important
  correctness property for an MSc defence, and it is present.
- **Experimental hygiene:** multi-seed `[42,123,2026]`, validation-based model
  selection on **macro-F1** (correct for 4-class, likely imbalanced), explicit
  `test_locked=true`. Classical baselines (LR, linear/RBF SVM, RF, ExtraTrees,
  HistGB) and a ResNet18 transfer config with class weighting, freeze epochs,
  differential LRs, early stopping, reduce-on-plateau — all defensible.
- **Calibration + uncertainty as first-class citizens** (temperature scaling on
  valid; entropy/margin/category outputs) — rare in student projects, strong
  thesis value.
- **Psychologist rules registry is exemplary for safety:** every rule carries
  `scientific_support: not_found_for_specific_claim` (or
  `indirect_expression_research_only`), a low `confidence_ceiling` (0.05–0.25),
  real DOI/PMID references (incl. the skeptical 1998 & 2025 reviews),
  `parent_safe_wording`, and `limitations`. This is exactly the non-diagnostic,
  evidence-cited posture the brief demands — and it already digests the Arabic
  notes.
- **Config reproducibility:** TOML + hashing + resolved-config persistence.

### 2.2 Confirmed defects (verified by full module + docs pass)

| # | Defect | Severity | Fix |
|---|---|---|---|
| D1 | **Calibration is written correctly (valid-only NLL grid search) but not wired** — trainer/fusion/inference all emit `calibration_status:"uncalibrated"`; no CLI calls `fit_temperature`. | High (RQ4 depends on it) | Wire a `calibrate` step after training; apply T at inference. |
| D2 | **Late-fusion / stacking / ensemble-uncertainty exist only as unused library code** (`probability.py`, `uncertainty.py`); `oof_stacking` & `logistic_probability_meta` are named in `PROBABILITY_METHODS` but unimplemented. The *executed* primary path is **early fusion** (concat features+embeddings → one classifier). | Medium | Either wire the late-fusion CLI (for RQ3) or scope the thesis to early-fusion + document. |
| D3 | **Cross-split duplicate/near-duplicate leakage detected but NOT blocked** (`duplicates_block_training:false`); no perceptual-hash near-dup check. | High (leakage) | **Graft my branch's perceptual-hash dedup + `leakage_ok` gate**; block on cross-split exact/near dupes. |
| D4 | **Shape feature family stubbed** — `shape.enclosed_shape_count` & `shape.repetition_score` hardcoded `0.0`; `contour_proxy_count` reuses component count. | Medium | Implement or mark `not_evaluated` (brief forbids fake features). |
| D5 | **Embedding `preprocessing_hash` mislabeled** `imagenet_v1`/224 for CLIP & DINOv2 (they use their own preprocess) — reproducibility-metadata bug. | Medium | Record the true preprocess per backbone. |
| D6 | **Concern-convergence engine is a stub** — `evaluate_rules` always returns `[]` concerns (safety-preserving but incomplete). | Medium | Build multi-source convergence (require ≥2 independent evidence IDs) — never from a single symbol. |
| D7 | **`safety_judge` regex is English-only and narrow** — does not scan Arabic `original_arabic` or parent wording; misses "shows signs of trauma". | High (safety) | Broaden patterns; scan Arabic + parent text; add tests. |
| D8 | **`bilingual.html` is malformed** (two `<!doctype>` roots joined by `<hr>`). | Low | Render one document with two sections. |
| D9 | **Cosmetic placeholder artifacts** shown as if meaningful — `page_mask` constant, `density_map`/`stroke_map` blurs, `quality.supported` hardcoded `True` (no real blur/res gating). | Medium | Implement real quality gating or label artifacts "illustrative". |
| D10 | **`confidence_ceiling` never numerically enforced**; Streamlit Q&A tab is inert (prints a CLI string). | Low/Med | Enforce ceiling on any rule-derived score; wire the Q&A tab to `qa.answer`. |
| D11 | Only `tests/test_objective.py` (17 synthetic tests) — good invariants but far below the brief's critical-path list; no Arabic-safety, no calibration-wired, no leakage-block test. | Medium | Expand suite (mocks/tiny models/synthetic). |

**Provenance caveat (important):** `PSYCHOLOGIST_SOURCE_AUDIT.md` states the real
source PDF (`التحليل النفسي للصور.pdf`) was **never readable** in the v3 authors'
sandbox either — import is formally "blocked" and the current registry is from
**pasted text**, not the PDF. This matches my own finding that the PDF/branch are
unreachable here. Treat the registry as a faithful transcription pending a
provenance-preserving PDF ingestion (the workflow is stubbed and documented).

### 2.3 What is duplicated / to reconcile
- Two parallel report stacks now exist (v3 `reports.py`/`case_output.py` vs my
  branch's `html_reports.py`). Keep **one**; graft the psychologist-review form +
  kappa from mine into v3's report module.
- Feature extraction exists in v3 (`features.py`) and my branch (`pipeline.py`).
  v3's is the keeper; delete the duplicate.

### 2.4 What is over-/under-engineered
- **Not over-engineered.** The module count is justified by the pipeline stages.
- **Under-developed:** tests, error-analysis galleries, thesis figure automation,
  and the psychologist review-save loop (my branch has the kappa half).

### 2.5 Is the research contribution clear? — **Yes.**
"Validation-selected multimodal fusion of objective drawing features + deep
visual representations, with calibration, uncertainty, and evidence-traceable —
but strictly non-diagnostic — clinical decision support." That is a legitimate,
defensible MSc contribution provided the fusion is shown to add value over the
best single modality (an ablation the framework is already set up to run).

---

## 3. Three candidate architectures

### Candidate A — Low-complexity, maximally defensible
**Objective-feature classifier only.** Handcrafted features (colour, composition,
stroke, shape) → HistGradientBoosting / SVM, multi-seed, calibrated.
- *Repr:* tabular features. *Train:* sklearn. *Fusion:* none.
- *Calib:* isotonic/Platt. *Uncertainty:* entropy/margin. *XAI:* permutation
  importance, per-feature contributions (fully valid for tabular).
- **+** Tiny overfitting risk on a small dataset; fully reproducible on CPU;
  maximally interpretable. **−** Likely lower ceiling; limited novelty alone.
- *Cost:* trivial. *Thesis value:* strong as a **baseline/ablation**, weak as the
  sole contribution.

### Candidate B — **Recommended:** validation-selected multimodal fusion
**ResNet18 (transfer) deep branch + objective-feature branch → fusion**, selected
on validation macro-F1, temperature-calibrated, with uncertainty and Grad-CAM
(deep) + permutation importance (features). The **shipped, leakage-safe primary
path is early fusion** (block-scaled concat of features + off-the-shelf
embeddings, transforms fit on train only, test never loaded); **late-fusion /
stacking variants** (`probability.py`) are available to wire as the RQ3 arm.
- *Repr:* image (224²) + tabular features + optional frozen CLIP embedding.
- *Train:* transfer-learn ResNet18 (freeze→fine-tune, class weights); extract
  features + embeddings; train fusion head (concat / PCA / small MLP) on
  train, select on valid, **test once**.
- *Calib:* temperature scaling on valid. *Uncertainty:* calibrated probs,
  margin, entropy, category. *XAI:* Grad-CAM for the CNN, permutation/coeff for
  the feature branch — **kept conceptually distinct**.
- **+** Best accuracy/interpretability/reproducibility trade-off for a small
  drawing dataset; the fusion-adds-value question is itself the contribution;
  trains on a single Colab GPU. **−** More moving parts than A; must guard
  fusion against overfitting (few seeds, PCA/regularised head).
- *Cost:* moderate (Colab T4). *Thesis value:* **high.**

### Candidate C — Advanced experimental
**ViT/ConvNeXt backbone + CLIP embeddings + attention-based fusion**, optionally
multi-task (emotion + auxiliary objective-feature regression), ensembled.
- **+** Higher ceiling, richer XAI (attention rollout), novelty. **−** Serious
  **overfitting risk** on a small children's-drawing set; heavier to train,
  calibrate, and defend; attention-as-explanation is contested.
- *Cost:* high. *Thesis value:* high **only if** the dataset is large enough and
  results are stable across seeds — otherwise it weakens the defence.

---

## 4. Recommendation — **Candidate B (the v3 fusion design)**

Adopt **v3's validation-selected fusion (Candidate B)** as the primary thesis
system. Reasons it beats A, C, and both repo branches:

1. **It matches the dataset reality.** Children's-drawing sets are small and
   noisy; B's transfer learning + handcrafted features + regularised fusion is
   the sweet spot between A's low ceiling and C's overfitting risk.
2. **The contribution is a *question the design answers*** ("does fusing
   objective features with deep representations help, and does calibration make
   it safer?"), not a bet on one model — robust to whatever the accuracy turns
   out to be. No fabricated accuracy needed.
3. **Safety and separation are already enforced in code/config**, satisfying the
   clinical constraints without extra machinery.
4. **Reproducibility is built in** (TOML hashing, seeds, test-lock).
5. Candidate A becomes a **baseline/ablation** inside B (feature-only arm), and a
   ViT can be added as **one** extra experimental arm (toward C) *if* seed
   stability holds — without betting the thesis on it.

I am **not** forcing my earlier branch; v3 is better and becomes the trunk.

---

## 5. Integration plan (retain / replace / remove / add)

**Retain (v3):** package layout, config/hashing, `features.py`, deep registry +
trainers + embeddings, fusion, calibration, uncertainty, experiments framework,
rules registry, judges, Q&A, Streamlit, docs.

**Replace / reconcile:** collapse the two report stacks into v3's, keeping my
branch's HTML technical/parent-EN/AR/psychologist templates where richer; make
the feature pipeline single-source (v3).

**Remove:** duplicated feature extraction and the notebook-era Colab cells; any
placeholder feature that cannot be honestly measured (or relabel `not_evaluated`).

**Add (from my branch + new):** leak-safe split with perceptual-hash dedup +
`leakage_ok` assertion; Cohen's/Fleiss' kappa + review-master CSV + save loop;
per-detection crop + annotated image; Grad-CAM figure disclaimer; expanded test
suite covering the brief's critical paths; thesis-figure automation; final-test
audit gate.

---

## 6. Thesis contribution statement

> *A reproducible, uncertainty-aware system for four-class emotion recognition
> from children's drawings that (a) compares objective-feature, deep, and
> validation-selected multimodal-fusion classifiers under identical leak-safe
> splits and multiple seeds; (b) quantifies the complementary value of
> handcrafted drawing features via ablation; (c) improves reliability through
> validation-fitted calibration and explicit uncertainty; and (d) delivers an
> evidence-traceable, strictly non-diagnostic clinical decision-support layer,
> kept architecturally separate from the classifier, with bilingual reports and a
> psychologist review workflow.*

---

## 7. Recommended research questions (refined)

RQ1 Which model family (objective-feature / deep / fusion) maximises validation
macro-F1 for 4-class drawing emotion recognition?
RQ2 Do objective drawing features add value **beyond** deep representations
(fusion vs best single modality)?
RQ3 Which fusion strategy (scaled-concat / PCA / MLP) is most effective and least
overfitting-prone?
RQ4 How much does temperature-scaling calibration reduce ECE / Brier / NLL?
RQ5 How stable are results across seeds [42,123,2026]?
RQ6 Do uncertainty + explainability improve the safety/usefulness of psychologist
review (qualitative + review-agreement)?

---

## 8. Experiment matrix (all: seeds 42/123/2026, select on valid macro-F1, test locked)

| ID | Family | Models | Purpose |
|---|---|---|---|
| E1 | Objective features | LR, linSVM, rbfSVM, RF, ExtraTrees, HistGB | baseline + RQ1/RQ2 |
| E2 | Deep transfer | ResNet18 (primary); MobileNetV3 (light comp) | RQ1 |
| E3 | Frozen embeddings | CLIP/ResNet embedding + linear head | RQ1/RQ2 |
| E4 | **Fusion (primary)** | scaled-concat / PCA / MLP | RQ2/RQ3 |
| E5 | Calibration | temp-scaling on E2/E4 | RQ4 |
| E6 | Ablations | −colour / −composition / −stroke / −shape; feat-only vs deep-only vs fusion | RQ2 |
| E7 | (optional) ViT arm | one ViT/ConvNeXt | RQ1 ceiling, only if seed-stable |
| S1 | Supplementary | semantic/CLIP-text features | explicitly separate, never in primary |

Prefer these **well-justified** experiments over a large sweep.

---

## 9. What requires the real dataset / GPU / psychologist (cannot be faked here)

- **Real dataset:** dataset validation counts, dedup/leakage report, all training,
  calibration, evaluation, confusion matrices, example galleries. *(Drive folder
  is not reachable from my environment; runs happen on your machine/Colab.)*
- **GPU:** E2/E4/E7 training within reasonable time (Colab T4+).
- **Psychologist:** real review labels → real Cohen's/Fleiss' kappa (the code is
  built; it must never be run on fabricated ratings).

No metrics will be invented; unexecuted experiments are marked as such.

---

## 10. Is this genuinely MSc-level? — Yes, conditionally.

The v3 design **is** MSc-grade: clear contribution, correct experimental method,
calibration+uncertainty, enforced safety separation, reproducibility. To reach
**submission-ready**, the remaining work is: finish/verify feature reality and the
deep smoke path, unify reports, expand tests to the critical-path list, automate
thesis figures, add the review-save loop + kappa, and — the irreducible part —
**run the frozen experiment suite on the real dataset with a GPU and collect real
psychologist reviews.** None of that requires abandoning the current direction; it
requires completing and executing it.
