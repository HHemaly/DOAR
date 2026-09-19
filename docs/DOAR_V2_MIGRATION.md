# DOAR V2 migration catalog

## 0. Baseline recorded before any V2 ontology work

- **Branch**: `claude/doar-v2-evidence-foundation-klkn1m` (the branch name
  this environment's own instructions designated for this task; already
  existed at session start).
- **Baseline commit (V1, for future V1-vs-V2 comparison)**: `247562d`
  ("Add training experiments and evidence pipeline") — `HEAD` of this
  branch at the moment this milestone's work began. `git status` was
  clean (no uncommitted changes) before any file in this milestone was
  touched.
- **Baseline test run** (after installing this environment's missing
  runtime dependencies — `numpy`, `Pillow`, `pytest`, `scikit-learn`,
  `pandas`, `joblib`; none of `torch`/`opencv-python-headless`/
  `streamlit` were installed, so deep-learning/CV/UI-dependent tests
  correctly skip rather than fail): **471 tests, 2 pre-existing
  failures, 0 errors, 47 skipped.** The 2 pre-existing failures
  (`test_label_provenance.py::test_recognizes_known_class_folder`,
  `test_thesis_cross_family.py::test_all_three_new_leaderboard_types_generate_figures_when_present`)
  are unrelated to evidence/rules/features and were not introduced by
  this milestone — confirmed present on `247562d` before any new file
  was added. See §5 for the post-milestone re-run.

## 1. KEEP (working, validated by execution, not touched)

| Component | Why |
|---|---|
| `src/doar/dataset.py`, `leakage.py`, `partition.py`, `label_provenance.py` | Real leakage/duplicate/partition machinery, exercised across Phases 1/7A/7B with real dataset counts. Out of scope for an evidence/rule-architecture milestone; touching it risks the dataset-safety guarantees the thesis depends on. |
| `src/doar/models.py`, `deep/*` (trainers, registry, embeddings, embedding_classifier) | The Happy/Sad/Angry/Fear classifier arm. Kept fully independent of rule reasoning — verified: `rules.py` never reads a classifier prediction to decide `weak_support`; the only place a classifier prediction reaches rule-adjacent code is `concerns.py`'s (disabled) convergence check, which treats it as one of ≥2 required independent sources, never a rule trigger by itself. |
| `src/doar/features.py` (59 features), `evidence_adapter.py` (6 canonical + 10 declared-unavailable) | See `docs/DOAR_V2_FEATURE_AUDIT.md` — zero interpretation mixed into extraction anywhere. |
| `src/doar/evidence_schema.py`, `rule_schema.py`, `evidence_rule_engine.py`, `evidence_adapter.py`, `evidence_pipeline.py` | The pre-existing "evidence pipeline" (see `DOAR_V2_ARCHITECTURE.md` §0 for why this is not the same thing as this milestone's "V2"). Real, tested (`tests/test_evidence_rule_engine_v2.py`, 396 lines), wired to `main.py evidence-pipeline`. Reused via adapters, not rewritten. |
| `src/doar/concerns.py` | Disabled by design (`CONCERNS_ENABLED = False`), correctness-verified. Not touched — no reason to. |
| `src/doar/qa.py`, `english_report.py`, `judges.py` | Deterministic, evidence-grounded, tested; `judges.py`'s diagnostic-language guard is reused as-is by the evidence pipeline's own tests. |
| `src/doar/human_review.py`, `phase7b_review_app.py` | Real, tested, blind human-review tooling for the dataset-duplicate question — unrelated to feature/rule architecture. |
| `tests/` (pre-existing 471 tests) | All of it — none of the new ontology work required modifying an existing test. |

## 2. ADAPT (kept, wrapped/extended for V2, not rewritten)

| Component | What changed | What did not |
|---|---|---|
| `features.py`'s 59 objective features | Declared `ADAPT` in `artifacts/feature_audit.csv`; a mechanical wrapping pattern into `FeatureObservation` now exists (`ontology/legacy_adapters.py::evidence_item_to_observation`) but was only actually applied to the 6 features `evidence_adapter.py` already promotes — wrapping all 59 is `NEXT`, not done here (see §4), to avoid touching a tested module wholesale in one milestone. | The extraction code itself: zero lines of `features.py` changed. |
| `evidence_adapter.py`'s 6 canonical `EvidenceItem`s | Converted into `FeatureObservation`s and fed into the new `FeatureStore`, with `composition.centroid_normalized` additionally decomposed into two derived scalar features (`composition.centroid_x`/`_y`) so ontology `FeatureCondition`s can express simple thresholds. | `evidence_adapter.py` itself: zero lines changed. The decomposition lives entirely in the new adapter (`ontology/legacy_adapters.py::evidence_set_to_store`). |
| The 19-rule registry (`rule_schema.py`/`resources/psychology_sources/rules_registry.json`) | Re-expressed as declarative `ontology.rule.Rule`s with composable premises (`ontology/legacy_adapters.py::legacy_rule_to_ontology_rule`), each tagged `CLASSICAL_PROJECTIVE` with a mapped `EvidenceGrade`. | The registry JSON, `rule_schema.py`, and `evidence_rule_engine.py`: zero lines changed. Citations, wording, and confidence ceilings are read, never altered. |

## 3. LEGACY (isolated, not imported by any live path — re-confirmed this session)

| Component | Status |
|---|---|
| `legacy/src/*.py` (`emotion_heuristic.py`, `psychological_rules_v2.py`, `parent_ai_helper.py`, `visual_claim_validator.py`, `numeric_validator.py`, `psych_safety_validator.py`, `ocr_validator.py`, `claim_builder.py`, `arabic_translator.py`, `final_response_judge.py`, `output_manager.py`, `safety_policy.py`) | Re-verified this session: `grep -rn "import legacy\|from legacy"` across the repo returns **zero** matches from any live path (`src/doar/review.py`'s docstring says logic was "ported and adapted", i.e. reimplemented, not imported). `docs/LEGACY_AUDIT.md`'s existing findings stand (legacy colour-ratio extraction mixed page pixels into the calculation; legacy dataset code used `val` instead of `valid`). No action needed — already correctly isolated. |
| `src/doar/chat.py`'s `OpenAIChatProvider`/`GeminiChatProvider`-equivalent stubs | Re-verified this session (`grep -n "openai\|anthropic\|api_key\|requests\." src/doar/chat.py`): both provider stubs explicitly `raise NotImplementedError` — **no live VLM/LLM call exists anywhere in this repository.** This is the opposite of the risk the task warns about (`image -> VLM -> psychological conclusion`); nothing needed marking as legacy here because nothing live does this. |

**No new legacy code was created or discovered this session that needed
isolating.** The task's Section 12 concern (a VLM/heuristic directly
producing psychological interpretation from an image) does not apply to
any currently-live path in this repository.

## 4. NEXT (recommended follow-on work, not started)

In priority order, based on evidence from this audit (not assumption):

1. **Wrap the remaining 59 `features.py` features as `FeatureObservation`s
   and declare them in `config/feature_ontology.yaml`.** Mechanical
   (the pattern is proven for 6/65 already-canonical features); highest
   leverage because it makes the entire existing objective-feature set
   usable by the new declarative rule engine, not just the 4 composition
   features the 19-rule registry currently touches.
2. **Resolve the `features.py` vs. `evidence_adapter.py` naming
   duplication** for the composition features (`docs/DOAR_V2_FEATURE_AUDIT.md`
   §1) — cosmetic today, but will cause real confusion once both are
   declared in the same ontology file under different names for the
   same underlying number.
3. **Widen `semantic.face.eyes` and `semantic.object.symbol` to
   categorical features with explicit `allowed_values`** (already done
   in `config/feature_ontology.yaml` for these two) **before** building
   any face/eye or symbol detector — otherwise a real detector still
   could not distinguish `wide_eyes` from `stern_eyes`/`closed_eyes`, or
   `stars` from `hearts` (`docs/DOAR_V2_RULE_AUDIT.md` §4).
4. **A first real `EMPIRICAL_EXPRESSIVE` rule**, once a genuine
   peer-reviewed expressive-drawing study is identified and its citation
   filled in — the schema (`ontology/rule.py`) already refuses to accept
   this framework tag without one, so this is "find real support and add
   one rule," not a code change.
5. **A first genuinely context-dependent rule** (age- or task-gated),
   once one is sourced from real literature — `ContextRequirements` and
   `CaseContext` already exist and are tested, but exercised only by
   synthetic rules today (`docs/DOAR_V2_RULE_AUDIT.md` §1).
6. Everything explicitly deferred by the task's own Section 14 (new
   detectors, VLM/LLM integration, calibration, aggregation, scoring,
   reporting, Q&A/counterfactual agents, dashboard, literature
   expansion, graphology, age estimation) — correctly not started.

## 5. Post-milestone verification

Full test suite (pre-existing 471 + this milestone's new ontology tests)
was re-run after all changes in this document — see the top-level
summary in the task's final report for the exact command and result.
No pre-existing test was modified to make it pass; the 2 pre-existing
failures from §0 are unchanged (same 2, not more).
