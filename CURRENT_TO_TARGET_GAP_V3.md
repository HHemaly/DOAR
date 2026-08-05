# Current-to-Target Gap (v3) — DOAR-TRACE Phase 1.5

**Status: post-implementation record.** Supersedes `CURRENT_TO_TARGET_GAP_V2.md`
(Phase 1) — that document's gaps are re-assessed here against what Phase
1.5 actually closed, not re-derived from scratch.

## Phase 1 audit findings (Section 3 of the Phase 1.5 task, answered before implementation)

1. **Which Phase 1 components genuinely worked**: `trace_evidence.py`
   (schema only, never wired into a live path), `objective_features.json`
   persistence (real, wired into every case), `structured_analysis.json`
   generation (real, but structurally flawed — see #6), `claim_verifier.py`
   /`judge_schemas.py` (real logic, but only callable ad hoc, never
   automatic — see #2).
2. **Callable only manually vs. wired into every case**: `claim_verifier.py`
   and `judge_schemas.py` were fully real but **never invoked by
   `case_output.py`** — a case directory never got a `judges_v2.json`,
   `generated_claims.json`, or `verification_report.json` unless a
   developer called the functions directly in a Python shell. **Fixed**:
   `case_output.py::finalize_case` now calls both automatically for
   every case (Phase 1.5 Section 8).
3. **Why real combined hypotheses could not occur**: `structured_report.py`'s
   `aggregate_candidate_themes` grouped by a bespoke, 1:1
   `target_construct` per rule — so a SINGLE triggered rule already
   formed a "group" and was immediately shown as a candidate theme (e.g.
   `coverage_full` alone → `"self_esteem"`). There was no way for two
   independent rules to converge because no two rules ever shared a
   construct by design. **Fixed**: the 12-construct system + Level A/B/C
   split (Phase 1.5 Section 6) — a single rule now structurally cannot
   reach Level C.
4. **Every place that assumed exactly 19 rules**: `rule_schema.py::RULE_PROVENANCE`
   (a hard-coded dict of exactly the 19 production rule IDs,
   `load_rules_v2()` raises if the production registry ever has a rule
   not in this dict) and the Phase 1 `registry_v2_build.py` (iterated
   `RULE_PROVENANCE.items()` directly, so it could only ever emit 19
   rows). **`rule_schema.py` is intentionally left unchanged** — it is
   the real, tested provenance layer for `evidence_rule_engine.py`
   (unrelated to this task) and still correctly describes the 19
   production rules. The Phase 1.5 `registry_v2_build.py` is a full,
   independent rewrite that reads from `source_catalog_build.py` instead.
5. **Every place that hard-coded the Arabic PDF as sole source**:
   `rule_schema.py::SOURCE_PDF` (unconditionally assigns the Arabic PDF
   to `source_document` for all 19 rules — again, correct and left
   unchanged, since it genuinely is the only source for those 19) and
   the Phase 1 `registry_v2_build.py`'s hard-coded
   `"source_pdf": "التحليل النفسي للصور.pdf"` top-level field (**fixed**:
   Phase 1.5's registry now lists `source_documents` plural, sourced
   from `source_catalog_build.py`'s real per-entry `source_document` field).
6. **Was a single rule displayed as a candidate theme?** **Yes, confirmed
   directly**: a synthetic full-page image reproducibly produced
   `candidate_drawing_level_themes: [{"target_construct": "self_esteem", ...}]`
   from `PSY_AR_SIZE_FULL_015` alone. **Fixed and regression-tested**
   (`tests/test_structured_report.py::CombinedHypothesesTests::test_single_rule_alone_never_produces_a_combined_hypothesis`).
7. **Could the output misleadingly present "self_esteem" as stronger than
   intended?** **Yes** — the old Parent-view UI rendered this theme in a
   section titled "Possible drawing-level themes" with an expander
   labeled `self_esteem (question_generating)`, which reads as a
   drawing-level finding despite the low `allowed_output_level`. **Fixed**:
   the same rule now renders only in the new Section 3 ("Individual rule
   suggestions"), explicitly separate from Section 2 ("Combined patterns,
   levels 2-4 only"), which stays empty for this exact case.
8. **Evidence-schema relationship — adapter/facade vs. risky merge?**
   Confirmed the adapter approach (Phase 1's `trace_evidence_adapters.py`)
   remains the right call: merging `evidence_schema.py` and
   `trace_evidence.py` would have touched 493+ tests depending on
   `evidence_schema.py`'s exact field names for zero functional gain.
   Not revisited in Phase 1.5 — no new pressure to merge arose.

## Gaps Phase 1.5 closed

1. `child_drawing_rules_compiled.pdf` ingested — 48 English rows
   catalogued, 22 new registry-v2 rules added.
2. Both PDFs fully catalogued in a single, explicit, relationship-aware
   `source_rule_catalog.json` (67 rows, 31 relationships) — no silent
   merging anywhere.
3. 12-construct system replacing 1:1 rule-to-construct mapping.
4. Level A/B/C output separation with a real, tested, ordinal (0-4)
   aggregation policy — the single-rule-as-theme bug is fixed and
   regression-tested.
5. The calibrated expressive-content model wired in as an independent
   evidence family with mandatory non-diagnostic wording.
6. `judges_v2.json`/`generated_claims.json`/`verification_report.json`
   now written automatically for every case.
7. `aggregation_judge` is now real (independently re-verifies the
   construct policy per case), not a stub.
8. Parent/Technical views rewritten to the required 12-section order,
   surfacing source-catalog coverage, construct mapping, judge verdicts,
   and claim verification.
9. Generated (not hand-written) expert-review instruments for rules,
   constructs, and report sentences.

## Gaps that remain (honestly, not silently)

1. **No new detector was built.** 35 of 41 registry-v2 rules remain
   `allowed_output_level: disabled`. Only the original 6 tier-1 rules
   are individually executable.
2. **Only one construct (`fear_or_insecurity_pattern`) can reach Level C
   on real data today**, and only via the expressive model, not two
   registry rules (see `docs/AGGREGATION_POLICY.md`).
3. **`language_judge` remains not_implemented** — no LLM exists in this
   phase, by explicit instruction.
4. **`detection_judge`/`relation_judge` remain not_implemented** — no
   object detector or spatial-relationship extractor exists.
5. **Construct mappings (Section 5) are this session's judgment calls**,
   not independently reviewed — the generated `construct_review_form.csv`
   exists precisely to close this gap, but no review has happened yet.
6. **`static_proxy` rules' underlying features
   (`stroke.intensity_proxy`, `stroke.fragmentation`) are real and
   already computed for every case, but not wired to any rule
   evaluator** — the single fastest remaining lever for expanding real
   executable coverage without new detector work (see recommendation in
   `PHASE1_5_IMPLEMENTATION_REPORT.md`).
7. **No object detector, no Gemini/LLM integration, no expressive-model
   retraining** — all explicitly out of scope for this phase, per
   instruction, and none was attempted.
