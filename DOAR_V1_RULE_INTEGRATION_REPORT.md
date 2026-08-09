# DOAR V1 — Rule Integration Report

**Status: execution phase, complete.** Branch `feature/doar-phase2c-annotation-expansion`,
starting HEAD `d4fd655` (verified clean, matches `d4fd655`/`5d8fde3` expectation). Closes the one
missing connection in the DOAR reasoning path: validated visual detections now reach the REAL rule
engine, and any resulting rule trigger flows through the existing aggregation/synthesis machinery
into Parent View, Technical View, and Q&A — with zero new detector work, zero fine-tuning, zero new
rule ontology, and zero rules enabled "to make the demo richer."

## 1. Starting branch/HEAD

`feature/doar-phase2c-annotation-expansion` at `d4fd655` (DOAR MVP commit), with `5d8fde3` (Phase
2C.7 detector finalization) as its parent. `git status` clean before any change in this phase.

## 2. What was already working

- Real objective measurements (composition/colour/stroke/quality) → canonical `Evidence` records
  (`schemas.py`) → the historical rule engine (`rules.py::evaluate_rules`, `rules_registry.json`) and
  the v2 engine (`rule_engine_v2.py::evaluate_v2_rules`, 4 hardcoded placement/stroke-pressure rules)
  → `structured_report.py::build_structured_analysis` (Level A/B/C synthesis) → Parent View (exactly
  5 sections) / Technical View (11 sections) / Q&A (`qa.py`) / expert review
  (`clinician_review.json`) — all real, all tested, unchanged by this phase.
- The DOAR MVP's visual pipeline (`visual_evidence.py`): a real, automatic, image-only visual scan
  (Phase 2C.7's frozen detector policy + broad open-vocabulary extras) writing `detections.json` with
  full provenance and a `validation_status`/`evidence_status`/`rule_mapping_status` taxonomy, plus a
  *descriptive* `build_rule_evidence_trace` and Q&A with on-demand visual search.

## 3. Exact missing integration discovered

Confirmed by reading, not assuming:

- `rules.py::evaluate_rules` hardcodes every `tier_2_content_conditional` rule (the ones that would
  need object detection) to `"missing_detector"` **unconditionally** — the `evidence` list it
  receives is only used for concern-convergence bookkeeping, never to decide an individual rule's
  status.
- `rule_engine_v2.py::evaluate_v2_rules` evaluates exactly 4 hardcoded rule IDs, all driven by
  `objective_features` (placement/stroke-pressure). No code path anywhere referenced
  `detections.json` or `VisualFinding`.
- Every rule in `rules_registry_v2.json` with `observability_class == "static_detector"` (24 rules —
  eyes, animals, house, tree, hands, etc.) has `allowed_output_level: "disabled"` and
  `validation_status: "DETECTOR_UNAVAILABLE"` — including `EN_COMPILED_HOUSE_023`/
  `EN_COMPILED_TREE_024`, whose `observable` exactly matches two `VALIDATED_AUTOMATIC` Phase 2C.7
  targets.
- The MVP's `build_rule_evidence_trace` is purely descriptive (which rule_ids a label's curated
  `related_rule_ids` mention) — it never calls either real rule engine.

**Conclusion: visual findings did not enter the real rule engine at all.** The connection was real
and missing, not imagined.

## 4. Did visual findings previously enter the real rule engine? No.

Verified from source, confirmed by the new `RealRegistryTodayTests`/`RealRegistryTodayIntegrationTests`
test classes (which use the real, unmodified `rules_registry_v2.json`, not synthetic data).

## 5. What adapter/wiring was added

Reused the existing `schemas.Evidence` dataclass — **no third evidence schema invented**.

- **`visual_evidence.py`**: added `finding_id` (a stable, deterministic provenance id) to
  `VisualFinding`; `visual_finding_to_evidence`/`visual_findings_to_evidence` (the
  `VisualFinding → Evidence` adapter, preserving label/bbox/confidence/detector/checkpoint/
  prompt/source/query/`source_finding_id`/validation_status/rule_mapping_status/related_rule_ids);
  `rule_eligible_visual_evidence` (filters to `evidence_status == "validated_evidence"` only);
  `integrate_visual_findings_into_case` (the orchestration: converts findings → evidence, calls the
  real rule engine, merges results into `analysis.json`, re-runs synthesis). Wired as the last step of
  `run_and_persist_initial_scan` — **never** called by `search_visual` (on-demand search never
  triggers rule integration, per instruction).
- **`rule_engine_v2.py`**: added `evaluate_visual_object_presence_rules` — extends this SAME real
  engine (not a parallel one) to `static_detector` rules. Two independent, non-bypassable gates: (1)
  `allowed_output_level == "individual_heuristic_only"` in the *current* registry (never flipped by
  this session — see §7); (2) `rule_eligible=True` on the evidence record (`VALIDATED` findings only).
  Also fixed a real latent bug in the shared `_base_eval` helper (`confidence_ceiling` is `None` for
  18 not-yet-curated registry rules; `_base_eval` crashed on `float(None)` — the 4 original callers
  never hit this since they're all fully curated). Fixed with a `None`-guard, zero behavior change for
  existing callers.
- **`case_output.py`**: added `resynthesize_case_with_visual_evidence` — re-runs `finalize_case`'s own
  synthesis tail (judges, `structured_analysis`, judges_v2, generated claims/verification) against the
  updated `analysis` dict, using the exact same functions `finalize_case` uses. Necessary because
  visual findings only exist *after* `analyze_image`'s synchronous call already ran `finalize_case`
  once. Deliberately never touches `detections.json` (would overwrite the real scan) or resets
  `clinician_review.json` (preserves an existing "submitted" review's status into the refreshed
  `judges.json`).
- **`doar_prototype_app.py`**: Technical View's "10. Visual object detections" section gained a
  second table — "Visual evidence actually used in rule reasoning" — cross-referencing
  `analysis["rule_evaluations"]` for rows tagged `visual_evidence_sourced=True`, showing the *real*
  status (`weak_support`/`missing_detector`) next to the pre-existing *descriptive* trace. This is the
  literal "detected" vs. "used in psychological reasoning" distinction Stage 6 asked for.

## 6. Canonical evidence flow

```
VisualFinding (label, bbox, confidence, detector, checkpoint, prompt, validation_status, ...)
    → visual_finding_to_evidence()          [ALWAYS, every finding, every status]
    → schemas.Evidence(evidence_id="ev_visual_<finding_id>", kind="visual_detection",
                        value={..., "rule_eligible": <True only if VALIDATED>})
    → analysis["evidence"] (merged, written to evidence.json)
    → rule_eligible_visual_evidence() filters to VALIDATED-only
    → evaluate_visual_object_presence_rules(rules_v2_by_id, ...) [registry-gated]
    → analysis["rule_evaluations"] (merged, written to rules.json, tagged visual_evidence_sourced=True)
    → resynthesize_case_with_visual_evidence()
    → structured_analysis.json (Level B individual_rule_suggestions / Level C combined hypotheses)
    → Parent View (parent_safe_wording only) / Technical View (full trace) / Q&A (qa.py's existing
      "rule"/"evidence" branches, unmodified, now see the merged lists automatically)
```

## 7. Rule-safety logic

- **VALIDATED** evidence (`evidence_status=="validated_evidence"`) → `rule_eligible=True` → the ONLY
  evidence `evaluate_visual_object_presence_rules` will ever consider.
- **EXPERIMENTAL/UNKNOWN/DISABLED_FOR_RULES** → `rule_eligible=False` → structurally invisible to the
  rule engine (filtered out before the matching loop even runs) — proven by
  `test_experimental_finding_never_triggers_even_an_enabled_rule`, which uses a registry with the gate
  **open** and still gets zero rows.
- **Registry gate**: `allowed_output_level` in `rules_registry_v2.json` is read, never written, by this
  phase. As of this session every `static_detector` rule is `"disabled"` — this phase does **not**
  flip that field. A validated match against a disabled rule produces `status="missing_detector"`
  (never a fabricated `"weak_support"`), with `missing_evidence=["rule_disabled_pending_registry_review:<rule_id>"]`.
- **No absence-reasoning**: the function never emits `"not_matched"` — a real negative/absence claim
  would require established detector *recall*, which this project has not validated for any target
  (Phase 2C.7's own hand-detector caveat: low recall means absence-of-detection must never be read as
  absence-of-object). "Nothing to say" always resolves to `missing_detector`, never a fabricated
  negative.

## 8. One concrete validated visual → rule trace

**Not legitimately available in production today** — see §9/§10 for why, and the honest reasoning
behind that being the correct answer, not a shortfall.

A **controlled, clearly-labeled test scenario** (`tests/test_visual_rule_integration.py`'s
`SyntheticGateOpenTests`, `_registry_with_house_enabled()`) proves the exact mechanism works
end-to-end: a deep copy of the real, current `rules_registry_v2.json` with *only*
`EN_COMPILED_HOUSE_023.allowed_output_level` flipped to `individual_heuristic_only` (every other field
real, curator-authored content) — a validated `house` finding then produces
`status="weak_support"`, `matched_evidence_ids=["ev_visual_vf_test_house_001"]`, and reaches
`structured_analysis.json`'s `individual_rule_suggestions` with the rule's own real
`parent_safe_wording` ("The drawing includes a feature (house) some sources associate with family
life, safety, privacy, or emotional openness...") — confirmed to contain **no** model name, checkpoint,
confidence number, bbox, or evidence ID.

## 9. Aggregation path

`analysis["rule_evaluations"]` (old registry + v2 4 rules + any visual-sourced rows) →
`structured_report.py::build_structured_analysis` (unmodified) → Level B `individual_rule_suggestions`
(one entry per triggered rule) and Level C `combined_drawing_level_hypotheses` (≥2 independent
evidence families converging, per `construct_registry.json` policy) → both already read by
Parent View section 3 ("Possible meaning") and Technical View. A visual-sourced `weak_support` row is
indistinguishable, from `build_structured_analysis`'s perspective, from any other triggered rule — no
special-casing was added or needed downstream of `rule_evaluations`.

## 10. Parent View behavior

**Unchanged, not redesigned.** Section 3 ("Possible meaning") reads `structured_analysis.json`'s
`individual_rule_suggestions`/`combined_drawing_level_hypotheses` — fields already limited to
`parent_safe_wording`, `possible_interpretation`, friendly evidence-family names, and source
citations. No model name, checkpoint, confidence number, bbox, or `rule_id`/`evidence_id` reaches this
rendering path (confirmed by `test_parent_safe_content_present_with_no_technical_internals_leaked`,
which checks the exact fields the app reads). This is a property of the *existing*
`structured_report.py`/Parent-View design, inherited for free — no new code needed.

## 11. Technical View behavior

"10. Visual object detections" now shows two tables: the pre-existing descriptive rule-evidence trace
("a rule mentions this label", `can_activate` = eligibility only) and a new "actually used in rule
reasoning" table (the real `weak_support`/`missing_detector` outcome, `matched_evidence_ids`,
`missing_evidence`) sourced directly from `analysis["rule_evaluations"]`. Every finding (validated,
experimental, unmapped, unknown) remains visible regardless of rule outcome.

## 12. Q&A behavior

**No changes to `qa.py` or `visual_qa.py` were needed or made.** Because the new evidence/rule rows
are merged into the *same* `analysis["evidence"]`/`analysis["rule_evaluations"]` lists every other
part of the system already reads, `qa.py`'s existing `"rule"` branch ("which rules...") and
`"evidence"` branch ("what evidence...") automatically surface visual-sourced rule IDs and
`ev_visual_*` evidence IDs with zero code change — proven, not assumed, by
`test_qa_can_explain_which_rule_used_which_evidence`/`test_qa_evidence_question_surfaces_the_visual_evidence_id`
against the gate-open scenario. Free-form "why did the system mention this" questions (no keyword
match in `qa.py`'s vocabulary) fall through to the existing grounded-refusal default — a pre-existing,
documented limitation, not something this phase introduced or was asked to fix.

## 13. Expert-review behavior

Unchanged. `resynthesize_case_with_visual_evidence` reads `clinician_review.json` (to preserve a
"submitted" status into the refreshed `judges.json`) but never writes to it, and
`integrate_visual_findings_into_case` never touches it either — confirmed by
`test_existing_submitted_review_status_survives_resynthesis`. An expert action never mutates
`rule_evaluations`/`evidence.json`/`detections.json`; it stays a separate, appended layer in
`clinician_review.json`'s own `history`.

## 14. Eye 170/43 vs. 169/42 explanation

Reconstructed precisely from real data (`scripts/phase2c7_evaluate_eye.py`'s exact loading calls),
not speculation. The dev/holdout **split** partitions all 213 reviewed pilot_ids: 170 dev + 43
holdout. The **evaluable metrics** (169/42) additionally require a stored *prediction* to exist for
that pilot_id (`usable_ids = set(pred_positive) & set(gt_status)`), and `pred_positive` is built from
`have_inference = {row["pilot_id"] for row in raw_rows}` — every raw-proposal-CSV row, across every
model and every target for that image. Two specific images produced **zero** rows across every
target/model entirely (Grounding DINO and OWLv2 both found nothing at all, for anything, on that
image) — `p2c6_0059` (dev, GT=absent) and `p2c6_0052` (holdout, GT=present). Confirmed **not** a
GT-uncertainty artifact (one is GT-absent, one is GT-present) and **not** a pipeline failure
(`n_failed=0`, `n_skipped_not_assessable=0` in the original batch run's own summary) — a benign
consequence of the raw-proposals CSV format only recording positive detections, with no explicit
"checked everything, found nothing" placeholder row. No rerun performed — documentation only, per
instruction.

## 15. Real end-to-end case result

`scripts/doar_v1_rule_integration_e2e_check.py`, real models, real image
(`outputs/phase2c1/private_images/p2b_0000.jpg`), no mocking anywhere:

All 10 steps passed. Key results:

| Step | Result |
|---|---|
| Objective measurements + baseline rules | 23 baseline `rule_evaluations`, 6 baseline evidence records (before the visual scan even ran) |
| Automatic visual scan | 37 findings: `person`/`face`/`tree`/`house` = VALIDATED, `animal`/`star`/`eye` = EXPERIMENTAL, 30 broad-scan extras (sun/moon/cloud/flower/bird/cat/dog/door/window/road/weapon-like) = UNKNOWN |
| **Real rule-engine outcome** | `evaluate_visual_object_presence_rules` produced exactly **2 rows** — `EN_COMPILED_TREE_024` and `EN_COMPILED_HOUSE_023` — both `status="missing_detector"` (never `weak_support`), `missing_evidence=["rule_disabled_pending_registry_review:<rule_id>"]`. **This is OUTCOME B**, confirmed on a real image with real, validated, rule-eligible detections that genuinely matched a rule's observable — the registry gate is the only reason nothing triggered, exactly as the audit (§3–4) predicted. Not fabricated, not a shortfall. |
| Canonical evidence | 37 `visual_detection`-kind Evidence records created (one per finding, every status) — full technical traceability regardless of trigger outcome |
| Q&A, existing evidence | "Is there a person in the drawing?" → `availability="available"`, grounded in the real stored finding |
| Q&A, rule linkage (zero new code) | "Which rules were evaluated?" → correctly lists `missing_detector: PSY_AR_EYES_WIDE_001, ...` via `qa.py`'s existing, unmodified "rule" branch |
| Q&A, on-demand search | "Is there a kite?" → not in initial evidence → live on-demand query → found (confidence 0.40, status UNKNOWN) → **confirmed** `rule_evaluations` unchanged (on-demand search never calls rule integration) |
| Expert review | Submitted (confirm, target=`person`) → `clinician_review.json` status="submitted" → `judges.json` module_availability correctly shows `detection="available"`, `clinician_review="submitted"` |
| Case reload | 38 detections, review status "submitted", 2 visual-sourced rule rows — all persisted, all reloaded correctly |

Case: `outputs/prototype_cases/v1_rule_integration_check_1786300628` (gitignored, not committed).

## 16. Limitations

- No `static_detector` rule is currently eligible to trigger in production — every such rule's
  `allowed_output_level` in `rules_registry_v2.json` is `"disabled"`, a pre-existing registry-curation
  decision this phase respects rather than overrides. Flipping it (e.g. for `EN_COMPILED_HOUSE_023`/
  `EN_COMPILED_TREE_024`, whose observable exactly matches a `VALIDATED_AUTOMATIC` target) is a
  content/clinical-governance decision for the registry curator, not an engineering one — the wiring
  is ready the moment that decision is made, with zero further code changes.
- `judges_v2.json`/`generated_claims.json`/`verification_report.json` ARE recomputed by
  `resynthesize_case_with_visual_evidence` (reusing `finalize_case`'s own functions), so they stay
  consistent with any visual-sourced rule row — but concern-profile aggregation (`concerns.py`) is
  *not* recomputed on this path; harmless today since `CONCERNS_ENABLED=False` project-wide regardless.
- `evaluate_visual_object_presence_rules` only fires for a rule whose `observable` *exactly* equals a
  detected label string — no synonym/paraphrase matching (matches the project's existing exact-match
  convention elsewhere, e.g. `detector_policy.py`'s own target keys).
- Free-form "why did the system mention this" Q&A questions (no keyword match) fall through to the
  pre-existing grounded-refusal default — not a regression, not addressed by this phase.

## 17. Tests

24 new tests across 3 new files (`test_visual_canonical_evidence.py`,
`test_rule_engine_v2_visual.py`, `test_visual_rule_integration.py`) covering all 14 requested
categories, plus a regression fix to 2 pre-existing test-helper fixtures (`finding_id` addition).

- `ruff check` on every new/modified file — all checks passed.
- `compileall` on every new/modified file — clean.
- Targeted suite (this phase's new/modified tests, 14 files) — **151 passed, 1 pre-existing skip, 0
  failed**.
- Full repository suite — **1435 passed, 7 pre-existing skips, 0 failed** (1400 baseline + exactly the
  35 new tests added this phase — 11 + 11 + 13 across the three new files), 2177s wall clock.
- Real end-to-end acceptance case (`scripts/doar_v1_rule_integration_e2e_check.py`) — **PASSED**, all
  10 steps, real models, real image, no mocking (§15).

## 18. Commit hash

`c61f3c2` on `feature/doar-phase2c-annotation-expansion` (parent: `d4fd655`).

## 19. Pushed?

**No. Nothing was pushed at any point in this phase.**
