# DOAR Research Experiment Log

Chronological log of every reproducible research experiment run against DOAR.
One entry per experiment; append-only (do not rewrite past entries -- add a
follow-up entry if a past experiment needs correction/extension).

---

## 2026-08-19 -- E6: Rule / Evidence Aggregation Ablation

**Checkpoint base:** `checkpoint/human-interaction-v1`
**Directory:** `experiments/E6_rule_aggregation/`
**Status:** Complete.

**What was done:**
1. Data-readiness audit of all 15 `DEVELOPMENT_SET_15.json` cases (see
   `experiments/E6_rule_aggregation/PROTOCOL.md` Section 1) -- 15/15 usable,
   0 excluded, cohort frozen before any strategy comparison.
2. Implemented 5 rule/evidence aggregation strategies (R1-R5) as pure
   re-aggregations of the already-computed `EligibleAtomicRuleMatch` list
   each cached case already has -- zero live Gemini calls, zero changes to
   any frozen scientific file.
3. R6 (weighted aggregation) deliberately NOT implemented -- rule-strength
   fields (`confidence_ceiling`, `evidence_strength_as_written`) are not
   consistently/numerically defined for 44% of rules; implementing R6 would
   require inventing weights (post-hoc tuning), explicitly out of scope.
4. Evaluated every strategy's every triggered interpretation against 6
   independent structural-violation criteria (duplicate-family inflation,
   cross-domain convergence, insufficient independent evidence, ineligible-
   evidence use, unverified-evidence use, ignored contradiction) -- R4 was
   never treated as ground truth.
5. Full statistics: Wilson CIs, case-resampling bootstrap paired-diff CIs,
   exact McNemar (Holm-corrected), Cochran's Q, risk-difference/odds-ratio
   effect sizes.
6. Generated all required raw/statistics/tables/figures/error_analysis/
   thesis/paper outputs (see `experiments/E6_rule_aggregation/
   artifact_manifest.json` for the full list).

**Headline result:** R1/R2/R3/R5 each had a 100% structural-violation rate
among their own triggers on this cohort (R1/R5: insufficient independent
evidence; R2: duplicate-family inflation on its 1 trigger; R3: cross-domain
convergence on all 3 of its triggers). R4 (current DOAR) triggered on 0/15
cases -- never over-claimed, but also never exercised its true-positive path
(the cohort contains no case with >=2 independent families within one
domain).

**Decision:** KEEP current aggregation (R4). See `DECISION_LOG.md`.

**Key limitation carried forward:** the 15-case development cohort cannot
demonstrate R4 correctly ACCEPTING genuine convergence -- only its (perfect)
rejection of unsupported convergence was tested. See `KNOWN_LIMITATIONS.md`.

**Scientific files touched:** none. `RULE_EVIDENCE_MATRIX.csv`,
`RULE_RELATIONSHIP_GRAPH.json`, `CONCERN_DOMAIN_MAP.json`, `concerns.py`,
`reasoning_chain.py`, `drawing_synthesis.py` are all unmodified;
`checkpoint/psychologist-prototype-v1` and `checkpoint/human-interaction-v1`
are both untouched.

**Note (2026-08-20):** This run is now labeled E6-PILOT / provisional. See
the E6-EXPANDED entry below for the corrected-methodology, expanded-cohort
follow-up. This entry is retained unmodified as the historical record.

---

## 2026-08-20 -- E6-EXPANDED: Rule / Evidence Aggregation Ablation (corrected methodology, expanded cohort)

**Checkpoint base:** `checkpoint/human-interaction-v1`
**Directory:** `experiments/E6_rule_aggregation_expanded/`
**Status:** Complete for available quota (34/74 candidate cases processed;
40 remain unprocessed due to a Gemini API quota limit hit mid-run; disclosed,
not silently dropped).

**Why this run exists:** E6-PILOT's 15-case result was reviewed and found to
have four methodological issues that needed correcting BEFORE any cohort
expansion could be scientifically meaningful (see
`experiments/E6_rule_aggregation_expanded/PROTOCOL.md` for the full audit
trail of each).

**What was done:**
1. **Rule-eligibility governance audit (HIGH PRIORITY, done first, before
   any other change).** Traced the frozen production pipeline
   (`reasoning_chain.py::check_visual_preconditions` /
   `build_eligible_matches`, `drawing_synthesis.py::
   build_literature_linked_associations` / `build_overall_synthesis`) line
   by line for any filter on a rule's own `allowed_output_level` field.
   **Found: none exists anywhere in the synthesis pipeline.** Confirmed with
   concrete evidence: 13 of E6-PILOT's own 24 matched associations (54%)
   were rules marked `disabled` in `RULE_EVIDENCE_MATRIX.csv` (31 of 41
   rules are `disabled`; only 10 are `individual_heuristic_only`). Per
   instruction, this is reported as a scientific-governance defect and left
   UNMODIFIED in production; E6's own external re-aggregation layer applies
   the correct `allowed_output_level == "individual_heuristic_only"` filter
   for its own comparison only.
2. **Contradiction redefinition.** Separated true literature contradiction
   (`literature_level_contradictions`, 1 topic, tied to a rule that is
   itself `disabled` -- so structurally 0% once the eligibility filter is
   applied) from contextual/cultural limitation (`context_limited_by_edges`,
   6 edges -- no longer counted as a contradiction) from
   requires/precondition limitation from duplicate/same-family relationship
   (`same_evidence_family_edges`, 10 edges).
3. **Fair R1-R5 domain scope.** R1 and R2 restricted to the same
   positive/concern-domain scope R3/R4/R5 already use, isolating exactly one
   aggregation variable per strategy for the primary comparison. Broader,
   all-domain variants (`R1_broad`, `R2_broad`) retained as clearly-labeled
   secondary stress tests only.
4. **Non-circular outcome definitions.** "Fewer than two independent
   evidence families" is no longer counted as a structural error -- it is
   R4's own policy choice, now tracked separately as
   `policy_disagreement_vs_r4` (`tables/E6X_T5_policy_disagreement.csv`).
   "Supported coverage" now reflects only genuine, strategy-independent
   evidence errors (`tables/E6X_T6_structural_failures.csv`):
   duplicate-family inflation, cross-domain pooling, disabled-rule use,
   unavailable page-dependent evidence, unverified-evidence misuse, ignored
   true contradiction.
5. **Page assessability fix.** `raw/case_level_summary.csv` now saves the
   actual per-case `page_assessable` value computed from cached evidence,
   not a hardcoded `True`.
6. **Figure reproducibility fix.** All jitter/plotting seeds are explicit
   fixed integers (`RNG_SEED = 20260820`; per-strategy `JITTER_SEEDS`
   dict), not Python's `hash()`.
7. **Cohort expansion.** Built `raw/cohort_manifest.csv` (80 rows: 15
   E6-PILOT originals, 59 new candidates, 6 reserved locked-benchmark
   images never processed). Ran live Gemini perception
   (`scripts/e6_expand_run_live_perception.py`, frozen Observer=
   `gemini-3.6-flash` / Verifier=`gemini-3.5-flash-lite`, unchanged from
   production) on the 59 new candidates: 19 succeeded before a persistent
   HTTP 429 quota exhaustion stopped the run cleanly after 5 consecutive
   failures (checkpointed, resumable; resume attempt confirmed quota still
   exhausted). Final usable cohort: 34 cases (15 original + 19 new).
8. Re-ran the 5 primary strategies (R1-R5) plus 2 secondary stress-test
   variants (R1_broad, R2_broad) against the corrected, non-circular
   structural-violation criteria. Full statistics: Wilson CIs, case-
   resampling bootstrap paired-diff CIs, Cochran's Q (hand-implemented --
   `statsmodels` unavailable in this venv), Holm-corrected pairwise tests,
   risk-difference/odds-ratio effect sizes.
9. Computed all 13 Section-D cohort-adequacy diagnostics
   (`tables/E6X_T2_cohort_diagnostics.csv`).
10. Generated all required raw/statistics/tables/figures/error_analysis/
    thesis/paper outputs (see `experiments/E6_rule_aggregation_expanded/
    artifact_manifest.json`). F9 (same-domain convergence examples) was
    explicitly skipped with a recorded null-result file
    (`E6X_F9_SKIPPED_no_r4_triggers.txt`) since 0 qualifying cases exist.

**Headline result (N=34):** R1 (58.8% triggered, 58.8% non-circular
supported coverage, 0% structural violation) and R5 (50.0% / 50.0% / 0%)
now show genuine coverage once the circular outcome definition is corrected
-- a marked change from E6-PILOT's artifactual 0% supported coverage for
both. R2 and R4 are numerically IDENTICAL (0/34 both; 0 discordant pairs),
because 0/34 cases contain any duplicate-family rule pair, so family
deduplication never had a case where it could matter. R3 triggered on 3/34
cases (8.8%) and **100% of its own triggers were cross-domain**, replicating
E6-PILOT's finding at more than double the sample. Cochran's Q across
R1/R2/R3/R5 = 48.324 (df=3, p<0.001). **Critically, 0/34 cases contain >=2
independent evidence families within a single concern domain** -- R4's
positive (accept) path remains completely unexercised even after cohort
expansion; 5/34 cases (14.7%) do show multi-family convergence, but only
when pooled ACROSS domains (which R3, not R4, would accept, and which is
flagged as a structural violation).

**Decision:** KEEP R4 PROVISIONALLY. See `DECISION_LOG.md`.

**Key limitation carried forward (strengthened, not resolved):** even at
34 cases, no case has ever given R4 the opportunity to correctly ACCEPT a
genuine same-domain, multi-family pattern. See `KNOWN_LIMITATIONS.md`.

**E6-STRESS cohort:** not built. The Section-D diagnostics already show 0
duplicate-family pairs, 0 true-contradiction cases, and 0 same-domain
multi-family cases across the full 34-case cohort -- there is no naturally-
occurring material beyond the 5 cross-domain / 3 R3-triggered cases already
captured in the main tables to select into a separate stress cohort. Noted
here as a reasoned decision, not a silent omission.

**Scientific files touched:** none. `RULE_EVIDENCE_MATRIX.csv`,
`RULE_RELATIONSHIP_GRAPH.json`, `CONCERN_DOMAIN_MAP.json`, `concerns.py`,
`reasoning_chain.py`, `drawing_synthesis.py`, `DEVELOPMENT_SET_15.json` are
all unmodified; `checkpoint/psychologist-prototype-v1` and
`checkpoint/human-interaction-v1` are both untouched. No commit/push made.

---

## 2026-08-20 -- E6-EXPANDED Phase 2: R4 Feasibility Audit + Production Eligibility-Governance Fix

**Checkpoint base:** `checkpoint/human-interaction-v1`
**Directory:** `experiments/E6_rule_aggregation_expanded/` (additions),
`src/doar/reasoning_chain.py` + `src/doar/drawing_synthesis.py` (production
fix), `tests/test_reasoning_chain.py` + `tests/test_drawing_synthesis.py`
(regression tests).
**Status:** Complete.

**Why this phase exists:** before spending further Gemini API quota on
E6-EXPANDED cohort expansion, this phase asked whether R4's positive
(same-domain, ≥2-independent-family) path is even structurally reachable
under the current 41-rule registry, and separately, whether the rule-
eligibility governance defect the earlier phase reported (not fixed) but
never remediated in production should now be corrected.

**What was done:**
1. **Static R4 feasibility audit** (no cases, no API calls) --
   `scripts/build_registry_feasibility.py` /
   `tables/E6X_T8_registry_feasibility.csv` /
   `REGISTRY_FEASIBILITY_REPORT.md`. Found: exactly ONE of the 8 registry
   concern domains (`anxiety_or_stress_related`) currently has 2 enabled
   rules from 2 distinct evidence families (`PSY_AR_SIZE_SMALL_016` +
   `EN_COMPILED_LINE_LIGHT_PRESSURE_031`) -- R4 is reachable, but only
   through this one narrow path, never through any other of the 8 domains.
2. **Production eligibility-governance fix.** Snapshotted the before-state
   (`eligibility_fix/BEFORE_STATE_test_baseline.txt`: 95 tests passing;
   `eligibility_fix/BEFORE_STATE_defect_demonstration.txt`: concrete proof
   that 7/7 semantic-path matches for a representative fixture were
   `disabled` rules). Added regression tests proving the defect
   (`EligibilityGateRegressionTests` in `test_reasoning_chain.py`,
   `EligibilityGateEndToEndRegressionTests` in `test_drawing_synthesis.py`)
   -- these fail against the unfixed code. Then added the smallest correct
   gate: `reasoning_chain.build_eligible_matches` and
   `drawing_synthesis.build_deterministic_eligible_matches` now skip any
   `satisfied` precondition check whose rule's `allowed_output_level !=
   "individual_heuristic_only"` (new `rc.ELIGIBLE_OUTPUT_LEVEL` constant).
   `check_visual_preconditions`/`check_deterministic_preconditions`
   themselves are unchanged -- the Technical/debug view
   (`clinician_review_app.py`'s `bundle["checks"]`) still shows every
   disabled rule's raw precondition status; only PROMOTION into governed
   evidence is gated. Updated 6 existing tests whose fixtures had
   (unknowingly) relied on disabled-rule matches to instead use hand-
   constructed `EligibleAtomicRuleMatch` objects for their real purpose
   (dedup/aggregation/hypothesis/package logic). Targeted test run after
   the fix: 166 passed, 1 skipped across `test_reasoning_chain.py` +
   `test_drawing_synthesis.py` + `test_clinician_review_app.py`.
   `RULE_EVIDENCE_MATRIX.csv`/`RULE_RELATIONSHIP_GRAPH.json`/
   `CONCERN_DOMAIN_MAP.json` untouched throughout.
3. **Before/vs/after impact analysis** on the 34 already-cached
   E6-EXPANDED cases (zero new API calls) --
   `scripts/build_eligibility_impact_analysis.py`,
   `raw/eligibility_before_after_per_case.csv`,
   `tables/E6X_T9_eligibility_fix_impact.csv`,
   `figures/E6X_F10_before_after_disabled_rule_impact.{png,pdf,csv}`,
   `eligibility_fix/ELIGIBILITY_FIX_IMPACT_REPORT.md`. Found: 19/34 cases
   (55.9%) affected, 27 disabled matches removed total, 9 cases'
   `overall_synthesis.level` changed (always toward less evidence, never
   more -- verified via an in-script assertion that held for all 34
   cases), and **1 case (`p2b_0007`) lost a real candidate hypothesis**
   that had been partly generated from a disabled rule
   (`PSY_AR_GEOMETRY_008`) -- a genuine, previously-hidden false positive
   in production, now corrected.
4. **Predeclared E6-STRESS-TARGETED screen** (Section 5 decision) --
   `scripts/screen_e6_stress_targeted.py`,
   `eligibility_fix/E6_STRESS_TARGETED_screen.csv`. Screened all 34 cached
   cases' already-computed deterministic features (no new API calls) for
   the one reachable combination found in step 1
   (`bounding_box_coverage<=0.20` AND `stroke.intensity_proxy<=
   INTENSITY_PROXY_LIGHT_THRESHOLD`). Result: 0/34 meet both; more
   specifically 0/34 even satisfy the size-small condition alone
   (`bounding_box_coverage` is at/near 1.0 for nearly every case in this
   corpus -- plausibly a photography/cropping artifact of how these images
   were captured, not evidence about drawing content).

**Headline result:** R4's positive path is NOT structurally impossible
(contradicting a premature reading of E6-EXPANDED's 0/34 result), but it
is reachable through exactly one narrow rule-pair in one domain, and that
exact pair was screened for (at zero API cost) and found absent in all 34
cached cases -- with the size-small half of the pair being essentially
unmeasurable in this image corpus specifically. The separate production
defect this phase set out to evaluate was confirmed real and is now fixed,
with measured, disclosed impact including one genuine corrected false
positive.

**Decision:** Option C (predeclared, prerequisite-based targeted screen,
not further blind cohort expansion) -- see `E6X_PHASE2_DECISION.md` and
`DECISION_LOG.md`.

**Scientific files touched:** `RULE_EVIDENCE_MATRIX.csv`,
`RULE_RELATIONSHIP_GRAPH.json`, `CONCERN_DOMAIN_MAP.json`,
`DEVELOPMENT_SET_15.json` all unmodified. `src/doar/reasoning_chain.py`
and `src/doar/drawing_synthesis.py` WERE modified (the eligibility gate
fix) -- this is a production CODE fix, not a change to any rule's content,
domain, family, or threshold; both checkpoint tags untouched. No
commit/push made.
