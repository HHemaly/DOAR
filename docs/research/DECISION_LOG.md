# DOAR Architecture Decision Log

Append-only. One entry per architecture decision made (or explicitly
deferred) on the basis of a reproducible experiment. Only updated when an
experiment's result changes, or could change, DOAR's production code path --
not for every experiment (an experiment that concludes "no change" without
any live tension with production is logged in `EXPERIMENT_LOG.md` only,
unless the "keep" itself is a decision worth recording, as here).

---

## 2026-08-19 -- E6: Rule/evidence aggregation strategy -- KEEP current (R4)

**Experiment:** `experiments/E6_rule_aggregation/`
**Decision:** **KEEP** `drawing_synthesis.build_overall_synthesis`'s existing
same-domain, family-deduplicated, >=2-independent-family convergence rule
(R4), unchanged.

**Alternatives considered:** R1 (any eligible rule), R2 (raw rule count, no
family dedup), R3 (evidence-family convergence pooled across concern
domains), R5 (support ratio = matched/assessable independent families,
same domain scope as R4).

**Evidence:** On the frozen 15-case development cohort, every trigger
produced by R1, R2, R3, and R5 was structurally flawed (100% violation rate
within each strategy's own triggers): R1/R5 relied on a single evidence
family (below the independence floor) in every case they fired;
R2's one trigger double-counted two rules sharing one evidence family; all 3
of R3's triggers pooled evidence across unrelated concern domains. R4
triggered on 0/15 cases and consequently had 0% of any violation type.
Cochran's Q across R1/R2/R3/R5 = 28.364 (df=3, p<0.001) confirms the
strategies' trigger behavior differs sharply; full paired McNemar/effect-size
tables in `experiments/E6_rule_aggregation/statistics/`.

**Why KEEP and not CHANGE:** no alternative demonstrated a supported-coverage
gain that was not fully offset by an equal or larger over-claiming cost. The
research question was explicitly "best trade-off between coverage and
resistance to unsupported convergence" -- on the measured trade-off, R4
strictly dominates every alternative tested (0% coverage tied with all
others, but also 0% over-claiming vs. 100% for every alternative that fires
at all).

**Important caveat attached to this decision:** the cohort used contains
zero cases where R4 has the opportunity to correctly ACCEPT genuine
same-domain, multi-family convergence -- R4's demonstrated property here is
a perfect false-positive rate, not a validated true-positive rate. This
decision should be revisited once a cohort (or cohort subset) exists with
at least a handful of genuine multi-family same-domain cases, so R4's
sensitivity (not just its specificity) can be measured. See
`KNOWN_LIMITATIONS.md` for the specific data that would be needed.

**What would change this decision:** a future cohort in which (a) R4
demonstrates poor sensitivity (misses cases a domain expert would judge as
genuine convergence) while (b) another strategy (most plausibly R5, the
ratio-based variant, once assessable-family denominators are less sparse)
demonstrates comparable specificity with better sensitivity. Until such a
cohort exists, this decision stands.

**No frozen scientific file was modified to reach this decision.**
`RULE_EVIDENCE_MATRIX.csv`, `RULE_RELATIONSHIP_GRAPH.json`,
`CONCERN_DOMAIN_MAP.json`, `concerns.py::MIN_EVIDENCE`,
`drawing_synthesis.build_overall_synthesis`, and both checkpoint tags are
unchanged.

**Status (2026-08-20): superseded by the entry below, which reaffirms the
same KEEP decision on corrected methodology / expanded evidence, but
downgrades confidence from "KEEP" to "KEEP PROVISIONALLY." This entry is
retained as the historical record; do not delete.**

---

## 2026-08-20 -- E6-EXPANDED: Rule/evidence aggregation strategy -- KEEP R4 PROVISIONALLY

**Experiment:** `experiments/E6_rule_aggregation_expanded/`
**Decision:** **KEEP** `drawing_synthesis.build_overall_synthesis`'s existing
same-domain, family-deduplicated, >=2-independent-family convergence rule
(R4), unchanged, but **PROVISIONALLY** -- not treated as fully validated.

**Why this decision was reopened:** the prior E6-PILOT decision (above) was
reached on a cohort and methodology with four identified issues: a rule-
eligibility governance defect contaminating 54% of matched evidence, a
circular outcome definition, a domain-scope confound between R1/R2 and
R3/R4/R5, and a contradiction-type conflation. All four were corrected
(see `EXPERIMENT_LOG.md`'s E6-EXPANDED entry and
`experiments/E6_rule_aggregation_expanded/PROTOCOL.md`) and the cohort was
expanded from 15 to 34 cases before this decision was re-evaluated.

**Alternatives considered:** same five -- R1 (any eligible rule, domain-
scoped), R2 (raw rule count, family-deduplicated, domain-scoped), R3
(evidence-family convergence pooled across concern domains), R5 (support
ratio, same domain scope as R4). R1_broad/R2_broad (all-domain) retained
only as secondary stress-test baselines, not part of this decision's
primary evidence.

**Evidence (N=34, corrected non-circular metrics):** R1 triggered 58.8%
(20/34) with 0% structural violation -- genuine, non-circular supported
coverage, a marked change from E6-PILOT's artifactual 0%. R5: 50.0%
triggered, 0% structural violation, same pattern. R2 and R4 are numerically
IDENTICAL (0/34 both) -- family deduplication had no case in this cohort
where it could have mattered either way, since 0/34 cases contain any
duplicate-family rule pair at all. R3 triggered on 8.8% (3/34) and **100%
of its own triggers were cross-domain violations**, reproducing E6-PILOT's
finding at more than double the sample size -- treated as settled evidence
against cross-domain pooling for this rule set. Cochran's Q across
R1/R2/R3/R5 = 48.324 (df=3, p<0.001); full paired bootstrap/effect-size
tables in `experiments/E6_rule_aggregation_expanded/statistics/`.

**Why KEEP and not CHANGE:** as in E6-PILOT, no alternative demonstrates a
coverage gain not fully offset by an equal or larger structural-error cost,
now measured on the corrected, non-circular definition. R3 is the only
strategy with a nonzero structural-violation rate among the primary
strategies (100% of its own triggers), and that specific failure mode
(cross-domain pooling) is exactly what R4's domain-scoping already guards
against by design.

**Why PROVISIONALLY and not a full KEEP:** even after correcting the
methodology and more than doubling the cohort, **0 of 34 cases contain >=2
independent evidence families within a single concern domain** -- R4's own
trigger condition has still never had the opportunity to correctly ACCEPT a
genuine same-domain convergence pattern. This decision documents R4's
(continued, now better-evidenced) specificity -- it still never over-claims
-- but its sensitivity remains completely unmeasured. Per the governing
instruction for this phase: R4 must not be declared best merely because it
abstains most; the data available do not yet exercise both its safe-
rejection and its legitimate-acceptance paths, so full validation is
withheld.

**What would change this decision:** unchanged from the E6-PILOT entry --
a future cohort in which (a) R4 demonstrates poor sensitivity while (b)
another strategy demonstrates comparable specificity with better
sensitivity. Additionally, per Discussion Point 4 in
`experiments/E6_rule_aggregation_expanded/thesis/DISCUSSION_POINTS.md`, the
reframed open question is whether the missing signal requires more images
(cohort expansion, already attempted here without success) or more enabled
detectors within the SAME concern domain (only 10 of 41 rules are
currently enabled, spread thinly across six domains) -- the latter may be
the binding constraint, not sample size.

**No frozen scientific file was modified to reach this decision.**
`RULE_EVIDENCE_MATRIX.csv`, `RULE_RELATIONSHIP_GRAPH.json`,
`CONCERN_DOMAIN_MAP.json`, `concerns.py::MIN_EVIDENCE`,
`drawing_synthesis.build_overall_synthesis`, `DEVELOPMENT_SET_15.json`, and
both checkpoint tags are unchanged. No commit/push made.

---

## 2026-08-20 -- Rule-eligibility governance defect -- FIX (production code change)

**Experiment:** `experiments/E6_rule_aggregation_expanded/` Phase 2 (see
`E6X_PHASE2_DECISION.md`).
**Decision:** **FIX** the production eligibility-governance defect
reported (not fixed) in the two prior E6 entries above: add an explicit
`allowed_output_level == "individual_heuristic_only"` gate to
`reasoning_chain.build_eligible_matches` and
`drawing_synthesis.build_deterministic_eligible_matches` -- the two, and
only two, choke points where a satisfied visual/deterministic precondition
check becomes governed evidence (`EligibleAtomicRuleMatch`).

**Why this is a CODE decision, not a scientific-content decision:** the
fix touches `src/doar/reasoning_chain.py` and
`src/doar/drawing_synthesis.py` (production files previously described as
untouched by E6). This required its own explicit entry because it is the
first E6 phase to modify production code. It does NOT change any rule's
content, domain, family, threshold, or `allowed_output_level` value --
`RULE_EVIDENCE_MATRIX.csv`, `RULE_RELATIONSHIP_GRAPH.json`,
`CONCERN_DOMAIN_MAP.json` remain byte-for-byte unmodified. The fix makes
the CODE respect a field the registry already declared; it does not alter
what the registry declares.

**Evidence the defect was real, not a re-labeling exercise:** 13 of
E6-PILOT's own 24 matched associations (54%) were `disabled` rules,
confirmed by direct trace of `check_visual_preconditions` ->
`build_eligible_matches` -> `build_literature_linked_associations` ->
`build_overall_synthesis` finding no `allowed_output_level` filter at any
stage. Replaying the 34 already-cached E6-EXPANDED cases through both the
unfixed and fixed logic (zero new API calls) found: 19/34 cases (55.9%)
affected, 27 disabled matches removed, 9 cases' `overall_synthesis.level`
changed (always toward less evidence, never more -- machine-verified for
all 34), and **one case (`p2b_0007`) lost a real candidate hypothesis**
that had been partly generated from `PSY_AR_GEOMETRY_008` (a `disabled`,
`DETECTOR_UNAVAILABLE` rule with `confidence_ceiling=0.15`) -- a genuine,
previously-hidden false positive in deployed behavior, now corrected. Full
detail: `eligibility_fix/ELIGIBILITY_FIX_IMPACT_REPORT.md`.

**Why FIX and not "flag only" or "defer":** the defect was already fully
diagnosed and reported in the prior phase; this phase additionally proved
(a) a real behavioral consequence exists on real cached case data (not
merely a theoretical risk) and (b) the fix is small, local, and testable
in isolation (two functions, one shared constant, no rule-content edits).
Deferring further risked shipping known-incorrect clinician/parent-facing
hypotheses in any future case processed through the unfixed pipeline.

**Traceability preserved:** `check_visual_preconditions`/
`check_deterministic_preconditions` -- the raw, per-rule precondition
check list the Technical/debug view reads directly
(`clinician_review_app.py`'s `bundle["checks"]`) -- are byte-for-byte
unchanged; a disabled rule's satisfied precondition is still visible
there. Only promotion into governed scientific evidence is gated.

**Tests:** before-state snapshot preserved
(`eligibility_fix/BEFORE_STATE_test_baseline.txt`,
`BEFORE_STATE_defect_demonstration.txt`); new regression tests added
BEFORE the fix and confirmed to fail against the unfixed code
(`EligibilityGateRegressionTests`,
`EligibilityGateEndToEndRegressionTests`); 6 existing tests whose fixtures
had unknowingly relied on the defect updated to use hand-constructed
matches for their real (downstream-logic) purpose; 166 passed / 1 skipped
across the three directly-relevant test files after the fix.

**Checkpoint tags untouched.** `DEVELOPMENT_SET_15.json`,
`annotations/A1`, `annotations/A2` untouched. No commit/push made -- this
decision and its code change are staged in the working tree only, pending
the user's own review and explicit commit.
