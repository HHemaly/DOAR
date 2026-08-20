# E6 Protocol -- Rule / Evidence Aggregation Ablation

## 0. Scope and non-goals

This experiment re-AGGREGATES the already-computed, already-frozen
`EligibleAtomicRuleMatch` list every cached development-set case already has
(the same object `drawing_synthesis.synthesize_drawing` builds in production).
It does not re-implement rule matching, does not call Gemini, does not touch
`RULE_EVIDENCE_MATRIX.csv` / `RULE_RELATIONSHIP_GRAPH.json` /
`CONCERN_DOMAIN_MAP.json` / `concerns.py::MIN_EVIDENCE`, and does not modify
`checkpoint/psychologist-prototype-v1` or `checkpoint/human-interaction-v1`.

## 1. Data-readiness audit (performed BEFORE any strategy comparison)

Source: all 15 cases in `DEVELOPMENT_SET_15.json`, loaded via
`scripts/clinician_review_app.py::load_case_bundle` (cache-only, returns `None`
if no cached Observer/Verifier data exists -- never makes a live call).

**1. Total cached cases found:** 15 (all of `DEVELOPMENT_SET_15.json`).

**2. Cases usable for E6:** 15/15. Every case has cached Observer/Verifier
data AND a computable `synthesis` (deterministic function of that cached
data + the frozen rule/domain files).

**3. Excluded cases:** none. 0 cases lacked cached data.

**4. Structural counts across the 15-case cohort:**

| Property | Count |
|---|---|
| Cases with >1 matched rule | 6 |
| Cases with >1 evidence family | 6 (same 6 cases -- every multi-rule case in this cohort has each rule in a distinct family) |
| Cases with >1 concern domain touched | 6 (same 6 cases -- every multi-rule case in this cohort also spans multiple domains) |
| Cases with duplicate-family rules (>=2 rules sharing one family) | 1 (p2b_0068 only) |
| Cases with a contradiction flag (rule_id in `RULE_RELATIONSHIP_GRAPH.json`'s `context_limited_by_edges`) | 9 |
| Cases with ZERO matched rules at all | 2 (p2b_0040, p2b_0047) |
| Cases with any uncertain/unreviewed entity present | 12 |
| Cases where the page boundary is NOT confirmed assessable | 14/15 (only p2b_0061 has an assessable page) |

**5. Replayable from cache without rerunning Gemini/perception:** YES, in
full. `check_visual_preconditions` (semantic rules) is a pure function of the
already-cached `VisualEntity` list; `check_deterministic_preconditions`
(composition/line rules) is a pure function of the already-cached
deterministic-feature cache. Neither makes a network call. Verified by
running `scripts/run_e6.py` with zero network access and confirming the h38
cached verification JSON's SHA-256 is unchanged before/after (same guarantee
already established for the Human Interaction Layer's own visual-recheck
tests).

**6. Missing fields that weaken the experiment:**
- `confidence_ceiling` is the literal string `"not_set_pending_review"` for
  18/41 rules (44%) and `evidence_strength_as_written` is unstructured free
  text (5 distinct phrases, no numeric scale) -- this is why **R6 (weighted
  aggregation) was NOT implemented**, per this task's own explicit gating
  condition ("only add R6 if existing rule-strength values are already
  consistently defined and scientifically usable"). Inventing a numeric
  mapping now would be exactly the post-hoc tuning this experiment is
  required to avoid.
- Only 1/15 cases has any duplicate-family-eligible rule pair, so R2 vs. R4's
  distinguishing behavior is observed on a single case -- real, but thin.
- **Zero cases have >=2 independent evidence families within one concern
  domain** -- this means R4 (current DOAR) and R5 (its ratio-based variant,
  same domain scope) have zero opportunities in this cohort to correctly
  ACCEPT genuine convergence; only their (successful) REJECTION behavior is
  observed. See "Cohort adequacy decision" below.

## 2. Cohort adequacy decision (GO / STOP)

The task's own stop condition is: "If the available cohort cannot meaningfully
distinguish aggregation methods, stop after the audit."

**Decision: GO**, with the limitation above stated up front and repeated in
every results artifact.

Rationale: the cohort DOES meaningfully distinguish the methods on the
specific axis the research question asks about -- "resistance to duplicate,
cross-domain, or otherwise unsupported convergence." R2 vs. R4 diverges (on
the one duplicate-eligible case); R3 vs. R4 diverges on every multi-domain
case (3 of them); R1/R5 vs. R4 diverge on the majority of cases. These
divergences are exactly what the research question asks to be measured, and
they are measured cleanly. What the cohort CANNOT do is show R4 correctly
accepting a genuine multi-family same-domain pattern, because no such pattern
exists in it -- this is disclosed as a limitation, not treated as
disqualifying the whole comparison.

**Cohort frozen at:** all 15 development-set cases, before any strategy
comparison was run. No case was added or removed after seeing results.

## 3. Strategy definitions (exact, as implemented in `scripts/run_e6.py`)

Unit of analysis: one row per `(case_id, concern_domain, strategy)`, for
every domain that has >=1 satisfied (matched) rule in that case. All five
strategies read the SAME underlying `matched_rule_count` /
`unique_evidence_families` / `assessable_family_count` per (case, domain) --
they differ only in how they aggregate those numbers into a trigger decision.

- **R1 -- Any eligible rule.** Domain-local. Trigger iff `matched_rule_count
  >= 1`. Applies to ALL domains including neutral (broadest possible
  coverage baseline).
- **R2 -- Raw rule-count convergence.** Domain-local, no family
  deduplication. Trigger iff `matched_rule_count >= MIN_EVIDENCE` (2 rules,
  even if they share one evidence family). Applies to all domains, same as
  R1 -- isolates the effect of removing family-dedup, nothing else.
- **R3 -- Pooled (cross-domain) evidence-family convergence.** Direction-
  scoped (positive / concern; neutral domains excluded, see below), POOLING
  families across every domain in that direction within the case. Trigger
  iff the pooled family count `>= MIN_EVIDENCE`. Every domain that
  contributed to a triggering pool is marked `triggered=True`;
  `cross_domain_convergence=True` when more than one distinct domain
  contributed. This reproduces the domain-pooling behavior
  `drawing_synthesis.py`'s own module docstring documents as "the original
  synthesis's over-claiming bug" that the acceptance-audit fix (now R4)
  replaced.
- **R4 -- Same-domain independent evidence-family convergence (CURRENT
  DOAR).** Domain-local, family-deduplicated (via
  `reasoning_chain.deduplicate_by_evidence_family`, reused verbatim).
  Trigger iff `unique_family_count >= MIN_EVIDENCE` WITHIN this one domain.
  Exactly `drawing_synthesis.build_overall_synthesis`'s own logic,
  unmodified. Only applies to positive/concern-direction domains (neutral
  domains structurally excluded, exactly as in the real pipeline --
  `_domain_scoped_convergence` is only ever called on `positive_matches`/
  `concern_matches`).
- **R5 -- Support ratio.** Domain-local, same domain-type restriction as R4
  (positive/concern only, so R4 vs. R5 is a single-variable ablation:
  absolute floor vs. ratio, nothing else changes). `support_ratio =
  unique_family_count / assessable_family_count`, where "assessable" means
  the rule's own precondition check returned `satisfied` or `not_satisfied`
  for THIS case (i.e. it was structurally possible to evaluate at all --
  excludes `blocked_structural`, `blocked_needs_unbuilt_feature`,
  `not_applicable_to_this_module`, `not_assessable`). Trigger iff
  `support_ratio >= 0.5` (chosen before running any comparison, as the only
  non-arbitrary "majority" cutoff; never tuned against results).
  Deliberately has NO absolute floor, to test whether normalizing by
  assessability changes the coverage/over-claiming trade-off relative to
  R4's fixed floor of 2.
- **R6 -- weighted aggregation.** NOT implemented; see Section 1's missing-
  fields finding.

`MIN_EVIDENCE = 2` is imported conceptually from `concerns.py`'s own frozen
constant (never redefined) -- E6 pins the literal value `2` in
`scripts/run_e6.py` with an explicit comment tying it back to that source,
rather than importing `concerns.py` directly, to keep the experiment script
fully decoupled from the scientific package's internal API surface.

## 4. Independent evaluation criteria (never treats R4 as ground truth)

Every triggered `(case, domain, strategy)` row is checked against ALL of the
following, independent of which strategy produced it:

- `duplicate_inflation` -- matched rules include >=2 sharing one evidence
  family (only meaningful for R2, which is the only strategy that doesn't
  dedup).
- `cross_domain_convergence` -- the trigger pooled evidence from more than
  one concern domain (only meaningful for R3).
- `insufficient_independent_evidence` -- triggered with fewer than
  `MIN_EVIDENCE` independent families (checked for R1 and R5, the two
  strategies whose trigger rule does not itself enforce this floor).
- `ineligible_evidence_used` -- always False on this cohort: verified (no
  `allowed_output_level` gate exists anywhere in the frozen pipeline that
  could make a "disabled" rule contribute differently from an
  "individual_heuristic_only" one; both can become `EligibleAtomicRuleMatch`
  objects identically).
- `inappropriate_unverified_use` -- triggered using an entity whose
  `case_verification_status != "verified"`.
- `unavailable_page_dependent_evidence` -- always False on this cohort: a
  page-dependent rule can only ever reach `satisfied` when
  `page_relative_features_assessable` is already True (verified from
  `check_deterministic_preconditions`'s own logic).
- `ignored_contradiction` -- triggered using a rule_id present in
  `RULE_RELATIONSHIP_GRAPH.json`'s `context_limited_by_edges`.

"Supported coverage" = triggered AND none of the above true. No strategy is
credited with coverage merely for abstaining.

## 5. Statistical methods

- Paired design: all 5 strategies scored on the same 15 cases.
- Wilson score 95% CIs for each strategy's single-proportion metrics
  (trigger rate, coverage rate).
- Case-resampling bootstrap (10,000 reps, seed 20260819, fixed) for paired
  risk-difference CIs between strategies.
- Exact McNemar sign test (`scipy.stats.binomtest` on the smaller discordant
  count) for every pairwise comparison; Holm correction across all 10
  pairwise comparisons.
- Cochran's Q (hand-implemented, standard closed-form, chi-square(k-1)
  reference) across R1/R2/R3/R5 as a single omnibus test; R4 excluded
  because a constant all-zero vector makes Cochran's Q's variance term
  degenerate.
- Effect sizes: paired risk difference (primary; well-behaved at small N)
  and discordant-pair odds ratio (secondary; reported honestly as
  undefined/infinite when a discordant cell is 0, never silently clipped).
- No p-value is reported without an accompanying effect size and CI, per
  this task's own instruction not to chase p-values at small N.

## 6. Known deviations from the literal task wording, and why

- R3 is defined as pooling within a DIRECTION (positive/concern), not
  purely "any domain" -- this exactly reproduces the specific,
  already-documented pre-fix DOAR bug (the most scientifically meaningful
  cross-domain baseline available), rather than an unmotivated alternative
  pooling rule.
- R3 and R4/R5 exclude neutral domains from triggering (a neutral domain can
  fall back to being counted by R1/R2 only). This mirrors
  `build_overall_synthesis`'s own real behavior exactly and keeps R4 vs. R5
  a clean single-variable ablation; an earlier draft of this script let R3
  fall back to an R1-like single-rule trigger on neutral domains, which
  produced a misleading "20% supported coverage" figure driven entirely by
  trivial single-rule neutral-domain matches -- corrected before any
  results were reported (see git history of `scripts/run_e6.py` for the
  fix, made BEFORE the cohort/results were finalized in this document).
