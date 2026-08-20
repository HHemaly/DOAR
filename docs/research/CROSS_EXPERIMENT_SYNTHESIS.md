# DOAR Cross-Experiment Synthesis

Updated after every experiment that could interact with a prior one's
conclusions. Currently covers only E6 (the first experiment in this
program, now split into E6-PILOT and E6-EXPANDED); this document is the
place future experiments (E1-E5, E7+, in whatever order they are actually
run) synthesize AGAINST -- not a prediction of experiments not yet run.

---

## Current state after E6-EXPANDED

**Two runs complete: E6-PILOT (15 cases, provisional/superseded) and
E6-EXPANDED (34 cases, corrected methodology).** Decision: KEEP R4
PROVISIONALLY. See `DECISION_LOG.md` for the full entry. E6-PILOT's own
"KEEP" entry is retained as historical record, not deleted.

## Cross-cutting findings so far

1. **The same-domain-convergence blind spot survived a >2x cohort
   expansion, and is now better evidenced as a structural property, not a
   small-sample artifact.** E6-PILOT found 0/15 cases with independent
   multi-family convergence within a single concern domain; E6-EXPANDED
   deliberately re-tested this on 34 cases (19 newly processed) and found
   0/34 -- still zero. Any FUTURE experiment that depends on observing
   DOAR's convergence logic actually firing correctly (not just correctly
   abstaining) will hit the same wall. Discussion Point 4 in
   `experiments/E6_rule_aggregation_expanded/thesis/DISCUSSION_POINTS.md`
   reframes this: with only 10 of 41 rules currently enabled, spread across
   most of six concern domains, the prior probability of two matched rules
   landing in the SAME domain is inherently low regardless of sample size --
   the actual prerequisite may be building more detectors within a single
   domain, not collecting more images. This is worth flagging explicitly to
   any experiment that touches `build_overall_synthesis`,
   `build_candidate_hypotheses`, or the concern-domain convergence path
   generally, and especially to any future work on rule/detector coverage.

2. **The frozen pipeline has two verified true-negative safety properties**
   that E6-PILOT confirmed hold across all 5 aggregation strategies tested,
   not just the current one: no rule with `allowed_output_level="disabled"`
   or any page-dependent rule when the page is unassessable can ever
   contribute evidence to any aggregation strategy, because
   `check_visual_preconditions`/`check_deterministic_preconditions`
   themselves already gate that upstream of aggregation. **Important
   correction from E6-EXPANDED's own eligibility audit: this gating exists
   for VISUAL/deterministic preconditions, but there is NO separate filter
   anywhere in the synthesis pipeline on a rule's own `allowed_output_level`
   field itself** -- 13/24 (54%) of E6-PILOT's own matched associations were
   `disabled` rules that nonetheless satisfied their visual preconditions
   and were treated as valid evidence. This is a genuine, disclosed
   production defect (not fixed here; see `experiments/
   E6_rule_aggregation_expanded/PROTOCOL.md` Section 1), and any future
   experiment relying on "disabled rules never contribute" as a safety
   property must NOT assume it without applying its own
   `allowed_output_level` filter, exactly as E6-EXPANDED's re-aggregation
   layer now does.

3. **`RULE_RELATIONSHIP_GRAPH.json`'s `same_evidence_family_edges` remains
   load-bearing but structurally unexercised even after cohort expansion**
   (0/34 cases have any duplicate-family rule pair, vs. 1/15 in E6-PILOT --
   the rate did not scale up with N). `context_limited_by_edges` is
   correctly no longer conflated with contradictions as of E6-EXPANDED (see
   finding 4). A future cohort-composition experiment specifically
   targeting cases with duplicate-family evidence would let this protection
   be tested under real pressure, not just structurally proven present.

4. **Contradiction and contextual-limitation edges are distinct relationship
   types and must not be merged in future experiments.** E6-PILOT's
   original framing lumped `context_limited_by_edges` (6 edges, cultural/
   contextual caveats) together with genuine
   `literature_level_contradictions` (1 topic). E6-EXPANDED separated them;
   the one true contradiction is tied to a rule that is itself `disabled`,
   so true contradictions are structurally 0% once the eligibility filter
   (finding 2) is applied. Future experiments touching
   `RULE_RELATIONSHIP_GRAPH.json` should use E6-EXPANDED's four-way
   distinction (true contradiction / contextual limitation / requires-
   precondition limitation / duplicate-family), not E6-PILOT's original
   two-way one.

5. **R2 and R4 are numerically indistinguishable on both cohorts tested so
   far (0 triggers each, both N=15 and N=34), which is itself informative:**
   it demonstrates that family-deduplication vs. raw rule counting only
   diverges in cases with duplicate-family co-occurrence, and this dataset
   has essentially none. A future experiment specifically curating cases
   with duplicate-family evidence (see finding 3) is the only way to make
   R2 vs. R4 a live comparison rather than a persistent tie.

## Update after E6-EXPANDED Phase 2 (2026-08-20)

6. **The same-domain-convergence blind spot (finding 1) now has a
   mechanistic explanation, not just a repeated empirical null.** A static
   registry audit (`experiments/E6_rule_aggregation_expanded/REGISTRY_
   FEASIBILITY_REPORT.md`) found R4's positive path is reachable through
   exactly ONE domain/rule-pair combination (`anxiety_or_stress_related`:
   `PSY_AR_SIZE_SMALL_016` + `EN_COMPILED_LINE_LIGHT_PRESSURE_031`) out of
   8 registry domains -- not zero (R4 is not structurally impossible) and
   not many (it is not merely a matter of collecting more undifferentiated
   images). A zero-API-cost predeclared screen of that exact combination
   against all 34 cached cases found 0/34 qualify. Any future experiment
   asking "why doesn't R4 ever trigger" should start from this table, not
   re-derive it.

7. **A new measurement-validity limitation, distinct from data volume:**
   `segmentation.bounding_box_coverage` is at/near 1.0 for nearly every one
   of the 34 cached cases -- `PSY_AR_SIZE_SMALL_016`'s own `<=0.20`
   threshold is met by 0/34 cases, even considered alone. This most
   plausibly reflects how these specific images were captured (cropped/
   photographed close to the drawn content) rather than genuine absence of
   small-figure drawing. Any future experiment relying on
   `bounding_box_coverage` as a proxy for "how much of the page a child
   used" should first check whether the image corpus in use has this same
   near-universal-full-coverage property, since it may make that feature
   structurally uninformative regardless of sample size.

8. **The rule-eligibility governance defect (finding 2's correction) is no
   longer merely flagged -- it is FIXED in production**, as of this phase
   (`src/doar/reasoning_chain.py::build_eligible_matches`,
   `src/doar/drawing_synthesis.py::build_deterministic_eligible_matches`,
   both gated on `allowed_output_level == "individual_heuristic_only"`).
   Any future experiment can now rely on "a disabled rule never becomes
   governed evidence" as an ACTUAL property of the current codebase, not
   merely an intended one -- but should still verify this against the
   `reasoning_chain.ELIGIBLE_OUTPUT_LEVEL` constant's current definition
   rather than assuming it, per this document's own "verify before
   recommending" discipline. Measured, disclosed impact of turning the fix
   on: 1 of 34 cached cases (`p2b_0007`) lost a real candidate hypothesis
   that had been generated partly from a disabled rule -- confirming the
   defect was not merely theoretical.

## Open threads for future experiments

- Whether R5 (support ratio) becomes competitive once assessable-family
  denominators are less sparse (more detectors built) -- not testable with
  the current 41-rule / current-detector-coverage matrix. E6-EXPANDED found
  R5 remains near-identical to R1 (any-eligible-rule) at current detector
  density (50.0% vs. 58.8% triggered, both 0% structural violation),
  confirming rather than resolving this open thread.
- Whether a larger or differently-selected cohort changes E6's headline
  result (R3's 100% cross-domain-violation rate among its own triggers) --
  E6-EXPANDED replicated this exactly at >2x the pilot's sample size,
  raising confidence this is a stable property of R3's design, not this
  cohort's composition. Treated as settled for this rule set unless a
  future experiment finds otherwise.
- **New from E6-EXPANDED:** whether the same-domain-convergence gap (finding
  1) is resolved by processing the remaining 40 unprocessed candidate
  images (quota-blocked this session), or whether it requires expanding
  which of the 41 registry rules have built detectors (currently only 10
  are `individual_heuristic_only`) -- E6-EXPANDED's own evidence leans
  toward the latter but does not settle it, since the 40 remaining images
  were never inspected.
- **New from E6-EXPANDED:** the rule-eligibility governance defect (finding
  2) is flagged for separate remediation in production
  (`reasoning_chain.py` / `drawing_synthesis.py` / `RULE_EVIDENCE_MATRIX.csv`
  never modified by any E6 phase) -- this is an open item independent of
  the aggregation-strategy question E6 itself investigates.
