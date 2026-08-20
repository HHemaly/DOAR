# E6-EXPANDED Phase 2 — Feasibility Audit, Eligibility Fix, and Decision

This phase answers the question raised by E6-EXPANDED's own headline
result (R4 triggered 0/34, 0/34 cases with ≥2 same-domain independent
families): **is R4's positive path even structurally reachable given the
current registry, and is the aggregation comparison itself built on
scientifically clean evidence?** See `REGISTRY_FEASIBILITY_REPORT.md`
(Part 1) and `eligibility_fix/` (Parts 2–3) for full detail; this document
is the synthesis and decision (Parts 4–5).

## 1. R4 feasibility (static audit, no cases, no API calls)

Exactly **one** of the eight concern domains present in the registry —
`anxiety_or_stress_related` — currently has two enabled rules from two
distinct evidence families: `PSY_AR_SIZE_SMALL_016` (family
`size_composition`) and `EN_COMPILED_LINE_LIGHT_PRESSURE_031` (family
`line_intensity_quality`), both with their visual/deterministic
prerequisites already satisfied in DOAR. No other in-scope domain has more
than one enabled family; the two neutral domains have two enabled families
each but are structurally outside R4's scope regardless.

**R4 is therefore NOT structurally impossible — but it is reachable
through exactly one narrow path**: a case whose drawing is both small
(page coverage ≤20%) AND drawn with light/faint line pressure. Full table:
`tables/E6X_T8_registry_feasibility.csv`.

## 2. Eligibility fix

The registry audit (E6-EXPANDED's original PROTOCOL.md Section 1) found
no `allowed_output_level` filter anywhere in the frozen synthesis
pipeline. This phase remediated it:

- **Files changed**: `src/doar/reasoning_chain.py`
  (`build_eligible_matches`, + new `ELIGIBLE_OUTPUT_LEVEL` constant),
  `src/doar/drawing_synthesis.py` (`build_deterministic_eligible_matches`,
  defense-in-depth — this path was already a no-op today since
  `DETERMINISTIC_RULE_IDS` happens to equal the enabled set, but was not
  explicitly gated).
- **What changed**: both functions now skip a `satisfied` precondition
  check whose rule's `allowed_output_level != "individual_heuristic_
  only"`. `check_visual_preconditions`/`check_deterministic_preconditions`
  themselves are byte-for-byte unchanged — the Technical/debug view still
  sees every rule's raw precondition status; only promotion to governed
  evidence is gated.
- **Tests**: `tests/test_reasoning_chain.py` (new
  `EligibilityGateRegressionTests`, 3 tests; existing tests whose fixtures
  relied on disabled-rule matches rewritten to use hand-constructed
  `EligibleAtomicRuleMatch` objects for their real purpose — dedup/
  aggregation/hypothesis/package logic, which is indifferent to how a
  match was built); `tests/test_drawing_synthesis.py` (new
  `EligibilityGateEndToEndRegressionTests`, 3 tests, including a real-data
  sweep proving no `literature_linked_associations` entry is ever a
  disabled rule). Before-state snapshot:
  `eligibility_fix/BEFORE_STATE_test_baseline.txt` (95 passed, defect not
  yet caught — the new regression tests did not exist in that baseline)
  and `eligibility_fix/BEFORE_STATE_defect_demonstration.txt` (concrete
  proof: 7/7 semantic-path matches were disabled rules). After the fix:
  166 passed, 1 skipped across `test_reasoning_chain.py` +
  `test_drawing_synthesis.py` + `test_clinician_review_app.py`.

## 3. Measured impact on the 34 cached E6-EXPANDED cases

19/34 cases (55.9%) had ≥1 disabled-rule match removed; 27 disabled
matches removed in total; 9 cases' `overall_synthesis.level` changed
(always toward LESS evidence, never more); **1 case (`p2b_0007`) lost a
previously-generated candidate hypothesis** that had been partly supported
by a disabled rule (`PSY_AR_GEOMETRY_008`) — a genuine, previously-hidden
false positive now corrected. Full detail:
`eligibility_fix/ELIGIBILITY_FIX_IMPACT_REPORT.md`,
`tables/E6X_T9_eligibility_fix_impact.csv`,
`figures/E6X_F10_before_after_disabled_rule_impact.{png,pdf,csv}`.

**Note on E6-EXPANDED's own R1–R5 comparison**: this production fix does
NOT change E6-EXPANDED's earlier R1–R5 results — that comparison already
applied its own `ELIGIBLE_OUTPUT_LEVEL` filter as an external correction
layer (`run_e6_expanded.py`), built precisely because this defect was
already identified (not yet fixed) at that time. What this fix changes is
the REAL production pipeline's behavior on ordinary case processing (as
demonstrated by `p2b_0007`) — previously a real, deployable correctness
bug, not merely an analysis-layer inconsistency.

## 4. Reassessing R1/R5 as policy, not correctness

E6-EXPANDED's earlier framing reported R1's 58.8% and R5's 50.0%
"supported coverage" as non-circular, structurally clean results. That
remains true in the narrow sense used there (no *objective* structural
error: no duplicate-family inflation, no cross-domain pooling, no
disabled-rule use). **It must not be read as evidence that R1/R5's
underlying single-family trigger is psychologically correct.** A single
matched rule — even an enabled, precondition-satisfied one — is still one
unvalidated literature heuristic (`evidence_strength_as_written` graded
"Weak/mixed" or "Speculative" for most of the 10 enabled rules;
`confidence_ceiling` ≤0.2 for all of them). R1/R5's higher trigger rates
reflect a more PERMISSIVE AGGREGATION POLICY (1 family is enough) — not a
demonstration that what they flag is more often true. R4's stricter policy
(≥2 independent families, same domain) remains the only one of the five
that requires genuinely converging evidence before claiming anything.

## 5. Decision

**Option C — R4 reachable but extremely sparse: a predeclared,
prerequisite-based stress/challenge cohort, not further blind expansion.**

Rationale:

- Option A (unconditionally justified further collection) is not
  supported: 40 additional images already sit unprocessed from
  E6-EXPANDED's own quota-blocked run, and nothing about those 40 is known
  to differ from the 34 already seen (which contain 0 same-domain
  multi-family cases). Blindly collecting MORE undifferentiated images has
  a known, computed base rate of finding the required pattern: 0/34 so
  far, and the feasibility audit now explains why — only ONE specific
  two-rule combination anywhere in the registry can ever produce it.
- Option B (declare structurally impossible, stop) is not supported either
  — Part 1 found R4 IS reachable, just narrowly. Declaring it impossible
  would misstate the audit's own finding.
- Option C directly follows from the feasibility audit's own output: since
  the exact required combination is known in advance (page coverage ≤20%
  AND light/faint line pressure, both already-computed deterministic
  features, needing no new detector work and no live Gemini calls to
  check for on ALREADY-CACHED evidence), the next experiment can be a
  **targeted screen of the deterministic feature cache** (already
  computed for all 34 cases, `outputs/prototype_cases/development_
  deterministic_features_cache/`) for cases meeting BOTH thresholds
  simultaneously — a selection criterion declared here, BEFORE any
  strategy outcome is examined, exactly as this task's own governing rule
  requires.

**Next exact experiment, already run this phase (zero API calls, screen
only)**: `E6-STRESS-TARGETED` — screened all 34 cached cases' already-
computed deterministic features (`segmentation.bounding_box_coverage <=
0.20` AND `stroke.intensity_proxy <= INTENSITY_PROXY_LIGHT_THRESHOLD`,
`scripts/screen_e6_stress_targeted.py`,
`eligibility_fix/E6_STRESS_TARGETED_screen.csv`) for any case meeting both
simultaneously.

**Result: 0/34 meet both.** More specifically: **0/34 satisfy the
size-small condition alone** (`bounding_box_coverage <= 0.20`); 10/34
satisfy the light-line-pressure condition alone. This is a sharper,
additional finding beyond "the co-occurrence never happened": the
`size_small` half of the ONLY reachable pair essentially never becomes
available in THIS image corpus, because `bounding_box_coverage` is at or
near 1.0 for the great majority of the 34 cases (median well above 0.9;
several cases are exactly 1.0). This most plausibly reflects how these
images were captured — photographs/scans of drawings that are typically
already cropped tightly to the drawn content, not full-page images with
visible surrounding margin — rather than a genuine absence of children
drawing small figures. **This is a new, distinct limitation from the
same-domain-convergence gap already documented**: it says the *feature*
needed for R4's one reachable path is close to unmeasurable given how
this specific image set was produced, independent of any psychological
question.

**Because deterministic features require only the raw image file (no
Gemini call at all)**, this same screen can be re-run against the 40
still-unprocessed candidate images with zero additional API cost — that
is the correct, lowest-cost next step before any further live perception
spending, and is recommended as the literal first action of any future
E6 continuation. If that screen also returns zero, the recommended next
step shifts from "collect more images" to "re-examine whether
`bounding_box_coverage` as currently computed is the right operationalization
of `PSY_AR_SIZE_SMALL_016`'s literature construct for cropped/photographed
input" — a measurement-validity question, not a data-volume one.

**Decision on the R4 aggregation choice itself**: unchanged from
E6-EXPANDED's own `KEEP R4 PROVISIONALLY` (see `DECISION_LOG.md`) — this
phase adds a mechanistic explanation for WHY the positive path remains
unexercised (narrow structural reachability, not small-sample bad luck)
and a concrete, low-cost next step to actually test it, rather than
either abandoning R4 or declaring it validated on absent evidence.
