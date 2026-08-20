# E6-EXPANDED Protocol -- Corrected Rule/Evidence Aggregation Ablation

Supersedes `experiments/E6_rule_aggregation/` (now labeled E6-PILOT,
provisional, preserved unmodified) for methodology; E6-PILOT's own
numbers are NOT overwritten or retracted, only superseded as the basis
for the architecture decision.

## 0. Scope

Re-aggregates the same already-computed, frozen `EligibleAtomicRuleMatch`
evidence every cached case already has via the unmodified
`drawing_synthesis.synthesize_drawing` pipeline. Does not touch
`RULE_EVIDENCE_MATRIX.csv`, `RULE_RELATIONSHIP_GRAPH.json`,
`CONCERN_DOMAIN_MAP.json`, `concerns.py::MIN_EVIDENCE`,
`GeminiVisualObserver`/`GeminiVisualVerifier` prompts/models, or either
checkpoint tag. The ONE new activity this phase performs is generating
NEW perception data (Observer+Verifier) for new candidate images, using
those FROZEN, unmodified classes exactly as production does.

---

## 1. HIGH-PRIORITY ELIGIBILITY GOVERNANCE AUDIT

**Question:** can a rule with `allowed_output_level == "disabled"` become
an `EligibleAtomicRuleMatch`, a literature association, and contribute to
`build_overall_synthesis`'s aggregation, in the production pipeline as it
exists today?

**Traced explicitly, not assumed** (`src/doar/reasoning_chain.py`,
`src/doar/drawing_synthesis.py`, read directly this phase):

| Step | Function | Filters on `allowed_output_level`? |
|---|---|---|
| Registry row | `RULE_EVIDENCE_MATRIX.csv` via `load_rule_matrix()` | n/a -- source data only |
| Precondition check | `check_visual_preconditions()` / `check_deterministic_preconditions()` | **NO** |
| -> EligibleAtomicRuleMatch | `build_eligible_matches()` / `build_deterministic_eligible_matches()` | **NO** -- only filters on `status == "satisfied"` |
| -> literature association | `build_literature_linked_associations()` | **NO** |
| -> synthesis | `build_overall_synthesis()` | **NO** -- partitions only by `concern_domain` (positive/distress/neutral), never by `allowed_output_level` |
| -> candidate hypothesis | `reasoning_chain.build_candidate_hypotheses()` | **NO** |

**Answer: YES, disabled rules CAN and DO contribute in production today.**

**Exact evidence** (from the 15 E6-PILOT cases' own real cached
Observer/Verifier data, re-queried directly this phase):

```
h38       EN_COMPILED_LINE_LIGHT_PRESSURE_031  -> individual_heuristic_only
p2b_0001  PSY_AR_GEOMETRY_008                  -> disabled
p2b_0001  EN_COMPILED_HOUSE_023                -> disabled
p2b_0001  EN_COMPILED_LINE_LIGHT_PRESSURE_031  -> individual_heuristic_only
p2b_0002  EN_COMPILED_TREE_024                 -> disabled
p2b_0003  EN_COMPILED_FACE_EXPRESSION_021      -> disabled
p2b_0004  EN_COMPILED_LINE_LIGHT_PRESSURE_031  -> individual_heuristic_only
p2b_0005  PSY_AR_GEOMETRY_008                  -> disabled
p2b_0005  EN_COMPILED_LINE_HEAVY_PRESSURE_030  -> individual_heuristic_only
p2b_0012  EN_COMPILED_TREE_024                 -> disabled
p2b_0012  EN_COMPILED_LINE_HEAVY_PRESSURE_030  -> individual_heuristic_only
p2b_0019  PSY_AR_CIRCLES_011                   -> disabled
p2b_0019  EN_COMPILED_HOUSE_023                -> disabled
p2b_0026  EN_COMPILED_LINE_LIGHT_PRESSURE_031  -> individual_heuristic_only
p2b_0033  EN_COMPILED_FACE_EXPRESSION_021      -> disabled
p2b_0054  EN_COMPILED_LINE_LIGHT_PRESSURE_031  -> individual_heuristic_only
p2b_0061  PSY_AR_GEOMETRY_008                  -> disabled
p2b_0061  PSY_AR_SIZE_HALF_014                 -> individual_heuristic_only
p2b_0061  EN_COMPILED_PLACEMENT_CENTER_029     -> individual_heuristic_only
p2b_0068  PSY_AR_GEOMETRY_008                  -> disabled
p2b_0068  EN_COMPILED_ANIMAL_CHOICE_GENERAL_022 -> disabled
p2b_0068  EN_COMPILED_HOUSE_023                -> disabled
p2b_0068  EN_COMPILED_TREE_024                 -> disabled
p2b_0068  EN_COMPILED_LINE_LIGHT_PRESSURE_031  -> individual_heuristic_only
```

**13 of 24 (54%) of E6-PILOT's own matched associations were "disabled"
rules.** `PSY_AR_GEOMETRY_008`, `EN_COMPILED_HOUSE_023`,
`EN_COMPILED_TREE_024`, `EN_COMPILED_FACE_EXPRESSION_021`,
`PSY_AR_CIRCLES_011`, `EN_COMPILED_ANIMAL_CHOICE_GENERAL_022` all reached
`synthesis.literature_linked_associations` and were counted in
`build_overall_synthesis`'s domain/family partition, despite their own
registry row saying their output is `disabled`.

### STOP -- reported as a scientific-governance defect, NOT silently fixed

Per explicit instruction, **this defect is reported here and NOT fixed in
`reasoning_chain.py`/`drawing_synthesis.py`** -- production continues to
behave exactly as before pending separate review/remediation by the
project owner. This is a genuine gap between what the registry's own
`allowed_output_level` column is documented to mean ("is it actually
computed and shown today" -- `registry_v2_build.py`'s own docstring) and
what the governed synthesis pipeline actually does with it.

**Resolution used for E6's OWN comparison (not a fix to production):**
E6-EXPANDED's re-aggregation layer (`scripts/run_e6_expanded.py`, entirely
outside `drawing_synthesis.py`/`reasoning_chain.py`) applies
`allowed_output_level == "individual_heuristic_only"` as its own
eligibility filter before computing R1-R5, exactly as R1-R5 already are
E6's own external re-aggregation of the SAME underlying evidence, never a
second production rule engine. Both the unfiltered (`allowed_output_level`
column preserved) and filtered evidence are saved in
`raw/input_atomic_evidence.csv` (`eligible_for_e6` boolean column) so the
defect's exact impact stays fully auditable, and a `_broad` stress-test
variant of R1/R2 is retained that intentionally includes disabled-rule
evidence, explicitly flagged via `ineligible_evidence_used`, to show what
happens if this governance defect is left entirely unaddressed.

**Recommendation to the project owner (not acted on here):** add an
explicit `allowed_output_level == "individual_heuristic_only"` filter to
`build_eligible_matches()`/`build_deterministic_eligible_matches()` (or
to `synthesize_drawing()`'s `all_matches` assembly) in production.

---

## 2. Contradiction redefinition

`RULE_RELATIONSHIP_GRAPH.json` distinguishes 4 relationship types,
previously conflated in E6-PILOT (which incorrectly treated
`context_limited_by_edges` as "contradictions"):

| Type | Source | Count | Meaning |
|---|---|---|---|
| **True literature contradiction** | `literature_level_contradictions` | 1 topic | Direct disagreement between cited references. Tied by name (in its own note text) to `EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038`, which is `disabled` -- can NEVER match under E6's eligibility filter, so `is_true_contradiction` is structurally 0% for every eligible-only comparison. Confirmed, not assumed. |
| **Contextual/cultural limitation** | `context_limited_by_edges` | 6 edges | A caveat on an otherwise-valid rule (e.g. figure-size rules confounded by culture) -- NOT a contradiction, tracked separately as `is_contextual_limitation`. |
| **Precondition/requires limitation** | `requires_edges` | 41 edges | `currently_satisfied_in_doar` is a STATIC, rule-level flag (10 True / 31 False) that maps 1:1 onto `allowed_output_level` (individual_heuristic_only / disabled) -- not an independent per-case signal; not double-counted as a separate metric. |
| **Duplicate/same-family** | `same_evidence_family_edges` | 10 edges | Already captured by `evidence_family` grouping (`duplicate_family_rule_count`) -- not a contradiction. |

## 3. Fair R1-R5 domain scope

E6-PILOT's R1/R2 were domain-agnostic (included neutral domains) while
R3/R4/R5 were positive/concern-domain-scoped -- a confound (R2 vs. R4
differed in BOTH raw-count-vs-family-dedup AND domain scope
simultaneously). Fixed: **R1 and R2 (primary)** are now restricted to the
same positive/concern-domain scope as R3/R4/R5, isolating exactly one
variable each:

- R1 vs R4: any-eligible-rule vs. same-domain-family-convergence, same
  domain scope.
- R2 vs R4: raw-rule-count vs. family-deduplicated-count, same domain
  scope, same >=2 threshold.

**R1_broad / R2_broad** (secondary stress-test only, never the primary
comparison): all domains including neutral, AND intentionally uses the
UNFILTERED (disabled-rule-inclusive) evidence, to demonstrate what the
eligibility governance defect (Section 1) would look like if left
completely unaddressed.

## 4. Non-circular outcome definitions

**OBJECTIVE STRUCTURAL FAILURES** (apply regardless of which strategy
produced the trigger):
- `duplicate_inflation` -- >=2 rules sharing one evidence family counted
  as independent (R2-specific risk).
- `cross_domain_convergence` -- evidence pooled across >1 concern domain
  (R3-specific risk).
- `ineligible_evidence_used` -- a `disabled` rule contributed (only
  possible for the `_broad` variants by construction).
- `unavailable_page_dependent_evidence` -- structurally impossible on
  this pipeline (a page-dependent rule's status can only be `satisfied`
  when the page is already confirmed assessable; verified directly from
  `check_deterministic_preconditions`'s own logic, not assumed).
- `inappropriate_unverified_use` -- a matched entity's
  `case_verification_status != "verified"`.
- `ignored_true_contradiction` -- a matched rule ties to a TRUE (not
  contextual) literature contradiction.

**AGGREGATION POLICY DIFFERENCES** (NOT errors -- what E6 is comparing):
- Whether a strategy requires >=2 independent families (R4) vs. >=1
  (R1) vs. a ratio (R5) vs. a raw count (R2) vs. pooled (R3).
- Recorded per (case, domain, strategy) as `policy_disagreement_vs_r4`
  (this strategy's trigger decision differs from what R4's own >=2-
  same-domain-family rule would say) -- purely descriptive, never
  folded into `structural_violation`.

`supported_coverage` = triggered AND zero objective structural failures
(the policy-difference flag never disqualifies a trigger from being
"supported").

## 5. Page assessability

Uses the real per-case `page_relative_features_assessable` value from
`deterministic_features["page_reference"]` -- already correctly computed
in E6-PILOT's own final saved output (verified: 14/15 pilot cases `False`,
1/15 `True`, matching this phase's independent re-check exactly). Carried
forward unchanged; `raw/case_level_summary.csv` records it per case
(covers cases with zero matched rules too, which never appear in the
per-domain table).

## 6. Figure seeds

`scripts/make_figures.py::JITTER_SEEDS` -- fixed explicit integers per
strategy (1-7), never `np.random.default_rng(hash(strategy) % ...)`
(E6-PILOT's own bug: `hash()` on a string is salted per-process by
`PYTHONHASHSEED` and is not reproducible across separate Python
processes/runs).

---

## 7. Dataset / cohort provenance

See `README.md` for the full inventory. Summary:

- **Source pool**: `outputs/phase2c1/private_images/` (80 images),
  itself a curated private-pilot subset of the `Combined_Drawing`
  emotion-classification corpus (a SEPARATE dataset/experimental track
  from DOAR's own rule-based pipeline -- Phase 3-7's CNN/detector work,
  not touched by E6).
- **Provenance record**: `outputs/phase2c1/private_pilot_mapping.csv`
  retains each pool image's `original_split` (train/valid/test) from
  that source corpus's OWN, now-known-to-be-leakage-affected split
  (Phase 7A's finding, a fully separate concern from E6).
- **Locked benchmark**: the 6 pool images with `original_split=="test"`
  that were NEVER used in E6-PILOT are reserved, unprocessed, uninspected
  -- `p2b_0013, p2b_0015, p2b_0048, p2b_0057, p2b_0064, p2b_0073`.
  (`p2b_0061`, ALSO `original_split=="test"`, was already used in
  E6-PILOT/`DEVELOPMENT_SET_15.json` before this phase began -- disclosed,
  not hidden; cannot be retroactively un-exposed, so it remains in the
  working cohort going forward, exactly as `DEVELOPMENT_SET_15.json`'s own
  "PERMANENTLY DEVELOPMENT-ONLY" governance already requires.)
- **No formal 100-image held-out benchmark exists yet** -- referenced as
  aspirational future work in `BENCHMARK_SCHEMA.md`/`DEVELOPMENT_SET_15.
  json`'s own note, never materialized (would need more images than the
  80-image pool currently contains). The 6-image locked benchmark above is
  the best currently-available approximation, reusing the source
  dataset's own pre-existing "test" designation rather than inventing a
  new one.
- **DEVELOPMENT_SET_15.json is NOT modified** -- read-only throughout.

## 8. Live perception generation

`scripts/e6_expand_run_live_perception.py` (repo root `scripts/`, not
under this experiment directory, since it is a general-purpose,
reusable cache-population tool) runs the FROZEN, unmodified
`GeminiVisualObserver`/`GeminiVisualVerifier` (via
`run_development_benchmark.run_live_observer_and_verifier`, imported not
reimplemented) against each new candidate image and saves to the SAME
`development_live_cache/` the original 15 cases already use. See
`environment.json` for the exact call count/timing actually incurred.
