# E6 Error / Qualitative Analysis Summary

Source data: `raw/aggregation_per_domain.csv`, `error_analysis/disagreement_cases.csv`,
`error_analysis/failure_cases.csv`. All numbers below are reproducible by re-running
`scripts/run_e6.py` against the frozen 15-case development-set cache.

## Headline pattern

**On this cohort, every alternative strategy's triggers are 100% structurally
flawed** (by whichever violation type is native to that strategy):

| Strategy | Triggers | Of those, structurally violated |
|---|---|---|
| R1 | 13/15 | 13/13 (100%) -- insufficient independent evidence every time |
| R2 | 1/15  | 1/1 (100%) -- duplicate-family inflation |
| R3 | 3/15  | 3/3 (100%) -- cross-domain convergence |
| R4 | 0/15  | n/a -- never triggers, never violates |
| R5 | 11/15 | 11/11 (100%) -- insufficient independent evidence |

R4 (current DOAR) is the only strategy in this cohort with a non-degenerate
distinction between "triggered" and "structurally clean" -- because it never
triggers at all. This is the central, honest finding of E6 and is repeated
verbatim in `KNOWN_LIMITATIONS.md`.

## Representative cases (not cherry-picked -- these are the ONLY instances of
each category that exist in the frozen 15-case cohort; where a category has
zero instances, that is stated, not hidden)

### 1. Duplicate-rule inflation
**p2b_0068, `neutral_descriptive_only_no_construct_proposed` domain, R2.**
3 raw rule matches (`EN_COMPILED_ANIMAL_CHOICE_GENERAL_022`, `EN_COMPILED_HOUSE_023`,
`EN_COMPILED_TREE_024`) but only 2 distinct evidence families -- `EN_COMPILED_HOUSE_023`
and `EN_COMPILED_TREE_024` are BOTH `object_symbolism`. R2 (raw rule count, no family
dedup) treats this as 3 independent pieces of evidence and triggers; R4's own
`deduplicate_by_evidence_family` (reused unchanged) correctly counts it as 2 families,
below what R2 sees. This is the ONLY duplicate-family-eligible case in the cohort
(1/15 cases even have >1 rule sharing a family) -- R2's 6.7% trigger rate is entirely
this one case.

### 2. Cross-domain false convergence
**p2b_0001, R3, "concern" direction.** `PSY_AR_GEOMETRY_008` (`shape_symbolism`,
domain `developmental_or_attention_related`) and `EN_COMPILED_LINE_LIGHT_PRESSURE_031`
(`line_intensity_quality`, domain `anxiety_or_stress_related`) are pooled together by
R3 because both point in the "concern" direction, reaching 2 independent families
and triggering -- despite the two rules belonging to two unrelated concern domains
with no shared clinical construct. R4's domain-scoped convergence correctly refuses
this (each domain individually has only 1 family). The SAME pattern recurs in
p2b_0005 (`developmental_or_attention_related` + `aggression_or_threat_related`) and
p2b_0068 (`developmental_or_attention_related` + `anxiety_or_stress_related`) -- all
3 of R3's triggers in this cohort are this exact failure mode; there are zero
"clean" R3 triggers in the cohort (see the table above).

### 3. Legitimate same-domain convergence
**None exist in this cohort.** Zero of the 15 cases have >=2 independent evidence
families within a single concern domain. This is stated as a genuine null result,
not glossed over -- see `KNOWN_LIMITATIONS.md` for what additional data would be
needed to observe this.

### 4. Appropriate abstention
**h38, R4, `anxiety_or_stress_related` domain.** Exactly 1 evidence family
(`line_intensity_quality`, from `EN_COMPILED_LINE_LIGHT_PRESSURE_031`). R4 correctly
abstains from calling this "convergent" -- one weak, `individual_heuristic_only`
line-pressure heuristic is not independent corroborating evidence. This is the
modal case in the cohort: 14 of the 15 cases where R4 abstains look exactly like
this (a single matched rule/family in a domain, correctly not promoted to
"convergent").

### 5. R4 vs. R5 disagreement
**p2b_0019, `social_withdrawal_related` domain.** R5's support ratio here is
1/1 = 1.0 (the ONE evidence family in this domain that could even structurally be
assessed for this case WAS matched) -- R5 triggers. R4 requires an absolute floor
of 2 independent families regardless of how many were assessable, and abstains.
This is the cleanest illustration of R5's core design trade-off: it rewards
"everything assessable was found" even when only one thing was ever assessable,
which is a fundamentally different, and in this cohort ALWAYS single-family,
notion of "support." R4 and R5 disagree on 11/15 cases (73.3%) -- every case where
R5 triggers, since R4 never does.

### 6. Cases where another strategy appears better than current DOAR
**None identified.** "Better" would require a strategy that reaches a genuinely
clean (non-structurally-violated) trigger that R4 misses. In this cohort, every
alternative strategy's triggers are 100% structurally flawed (table above) -- there
is no case where R2, R3, R1, or R5 finds real, clean, independent convergence that
R4's stricter rule denies. This is itself the main finding: on the evidence
available, no alternative strategy demonstrates an advantage over R4 that isn't
also an increase in false/inflated convergence. See `DECISION_LOG.md`.

## Negative / null results preserved (not discarded)

- R4 (current DOAR) never once reaches its own "convergent" state on this cohort
  (0/15). This is a null result about the COHORT, not necessarily about R4's
  design -- see `KNOWN_LIMITATIONS.md`.
- No case in the cohort exhibits legitimate multi-family same-domain convergence,
  so R4's TRUE positive rate cannot be measured here, only its (perfect, 0%) false
  positive rate.
- `ineligible_evidence_used` and `unavailable_page_dependent_evidence` are 0% for
  ALL 5 strategies -- the frozen pipeline structurally prevents both failure modes
  before evidence ever reaches the aggregation step (see `run_e6.py`'s own inline
  documentation of why). This is a genuine, reportable true-negative safety
  property of the existing DOAR pipeline, not a limitation of this experiment.
