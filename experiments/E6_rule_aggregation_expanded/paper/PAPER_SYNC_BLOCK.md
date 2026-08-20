# E6-EXPANDED Paper Sync Block

**Title fragment:** Rule/Evidence Aggregation Ablation for Drawing-Based
Psychological-Indicator Screening -- Corrected Methodology and Expanded
Cohort

**One-line summary:** Correcting four methodological issues (an
eligibility-governance defect letting disabled rules shape 54% of
matched evidence, a circular outcome definition, a domain-scope
confound, and a contradiction-type conflation) and expanding the
development cohort from 15 to 34 cases changes the measured coverage of
permissive strategies substantially (0%->58.8% for the any-eligible-rule
baseline) but leaves the central finding unchanged: the currently-
deployed same-domain convergence rule never triggers on any of 34
naturally-sampled cases, so its positive-path sensitivity remains
unvalidated even after cohort expansion.

**Method (1 sentence):** A registry-level trace confirmed a rule-
eligibility governance defect in production (no `allowed_output_level`
filter anywhere in the synthesis pipeline); E6's own analysis layer
applies the correct filter without modifying production code, then
compares 5 aggregation strategies (plus 2 labeled stress-test variants)
against 6 independent structural-error criteria, kept strictly separate
from each strategy's own policy threshold.

**Result (1 sentence, with numbers):** At N=34, R1/R5 (any-rule / ratio-
based) show genuine non-circular coverage of 58.8%/50.0% with zero
structural errors; R3 (cross-domain pooling) shows 100% structural-error
rate among its own 3 triggers; R2 and R4 (current DOAR) are numerically
identical (0/34 both); Cochran's Q=48.324 (df=3, p<0.001) across R1/R2/
R3/R5.

**Limitation to disclose:** 0/34 cases contain independent multi-family
evidence within one concern domain; 40/59 targeted new images remain
unprocessed due to a Gemini API quota limit encountered mid-run; a
6-image benchmark is reserved, unprocessed, for a future genuinely blind
check.

**Reproducibility statement:** All results regenerate deterministically
from `experiments/E6_rule_aggregation_expanded/scripts/` against the
frozen `raw/cohort_manifest.csv` and cached Observer/Verifier JSON --
zero live model calls in the analysis itself; the ONE live-call phase
(perception generation for new images) is logged in `environment.json`
with its exact achieved/failed counts.
