# E6 Paper Sync Block

Condensed block for a conference/journal paper draft (shorter than
`thesis/THESIS_SYNC_BLOCK.md`, same underlying numbers).

**Title fragment:** Rule/Evidence Aggregation Ablation for Drawing-Based
Psychological-Indicator Screening

**One-line summary:** Across five candidate rule-aggregation strategies
evaluated on a frozen 15-case cohort, the currently-deployed same-domain
independent-evidence-family convergence rule was the only strategy that never
produced a structurally unsupported (duplicate-inflated, cross-domain, or
under-evidenced) triggered interpretation -- every alternative strategy's
triggers were 100% structurally flawed in this cohort, at the cost of zero
triggered interpretations from the current rule on this same cohort.

**Method (1 sentence):** Five aggregation strategies were applied, unchanged,
to the same already-computed rule-eligibility evidence for each case, and every
triggered interpretation was checked against six independent structural
criteria (duplicate-family inflation, cross-domain convergence, insufficient
independent evidence, ineligible-evidence use, unverified-evidence use,
ignored contradiction).

**Result (1 sentence, with numbers):** Trigger rates ranged from 0% (current
rule) to 86.7% (any-eligible-rule baseline); every strategy other than the
current rule had a 100% structural-violation rate among its own triggers
(Cochran's Q=28.364, df=3, p<0.001 across the four non-degenerate strategies;
full paired statistics in `statistics/`).

**Limitation to disclose:** the evaluation cohort (N=15) contains zero cases
with independent multi-family convergence within a single concern domain, so
the current rule's true-positive sensitivity remains untested; only its
false-positive resistance was measured.

**Reproducibility statement:** All results regenerate deterministically from
`experiments/E6_rule_aggregation/scripts/run_e6.py` against the cached,
already-frozen Observer/Verifier development-set JSON -- no live model calls
were made for this experiment.
