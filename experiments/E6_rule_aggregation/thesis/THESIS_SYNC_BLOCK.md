# E6 Thesis Sync Block

Paste-ready block summarizing E6 for the thesis document. Every number here is
sourced from `tables/E6_T1_strategy_comparison.csv` /
`statistics/statistical_tests.csv` / `statistics/confidence_intervals.csv` --
regenerate those before updating this block if the underlying cache changes.

---

**Experiment:** E6 -- Rule / Evidence Aggregation Ablation
**Status:** Complete. Architecture decision: KEEP current aggregation (R4).
**Cohort:** 15/15 DOAR development-set cases (frozen before comparison; 0 excluded).
**Checkpoint:** `checkpoint/human-interaction-v1` (no scientific code changed).

**Research question:** Which rule/evidence aggregation strategy gives the best
trade-off between useful supported coverage and resistance to duplicate,
cross-domain, or otherwise unsupported convergence?

**Methods compared (N=15 cases each, paired):**
- R1 Any eligible rule
- R2 Raw rule-count convergence (no evidence-family dedup)
- R3 Pooled (cross-domain) evidence-family convergence
- R4 Same-domain independent evidence-family convergence -- **current DOAR**
- R5 Support ratio (matched / assessable independent families, same-domain)

R6 (weighted aggregation) was NOT implemented: `confidence_ceiling` is
`not_set_pending_review` for 18/41 rules (44%) and `evidence_strength_as_written`
is free text with no defined numeric mapping -- inventing weights would be
post-hoc tuning, explicitly out of scope.

**Key result table (Wilson 95% CI in parentheses):**

| Strategy | Trigger % | Supported coverage % | Structural violation % (of all triggers) |
|---|---|---|---|
| R1 | 86.7 (62.1-96.3) | 0.0 | 86.7 (100% of R1's own triggers) |
| R2 | 6.7 (1.2-29.8)   | 0.0 | 6.7 (100% of R2's own triggers) |
| R3 | 20.0 (7.0-45.2)  | 0.0 | 20.0 (100% of R3's own triggers) |
| R4 | 0.0 (0.0-20.4)   | 0.0 | 0.0 |
| R5 | 73.3 (48.0-89.1) | 0.0 | 73.3 (100% of R5's own triggers) |

**Statistics:** Cochran's Q (R1,R2,R3,R5; R4 excluded, constant zero vector) =
28.364, df=3, p<0.001 -- the four alternative strategies' trigger rates differ
sharply from each other, but sharp differences in trigger RATE alone do not
establish which is scientifically better (see failure-mode breakdown). Pairwise
exact McNemar tests, Holm-corrected, in `statistics/statistical_tests.csv`. N=15
is small; every test result is reported with its paired risk-difference effect
size and bootstrap 95% CI (`statistics/effect_sizes.csv`), not p-value alone.

**Central, honest finding:** every trigger produced by R1, R2, R3, and R5 in
this cohort is structurally flawed (100% violation rate within each strategy's
own triggers -- insufficient independent evidence for R1/R5, duplicate-family
inflation for R2, cross-domain convergence for R3). R4 (current DOAR) never
triggers at all on this cohort (0/15), so it never over-claims -- but this
cohort also contains ZERO cases where R4 has the opportunity to correctly
accept genuine same-domain, multi-family convergence. R4's demonstrated
property in this experiment is a perfect false-positive rate, not a validated
true-positive rate.

**Architecture decision:** KEEP current aggregation (R4). No alternative
strategy demonstrated a coverage gain that was not fully offset by an equal or
larger over-claiming cost; several (R2, R3, R5) demonstrated the SPECIFIC
failure modes R4 was designed to prevent, at 100% of their own trigger volume.

**Limitation to always state alongside this result:** the 15-case development
cohort contains no case with >=2 independent evidence families inside a single
concern domain, so R4's true-positive behavior is untested here. A larger or
differently-selected cohort is required before this decision can be treated as
final; see `KNOWN_LIMITATIONS.md`.

**Figures/tables to cite:** `E6_F1_coverage_vs_overclaiming` (primary trade-off
plot), `E6_T1_strategy_comparison` (full metrics table).
