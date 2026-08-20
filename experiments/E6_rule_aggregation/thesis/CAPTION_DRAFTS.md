# E6 Figure/Table Caption Drafts

Each caption states which part of the E6 research question the item answers.
Mirrors `FIGURE_TABLE_REGISTER.md`'s entries (kept in sync manually -- update
both if wording changes).

**E6_F1_coverage_vs_overclaiming.** Supported coverage (%) versus structurally
unsupported convergence (%) for each of the five aggregation strategies, N=15
development-set cases. A strategy in the bottom-right would offer high useful
coverage at low over-claiming risk; none does. R4 (current DOAR) sits at the
origin (0% coverage, 0% over-claiming); R1/R5 sit in the top-left (high
over-claiming, no coverage); R3 sits at intermediate over-claiming with no
coverage; R2 is close to the origin, its one trigger fully over-claimed.
Answers: "which strategy gives the best coverage/over-claiming trade-off?"

**E6_F2_trigger_abstention.** Stacked bar of trigger vs. abstention rate per
strategy. Answers: "how permissive is each strategy, in isolation from whether
its triggers are supportable?"

**E6_F3_duplicate_inflation.** Bar of the percentage of cases where a
strategy's triggered interpretation relied on >=2 rules sharing one evidence
family (double-counted evidence). Answers: "which strategies are vulnerable to
duplicate-evidence inflation?"

**E6_F4_cross_domain_convergence.** Bar of the percentage of cases where a
strategy's triggered interpretation pooled evidence across more than one
concern domain. Answers: "which strategies reproduce the documented
pre-acceptance-audit-fix cross-domain over-claiming bug?"

**E6_F5_rules_vs_families.** Scatter of raw matched-rule count vs. independent
evidence-family count, for every triggered (case, domain, strategy) instance,
colour-coded by strategy, with the rules=families reference line. Points below
the line indicate duplicate-family inflation (more raw rules than independent
families). Answers: "how much of each strategy's apparent evidence volume is
actually independent?"

**E6_F6_case_strategy_heatmap.** Binary trigger/abstain heatmap, 15 cases x 5
strategies. Answers: "does agreement/disagreement between strategies cluster on
particular cases, or is it spread evenly across the cohort?" (It is spread
evenly -- no single case drives the aggregate pattern; see
`error_analysis/disagreement_cases.csv`.)

**E6_T1_strategy_comparison.** Full per-strategy metrics table: N, trigger/
abstention/coverage rates, mean/median rules and families per triggered
concern, and duplicate/cross-domain/structural-violation/contradiction rates,
with absolute and relative differences vs. R4. The primary results table for
the chapter.

**E6_T2_case_comparison.** Per-case (row) x per-strategy (column) trigger
matrix -- the exact source data behind Figure F6, kept as a table for
appendix/supplementary use.

**E6_T3_domain_comparison.** Per-(case, concern-domain) x per-strategy trigger
matrix -- one row per domain actually touched by evidence in a case, finer
grain than T2.

**E6_T4_failure_cases.** Every (case, domain, strategy) row flagged with any
structural violation (duplicate inflation, cross-domain convergence,
insufficient independent evidence, ignored contradiction), with the specific
reason recorded. The evidentiary backing for every qualitative example in
`error_analysis/error_summary.md`.

**E6_T5_pairwise_effects.** All 10 pairwise strategy comparisons' risk
differences with bootstrap 95% CIs and discordant-pair odds ratios.
