# E6-EXPANDED Figure/Table Caption Drafts

**E6X_F1_coverage_vs_structural_risk.** Supported coverage (%, non-
circular -- triggered AND free of objective structural errors) versus
structurally unsupported convergence (%), five primary strategies, N=34.
Answers: "which strategy gives the best coverage/structural-risk
trade-off, once policy differences are separated from genuine errors?"

**E6X_F2_trigger_abstention.** Trigger vs. abstention rate, all 7
strategies including the `_broad` secondary stress-test variants.
Answers: "how permissive is each strategy/scope combination?"

**E6X_F3_duplicate_inflation.** % of cases per strategy relying on
duplicate-family-inflated evidence. At N=34, only the `_broad`/raw-count
variants show any (still low, since only 0/34 cases have any duplicate-
family pair at all).

**E6X_F4_cross_domain_convergence.** % of cases per strategy relying on
evidence pooled across multiple concern domains. R3 = 100% of its own
triggers, replicated from E6-PILOT at more than double the sample.

**E6X_F5_policy_disagreement.** % of cases where each primary strategy's
trigger decision differs from R4's, WITHOUT implying either side is
wrong -- a policy-choice map, not an error map. Answers: "how often, and
on what basis, would an alternative strategy have decided differently
from current DOAR?"

**E6X_F6_rules_vs_families.** Matched (eligible) rule count vs.
independent evidence-family count, triggered domain-instances, primary
strategies only, WITH the eligibility filter already applied (so this
plot cannot show disabled-rule inflation -- see F2's `_broad` bars for
that). Fixed jitter seeds (1-7), not `hash()`.

**E6X_F7_case_strategy_heatmap.** Binary trigger/abstain heatmap, 34
cases x 7 strategies.

**E6X_F8_domain_strategy_heatmap.** Trigger RATE (fraction of that
domain's own cases) by concern domain x primary strategy -- shows which
domains are most permissive under R1/R5 vs. which never trigger under
any strategy.

**E6X_F9 (skipped, `E6X_F9_SKIPPED_no_r4_triggers.txt`).** No same-domain
multi-family convergence example exists anywhere in the 34-case cohort
to plot -- explicitly recorded as a null result rather than silently
omitted from the figure set.

**E6X_T1_strategy_comparison.** Primary results table: full corrected,
non-circular per-strategy metrics with policy-disagreement and
structural-violation columns kept separate.

**E6X_T2_cohort_diagnostics.** The Section D cohort-adequacy table --
counts/percentages for all 13 required diagnostic properties, corrected
contradiction definitions.

**E6X_T3/T4.** Per-case / per-domain wide trigger matrices (source for
F7/F8).

**E6X_T5_policy_disagreement.** Every (case, domain, strategy) row where
a primary strategy triggered but disagreed with R4's own floor -- the
evidentiary backing for every "policy, not error" claim in the text.

**E6X_T6_structural_failures.** Every (case, domain, strategy) row with
a genuine objective structural violation -- much shorter than E6-PILOT's
equivalent table, since insufficient-independent-evidence rows have been
correctly moved out.

**E6X_T7_pairwise_effects.** All 10 pairwise primary-strategy risk
differences with bootstrap 95% CIs and discordant odds ratios.
