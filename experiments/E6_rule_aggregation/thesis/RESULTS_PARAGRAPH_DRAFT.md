# E6 Results Paragraph (draft, ready to adapt into the Results chapter)

We compared five rule/evidence aggregation strategies (R1-R5, defined in
`PROTOCOL.md`) against the frozen 15-case DOAR development-set cohort, evaluating
every strategy against the same underlying `EligibleAtomicRuleMatch` evidence each
case's existing, unmodified `drawing_synthesis.synthesize_drawing` pipeline had
already computed. R1 ("any eligible rule") triggered on 13/15 cases (86.7%, 95%
Wilson CI 62.1-96.3%); R2 (raw rule-count convergence, no evidence-family
deduplication) triggered on 1/15 (6.7%); R3 (evidence-family convergence pooled
across concern domains) triggered on 3/15 (20.0%); R4, the currently-deployed
same-domain independent evidence-family convergence rule, triggered on 0/15
(0.0%); and R5 (support ratio of matched to structurally-assessable independent
families within one domain) triggered on 11/15 (73.3%).

Critically, when every triggered interpretation was independently checked for
duplicate-family inflation, cross-domain pooling, and insufficient independent
evidence, **100% of R1's, R2's, R3's, and R5's own triggers exhibited at least one
of these structural violations** -- R1 and R5 relied on a single evidence family
below the two-family independence floor in every trigger; R2's one trigger
double-counted two rules from the same evidence family
(`object_symbolism`, case p2b_0068); and all three of R3's triggers pooled
evidence across concern domains with no shared clinical construct (e.g. combining
a `developmental_or_attention_related` shape-symbolism rule with an
`anxiety_or_stress_related` line-pressure rule into one "concern" reading). R4
produced zero triggers, and therefore zero violations, on this cohort.

A Cochran's Q test across R1, R2, R3, and R5 (R4 excluded as a constant, all-zero
vector) confirmed the four alternative strategies' trigger rates differ sharply
from one another (Q=28.364, df=3, p<0.001); paired exact McNemar comparisons and
Holm-corrected p-values, together with bootstrap 95% CIs on each pairwise risk
difference, are reported in full in `statistics/statistical_tests.csv` and
`statistics/effect_sizes.csv`. Given N=15, these results are interpreted primarily
through effect sizes and the qualitative structural-violation breakdown, not
through p-values in isolation.

No alternative strategy demonstrated a supported-coverage gain over R4 that was
not accompanied by an equal or larger increase in structurally unsupported
convergence; consequently R4's aggregation logic was retained unchanged. This
result should be read alongside an important cohort limitation: none of the 15
development-set cases contain two or more independent evidence families within a
single concern domain, so R4's true-positive behavior -- correctly accepting
genuine convergence when it exists -- could not be exercised or evaluated in this
experiment; only its false-positive resistance was tested, and on that axis it
was perfect.
