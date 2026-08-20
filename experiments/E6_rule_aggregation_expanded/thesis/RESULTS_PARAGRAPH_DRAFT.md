# E6-EXPANDED Results Paragraph (draft)

Following E6-PILOT's own initial 15-case comparison (now labeled
provisional; see `../../E6_rule_aggregation/`), four methodological
issues were identified and corrected, and the development cohort was
expanded from 15 to 34 cases (the largest cohort achievable within this
session's Gemini API quota; 40 additional candidate images remain
unprocessed and are disclosed as pending, not silently dropped). First,
tracing the frozen production pipeline directly (`reasoning_chain.py`,
`drawing_synthesis.py`) revealed that `build_overall_synthesis` applies
no filter on a rule's own `allowed_output_level` field at any stage --
rules explicitly marked `disabled` in `RULE_EVIDENCE_MATRIX.csv` (31 of
41 rules) can and do contribute to the governed synthesis today; 13 of
E6-PILOT's own 24 matched associations (54%) were such rules. This is
reported here as a scientific-governance defect and left unmodified in
production pending separate review; E6's own re-aggregation layer
applies the correct `allowed_output_level == "individual_heuristic_only"`
filter for its comparison. Second, `context_limited_by_edges` (contextual/
cultural caveats) and `literature_level_contradictions` (true literature
disagreement) were previously conflated as "contradictions" and are now
tracked separately -- only 1 true-contradiction topic exists in the
frozen registry, tied to a rule that is itself disabled and therefore
never eligible, so true contradictions are structurally 0% in any
eligible-only comparison. Third, R1 and R2 were restricted to the same
positive/concern-domain scope R3/R4/R5 already use for their primary
comparison, isolating exactly one aggregation variable each; broader,
all-domain variants (R1_broad, R2_broad) are retained as clearly labeled
secondary stress tests only. Fourth, "fewer than two independent evidence
families" is no longer counted as a structural error -- it is R4's own
policy choice, now tracked separately as `policy_disagreement_vs_r4`,
so a strategy's "supported coverage" reflects only genuine, strategy-
independent evidence errors (duplicate-family inflation, cross-domain
pooling, disabled-rule use, unverified-evidence misuse, ignored true
contradiction).

On the resulting 34-case cohort, R1 (any eligible rule, primary,
domain-scoped) triggered on 20/34 cases (58.8%, 95% Wilson CI 42.2-73.6%)
with zero structural violations, giving genuine, non-circular supported
coverage of 58.8% -- a marked change from E6-PILOT's artifactual 0%,
which resulted from counting single-family evidence as an error rather
than a policy difference. R5 (support ratio) showed the same pattern:
50.0% triggered, 50.0% supported coverage, 0% structural violations. R2
(raw rule count, family-deduplicated domain scope) and R4 (current DOAR)
were numerically IDENTICAL on this cohort -- both 0/34 (0%), 0 discordant
pairs -- because 0/34 cases contain any duplicate-family rule pair at
all, so family deduplication has had no case in this cohort where it
could matter either way. R3 (evidence pooled across concern domains)
triggered on 3/34 cases (8.8%), and **100% of its own triggers were
flagged cross-domain** -- reproducing, at 34-case scale, the exact
over-claiming pattern `drawing_synthesis.py`'s own module docstring
already documents as a previously-fixed production bug. A Cochran's Q
test across R1, R2, R3, and R5 (R4 excluded, a constant zero vector)
confirmed the four strategies' trigger rates differ sharply (Q=48.324,
df=3, p<0.001); paired risk differences and bootstrap 95% CIs for every
comparison are in `statistics/effect_sizes.csv`.

The central, cohort-level finding of this phase is that **expanding the
development cohort from 15 to 34 cases did not resolve the fundamental
data-adequacy gap identified in E6-PILOT**: zero of the 34 cases contain
two or more independent evidence families within a single concern
domain -- R4's own trigger condition remains completely unexercised on
its positive (accept) path, even after more than doubling the sample.
5/34 cases (14.7%) do contain two or more independent families when
pooled across DIFFERENT domains, consistent with R3's cross-domain
violation pattern: when this dataset's cases do accumulate multiple
pieces of eligible evidence, that evidence consistently spans domains
rather than converging within one. Given this, the architecture decision
from this phase is KEEP R4 PROVISIONALLY: the corrected comparison
strengthens confidence in R4's underlying design principle (R2 and R4
tying exactly; R3's 100% cross-domain violation rate), but the decision
cannot yet be treated as fully validated, because no case in either the
15- or 34-case cohort has ever given R4 the opportunity to correctly
accept a genuine same-domain, multi-family pattern.
