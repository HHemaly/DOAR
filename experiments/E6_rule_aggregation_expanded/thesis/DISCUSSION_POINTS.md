# E6-EXPANDED Discussion Points

1. **The eligibility governance defect is arguably the single most
   important finding of the whole E6 program, independent of which
   aggregation strategy is eventually chosen.** 54% of E6-PILOT's matched
   associations, and a comparably large share at 34 cases, came from
   rules the registry itself marks `disabled` -- meaning their detectors
   are documented as unbuilt/unvalidated, yet they were silently shaping
   every synthesis result. This is a production defect, not an E6
   artifact, and is reported for separate remediation (Section 1 of
   `PROTOCOL.md`) rather than fixed here.

2. **R2 and R4 tying exactly (0/34 both) is a genuinely informative null
   result, not a non-finding.** It demonstrates that evidence-family
   deduplication and raw rule counting are mathematically indistinguishable
   on THIS cohort specifically because duplicate-family co-occurrence
   never happens in it (0/34 cases). The methodological correction (R2's
   domain scope matched to R4's) is what makes this tie visible and
   interpretable -- in E6-PILOT's confounded comparison, R2's 6.7% vs.
   R4's 0% trigger rate looked like a real difference, but was actually
   driven by R2's broader (neutral-inclusive) domain scope, not by its
   lack of family dedup.

3. **R3's 100% cross-domain-violation rate, replicated exactly at 3x the
   pilot's cohort size, is strong, repeated evidence that pooling
   evidence across concern domains is not a viable alternative for this
   rule set.** Every single time R3 diverges from R4 in this expanded
   cohort, it does so by manufacturing a cross-domain reading. This
   should be treated as settled for this rule set/detector coverage, not
   requiring further cohort expansion to re-confirm.

4. **The 0/34 same-domain convergence finding, despite genuine cohort
   expansion, shifts the interpretation from "small-sample artifact" (a
   plausible read of the 15-case pilot alone) toward "a structural
   property of this specific 10-enabled-rule, sparse-detector-coverage
   configuration."** With only 10 of 41 rules currently enabled, and
   those 10 spread across most of the six substantive concern domains,
   the prior probability that any two of a single case's matched rules
   land in the SAME domain is inherently low, independent of sample size,
   until more rules are built/enabled. This reframes the open question:
   it may not be "collect more images" but "build more detectors within
   the SAME domain" that is the actual prerequisite for observing R4's
   positive path.

5. **The quota-exhaustion blocker is disclosed as an environmental
   constraint, not folded into the scientific conclusion.** 40/59 targeted
   images remain unprocessed. Because the achieved 19 new cases showed the
   SAME pattern (0 same-domain convergence) as the first 15, there is no
   positive signal suggesting the remaining 40 would look different --
   but this is stated as an expectation, not a claim, since those 40
   images were never inspected.

6. **The locked 6-image benchmark remains untouched and should stay that
   way until an aggregation strategy decision is treated as final** --
   its value is entirely in never having been used to shape that
   decision. Processing it now, even just to "check," would compromise
   its use as a genuinely blind final confirmation later.

7. **R5's near-identical behavior to R1 in this cohort (50.0% vs. 58.8%
   trigger, both 0% structural violation, both driven by single-family
   evidence) suggests R5's ratio-based normalization is not yet doing
   meaningfully different work from a plain any-eligible-rule baseline
   at current detector density.** This mirrors E6-PILOT's own finding and
   is now confirmed, not merely suggested, at more than double the
   sample size.
