# E6 Discussion Points (draft talking points for the Discussion chapter)

1. **R4's perfect false-positive resistance is a genuine, meaningful result --
   but it is a resistance result, not a validation of R4's sensitivity.** Every
   alternative strategy, when it triggers at all, triggers on unsupported
   grounds 100% of the time in this cohort. That is strong evidence R4's design
   (same-domain, family-deduplicated, >=2-independent-family floor) is doing
   real, necessary work. It is not evidence that R4 correctly recognizes genuine
   convergence when it exists, because no case in the cohort offers it that
   opportunity.

2. **The "0/15 trigger" result for R4 is itself a data-adequacy finding, not
   just a safety finding.** A 15-image, evenly-spaced systematic sample
   (`DEVELOPMENT_SET_15.json`) was never designed to guarantee multi-family
   same-domain co-occurrence; it was designed for broad pipeline coverage. E6
   surfaces, for the first time with numbers, that this specific development
   cohort cannot exercise DOAR's own convergence logic at all. This directly
   motivates prioritizing cohort composition (not just cohort size) in any
   future benchmark-set design work.

3. **R2's single trigger is the cleanest, smallest possible illustration of why
   evidence-family deduplication exists.** Two rules (`EN_COMPILED_HOUSE_023`,
   `EN_COMPILED_TREE_024`) both being `object_symbolism` is exactly the
   "frown+tears+sad-face" double-counting scenario
   `RULE_RELATIONSHIP_GRAPH.json`'s `same_evidence_family_edges` was built to
   prevent (see that file's own inline note). E6 confirms this protection is
   still load-bearing: remove it (R2) and the ONE case in the cohort where it
   would matter immediately over-triggers.

4. **R3 reproduces a bug DOAR's own codebase already documented and fixed.**
   `drawing_synthesis.py`'s module docstring explicitly calls domain-pooled
   convergence "the original synthesis's over-claiming bug." E6 is the first
   experiment to quantify what reverting that fix would cost: 3/15 cases
   (20%) would newly trigger, and all 3 would be cross-domain false
   convergence with zero exceptions.

5. **R5 (support ratio) is the most interesting near-miss, and worth revisiting
   with better denominators.** R5's 73.3% trigger rate is driven almost
   entirely by domains where only ONE evidence family could ever structurally
   be assessed (many rules are `blocked_structural`/`blocked_needs_unbuilt_
   feature` for a given case), so a single weak match reaches ratio=1.0.
   The idea behind R5 -- normalizing by what could realistically be checked,
   rather than an absolute count -- is scientifically reasonable and could
   plausibly help once more detectors exist and denominators are less sparse.
   Today, with a median assessable-family count of ~2 per domain, a ratio
   metric without an absolute floor is too easily saturated by N=1 evidence.
   A natural follow-up (not run here, to avoid post-hoc tuning against this
   cohort) is R5' = R5 AND matched_families>=2, which would collapse to
   R4's own floor and is worth testing once a cohort exists where the two
   can meaningfully diverge.

6. **No strategy "won" by abstaining on everything except R4, and that
   distinction matters.** R4 is not the best strategy merely because it is
   the most conservative -- it is deliberately included alongside R1-R5 as
   the CURRENT baseline, not as ground truth, per the experiment's own
   design constraint. Its 0% coverage in this specific cohort is a fact
   about this cohort intersecting with R4's design, not proof that R4 is
   optimal in general.

7. **Statistical caution.** With N=15 and several strategies producing very
   few (or zero) triggers, most pairwise McNemar comparisons are "degenerate"
   (one discordant cell is exactly 0), which mechanically produces small
   p-values without necessarily reflecting a robust, generalizable effect.
   Every comparison in this experiment is reported with its risk-difference
   effect size and bootstrap CI specifically so a reader is not misled by
   p-values alone -- see `statistics/effect_sizes.csv`.
