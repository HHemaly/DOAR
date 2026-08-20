# DOAR Research Program -- Known Limitations

Cross-experiment limitations register. Append a section per experiment;
never delete -- a limitation resolved by a later experiment gets a
"RESOLVED by E<n>" note added, not removed.

---

## E6-PILOT -- Rule / Evidence Aggregation Ablation (provisional, superseded by E6-EXPANDED below)

1. **Cohort cannot exercise R4's true-positive path.** Zero of the 15
   development-set cases have >=2 independent evidence families within a
   single concern domain. Every comparison of R4 to an alternative strategy
   in E6 is therefore a comparison of an always-abstaining baseline against
   increasingly permissive alternatives -- it measures R4's (perfect)
   resistance to false convergence, not its sensitivity to true convergence.
   **What would resolve this:** a cohort (or cohort extension) containing at
   least a handful of cases with genuine multi-family, same-domain evidence
   co-occurrence, ideally verified by a domain expert as a legitimate
   pattern rather than selected post-hoc to favor any one strategy.

2. **N=15 is small for the paired statistical tests used.** Several pairwise
   McNemar comparisons are "degenerate" (one discordant cell = 0), which
   produces small p-values somewhat mechanically. Every such result is
   reported alongside its risk-difference effect size and bootstrap CI
   specifically to avoid over-interpreting the p-value alone (see
   `experiments/E6_rule_aggregation/statistics/`). Do not treat any single
   p-value in this experiment as conclusive; treat the CONSISTENT PATTERN
   (100% violation rate across 4 independently-computed alternative
   strategies) as the primary evidence.

3. **R6 (weighted aggregation) was not implemented, not evaluated.**
   `confidence_ceiling` is `"not_set_pending_review"` for 18/41 rules (44%)
   and `evidence_strength_as_written` is unstructured free text. Any future
   weighted-aggregation experiment first requires a scientifically-justified
   (not post-hoc) numeric strength scale to be defined and populated for
   all 41 rules -- that is a prerequisite research task in itself, not a
   parameter this experiment could safely choose.

4. **Duplicate-family and contradiction scenarios are each observed on only
   1 and 9 cases respectively (of 15), and never in combination with an
   actual convergent trigger.** `RULE_RELATIONSHIP_GRAPH.json`'s protective
   edges are confirmed structurally present and load-bearing (E6 shows
   removing family-dedup immediately causes over-triggering on the one case
   where it matters) but are not stress-tested at scale.

5. **Only 41 rules exist in the frozen matrix, and only 10 currently carry
   `allowed_output_level="individual_heuristic_only"`** (the rest are
   `disabled`, i.e. their detectors are not built) -- E6's findings describe
   the aggregation LOGIC's behavior given the CURRENT detector coverage;
   they do not predict how the same logic will behave once more of the 41
   rules' detectors exist and more evidence families become simultaneously
   assessable per case.

**RESOLVED by E6-EXPANDED:** limitations 1 and 4 above were directly
re-tested on an expanded, corrected cohort (see below) -- both persisted
rather than resolved, which is itself the key finding of that phase.
Limitation 2 (N=15 statistical power) is improved but not eliminated at
N=34. Limitations 3 and 5 remain fully open, untouched by E6-EXPANDED.

---

## E6-EXPANDED -- Rule / Evidence Aggregation Ablation (corrected methodology, expanded cohort)

1. **The same-domain-convergence blind spot survived cohort expansion --
   it is now better evidenced as a structural property, not a small-sample
   artifact.** Zero of 34 cases (up from zero of 15) have >=2 independent
   evidence families within a single concern domain. Every comparison of
   R4 to an alternative strategy in this experiment remains, as in
   E6-PILOT, a comparison of an always-abstaining baseline against
   increasingly permissive alternatives on their FALSE-convergence
   resistance only -- R4's sensitivity to TRUE convergence is still
   entirely unmeasured. **PARTIALLY RESOLVED/EXPLAINED by E6-EXPANDED
   Phase 2 (2026-08-20):** a static registry feasibility audit
   (`experiments/E6_rule_aggregation_expanded/REGISTRY_FEASIBILITY_
   REPORT.md`) found R4's positive path is reachable through exactly ONE
   two-rule combination in ONE domain (`anxiety_or_stress_related`:
   `PSY_AR_SIZE_SMALL_016` + `EN_COMPILED_LINE_LIGHT_PRESSURE_031`), never
   any other of the 8 registry domains -- this is now understood to be a
   narrow structural-reachability constraint, not merely "not enough
   images yet". A predeclared, zero-API-cost screen of that exact
   combination against all 34 cached cases found 0/34 meet it, and more
   specifically 0/34 even satisfy the size-small half alone (see
   limitation 8 below for why). **What would still resolve this:**
   re-running the same zero-cost screen (`scripts/
   screen_e6_stress_targeted.py`) against the 40 still-unprocessed
   candidate images (needs only the raw image file, no Gemini call) is the
   correct next, lowest-cost step -- recommended ahead of any further live
   perception spending.

2. **A production scientific-governance defect was found and is NOT fixed
   by this experiment.** `build_overall_synthesis` and the rest of the
   frozen synthesis pipeline apply no filter anywhere on a rule's own
   `allowed_output_level` field; rules marked `disabled` in
   `RULE_EVIDENCE_MATRIX.csv` (31 of 41) can and do reach the governed
   synthesis today whenever their (separate) visual/deterministic
   preconditions are satisfied. Concrete evidence: 13 of E6-PILOT's own 24
   matched associations (54%) were such rules. Per the governing
   instruction for this phase, this is reported here as a defect requiring
   separate remediation, not silently patched; E6's own external
   re-aggregation layer applies the correct filter for its own comparison
   only, and `reasoning_chain.py`/`drawing_synthesis.py`/
   `RULE_EVIDENCE_MATRIX.csv` remain completely unmodified. **RESOLVED by
   E6-EXPANDED Phase 2 (2026-08-20):** `reasoning_chain.
   build_eligible_matches` and `drawing_synthesis.build_deterministic_
   eligible_matches` now gate on `allowed_output_level ==
   "individual_heuristic_only"` before promoting a satisfied precondition
   check into governed evidence. Measured impact on the 34 cached cases:
   19/34 affected, 27 disabled matches removed, 9 cases' synthesis level
   changed, 1 case (`p2b_0007`) lost a real candidate hypothesis that had
   been partly generated from a disabled rule -- a genuine, previously-
   hidden false positive, now corrected. See `experiments/
   E6_rule_aggregation_expanded/eligibility_fix/
   ELIGIBILITY_FIX_IMPACT_REPORT.md` and `DECISION_LOG.md`.
   `RULE_EVIDENCE_MATRIX.csv`/`RULE_RELATIONSHIP_GRAPH.json`/
   `CONCERN_DOMAIN_MAP.json` were NOT modified -- only the two code choke
   points that read them.

3. **40 of 59 targeted new candidate images remain unprocessed.** A
   persistent HTTP 429 quota exhaustion stopped live Gemini perception
   generation after 19/59 successes (5 consecutive failures, checkpointed
   cleanly; a resume attempt confirmed the quota was still exhausted). This
   is disclosed as an environmental constraint, not folded into any
   scientific conclusion -- the 19 images that WERE processed showed the
   same 0-same-domain-convergence pattern as the original 15, offering no
   positive signal the remaining 40 would differ, but this is stated as an
   expectation only, since those 40 images were never inspected. **What
   would resolve this:** re-running
   `scripts/e6_expand_run_live_perception.py` once Gemini API quota
   resets/increases; the script is idempotent and checkpointed
   (`outputs/human_interaction_v1/e6_expand_progress.jsonl`), so a re-run
   will only process the remaining unprocessed candidates.

4. **N=34 remains modest for the paired statistical tests used, though
   improved from E6-PILOT's N=15.** As before, every pairwise comparison is
   reported alongside its risk-difference effect size and bootstrap CI, not
   the p-value alone (see `experiments/E6_rule_aggregation_expanded/
   statistics/`). The CONSISTENT PATTERN across both cohort sizes (R2/R4
   tying exactly; R3 at 100% cross-domain violation on its own triggers in
   both) is treated as the primary evidence, not any single test.

5. **R6 (weighted aggregation) remains not implemented, not evaluated** --
   unchanged from E6-PILOT; `confidence_ceiling` and
   `evidence_strength_as_written` are still not consistently numeric across
   the 41 rules. Not revisited in this phase.

6. **The 6-image locked benchmark
   (`p2b_0013, p2b_0015, p2b_0048, p2b_0057, p2b_0064, p2b_0073`) remains
   reserved, unprocessed, and uninspected** -- deliberately, so it retains
   its value as a genuinely blind final check once an aggregation-strategy
   decision is treated as final rather than provisional. Note:
   `p2b_0061` (also `original_split=="test"`) was already used in
   E6-PILOT before this phase began and is disclosed here as a pre-existing
   fact, not retroactively excludable.

7. **An optional E6-STRESS cohort (naturally-occurring same-domain
   multi-family / duplicate-family / cross-domain / contradiction cases)
   was considered but not built.** The Section-D diagnostics computed for
   the full 34-case cohort already show 0 duplicate-family pairs, 0
   true-contradiction cases, and 0 same-domain multi-family cases --
   there is no additional naturally-occurring material beyond the 5
   cross-domain-convergence cases / 3 R3-triggered cases already fully
   captured in `tables/E6X_T5_policy_disagreement.csv` and
   `tables/E6X_T6_structural_failures.csv`. Building a separate stress
   cohort from this same pool would not add distinct cases, only relabel
   ones already reported. **UPDATE, E6-EXPANDED Phase 2 (2026-08-20):** a
   PREDECLARED, criteria-first version of this idea was built --
   `E6-STRESS-TARGETED` (see limitation 1 and 8) -- but its selection
   criteria (the one registry-reachable rule pair) is far narrower than
   the general notion originally considered here, and also returned zero
   qualifying cases. This limitation's original "not built" framing still
   stands for the general/undifferentiated version; only the specific,
   registry-motivated targeted version was attempted.

8. **`segmentation.bounding_box_coverage` is at or near 1.0 for nearly
   every one of the 34 cached cases, added by E6-EXPANDED Phase 2
   (2026-08-20).** The predeclared `E6-STRESS-TARGETED` screen
   (`scripts/screen_e6_stress_targeted.py`,
   `eligibility_fix/E6_STRESS_TARGETED_screen.csv`) found 0/34 cases
   satisfy `PSY_AR_SIZE_SMALL_016`'s own `<=0.20` threshold at all -- not
   merely "0/34 satisfy it in combination with light line pressure"
   (limitation 1) but 0/34 satisfy it AT ALL, alone. This most plausibly
   reflects how these specific images were captured -- photographs/scans
   of drawings typically already cropped tightly to the drawn content,
   rather than full-page images with visible surrounding margin -- rather
   than genuine absence of small-figure drawing among these children. This
   is a MEASUREMENT-VALIDITY concern distinct from limitation 1's data-
   volume framing: even unlimited further images from the SAME capture
   pipeline may never exercise this feature meaningfully. **What would
   resolve this:** verify whether any of the 74 candidate images (34
   processed + 40 pending) were captured with visible page margin intact,
   and/or re-examine whether `bounding_box_coverage` computed against a
   cropped/photographed input is the right operationalization of the
   literature construct `PSY_AR_SIZE_SMALL_016` is actually describing
   (full-page coverage fraction as originally intended by the source
   material) before concluding anything about this feature's real-world
   prevalence from this corpus.
