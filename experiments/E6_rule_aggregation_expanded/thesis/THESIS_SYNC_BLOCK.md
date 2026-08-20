# E6-EXPANDED Thesis Sync Block

Supersedes E6-PILOT's own sync block (`../../E6_rule_aggregation/thesis/
THESIS_SYNC_BLOCK.md`, now provisional). Every number here is sourced
from `tables/E6X_T1_strategy_comparison.csv` / `tables/E6X_T2_cohort_
diagnostics.csv` / `statistics/`.

---

**Experiment:** E6-EXPANDED -- Rule/Evidence Aggregation Ablation
(corrected methodology, expanded cohort)
**Status:** Complete for this session's achievable cohort. Architecture
decision: **KEEP R4 PROVISIONALLY -- insufficient positive-path
evidence.**
**Cohort:** 34 usable cases (15 from E6-PILOT + 19 newly perceived this
phase), frozen before comparison. 6 images reserved as a locked,
unprocessed, uninspected benchmark. 40 candidate images remain
unprocessed (Gemini API quota exhausted mid-run; genuine, disclosed
blocker, not a silent reduction).
**Checkpoint:** `checkpoint/human-interaction-v1` (no scientific code
changed).

**Methodology corrections applied vs. E6-PILOT** (see `../PROTOCOL.md`):
a rule-eligibility governance defect was found and reported (production
does not filter by `allowed_output_level`; 54% of E6-PILOT's own matched
associations were `disabled` rules) -- E6's own analysis layer now
applies the correct filter, production is left unmodified pending
separate review; contradiction metric corrected (true literature
contradiction vs. contextual limitation are no longer conflated); R1/R2
now domain-scope-matched to R3/R4/R5 for the primary comparison;
outcome definitions are non-circular ("<2 families" is a policy choice,
never a structural error); figure jitter uses fixed seeds.

**Key result table (N=34, Wilson 95% CI):**

| Strategy | Trigger % | Supported coverage % | Structural violation % |
|---|---|---|---|
| R1 (primary) | 58.8 (42.2-73.6) | 58.8 | 0.0 |
| R1_broad (stress-test) | 85.3 (69.9-93.6) | 29.4 | 55.9 (all ineligible-evidence use) |
| R2 (primary) | 0.0 (0.0-10.2) | 0.0 | 0.0 |
| R2_broad (stress-test) | 8.8 (3.0-23.0) | 0.0 | 8.8 |
| R3 | 8.8 (3.0-23.0) | 0.0 | 8.8 (100% of own triggers, cross-domain) |
| **R4 (current DOAR)** | **0.0 (0.0-10.2)** | **0.0** | **0.0** |
| R5 | 50.0 (34.1-65.9) | 50.0 | 0.0 |

**Cohort adequacy (the central finding of this phase):** even at N=34
(>2x the pilot), **0/34 cases have >=2 independent evidence families
within a single concern domain** -- R4's own trigger condition remains
completely unexercised on its positive (accept) path. 5/34 (14.7%) cases
have >=2 independent families overall (i.e. across DIFFERENT domains),
confirming that when multiple pieces of evidence co-occur in this
dataset, they consistently span domains rather than converging within
one -- exactly what R3's 100% cross-domain-violation rate independently
corroborates. R2 and R4 are now numerically IDENTICAL on this cohort
(0/34 both, 0 discordant pairs) because 0/34 cases have any duplicate-
family pair either -- family deduplication has had no opportunity to
matter in either direction yet.

**Statistics:** Cochran's Q (R1,R2,R3,R5) = 48.324, df=3, p<0.001.
R1_vs_R4 and R5_vs_R4 both Holm-significant with large paired risk
differences (+0.588 and +0.500 respectively, bootstrap 95% CI excluding
0). R2_vs_R4: 0 discordant pairs, p=1.0 -- genuinely tied, not merely
underpowered.

**Central finding:** with the eligibility/circularity/domain-scope
corrections applied, R1 and R5's "supported coverage" is no longer
artificially zero (58.8% and 50.0% respectively) -- but this coverage is
built entirely on SINGLE-family evidence (a POLICY choice, not an error)
inside domains where R4 requires two. R3 remains the only primary
strategy whose triggers are majority structurally invalid (cross-domain
pooling, 100% of its own triggers). R4 still never triggers, at N=34, so
its true-positive sensitivity remains as untested as it was at N=15.

**Architecture decision:** KEEP R4 PROVISIONALLY. The corrected analysis
strengthens confidence that R4's design PRINCIPLE (same-domain,
deduplicated, >=2-independent-family floor) is sound -- R2/R4 tying
exactly and R3's 100% cross-domain-violation rate both independently
support requiring same-domain convergence specifically. But the decision
remains provisional because the expanded cohort still cannot show R4
correctly ACCEPTING a genuine positive case -- 34 real, naturally-sampled
children's drawings from this pool have simply never produced one.

**Limitation to always state alongside this result:** 40/59 targeted new
images remain unprocessed due to Gemini API quota exhaustion (a genuine,
disclosed environmental constraint, not a scientific limitation); a
locked 6-image benchmark remains reserved and untouched for a future,
genuinely blind final check once an aggregation decision is finalized.

**Figures/tables to cite:** `E6X_F1_coverage_vs_structural_risk.png`
(primary trade-off plot, now non-circular), `E6X_T2_cohort_diagnostics.
csv` (the 0/34 same-domain-convergence finding), `E6X_F9_SKIPPED_no_r4_
triggers.txt` (explicit record that no same-domain convergence example
exists to show).

---

## Phase 2 addendum (2026-08-20): feasibility audit + production fix

**Thesis-safe statement.** A static audit of `RULE_EVIDENCE_MATRIX.csv`
(`../REGISTRY_FEASIBILITY_REPORT.md`, `E6X_T8_registry_feasibility.csv`)
found that R4's positive path is reachable through exactly ONE
domain/rule-pair combination in the current registry
(`anxiety_or_stress_related`: `PSY_AR_SIZE_SMALL_016` +
`EN_COMPILED_LINE_LIGHT_PRESSURE_031`, out of 8 registry domains) -- **not
zero (R4 is not structurally impossible) and not many**. Separately, this
phase traced and CONFIRMED a production defect first reported in Phase 1:
`build_eligible_matches`/`build_deterministic_eligible_matches` applied no
filter on a rule's `allowed_output_level`, letting `disabled` rules (31 of
41) reach governed synthesis whenever their (separately gated) visual
precondition was satisfied. **This defect is now FIXED** in
`src/doar/reasoning_chain.py` and `src/doar/drawing_synthesis.py` (an
explicit `allowed_output_level == "individual_heuristic_only"` gate at the
two, and only two, choke points where a match is constructed).
**Measured impact**, replaying the same 34 cached cases before/after (zero
new API calls): 19/34 cases affected, 27 disabled matches removed, and
**one case (`p2b_0007`) lost a real candidate hypothesis** that had been
partly generated from a `disabled` rule -- a genuine, previously-hidden
false positive in deployed behavior, now corrected. A predeclared,
zero-API-cost screen of the one reachable domain/rule-pair against all 34
cached cases (`../eligibility_fix/E6_STRESS_TARGETED_screen.csv`) found
**0/34 qualify**, and more specifically 0/34 even satisfy the size-small
half alone -- `bounding_box_coverage` sits at or near 1.0 for nearly every
case in this corpus, most plausibly a photography/cropping artifact of how
these images were captured, a new measurement-validity limitation distinct
from sample size. **Decision: KEEP R4 PROVISIONALLY stands**; the
recommended next step is re-running the same zero-cost screen against the
40 still-unprocessed candidate images before any further live perception
spending. Full detail: `../E6X_PHASE2_DECISION.md`.
