# E6-EXPANDED -- Corrected, Expanded Rule/Evidence Aggregation Ablation

**Supersedes** `experiments/E6_rule_aggregation/` (E6-PILOT) for
methodology and cohort size. E6-PILOT is preserved **unmodified** and
labeled provisional -- see its own README.

**Status:** see `docs/research/EXPERIMENT_LOG.md` for the frozen-cohort
completion status and `docs/research/DECISION_LOG.md` for the
architecture decision.

## What changed vs. E6-PILOT (see `PROTOCOL.md` for full detail)

1. **Eligibility governance defect found and reported** (not silently
   fixed): production's `drawing_synthesis.build_overall_synthesis` does
   not filter by `allowed_output_level` -- 13/24 (54%) of E6-PILOT's own
   matched associations were `disabled` rules. E6's own re-aggregation
   layer now applies the correct filter; production is left unmodified
   pending separate review.
2. Contradiction metric corrected: true literature contradictions,
   contextual/cultural limitations, precondition limitations, and
   duplicate/same-family relationships are now tracked as 4 distinct
   things (E6-PILOT conflated the first two).
3. R1/R2 now have a domain-scope-matched PRIMARY variant (isolating
   exactly one aggregation variable vs. R4) plus a clearly separate
   `_broad` secondary stress-test variant.
4. Outcome definitions are non-circular: "fewer than 2 independent
   families" is a POLICY difference (`policy_disagreement_vs_r4`), never
   a "structural violation" -- only genuine objective errors count as
   violations now.
5. `page_assessability` uses the real per-case value (was already fixed
   correctly within E6-PILOT's own final run; carried forward).
6. Figure jitter uses fixed explicit seeds, never `hash()`.
7. **Cohort expanded** from 15 to the largest valid non-held-out pool
   available (up to 74 candidate images: 15 original + up to 59 newly
   perceived), with 6 images reserved as a locked, untouched benchmark.

## Directory guide

| Path | Contents |
|---|---|
| `README.md` | This file. |
| `PROTOCOL.md` | Full corrected method definitions, eligibility audit, cohort provenance. |
| `config.yaml` | Fixed parameters. |
| `environment.json` | Package versions, live-call counts actually incurred. |
| `artifact_manifest.json` | Every output file, generating script. |
| `raw/` | Cohort manifest, atomic evidence, per-case/domain aggregation (machine-readable). |
| `statistics/` | Confidence intervals, paired tests, effect sizes. |
| `tables/` | Thesis-ready CSV + LaTeX tables (E6X_T1-T7). |
| `figures/` | Thesis-ready PNG+PDF figures (E6X_F1-F9) with source CSVs. |
| `error_analysis/` | Disagreement/failure cases, not cherry-picked. |
| `thesis/` | Paste-ready thesis blocks. |
| `paper/` | Condensed paper-draft sync block. |

## Quick start (reproduce from the frozen cache)

```bash
python experiments/E6_rule_aggregation_expanded/scripts/build_cohort_manifest.py
python experiments/E6_rule_aggregation_expanded/scripts/run_e6_expanded.py
python experiments/E6_rule_aggregation_expanded/scripts/compute_stats.py
python experiments/E6_rule_aggregation_expanded/scripts/make_figures.py
python experiments/E6_rule_aggregation_expanded/scripts/build_error_analysis.py
```

New perception data (if needed) is generated separately by
`scripts/e6_expand_run_live_perception.py` (repo-root `scripts/`) --
already run once for this experiment; re-running it is a no-op for every
already-cached image (checked via `find_saved_verification_rows` before
any live call).
