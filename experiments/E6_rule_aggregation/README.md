# E6-PILOT -- Rule / Evidence Aggregation Ablation

> **THIS RUN IS NOW LABELED E6-PILOT / PROVISIONAL, SUPERSEDED.**
> Preserved unmodified as a historical record. Its methodology had four
> issues later found and corrected (see
> `../E6_rule_aggregation_expanded/PROTOCOL.md`): an eligibility-
> governance defect (disabled rules were silently contributing to 54% of
> matched associations), circular outcome definitions ("<2 families"
> counted as a structural error rather than a policy choice), an R1/R2
> domain-scope confound vs. R3/R4/R5, and a contradiction-vs-contextual-
> limitation conflation. **Do not cite the numbers below as final** -- see
> `../E6_rule_aggregation_expanded/` for the corrected, expanded result
> and `../../docs/research/DECISION_LOG.md` for the current architecture
> decision.

**Status (as originally run):** Complete. **Decision (as originally
made, now superseded):** KEEP current aggregation (R4). See
`../../docs/research/DECISION_LOG.md` for the CURRENT decision.

**Research question:** Which rule/evidence aggregation strategy gives the best
trade-off between useful supported coverage and resistance to duplicate,
cross-domain, or otherwise unsupported convergence?

## Quick start (reproduce everything from the frozen cache)

```bash
python experiments/E6_rule_aggregation/scripts/run_e6.py       # raw/ extraction
python experiments/E6_rule_aggregation/scripts/compute_stats.py # statistics/, tables/
python experiments/E6_rule_aggregation/scripts/make_figures.py  # figures/
```

No live Gemini/model calls are made by any of these three scripts -- every
number comes from the already-cached development-set Observer/Verifier JSON
plus the frozen `RULE_EVIDENCE_MATRIX.csv`/`RULE_RELATIONSHIP_GRAPH.json`/
`CONCERN_DOMAIN_MAP.json`, read via the same production loaders
(`scripts/clinician_review_app.py`, `src/doar/reasoning_chain.py`,
`src/doar/drawing_synthesis.py`) the live app uses.

## Directory guide

| Path | Contents |
|---|---|
| `README.md` | This file. |
| `PROTOCOL.md` | Full method definitions, data-readiness audit, and every design decision's rationale. |
| `config.yaml` | Fixed parameters (MIN_EVIDENCE, R5 threshold, bootstrap seed/reps). |
| `environment.json` | Package versions used to produce these results. |
| `artifact_manifest.json` | Every output file this experiment produces, with its generating script. |
| `raw/` | Per-case, per-domain, per-strategy raw data (machine-readable). |
| `statistics/` | Confidence intervals, paired tests, effect sizes. |
| `tables/` | Thesis-ready CSV + LaTeX tables (E6_T1-T5). |
| `figures/` | Thesis-ready PNG+PDF figures (E6_F1-F6) with their own source CSVs. |
| `error_analysis/` | Qualitative failure/disagreement cases, not cherry-picked. |
| `thesis/` | Paste-ready thesis blocks (results paragraph, discussion points, captions). |
| `paper/` | Condensed paper-draft sync block. |

## Headline result

| Strategy | Trigger % (N=15) | Supported coverage % | Structural violation % of its own triggers |
|---|---|---|---|
| R1 Any eligible rule | 86.7 | 0.0 | 100% |
| R2 Raw rule-count | 6.7 | 0.0 | 100% |
| R3 Pooled cross-domain families | 20.0 | 0.0 | 100% |
| **R4 Same-domain families (current DOAR)** | **0.0** | **0.0** | **n/a (never triggers)** |
| R5 Support ratio | 73.3 | 0.0 | 100% |

Every alternative to the current DOAR aggregation over-claims 100% of the time
it fires, in this cohort. The current rule never fires on this cohort either --
this is disclosed as a cohort limitation, not hidden. Full detail in
`PROTOCOL.md` and `thesis/THESIS_SYNC_BLOCK.md`.
