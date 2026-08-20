# Eligibility Fix — Before/vs/After Impact Report

**Method:** replayed the 34 already-cached E6-EXPANDED cases through the
pipeline twice — "before" (exact reconstruction of the unfixed
`build_eligible_matches`/`build_deterministic_eligible_matches`, promoting
every `satisfied` precondition check regardless of `allowed_output_level`)
and "after" (the real, now-fixed production functions, called directly, no
reimplementation). Zero API calls, zero new perception. Generating script:
`scripts/build_eligibility_impact_analysis.py`. Figure:
`figures/E6X_F10_before_after_disabled_rule_impact.{png,pdf,csv}`. Raw
per-case data: `raw/eligibility_before_after_per_case.csv` and
`tables/E6X_T9_eligibility_fix_impact.csv` (identical content).

## Headline numbers

| Metric | Value |
|---|---|
| N cases replayed | 34 |
| N cases affected by the fix (≥1 disabled match removed) | 19 (55.9%) |
| Total disabled matches removed across all cases | 27 |
| N cases where `overall_synthesis.level` changed | 9 |
| N cases where `candidate_hypotheses` changed | 1 |

## Every `overall_synthesis.level` transition observed

| Before → After | N cases |
|---|---|
| `limited_association` → `insufficient_interpretable_evidence` | 5 |
| `descriptive_only` → `insufficient_interpretable_evidence` | 2 |
| `limited_association` → `descriptive_only` | 1 |
| `convergent_concern_pattern` → `limited_association` | 1 |

**No unexpected regression**: every single transition moves toward LESS
evidence / a weaker claim, never the reverse (`insufficient_interpretable_
evidence` < `descriptive_only` < `limited_association` < `convergent_*` <
`mixed_evidence`). The analysis script asserts `rule_ids_after ⊆
rule_ids_before` for every case and this held for all 34 — the fix can only
remove evidence, never add it, and the observed level transitions are
exactly consistent with that.

## The one case where a human-facing candidate hypothesis actually changed

**`p2b_0007`** (`developmental_or_attention_related` domain):

- **Before the fix**: `EN_COMPILED_LINE_SHAKY_BROKEN_032` (enabled,
  family `line_fragmentation_quality`) + `PSY_AR_GEOMETRY_008` (**disabled**,
  family `shape_symbolism`, source: a weak-evidence Arabic pop-psychology
  claim about "repeated geometric shapes" indicating "planning, control,
  persistence, or stubbornness" — `DETECTOR_UNAVAILABLE`, `confidence_
  ceiling=0.15`) together reached 2 independent families in one domain →
  `overall_synthesis.level == "convergent_concern_pattern"` →
  **1 candidate hypothesis** at `WEAK_HYPOTHESIS` ("Possible
  attention/developmental-related pattern worth noting").
- **After the fix**: only `EN_COMPILED_LINE_SHAKY_BROKEN_032` remains (1
  family) → `overall_synthesis.level == "limited_association"` → **0
  candidate hypotheses**.

This is a genuine, previously-hidden false positive that the fix corrects:
a clinician/parent-facing hypothesis was being generated in production
partly on the basis of a rule whose own registry status says its detector
is unbuilt and unvalidated. This is exactly the class of error Section 2
of this phase's task was meant to close, and it is now measurably closed
— on real cached case data, not merely a synthetic example.

## What did NOT change

- Zero cases gained a hypothesis they didn't have before (only lost, or
  kept the same — 33/34 unchanged, 1/34 lost exactly one).
- `check_visual_preconditions`/`check_deterministic_preconditions`
  themselves are byte-for-byte unchanged — the Technical/debug view
  (`clinician_review_app.py`'s `bundle["checks"]`) still shows every
  disabled rule's precondition status exactly as before; only PROMOTION
  into governed evidence is gated.
- `RULE_EVIDENCE_MATRIX.csv`, `RULE_RELATIONSHIP_GRAPH.json`,
  `CONCERN_DOMAIN_MAP.json` are unmodified — nothing about which rules
  exist, their domains, families, or thresholds changed.
