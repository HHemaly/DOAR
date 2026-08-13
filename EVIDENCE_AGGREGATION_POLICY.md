# DOAR — Evidence Aggregation Policy

Defines how multiple pieces of evidence combine into a candidate clinical
hypothesis, without inventing a diagnostic probability. This is a policy
document — it formalizes and extends `src/doar/concerns.py`'s **already
implemented, already unit-tested** convergence engine; it does not change
that module's code or its `CONCERNS_ENABLED = False` production state.

## 1. Why no numeric probability

`RULE_EVIDENCE_AUDIT.md` Section 5 found two real, published, standardized
multi-indicator drawing-scoring systems (Koppitz Emotional Indicators,
DAP:SPED) — neither is transferable to DOAR (instructed protocol, DOAR
lacks the required detectors, and both show weak psychometrics even in
their native context: Koppitz test–retest phi <0.43 for 28/30 items;
DAP:SPED correlations reported as "at best small"). **No literature found
this session or previously supplies a validated weight, effect size, or
cutoff DOAR could borrow.** Inventing one — e.g. "this rule counts for
0.3, that one for 0.5" — would be indistinguishable from fabrication.
`SCIENTIFIC_LIMITATIONS.md` already states this as a standing rule
(§3: "DOAR must never produce an... probability... or similar clinical
score"); this document operationalizes it for the hypothesis layer
specifically.

## 2. What DOES exist: transparent counting, not scoring

`concerns.py::_aggregation_strength` already implements exactly the kind of
transparent, non-numeric aggregation this policy formalizes:

| Strength (existing, `concerns.py`) | Condition (existing code) |
|---|---|
| `INSUFFICIENT` | Fewer than 2 matched evidence IDs |
| `WEAK_HYPOTHESIS` | ≥2 evidence IDs, but only 1 source type (e.g. two clinician-rule hits — still one source, two guesses) |
| `POSSIBLE_FOR_EXPLORATION` | ≥2 evidence IDs from ≥2 distinct source types, ≥1 non-clinician-symbolic |
| `PROFESSIONAL_REVIEW_SUGGESTED` | ≥3 distinct source types (not reachable with real data until Phase 3 detectors ship) |

This policy extends that same transparent-counting principle to
per-hypothesis reporting, once `CONCERN_DOMAIN_MAP.json`'s domains are ever
implemented. **Nothing below changes `concerns.py`'s actual thresholds or
enables it in production** — this is the reporting format a future,
domain-aware version would use.

## 3. Candidate hypothesis report format (design only)

Replacing any single number, a candidate hypothesis reports:

```
Candidate hypothesis: <concern domain, e.g. "Depressive/low-mood presentation">
Evidence families represented: <n> / <total families relevant to this domain>
Supporting indicators: <n> (list of rule_ids / evidence_ids)
  Evidence strength breakdown: <n> strong / <n> moderate / <n> limited
    (strong/moderate/limited = the SOURCE's own `evidence_level_as_written`
     grade, e.g. "Moderate for basic emotion only" vs. "Speculative" —
     never a DOAR-invented number; see RULE_EVIDENCE_MATRIX.csv)
Contradicting evidence: <n> (list; e.g. a `SUPPORTS` rule co-occurring with
  literature that CONTRADICTS the same construct, per
  RULE_RELATIONSHIP_GRAPH.json's literature_level_contradictions)
Recurring findings: <n> / <total drawings in case> (longitudinal modifier,
  RULE_EVIDENCE_AUDIT.md Section 6 — first_occurrence/repeated/persistent/
  increasing/decreasing, never a probability)
Aggregation strength: INSUFFICIENT | WEAK_HYPOTHESIS | POSSIBLE_FOR_EXPLORATION
  | PROFESSIONAL_REVIEW_SUGGESTED  (concerns.py's existing 4-level scale)
```

Example (hypothetical — not a real case output):

```
Candidate hypothesis: Depressive/low-mood presentation
Evidence families represented: 2/6 (facial_feature_style, colour_mood_flags)
Supporting indicators: 2 (EN_COMPILED_FACE_EXPRESSION_021, EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038)
  Evidence strength breakdown: 0 strong / 1 moderate / 1 limited
Contradicting evidence: 1 (REF_CRAWFORD_COLOUR_CAUTION_2012 — colour use alone is not a reliable emotional indicator)
Recurring findings: 1/3 drawings (first_occurrence)
Aggregation strength: WEAK_HYPOTHESIS (single source type: both are clinician-symbolic rules)
```

Never: "73% probability of depression." Always: the counts above, plus the
existing `parent_safe_wording`/`professional_wording` hedged phrasing
(`rules_registry_v2.json`) and a disclaimer (see `CLINICIAN_OUTPUT_SCHEMA.md`).

## 4. Non-double-counting rule (evidence families)

`RULE_RELATIONSHIP_GRAPH.json`'s `same_evidence_family_edges` exists
specifically so a frown, tears, and a sad-face label (all
`facial_feature_style`) are never counted as 3 independent supporting
indicators — they are 1 family, however many individual rules within it
fire. "Evidence families represented: N/M" in Section 3's report format
is the number that matters; "supporting indicators: N" is reported
alongside it but is explicitly the weaker of the two signals, exactly the
lesson from `concerns.py`'s existing source-type-diversity requirement
(two same-source hits stay `WEAK_HYPOTHESIS`, never promoted just because
the raw count went up).

## 5. Explicit non-goals

- No weight, coefficient, or cutoff from Koppitz, DAP:SPED, or any other
  external scoring system is used, borrowed, or approximated anywhere.
- No index in this policy is ever labelled a probability, likelihood, risk
  score, or percentage. If a future implementation adds a numeric summary,
  it must be explicitly named an **Evidence/Review Index** (an ordinal
  count-derived signal for triage/sorting only) and carry, every time it is
  displayed, the sentence: *"This number reflects how much independent
  evidence converges, not the likelihood of any diagnosis."*
- Aggregation strength can never exceed what `concerns.py`'s existing,
  tested logic would compute from the same evidence — this policy adds
  reporting detail, not a new or looser threshold.
