# E6X Registry Feasibility Report — Is R4 Structurally Reachable?

**Method:** static audit only. No cases, no API calls, no aggregation runs.
Reads `RULE_EVIDENCE_MATRIX.csv` and `RULE_RELATIONSHIP_GRAPH.json` as they
exist today. Source table: `tables/E6X_T8_registry_feasibility.csv`.
Generating script: `scripts/build_registry_feasibility.py`.

**R4's real trigger condition** (`drawing_synthesis.py`, unchanged): within
a SINGLE concern domain, evidence from `>=2` distinct evidence families
(`RULE_RELATIONSHIP_GRAPH.json`'s `same_evidence_family_edges`) must
independently converge. R4's domain-scoped convergence check only ever runs
on the six "in-scope" domains — the five distress domains plus
`positive_affect_or_social_engagement` — never on the two neutral domains.

## Per-domain enabled-rule / enabled-family inventory

| Domain | In R4 scope | Enabled rules | Enabled families | Max independent families | R4 threshold reachable |
|---|---|---|---|---|---|
| anxiety_or_stress_related | yes | 2 (`PSY_AR_SIZE_SMALL_016`, `EN_COMPILED_LINE_LIGHT_PRESSURE_031`) | `size_composition`, `line_intensity_quality` | **2** | **YES** |
| aggression_or_threat_related | yes | 1 (`EN_COMPILED_LINE_HEAVY_PRESSURE_030`) | `line_intensity_quality` | 1 | NO |
| depressive_or_low_mood_related | yes | 0 | — | 0 | NO |
| developmental_or_attention_related | yes | 1 (`EN_COMPILED_LINE_SHAKY_BROKEN_032`) | `line_fragmentation_quality` | 1 | NO |
| positive_affect_or_social_engagement | yes | 1 (`PSY_AR_PLACE_RIGHT_019`) | `spatial_placement` | 1 | NO |
| social_withdrawal_related | yes | 1 (`PSY_AR_PLACE_LEFT_018`) | `spatial_placement` | 1 | NO |
| neutral_descriptive_only | no (out of R4 scope) | 2 | `size_composition`, `spatial_placement` | 2 | NO — never scored by R4 regardless of family count |
| neutral_descriptive_only_no_construct_proposed | no (out of R4 scope) | 2 | `size_composition`, `spatial_placement` | 2 | NO — never scored by R4 regardless of family count |

`maltreatment_or_safety_concern` carries 0 candidate rules by design
(`CONCERN_DOMAIN_MAP.json`) and is excluded from this table entirely.

## Finding

**Exactly one concern domain — `anxiety_or_stress_related` — is
theoretically reachable for R4's positive (accept) path under the CURRENT
registry.** It has two enabled (`individual_heuristic_only`) rules from two
distinct evidence families, both with their visual/deterministic
prerequisites already satisfied in DOAR:

- `PSY_AR_SIZE_SMALL_016` (family `size_composition`; precondition
  `segmentation_and_bounding_box_coverage_available`, satisfied) — page
  coverage below ~20%.
- `EN_COMPILED_LINE_LIGHT_PRESSURE_031` (family `line_intensity_quality`;
  precondition `stroke_pressure_appearance_proxy_feature_available`,
  satisfied) — light/faint line-pressure appearance.

**A single case that visually shows BOTH a small drawing (low page
coverage) AND light/faint line pressure would be the only currently
possible way for R4 to trigger in production.** No other domain has two
enabled families to draw on: five of the eight domains present in the
registry have 0 or 1 enabled family, and the two neutral domains (which do
have 2 enabled families each) are structurally out of R4's scope regardless
— `_domain_scoped_convergence` is never even called on them.

Therefore, the correct framing is **NOT** "R4 positive-path convergence is
structurally impossible under the current registry" (0/8 domains reachable
would be required for that statement) — it IS reachable, but through
exactly one narrow, specific rule-pair path in one domain. E6-EXPANDED's
0/34 empirical result is consistent with this path being real but rare: it
requires a case where a child's drawing is both small AND drawn with light
pressure, and 34 naturally-sampled cases happened to contain none.

## What this changes about the E6-EXPANDED interpretation

The prior framing ("0/34 same-domain families, may need more enabled
detectors in general") is now sharper: **the single most useful next
observation would specifically be a case satisfying BOTH
`PSY_AR_SIZE_SMALL_016` and `EN_COMPILED_LINE_LIGHT_PRESSURE_031`
simultaneously** — not detector expansion in general, and not blind image
volume. This directly motivates Section 4's Decision Option C (a
predeclared, prerequisite-based stress-cohort selection method targeting
this exact rule pair) rather than either unconditional further collection
(Option A) or a blanket stop (Option B).

**No registry file was modified to produce this report.**
`RULE_EVIDENCE_MATRIX.csv`, `RULE_RELATIONSHIP_GRAPH.json`,
`CONCERN_DOMAIN_MAP.json` are all read-only inputs, unchanged.
