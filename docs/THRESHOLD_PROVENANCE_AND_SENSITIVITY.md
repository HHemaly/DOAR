# Threshold Provenance and Sensitivity (DOAR-TRACE Phase 2A, Section 6)

## Provenance vocabulary (used consistently across all 10 executable rules)

| Category | Meaning |
|---|---|
| `directly_sourced` | The source text states a specific numeric value/percentage |
| `source_centre_with_invented_band` | The source gives a center point in words (e.g. "about 50%"); the band width around it is an implementation choice |
| `invented_operational_standin` | The source gives no numeric anchor at all (e.g. "covers the whole page", "top", "left") — a number was invented to operationalize the qualitative claim |
| `expert_defined` | A domain expert directly set the value (not used by any current rule — none of the current thresholds came from a live expert session, only from source text or real-data quartiles) |
| `empirically_exploratory` | Derived from a real, non-test, class-balanced data sample (quartile cut points) — **never** tuned to agree with folder labels; used for the 3 new stroke-proxy rules, which have no numeric anchor in either source PDF at all |

This replaces the Phase 1.5 vocabulary (`sourced_center_invented_band`,
`invented_numeric_stand_in`, `invented_no_anchor`) with the above
5-category scheme; `registry_v2_build.py`'s existing 6 rules were
migrated to it (the distinction previously captured by
`invented_no_anchor` vs. `invented_numeric_stand_in` is preserved in
each rule's free-text `rationale`, just not as a separate top-level
category).

## The 6 original tier-1 rules

| Rule | Feature | Current threshold | Source status | Rationale |
|---|---|---|---|---|
| `PSY_AR_SIZE_SMALL_016` | `bounding_box_coverage` | ≤ 0.20 | `directly_sourced` | Arabic source states this percentage exactly ("لا تتجاوز 20%") |
| `PSY_AR_SIZE_HALF_014` | `bounding_box_coverage` | 0.40 ≤ x ≤ 0.60 | `source_centre_with_invented_band` | Source states "about 50%"; the ±10pp band is an implementation choice |
| `PSY_AR_SIZE_FULL_015` | `bounding_box_coverage` | ≥ 0.90 | `invented_operational_standin` | Source says only "covers the whole page" (no percentage) |
| `PSY_AR_PLACE_TOP_017` | `centroid_normalized[1]` | cy < 0.40 | `invented_operational_standin` | Source gives no numeric boundary for "top" at all |
| `PSY_AR_PLACE_LEFT_018` | `centroid_normalized[0]` | cx < 0.40 | `invented_operational_standin` | Source gives no numeric boundary for "left" at all |
| `PSY_AR_PLACE_RIGHT_019` | `centroid_normalized[0]` | cx > 0.60 | `invented_operational_standin` | Source gives no numeric boundary for "right" at all |

## Real sensitivity sweep (300 real, non-test, class-balanced images; 75/class, seed=42)

Each threshold was swept across a documented range and the real trigger
rate measured — **never** to tune agreement with folder labels; the
`class` column is used only to build the balanced sample.

| Rule | Sweep value | Trigger rate |
|---|---|---|
| `PSY_AR_SIZE_SMALL_016` (≤) | 0.10 / 0.15 / **0.20** / 0.25 / 0.30 | 0% / 0% / **0%** / 0% / 0.33% |
| `PSY_AR_SIZE_HALF_014` (band) | ±5 / ±7.5 / **±10** / ±12.5 / ±15pp | 0.67% / 1.0% / **2.0%** / 3.33% / 3.33% |
| `PSY_AR_SIZE_FULL_015` (≥) | 0.80 / 0.85 / **0.90** / 0.925 / 0.95 | 93.0% / 90.0% / **86.0%** / 82.67% / 77.33% |
| `PSY_AR_PLACE_TOP_017` (cy <) | 0.30 / 0.35 / **0.40** / 0.45 | 0.33% / 1.0% / **4.67%** / 15.67% |
| `PSY_AR_PLACE_LEFT_018` (cx <) | 0.30 / 0.35 / **0.40** / 0.45 | 0.67% / 2.67% / **7.67%** / 17.67% |
| `PSY_AR_PLACE_RIGHT_019` (cx >) | 0.55 / **0.60** / 0.65 / 0.70 | 13.0% / **5.33%** / 2.67% / 1.33% |

(Current production value **bolded**.)

**Headline finding**: the one *directly-sourced* threshold
(`coverage_small` ≤ 0.20) triggers on effectively **0% of real sampled
images across the entire swept range** (0/300 at every value from 0.10
to 0.25, 1/300 only at 0.30), while the one most clearly *invented*
threshold (`coverage_full` ≥ 0.90) triggers on **86% of real images**
and stays above 77% across the entire swept range. This dataset's
drawings are almost always high-coverage relative to the whole image
frame — this both explains and precisely quantifies why the
Level-C aggregation convergence path (which needs ≥2 independent
evidence families) almost never fires on registry rules alone on this
dataset (see `docs/AGGREGATION_POLICY.md`).

Every rule's `impact_on_combined_hypotheses` column notes that this
impact is currently negligible: no two of these 6 rules share a
`target_construct`, so none can combine with each other regardless of
trigger rate; the only demonstrated Level-C convergence path on real
data uses the expressive-content model as an independent second family.

## The 4 newly-activated rules (Phase 2A, Section 7)

| Rule | Feature | Threshold | Source status |
|---|---|---|---|
| `EN_COMPILED_PLACEMENT_CENTER_029` | `centroid_normalized` | 0.40 ≤ cx,cy ≤ 0.60 | `invented_operational_standin` (matches the existing top/left/right convention exactly) |
| `EN_COMPILED_LINE_HEAVY_PRESSURE_030` | `stroke.intensity_proxy` | ≥ 0.6488 (p75) | `empirically_exploratory` |
| `EN_COMPILED_LINE_LIGHT_PRESSURE_031` | `stroke.intensity_proxy` | ≤ 0.4300 (p25) | `empirically_exploratory` |
| `EN_COMPILED_LINE_SHAKY_BROKEN_032` | `stroke.fragmentation` | ≥ 0.2573 (p75) | `empirically_exploratory` |

Quartile cut points computed from an 80-image real, non-test,
class-balanced sample (seed=11):
`stroke.intensity_proxy` p25=0.4300, p50=0.5423, p75=0.6488;
`stroke.fragmentation` p25=0.0943, p50=0.1694, p75=0.2573. Neither
source PDF discusses stroke pressure or fragmentation numerically at
all — these thresholds have no source anchor whatsoever, which is
exactly why `empirically_exploratory` (not `expert_defined` or any
"sourced" category) is the honest label: they describe where *this
dataset's* distribution happens to sit, not a validated cut point for
tension/anxiety.

Heavy and light pressure use the top/bottom quartile of the **same**
`stroke.intensity_proxy` distribution, with a gap between p25 and p75 —
this makes them **structurally** mutually exclusive (a value cannot be
≥ the 75th percentile and ≤ the 25th percentile simultaneously), not
just by convention. Verified by a dense 1001-value sweep in
`tests/test_phase2a_rule_engine_v2.py`.

## Real trigger counts for the 4 new rules (200 real, non-test images; see `docs/../artifacts/phase2a/rule_trigger_distribution.csv`)

| Rule | Trigger rate among assessable cases |
|---|---|
| `EN_COMPILED_PLACEMENT_CENTER_029` | 93.55% (29/31) |
| `EN_COMPILED_LINE_HEAVY_PRESSURE_030` | 18.09% (36/199) |
| `EN_COMPILED_LINE_LIGHT_PRESSURE_031` | 32.66% (65/199) |
| `EN_COMPILED_LINE_SHAKY_BROKEN_032` | 20.10% (40/199) |

`placement_center`'s 93.55% rate among the (few) page-frame-assessable
cases is notable: once a page-frame-assessable image exists, its
content's centroid is very often near the image center too — plausibly
because images that pass the page-frame check tend to also be
well-composed, centered scans, not because "drawing in the center" is a
common behaviour per se. **This is a trigger-frequency observation about
this implementation on this dataset, not evidence of psychological
prevalence** — see `docs/STATIC_PROXY_RULE_POLICY.md` and the explicit
task instruction: "Do not interpret trigger frequency as psychological
prevalence."

## Phase 2A.1 update: `coverage_full` redefined

`PSY_AR_SIZE_FULL_015`'s trigger condition changed from image-relative
area (`bounding_box_coverage >= 0.90`) to a margin-based "all 4 page
margins <= 0.15" test, gated on a real page reference
(`docs/PAGE_COVERAGE_DEFINITION_DECISION.md`) — its `threshold_source`
stays `invented_operational_standin` (still no numeric anchor in either
source). Real, measured effect on the same 200-image sample
(`per_class=50, seed=7`): trigger rate among assessable cases went from
**25.81% (8/31)** under the old definition to **58.06% (18/31)** under
the new one. Every other rule's real trigger rate is unchanged by this
phase.

## Reproducibility

`phase2a_threshold_sensitivity.py` and
`phase2a_rule_trigger_distribution.py` (plus `phase2a_page_frame_audit.py`,
`phase2a_feature_ground_truth.py`, and `phase2a_feature_invariance.py`)
were re-run twice independently on their full samples after every
Phase 2A.1 code change, including the final state; output was
byte-identical both times in every run (seeded sampling, no other
randomness).
