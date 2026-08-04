# Rule and Feature Coverage

**Status: evidence document only.** Rule table read directly from
`resources/psychology_sources/rules_registry.json` (19 entries). Feature
table computed live, this session, from a real image
(`END_TO_END_INFERENCE_TRACE.md`'s traced case) via
`src/doar/features.py::objective_feature_row` — every value below is a real
number from that run, not illustrative.

---

## 1. The 19-rule registry, in full

Source metadata (registry header): `source_id=PSYCHOLOGIST_NOTES_AR_2026_07`,
`source_type=psychologist_supplied_notes`, `scientifically_validated: false`
(the registry self-declares this). Every rule's `scientific_support` field
is one of `not_found_for_specific_claim`,
`indirect_expression_research_only`,
`indirect_size_research_conflicting/does_not_validate_claim`, or
`placement_effects_weak_not_specific` — none of the 19 claims are asserted
as validated, by the registry's own data.

| # | rule_id | tier | observable | activation_status | ceiling | claim (English gloss) |
|---|---|---|---|---|---|---|
| 1 | PSY_AR_EYES_WIDE_001 | tier_2 | wide_eyes | DETECTOR_UNAVAILABLE | 0.20 | Wide eyes → outgoing personality |
| 2 | PSY_AR_EYES_STERN_002 | tier_2 | stern_eyes | DETECTOR_UNAVAILABLE | 0.25 | Stern eyes → feeling angry |
| 3 | PSY_AR_EYES_CLOSED_003 | tier_2 | closed_eyes | DETECTOR_UNAVAILABLE | 0.10 | Closed eyes → refusal of self-reflection/faults |
| 4 | PSY_AR_ANIMAL_TIGER_WOLF_004 | tier_2 | tiger_or_wolf | DETECTOR_UNAVAILABLE | 0.10 | Tiger/wolf → anger |
| 5 | PSY_AR_ANIMAL_FOX_005 | tier_2 | fox | DETECTOR_UNAVAILABLE | 0.05 | Fox → thinking of doing something mean |
| 6 | PSY_AR_ANIMAL_SQUIRREL_006 | tier_2 | squirrel | DETECTOR_UNAVAILABLE | 0.10 | Squirrel → need for protection/attention |
| 7 | PSY_AR_ANIMAL_LION_007 | tier_2 | lion | DETECTOR_UNAVAILABLE | 0.05 | Lion → belief in own superiority |
| 8 | PSY_AR_GEOMETRY_008 | tier_2 | repeated_geometric_shapes | DETECTOR_UNAVAILABLE | 0.15 | Repeated shapes → many goals/plans, persistence |
| 9 | PSY_AR_STARS_009 | tier_2 | stars | DETECTOR_UNAVAILABLE | 0.10 | Stars → wanting everyone's attention |
| 10 | PSY_AR_FLOWERS_CLOUDS_SUN_010 | tier_2 | flowers_clouds_sun | DETECTOR_UNAVAILABLE | 0.15 | Flowers/clouds/sun → imagined goals, positivity |
| 11 | PSY_AR_CIRCLES_011 | tier_2 | circles | DETECTOR_UNAVAILABLE | 0.10 | Circles → loneliness, need for closeness |
| 12 | PSY_AR_TRANSPORT_012 | tier_2 | vehicles | DETECTOR_UNAVAILABLE | 0.10 | Vehicles → love of travel, sociability |
| 13 | PSY_AR_HEARTS_013 | tier_2 | hearts | DETECTOR_UNAVAILABLE | 0.15 | Hearts → affectionate/social personality |
| 14 | PSY_AR_SIZE_HALF_014 | tier_1 | coverage_about_half | IMPLEMENTED_UNVALIDATED | 0.20 | ~50% page coverage → situationally outgoing/introverted |
| 15 | PSY_AR_SIZE_FULL_015 | tier_1 | coverage_full | IMPLEMENTED_UNVALIDATED | 0.15 | Full-page coverage → high self-esteem |
| 16 | PSY_AR_SIZE_SMALL_016 | tier_1 | coverage_small | IMPLEMENTED_UNVALIDATED | 0.15 | ≤20% coverage → instability or fear |
| 17 | PSY_AR_PLACE_TOP_017 | tier_1 | placement_top | IMPLEMENTED_UNVALIDATED | 0.10 | Top placement → dreaminess/fantasy |
| 18 | PSY_AR_PLACE_LEFT_018 | tier_1 | placement_left | IMPLEMENTED_UNVALIDATED | 0.10 | Left placement → introverted |
| 19 | PSY_AR_PLACE_RIGHT_019 | tier_1 | placement_right | IMPLEMENTED_UNVALIDATED | 0.10 | Right placement → outgoing |

**Executable today: 6/19 (31.6%)** — every `tier_1` row. **Structurally
blocked: 13/19 (68.4%)** — every `tier_2` row, forever, until a real
eye/animal/shape/symbol detector exists and is validated
(`detectors/schema.py::DetectorResult.validated` must be independently
confirmed against `PHASE3_DETECTOR_EVALUATION_PLAN.md`'s acceptance bar —
see `rules.py:20-26`'s own comment). Zero `tier_3` rules exist in the
current registry (that dispatch branch exists only for future
prompt-dependent HTP/DAP/KFD-style rules).

## 2. Rule → feature dependency map

| Rule group | Depends on | Evaluated via |
|---|---|---|
| `coverage_about_half`/`coverage_full`/`coverage_small` | `composition.bounding_box_coverage` | `rules.py::_tier1_status`, thresholds 0.40–0.60 / ≥0.90 / ≤0.20 |
| `placement_top`/`placement_left`/`placement_right` | `composition.placement` (derived from `composition.centroid_normalized`) | string containment check against `{vertical}_{horizontal}` |
| All 13 tier-2 rules | An object/eye/animal/shape/symbol detector (does not exist) | Always `missing_detector`, `rules.py:56-60` |

Both tier-1 dependencies (`bounding_box_coverage`, `centroid_normalized`)
come from `analysis.py::_composition`, computed from the same
foreground/background mask segmentation used everywhere else in the
pipeline — there is no separate "rule-specific" feature extraction path.

## 3. Full objective feature inventory (59 features, `features.py::objective_feature_row`)

Computed live this session on the traced image
(`END_TO_END_INFERENCE_TRACE.md` §1). All are real values, not
illustrative placeholders. **This module is only invoked during
fusion-checkpoint inference (`emotion.py`'s `doar_fusion_bundle_v1`
branch)** — a plain deep-model or classical-checkpoint `analyze-image` run
(as in the main trace) does not compute or persist these 59 features in
`evidence.json`; they exist and are correct, but are not wired into the
default per-case output today.

| Family | Count | Missing | Sample real values (this session's image) |
|---|---|---|---|
| `quality.*` | 11 | 0 | width=225, contrast=73.9, blur_score=2411.9, entropy=5.45 |
| `segmentation.*` | 10 | 0 | foreground_coverage=0.535, component_count=24, border_touch_ratio=0.186 |
| `composition.*` | 16 | 0 | centroid=(0.552, 0.549), symmetry=0.576, occupied_quadrants=4 |
| `colour.*` | 13 | 0 | dark_ratio=0.369, colour.diversity=3, monochrome_flag=0.0 |
| `stroke.*` | 6 | 0 | edge_density=0.200, fragmentation=0.044 |
| `shape.*` | 3 | **2** | `contour_proxy_count=24` (real); `enclosed_shape_count`, `repetition_score` = **NaN, `missing=True`, `method="not_evaluated_no_detector"`** |

The two `shape.*` gaps are the only features in the entire 59-feature
schema that are honestly reported as unavailable rather than computed —
`features.py:137-140`'s own comment: *"No true shape detector exists in
this release. These are marked UNAVAILABLE (missing) rather than reported
as fabricated zeros (D4)."*

## 4. What this means for the target application (see `TARGET_APPLICATION_ARCHITECTURE.md`)

A parent- or clinician-facing view must be able to render, per rule, one
of exactly four honestly-distinguished states — never collapse them:

1. **`weak_support`** — evaluated, condition met, capped low-confidence
   interpretation attached (e.g. `PSY_AR_SIZE_FULL_015` in the traced
   case).
2. **`not_matched`** — evaluated, condition not met (a real negative
   result, not an error).
3. **`missing_detector`** — never evaluable in this release; missing
   evidence is explicitly **not** treated as negative evidence
   (`rules.py:58-60`).
4. **`not_assessable_context_unknown`** — reserved for future tier-3
   prompt-dependent rules; not currently reachable (no tier-3 rules exist
   yet).

Any UI that silently omits `missing_detector` rules, or presents them the
same as `not_matched`, would misrepresent what the system actually knows —
this is the specific failure mode `RULE_AND_FEATURE_COVERAGE.md` and the
prototype are designed to prevent.
