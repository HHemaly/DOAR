# Static-Direct / Static-Proxy Rule Activation Policy (DOAR-TRACE Phase 2A, Section 7)

## The 7 required conditions

A rule was activated in `rule_engine_v2.py` only if **all** of the
following held:

1. A real, already-computed feature exists (`objective_features` or
   `page_frame`) whose definition precisely matches the rule's
   `observable`.
2. The feature's semantics genuinely match what the source text
   describes — not merely a similarly-named feature (the task's
   explicit instruction: "Do not activate a rule merely because a
   similarly named feature exists").
3. Any input/page-frame requirements are satisfied or explicitly gated
   (`not_assessable` when not met, never silently ignored or
   misreported as `not_matched`).
4. A threshold policy is recorded with an honest provenance category
   (`docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md`) — never invented
   silently.
5. Wording is proxy-safe: never claims a proxy measurement is the
   literal physical quantity.
6. Real tests exist for the evaluator (`tests/test_phase2a_rule_engine_v2.py`).
7. The rule is still labeled psychologically unvalidated
   (`validation_status: IMPLEMENTED_UNVALIDATED`, never
   `SCIENTIFICALLY_VALIDATED` — that status does not exist anywhere in
   this codebase).

## The 4 rules activated

| Rule | Observable | Feature | Why it qualifies |
|---|---|---|---|
| `EN_COMPILED_PLACEMENT_CENTER_029` | `placement_center` | `composition.centroid_normalized` | Identical feature and semantics to the 3 already-executable placement rules (top/left/right); only the band changed (center: 0.40–0.60 on both axes, matching the existing convention exactly). Page-frame-gated like the other placement rules. |
| `EN_COMPILED_LINE_HEAVY_PRESSURE_030` | `heavy_line_pressure_appearance` | `stroke.intensity_proxy`, top quartile | A real proxy feature already computed for every case; ground-truth-validated (`docs/FEATURE_MEASUREMENT_VALIDATION.md`) and robust to 79/80 transform-invariance checks (`docs/FEATURE_ROBUSTNESS_RESULTS.md`). |
| `EN_COMPILED_LINE_LIGHT_PRESSURE_031` | `light_line_pressure_appearance` | `stroke.intensity_proxy`, bottom quartile | Same feature as above, opposite quartile — structurally mutually exclusive with the heavy-pressure rule. |
| `EN_COMPILED_LINE_SHAKY_BROKEN_032` | `shaky_or_broken_lines` | `stroke.fragmentation`, top quartile | A different, genuinely distinct feature from the intensity proxy; also ground-truth-validated and robust (79/80). |

All 4 carry `allowed_output_level: individual_heuristic_only` (never
higher) and a conservative, explicitly-invented `confidence_ceiling`
of 0.10 — matching the lowest ceiling already used for the most
comparable existing rules (the 3 placement rules), rather than
inventing a new number, since no clinician-assigned ceiling exists for
any of these 4 rules.

## Rules deliberately NOT activated

| Rule | Observable | Why not |
|---|---|---|
| `EN_COMPILED_EXCESSIVE_DETAIL_040` | `excessive_detail_or_fixation` | No existing feature maps precisely to "fixation on detail" as a construct. `stroke.edge_density`/`shape.contour_proxy_count` measure edge/component density in general, not a specific fixation pattern — activating on either would risk exactly the "similarly named feature" trap the task explicitly warns against. |
| `EN_COMPILED_NEGLECT_BACKGROUND_041` | `neglect_of_background` | The source itself frames this as an absence of background elements relative to a detected main subject — this requires a figure/subject detector to distinguish "subject with no background" from "subject that IS the whole drawing," which does not exist. `segmentation.empty_space_ratio` measures unfilled canvas area in general, not specifically an omitted background around a detected subject; using it would risk a misleadingly precise claim. |
| `EN_COMPILED_LINE_ZIGZAG_033` | `zigzag_lines` | No existing feature proxies a zigzag pattern specifically — `stroke.fragmentation`/`edge_density` do not distinguish zigzag from other line irregularity. |
| `EN_COMPILED_LINE_OVER_ERASING_034` | `over_erasing_appearance` | `observability_class: process_required` — erasure frequency describes the drawing *process* over time, not visible in a single flattened final scan. No amount of static-image feature engineering can recover this. |
| `EN_COMPILED_DARK_COLORS_SAD_ISOLATION_038` | `dark_colors_sad_faces_isolation` | Compound row: the colour component (`colour.dark_ratio`) is genuinely `static_direct`, but "sad faces" and "isolation" both require detectors that do not exist — the whole row is blocked at the weakest-link level, per the existing Phase 1.5 policy. |
| Every other registry-v2 rule (28 more) | — | `observability_class` is `static_detector` (needs an object/face/shape/symbol detector — none exists), `process_required`, `longitudinal_required`, or `not_operational` — none of these can be satisfied by any static single-image feature, regardless of naming similarity. |

## Evidence-family independence (Section 8)

`heavy_line_pressure_appearance`/`light_line_pressure_appearance` share
the evidence family `line_intensity_quality` and the dependency group
`["stroke.intensity_proxy"]` — they derive from the identical
underlying feature, so they can never both fire (mutual exclusivity by
construction) and, even hypothetically, could never count as 2
independent evidence families toward a Level-C combined hypothesis.

`shaky_or_broken_lines` uses a distinct evidence family
(`line_fragmentation_quality`) and dependency group
(`["stroke.fragmentation"]`) — genuinely different feature, so it is
allowed to count as an independent family from the intensity-based
pair, per the task's explicit requirement: "line-intensity and
line-fragmentation may count as separate families only if their
measurements and dependency groups are genuinely distinct."

`EN_COMPILED_PLACEMENT_CENTER_029` keeps the existing `spatial_placement`
family (shared with the 3 historical placement rules) and the
`composition.centroid_normalized` dependency group (identical to those
3 rules) — so it is subject to the same double-counting protection they
already had (a construct needing ≥2 independent families cannot combine
`placement_center` with `placement_top`/`left`/`right`, since they share
one dependency group).

## Wording safety

The 3 stroke-proxy rules use hand-written wording overrides (not the
generic auto-generated template) matching the task's mandatory
examples exactly:

- Heavy pressure: *"The lines appear relatively dark or thick in the
  uploaded image."*
- Light pressure: *"The lines appear relatively light or thin in the
  uploaded image."*
- Shaky/broken: *"The extracted line-fragmentation proxy is elevated
  for this drawing."*

All three explicitly state they do not measure actual physical pencil
pressure or the child's intent. Verified by
`tests/test_phase2a_rule_engine_v2.py::ProxySafeWordingTests` against
both a forbidden-phrase blocklist (`"pressed the pencil"`, `"drew
anxiously"`, `"the child is"`) and the required phrasing above.

## What "newly executable" does NOT mean

Per the task's explicit constraints: this does not "scientifically
validate" any of the 4 rules' psychological interpretations, does not
select or retune any threshold to agree with the Angry/Fear/Happy/Sad
folder labels (the quartiles were computed once, from a fixed seed, on
data that never touches the `class` column except to build a balanced
sample), and does not promote `registry_v2` into the production
`rules_registry.json` — `rule_engine_v2.py` remains a strictly
additive, parallel evaluator.
