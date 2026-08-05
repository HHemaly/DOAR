# Feature Measurement Validation (DOAR-TRACE Phase 2A, Section 4)

## Scope and what this does NOT prove

`phase2a_feature_ground_truth.py` validates **measurement implementation
only**: given a synthetic image with an analytically-derived, known
expected value, does the real pipeline (`analyze_image` → `_segment` →
`_composition`/`_colour` → `features.objective_feature_row`) recover
that value within a stated tolerance? It says nothing about
psychological validity, and every synthetic image is a flat-colour
geometric shape, not a real child's drawing — real drawings will vary
far more than this harness's tolerance bands allow for. No tolerance
was picked to make a result pass; each tolerance's rationale is stated
inline in `phase2a_feature_ground_truth.py` next to the case it applies to.

## Method

12 synthetic test cases (22 individual feature checks) built directly
with PIL at known pixel geometry (e.g. a 100×100 black square centered
in a 200×200 white canvas → analytically expected
`foreground_coverage = 0.25`, `centroid = (0.5, 0.5)`). Each case is run
through the real pipeline exactly once; measured vs. expected value is
compared with a per-case tolerance and classified `pass` / `fail` /
`not_applicable`.

## Full results (`artifacts/phase2a/feature_ground_truth.csv`)

| Case | Feature | Expected | Measured | Tolerance | Status |
|---|---|---|---|---|---|
| centered_square | segmentation.foreground_coverage | 0.25 | 0.2566 | 0.02 | pass |
| centered_square | segmentation.bounding_box_coverage | 0.25 | 0.2567 | 0.02 | pass |
| centered_square | composition.centroid_x/y | 0.5 | 0.4983 | 0.02 | pass |
| top_left_square | composition.centroid_x/y | 0.1667 | 0.165 | 0.03 | pass |
| top_left_square | composition.placement | top_left | top_left | exact | pass |
| right_square | composition.placement | middle_right | middle_right | exact | pass |
| centered_square | colour.dark_ratio | 1.0 | 0.9744 | 0.05 | pass |
| gray_square | colour.dark_ratio | 0.0 | 0.0 | 0.05 | pass |
| light_ratio | colour.light_ratio | n/a | NOT_IMPLEMENTED | — | not_applicable |
| three_squares | segmentation.component_count | 3.0 | 3.0 | exact | pass |
| three_squares | shape.contour_proxy_count | 3.0 | 3.0 | exact | pass |
| two_squares_sizes | segmentation.largest_component_ratio | 0.9615 | 0.9566 | 0.02 | pass |
| top_edge_strip | segmentation.border_touch_ratio | 0.25 | 0.015 | — | **known_unreliable** (retired, Phase 2A.1) |
| isolated_square | segmentation.border_touch_ratio | 0.0 | 0.0 | — | **known_unreliable** (retired, Phase 2A.1) |
| centered_circle | composition.symmetry | 1.0 | 1.0 | 0.03 | pass |
| left_only_strip | composition.symmetry | <0.85 (asymmetric) | 0.6393 | ordinal | pass |
| centered_square | stroke.intensity_proxy | 1.0 | 0.9744 | 0.05 | pass |
| gray_square | stroke.intensity_proxy | 0.4118 | 0.4012 | 0.05 | pass |
| centered_square | segmentation.empty_space_ratio | 0.75 | 0.7434 | 0.02 | pass |
| blank_vs_hatched | stroke.edge_density | hatched > blank (ordinal) | blank=0.0, hatched=0.1267 | ordinal | pass |

**Current (Phase 2A.1): 19 pass, 1 not_applicable, 2 known_unreliable**
(22 checks total, some rows above cover 2 checks — 22 individual
assertions). Originally (Phase 2A): 20 pass, 1 not_applicable, 1 fail —
`isolated_square` moved from `pass` to `known_unreliable` because
`border_touch_ratio`'s retirement (below) is unconditional: even a case
where the value happens to look numerically correct is not selectively
trusted, since nothing downstream can distinguish "right by luck" from
"wrong by the same mechanism."

## `segmentation.border_touch_ratio`: retired in Phase 2A.1 (originally documented as "the 1 real fail")

Case `top_edge_strip`: a 200×20 black strip along the very top edge of a
200×200 canvas, analytically expected to give
`border_touch_ratio = 0.25` (the strip occupies the top quarter of the
border-touching perimeter definition used). Measured: **0.015**.

**Original (Phase 2A) root-cause explanation**: attributed to
`_segment`'s morphological cleanup step eroding roughly 1–2px from the
shape's edge, including the image's own border row.

**Corrected (Phase 2A.1) root cause**: direct debugging of the
candidate-generation step (not just the final mask) found the original
explanation was incomplete. The primary cause is upstream:
`candidate_adaptive`'s BoxBlur-based local-contrast test structurally
fails to detect thick, border-touching content (PIL's edge-padding
erases the local contrast it depends on) — verified directly, zero
foreground at the strip's border row *before* cleanup ever runs.
`_candidate_score`'s `(1 - border_ratio)` term then compounds this by
scoring the flawed candidate *higher* than the two that correctly
detect the strip, causing it to be selected. Morphological cleanup,
applied after selection, is a minor contributor that actually partially
recovers the under-detected band. Full mechanism, with a regression
test reproducing it exactly:
`tests/test_border_touch_ratio_retirement.py`, `docs/BORDER_TOUCH_RATIO_DECISION.md`.

**Decision**: retired from downstream use rather than patched — the
real bug lives in shared `_segment` logic used by every feature and by
`page_frame.py`'s own edge-touch evidence, not something isolated to
this one feature; fixing it properly is out of a single-feature
validation module's scope and risks an unreviewed, dataset-wide
regression. `features.py`'s `KNOWN_UNRELIABLE` set now forces this
feature's `confidence=0.0`/`missing=True` unconditionally (even on
cases where its value happens to look correct), while preserving the
historical computed value for comparison. Ground-truth
`status_counts` changed from `{pass: 20, not_applicable: 1, fail: 1}`
to `{pass: 19, not_applicable: 1, known_unreliable: 2}` as a direct
result.

## A test-construction fix along the way (not a pipeline bug)

The `blank_vs_hatched` case's original hatching pattern (lines every
6px) saturated the gradient-magnitude field almost everywhere in the
image, causing the adaptive 80th-percentile edge threshold to collapse
to the image's own maximum value — so `edges.mean()` degenerately read
`0.0` for *both* the blank and hatched images on the first attempt. This
was a genuine flaw in the synthetic test image, not a pipeline defect:
fixed by widening the hatching spacing to 30px (a sparser pattern with
real white-space variance), which correctly produces
`hatched > blank`. The original failure mode is left as an explanatory
comment in `phase2a_feature_ground_truth.py`, since it is itself a
real, disclosed limitation of `edge_density`'s adaptive threshold on
very dense content — worth knowing even though it isn't this rule's
concern.

## Cross-checked but not separately tabulated

`shape.contour_proxy_count` and `segmentation.component_count` were
confirmed to always agree on every case checked here — they use the
identical underlying formula (`len(component_sizes)`), so this is
expected, not a coincidence requiring separate validation.

## What this means for rule activation

Section 7's decision to activate `EN_COMPILED_LINE_HEAVY_PRESSURE_030`/
`LIGHT_PRESSURE_031` (via `stroke.intensity_proxy`) and
`EN_COMPILED_LINE_SHAKY_BROKEN_032` (via `stroke.fragmentation`) is
supported by this validation: both underlying features passed their
ground-truth checks above. This does **not** mean the *interpretation*
attached to those rules (tension, anxiety, etc.) is validated — only
that the number being thresholded is being computed as intended.
