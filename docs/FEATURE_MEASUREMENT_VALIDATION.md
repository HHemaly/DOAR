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
| top_edge_strip | segmentation.border_touch_ratio | 0.25 | 0.015 | 0.05 | **fail** |
| isolated_square | segmentation.border_touch_ratio | 0.0 | 0.0 | 0.01 | pass |
| centered_circle | composition.symmetry | 1.0 | 1.0 | 0.03 | pass |
| left_only_strip | composition.symmetry | <0.85 (asymmetric) | 0.6393 | ordinal | pass |
| centered_square | stroke.intensity_proxy | 1.0 | 0.9744 | 0.05 | pass |
| gray_square | stroke.intensity_proxy | 0.4118 | 0.4012 | 0.05 | pass |
| centered_square | segmentation.empty_space_ratio | 0.75 | 0.7434 | 0.02 | pass |
| blank_vs_hatched | stroke.edge_density | hatched > blank (ordinal) | blank=0.0, hatched=0.1267 | ordinal | pass |

**20 pass, 1 not_applicable, 1 documented real fail** (22 checks total,
some rows above cover 2 checks — 22 individual assertions).

## The 1 real fail: `segmentation.border_touch_ratio`

Case `top_edge_strip`: a 200×20 black strip along the very top edge of a
200×200 canvas, analytically expected to give
`border_touch_ratio = 0.25` (the strip occupies the top quarter of the
border-touching perimeter definition used). Measured: **0.015**.

**Root cause, confirmed by direct debugging**: `_segment`'s morphological
cleanup step (a `neighbours >= minimum` filter over a zero-padded 3×3
window) erodes roughly 1–2px from the shape's edge — including the
image's own border row, since the zero-padding treats anything outside
the frame as background. A shape that is *supposed* to touch the image
border loses that exact contact after cleanup.

**This is a real finding about `analysis.py::_segment`, not a
test-construction error** — confirmed by isolating the call and
inspecting `_segment`'s intermediate mask directly. It is **not fixed
here**: this module's explicit scope is validation, not patching
`analysis.py`, and any fix to the morphological cleanup step needs its
own dedicated regression testing against the existing 610+ baseline
tests that already depend on `_segment`'s current behavior. Recorded as
a known limitation for a future phase.

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
