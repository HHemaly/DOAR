# Feature Transformation Robustness (DOAR-TRACE Phase 2A, Section 5)

## Method

`phase2a_feature_invariance.py` runs a **fixed real, non-test sample of
4 dataset images** (one per class: `happy_h42`, `fear_f42`, `sad_1_21`,
`angry_a11`) through **20 named transforms** (3 resize scales, 4 small
rotations, 4 JPEG qualities, brightness up/down, contrast up/down, a
white margin add, a border crop, a mild perspective warp, a mild
directional shadow, grayscale), extracting **11 features** per
(image, transform) pair — 880 rows total
(`artifacts/phase2a/feature_invariance.csv`).

No randomness anywhere in this script: fixed images, deterministic
transforms. Verified reproducible by running it twice and diffing the
output byte-for-byte — identical both times.

## Tolerances (stated once, applied uniformly, never picked per row)

| Group | Features | Tolerance | Rationale |
|---|---|---|---|
| Geometry | `foreground_coverage`, `bounding_box_coverage`, `centroid_x`, `centroid_y`, `symmetry` | 8% relative | Allows for resampling/anti-aliasing noise; scaled up from the ground-truth harness's own 2%-absolute coverage tolerance to account for compounded transform noise |
| Stroke/detail | `edge_density`, `intensity_proxy`, `fragmentation`, `component_count`, `largest_component_ratio` | 35% relative | Stroke/detail features are expected to be materially more sensitive to resampling/compression than coarse geometry |
| Colour | `dark_ratio` | 10% relative | — |

Each transform is also explicitly labeled, per feature group, as
`should_theoretically_preserve` or not (e.g. geometry features are
expected to survive resizing and brightness/contrast changes, but *not*
a border crop or perspective warp, which genuinely change geometry).
Rows where the transform is not expected to preserve a feature group are
scored `expected_change`, not `pass`/`fail` — this is not a silently
looser pass criterion, it is a declared, inline expectation.

## Results

| Status | Count | % |
|---|---|---|
| `pass` | 351 | 39.9% |
| `expected_change` | 508 | 57.7% |
| `fail` | 21 | 2.4% |

## The 21 real fails

All 21 fails are within transforms/features that *were* expected to be
preserved — genuine, disclosed sensitivity, not hidden by loosening the
expectation after the fact:

| Feature | Fail count |
|---|---|
| `segmentation.foreground_coverage` | 13 |
| `colour.dark_ratio` | 3 |
| `composition.symmetry` | 3 |
| `stroke.intensity_proxy` | 1 |
| `stroke.fragmentation` | 1 |

| Transform | Fail count |
|---|---|
| `resize_2x` | 6 |
| `resize_0.5x` | 5 |
| `brightness_down_0.7x` | 5 |
| `contrast_down_0.7x` | 3 |
| `brightness_up_1.3x` | 1 |
| `contrast_up_1.3x` | 1 |

**Interpretation**: `foreground_coverage` is the single most sensitive
feature — resizing (both directions) changes how many pixels the
border-colour-distance segmentation classifies as foreground vs.
background, because resampling blends edge pixels differently at
different scales. Brightness/contrast changes shift the same
segmentation threshold on real photographic content, secondarily
affecting `dark_ratio`, `symmetry`, and (once) the stroke proxies. This
is a real, useful finding about `analysis.py::_segment`'s sensitivity to
non-geometric image variation — not a defect in the invariance harness,
and not fixed here (out of this module's scope; any fix to
`_segment`'s thresholding needs its own dedicated regression testing).

## What this means for the newly-activated rules

`EN_COMPILED_LINE_HEAVY_PRESSURE_030`/`LIGHT_PRESSURE_031` (via
`stroke.intensity_proxy`) had exactly **1** fail out of 4 images × 20
transforms = 80 measurements (98.75% within the 35% stroke tolerance);
`EN_COMPILED_LINE_SHAKY_BROKEN_032` (via `stroke.fragmentation`) also
had exactly 1 fail out of 80. Both are meaningfully more robust than the
segmentation-derived features under this transform battery — a real
argument in the 4 rules' favour, though not proof of psychological
validity, only of relative measurement stability.
