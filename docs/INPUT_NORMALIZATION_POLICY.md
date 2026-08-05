# Input Normalization Policy (DOAR-TRACE Phase 2A.1, Section 6)

## The audit

Phase 2A's transformation-robustness testing measured how 60 features
respond to resolution/rotation/compression/brightness/contrast/margin/
crop/perspective/shadow/grayscale changes (`docs/FEATURE_ROBUSTNESS_RESULTS.md`).
This phase audited whether any of those transform categories could be
**safely normalized away before feature extraction** -- i.e., whether
some input variation is capture-pipeline noise (not evidence about the
child's drawing) versus variation that IS the evidence several rules
and features exist to measure.

| Transform category | Real fails found (Phase 2A) | Safe to normalize? |
|---|---|---|
| Resolution / resize | 13 (`foreground_coverage`), 3 (`colour.dark_ratio`), 1 (`composition.symmetry`) -- the largest single cluster | **Yes** -- image size is a capture-pipeline artifact, never evidence |
| JPEG compression | 0 (all 4 quality levels tested clean) | Already stable; no normalization needed |
| Brightness | 5 (`foreground_coverage`, `stroke.intensity_proxy`) | **No** -- line darkness/thickness is itself evidence for the 2 line-pressure rules; several colour/stroke features exist specifically to measure it |
| Contrast | 4 (`foreground_coverage`, `composition.symmetry`) | **No** -- same reasoning; contrast is not incidental |
| Shadow, margin, crop, perspective, grayscale | 0 each in Phase 2A's own measurements | No fails recorded; not touched by this phase |

**Decision**: normalize resolution only. Never touch brightness,
contrast, colour, shadow, or compression. This is not a partial
implementation of a broader plan -- it is the deliberate, complete
scope: the task's own instruction is explicit that darkness, colour
choice, line intensity, and contrast must never be silently normalized
away, and Phase 2A's real data shows resolution is the one category
that is both (a) responsible for real measurement instability and (b)
has zero legitimate claim to being evidence.

## Implementation (`canonical_input.py`)

`canonicalize_image()`: resizes the uploaded image **only if** its
longer side exceeds `CANONICAL_MAX_DIMENSION` (1600px), preserving
aspect ratio, using a high-quality (LANCZOS) resample. **Never
upscales** -- a small source image is left at native resolution rather
than fabricating detail from nothing. Verified (`tests/test_canonical_input.py::test_brightness_and_colour_are_never_altered_by_canonicalization`)
that when no resize is triggered, the canonical image's pixel values
are bit-identical to the original -- proof no other adjustment is
silently applied alongside the resize.

`compute_canonical_geometry_features()`: recomputes exactly the 4
features Phase 2A's real measurements showed were resize-sensitive
(`segmentation.foreground_coverage`, `segmentation.bounding_box_coverage`,
`composition.symmetry`, `colour.dark_ratio`) against the canonicalized
image, using the identical real formulas `analysis.py`/`features.py`
already use -- never a different, invented definition for the
canonical version.

## Which version every feature uses

| Version | Where stored | Feature count | `method` value |
|---|---|---|---|
| **Original** (unmodified upload) | `objective_features` (`Analysis.to_dict()`) | All 60 existing features, unchanged | `objective_features_v3_1` (or `retired_unreliable_v1_...`/`not_evaluated_no_detector` for the 3 marked-unreliable/unimplemented ones) |
| **Canonical** (resolution-normalized) | `canonical_features["features"]` (new, separate) | Exactly 4: `segmentation.foreground_coverage`, `segmentation.bounding_box_coverage`, `composition.symmetry`, `colour.dark_ratio` | `canonical_resized_v1` |

The two sets are **never merged**. Every rule evaluator in
`rule_engine_v2.py`/`rules.py` reads `objective_features` (the
original-image set) exclusively -- confirmed directly:
`canonical_features` never contains `stroke.intensity_proxy`/
`stroke.fragmentation` (the 2 features the line-proxy rules depend on),
so those rules are structurally incapable of reading a canonicalized
value even for an image large enough to trigger resizing
(`tests/test_phase2a1_integration.py::CanonicalEvidenceNeverConfusedWithLineProxyEvidenceTests`).

## What canonical features are for (today) and are not (yet)

They exist as a **separately-recorded, comparable robustness signal** --
useful for a future expert reviewer or a future rule wanting a
resolution-stable coverage/symmetry/dark-ratio reading, and for
detecting when a case's original-vs-canonical values diverge
meaningfully (a signal that resolution-driven noise may be affecting
that specific case). No rule currently consumes `canonical_features` --
this phase's scope was building and validating the canonical pipeline
itself, not re-pointing existing rules at it. Re-pointing a rule at the
canonical value instead of the original would be a deliberate, reviewed
decision for a future phase, not an automatic consequence of this one.

## Verification

Wired into every `analyze_image` run automatically (`Analysis.canonical_features`,
persisted in `analysis.json`). 12 dedicated tests
(`tests/test_canonical_input.py`) cover: no-resize/resize behavior,
aspect-ratio preservation, the never-upscale guarantee, determinism,
strict separation from `objective_features`, and pixel-identity when no
resize occurs. No existing test's pass/fail status changed as a result
of this section (purely additive).
