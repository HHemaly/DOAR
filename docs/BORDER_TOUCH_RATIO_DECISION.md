# border_touch_ratio Decision (DOAR-TRACE Phase 2A.1, Section 5)

## The corrected root cause

Phase 2A's ground-truth harness found `segmentation.border_touch_ratio`
measured ~0.015 against an analytically expected 0.25 for a synthetic
strip touching the top image edge, and attributed this to `_segment`'s
morphological cleanup pass eroding "~1-2px from shape edges." That
explanation was **incomplete**. Direct debugging (reproduced as a
regression test, `tests/test_border_touch_ratio_retirement.py::RootCauseReproductionTests`)
traced the exact mechanism:

1. **Primary cause — candidate generation**: `_segment` builds 3
   candidate foreground masks per image. `candidate_adaptive` (`gray <
   local_mean - 8`, where `local_mean` is a radius-9 box blur) fails
   structurally within ~9px of the image border whenever the
   border-touching region is thick/uniform: PIL's edge-padding during
   the blur pulls `local_mean` near the border toward the foreground's
   own value, erasing the local contrast the threshold depends on.
   Verified directly: `candidate_adaptive` has **zero** foreground at
   rows 0-1 of the test strip, before any cleanup ever runs.
2. **Compounding cause — candidate scoring**: `_candidate_score`
   includes a `+0.15 * (1 - border_ratio)` term that rewards *less*
   border-touching. Because `candidate_adaptive`'s failure happens to
   produce artificially low border touch, it scores **higher**
   (0.9966) than the two candidates that correctly detect the full
   strip (`colour_distance`/`global_grayscale`, both 0.9584) — so the
   flawed candidate gets selected.
3. **Minor, secondary contributor — morphological cleanup**: applied
   *after* selection, the cleanup pass actually *recovers* most of the
   under-detected band (row 1 goes from 0 to 198/200 foreground
   pixels) via neighborhood smoothing from the correctly-detected
   interior. It cannot recover row 0 at all, since row 0 has zero
   support in its 3x3 neighborhood either way (both row 0 and row 1
   started at `False` in the selected candidate). Cleanup is not the
   primary cause; if anything, it is fighting the earlier bug and
   partially winning.

## Why this changes the decision from Phase 2A's framing

The real bug lives in `_segment`'s shared candidate-generation and
-scoring logic — used by **every** feature (`foreground_coverage`,
`bounding_box_coverage`, centroid, all of it) and by `page_frame.py`'s
own, separately-implemented edge-touch evidence (`_content_touches_edge`/
`_edge_touch_ratio`, which operates on the same post-`_segment` mask).
It is not isolated to this one named feature. That reframes the 3
options Section 5 poses:

- **Option A (fix morphological cleanup)**: would not fix the actual
  bug — cleanup is a minor contributor, and even a perfect cleanup pass
  cannot manufacture foreground from a candidate that structurally
  never marked the border pixels in the first place.
- **Option B (compute from the pre-cleanup selected candidate)**: only
  partially helps (removes cleanup's small residual effect) but leaves
  the primary bug (flawed candidate selection) fully in place.
- **Option C (retire from downstream use)**: chosen.

Properly fixing the actual bug means changing `_candidate_score`'s
border-ratio term or `candidate_adaptive`'s edge behavior in `_segment`
itself — logic shared by every feature and by `page_frame.py`. That is
a change with dataset-wide blast radius (it could shift which candidate
gets selected for many real images, not just synthetic border-touching
ones, plausibly changing `page_frame_status` classifications broadly —
see Section 2's audit answer #3). Making that change correctly requires
its own dedicated re-validation of the full page-frame audit and every
downstream rule-gating behavior, which is out of Section 5's scope (one
named feature) and risks exactly the kind of unreviewed, broad
regression the task's "do not hide regressions by widening tolerances"
instruction warns against. It is recorded as a top remaining risk and a
candidate for a dedicated future phase, not attempted here.

## Decision: retire `segmentation.border_touch_ratio` from downstream use

**Preconditions verified before choosing retirement over a fix**:
confirmed, by direct inspection of both `rules_registry.json` and
`rules_registry_v2.json` (and by regression test
`tests/test_border_touch_ratio_retirement.py::NoDownstreamConsumerTests`),
that **no rule anywhere references this feature**. `page_frame.py` uses
its own separate, duplicate edge-touch computation, not this named
feature — so retiring the feature has zero effect on rule evaluation or
page-frame classification (confirmed empirically below).

**Implementation** (`features.py`): `segmentation.border_touch_ratio` is
added to a new `KNOWN_UNRELIABLE` set. Its value is still computed and
persisted (preserved for comparison, per the task's instruction), but
`confidence` is forced to `0.0` and `missing` to `True` — the same
signal already used for genuinely unimplemented features
(`shape.enclosed_shape_count`/`repetition_score`), so every existing
consumer of the `missing`/`confidence` fields (the Technical view's "N
missing" caption, any future per-feature judge) already treats it
correctly with no further code changes needed. `method` is set to
`retired_unreliable_v1_see_border_touch_ratio_decision`. Retirement is
**unconditional** — even a case where the measured value happens to
look numerically correct (an isolated shape with genuinely 0 border
touch) is still marked unreliable, since nothing downstream can
distinguish a case that is right by luck from one that is wrong by the
same mechanism.

## Before/after: synthetic ground-truth

| Metric | Before (Phase 2A) | After (Phase 2A.1) |
|---|---|---|
| `top_edge_strip` status | `fail` (tolerance-based, 0.25 expected vs. 0.015 measured) | `known_unreliable` (no numeric pass/fail attempted) |
| `isolated_square` status | `pass` (0.0 expected, 0.0 measured) | `known_unreliable` (retirement is unconditional) |
| Overall `status_counts` | `{pass: 20, not_applicable: 1, fail: 1}` (22 cases) | `{pass: 19, not_applicable: 1, known_unreliable: 2}` (22 cases) |

The drop from 20 to 19 `pass` is the `isolated_square` case, which is
no longer scored as a numeric pass even though its value happens to be
correct — this is the intended effect of unconditional retirement, not
a regression.

## Before/after: page-frame audit and robustness (Section 5's explicit rerun requirement)

Both rerun on the full real samples and diffed byte-for-byte against
the pre-change artifacts:

| Artifact | Before | After |
|---|---|---|
| `dataset_page_frame_audit.csv` (120 images) | 5 full_page_detected / 10 likely_full_page / 8 uncertain / 97 cropped_or_content_only (12.5% assessable) | **identical, byte-for-byte** |
| `feature_invariance_summary.json` (880 rows) | 351 pass / 508 expected_change / 21 fail | **identical, byte-for-byte** |

Confirms the retirement is a fully isolated change with zero effect on
page-frame classification or transform robustness — exactly as
predicted from the "no downstream consumer" precondition above. No
tolerance was widened anywhere to produce this result.
