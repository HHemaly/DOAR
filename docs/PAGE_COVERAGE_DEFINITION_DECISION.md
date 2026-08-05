# Page-Coverage Definition Decision (DOAR-TRACE Phase 2A.1, Section 4)

## The conflict

Phase 2A found that `PSY_AR_SIZE_FULL_015` (`coverage_full`, ≥0.90
bounding-box coverage relative to the whole uploaded image) and a
genuinely detectable full-page margin are close to mutually exclusive:
a margin thin enough to satisfy 90%+ area coverage (≤~2.57% per side
for simple centered content) is thinner than the ~3%-of-dimension band
`page_frame.py` needs to read as a clean, uniform border. On real data,
`coverage_full` triggered on 86% of page-frame-assessable images
(Phase 2A, `docs/THRESHOLD_PROVENANCE_AND_SENSITIVITY.md`) — a rate high
enough to suggest the rule was not discriminating meaningfully between
drawings that "cover the whole page" and drawings that merely occupy
most of a photo/scan frame.

## Five candidate definitions compared

| Option | Definition | Definitional match | Measurement stability | Explainability | Page-reference requirement | Expert-review suitability |
|---|---|---|---|---|---|---|
| A. Page-polygon-relative coverage | `bbox_area / confirmed_page_area` (Section 3's `segmentation.page_relative_bounding_box_coverage`) | Correct in principle ("covers the page," not "covers the image") but still an area-percentage read of "covers," which the source text doesn't actually specify numerically | Same fragility as B on this dataset today (page polygon = whole image absent user input) | Percentage is intuitive but doesn't map cleanly to "covers the whole page" as a lay description | Explicit — requires `page_relative_features_assessable` | A single number with a suggested-threshold field; easy to review, but reviewer must still pick an arbitrary % |
| B. Image-relative bounding-box coverage (original, Phase 1.5) | `bbox_area / image_area` | Weakest — conflates "the photo/scan frame" with "the page," exactly the bug Section 2/3 found | Sensitive to how tightly a photo is cropped, unrelated to the drawing itself | Easy to compute, hard to justify to a reviewer once the page-vs-image distinction is pointed out | None — the flaw itself | Poor — the definition is already known to be wrong |
| C. Foreground pixel-density coverage | `segmentation.foreground_coverage` (actual ink/mark pixels, not bounding box) | Weak — measures how much of the page is *filled with marks*, not how far the drawing's *extent* reaches; "covers the whole page" reads as a spatial/extent claim, not a density claim (a sparse stick-figure spanning corner-to-corner still "covers" the page in the source's apparent sense) | Phase 2A's robustness testing found `foreground_coverage` was the single most transform-sensitive feature measured (13/21 real invariance fails, mostly resize-driven) | Hard to explain as "covers the whole page" to a parent — reads more like "how much was drawn," a different claim | Same as A/B | Poor — conflates two different things a reviewer would want to judge separately (extent vs. density) |
| D. Qualitative "approaches page margins" | All 4 normalized margins (`composition.margins_normalized`, recomputed relative to the confirmed page when available) are within a threshold of 0 | **Strongest** — "covers the whole page" is naturally read as "the drawing's marks reach all four edges," which margins measure directly, not as an arbitrary area percentage | More forgiving of irregular (non-rectangular) drawing shapes than an area ratio — a drawing reaching all 4 edges via an irregular outline scores the same as a filled rectangle doing so, unlike bbox-area which penalizes non-rectangular fill | Directly explainable: "the drawing's lines come close to every edge of the page" | Explicit — same requirement as A, and shares Section 3's margin/polygon machinery | A single per-side threshold, individually reviewable and adjustable per margin if ever needed |
| E. Disable until expert review | `allowed_output_level: disabled` | N/A | N/A | N/A | N/A | Maximally safe, but discards a rule Phase 2A already found real, ground-truth-validated, transform-robust underlying measurements for (composition/margins) |

## Decision: Option D, gated by a valid page reference (Section 3)

`PSY_AR_SIZE_FULL_015`'s trigger condition is redefined (via
`rule_engine_v2.redefine_coverage_full`, applied only after
`apply_page_frame_gating` — `rules.py` itself is never edited) to: **all
four normalized margins are ≤ 0.15**, computed only when
`page_reference.page_relative_features_assessable` is `True`; otherwise
the rule stays `not_assessable`, exactly as before.

**Why D over A**: both require the same Section 3 page-reference
machinery and are numerically identical for the two whole-image modes
(`auto_detected_page`/`user_confirmed_full_frame`) on this dataset
today. D was chosen because it is the more faithful reading of "covers
the whole page" (extent-to-every-edge, not an aggregate area
percentage) and is measurably more forgiving of the exact same
geometric tension that caused the original conflict: a margin-based
test does not require a single tight area ratio, so it does not
recreate the same structural incompatibility with the page-frame
border-detection band that the original ≥0.90 area threshold did.

**Why not C**: rejected for definitional mismatch (density vs. extent)
and because it was independently flagged as the single least
transform-stable feature in Phase 2A's own robustness testing —
choosing it here would knowingly build on the weakest available
measurement.

**Why not E**: rejected because a defensible automatic option (D) does
exist, backed by the same ground-truth-validated `composition.margins_normalized`
values Phase 2A already checked (the `top_left_square`/`right_square`
ground-truth cases in `docs/FEATURE_MEASUREMENT_VALIDATION.md` directly
exercise margin/centroid computation). Disabling a rule with real,
validated underlying measurements available would be a step backward
from Phase 2A's own findings, not forward.

## The 0.15 margin threshold

`invented_operational_standin` — neither source PDF gives any number
for "covers the whole page" at all (the original ≥0.90 area threshold
carried the identical honest label). 0.15 was chosen, **not by sweeping
for the value that fires most often**, but as a round, defensible
"reaches toward the edges" criterion: wide enough that a real full-page
drawing with an ordinary amount of white space near the edges can still
qualify, narrow enough to exclude drawings that only occupy the central
region. This is stated plainly as an invented threshold requiring
expert review (`artifacts/expert_review/phase2a1_threshold_review_form.csv`,
Section 7), not a validated cut point.

## Observed effect on real behavior

On the baseline full-page smoke image used throughout this phase
(`train/Angry/a11.jpg`): under the old definition, `bounding_box_coverage
= 0.8067 < 0.90` → `not_matched`. Under the new definition, all 4
margins (`[0.068, 0.020, 0.087, 0.024]`) are ≤ 0.15 → **`weak_support`**.
This is a real, intended behavior change, not an accidental side effect
— the drawing genuinely reaches close to every page edge, which is what
the source rule describes, even though it does not occupy 90% of the
photographed frame's area (some of that area is a wider white margin
than the ≥0.90 threshold would have tolerated).

A second, unplanned but correct consequence: a synthetic centered
rectangle with symmetric ~13.3% margins now satisfies **both** the
redefined `coverage_full` and `EN_COMPILED_PLACEMENT_CENTER_029`
(Phase 2A) simultaneously — 2 independent rules, 2 independent evidence
families (`size_composition` + `spatial_placement`), both mapping to
`visual_dominance_or_prominence`. This is the **first real
2-registry-rule Level-C convergence** the system has ever mechanically
produced (previously the only demonstrated Level-C path required the
expressive-content model as a second family; see
`docs/AGGREGATION_POLICY.md`). Verified and regression-tested
(`tests/test_structured_report.py::StructuredAnalysisEndToEndTests::test_two_independent_rules_do_combine_under_the_redefined_coverage_full`).

## What this does NOT do

Does not change `rules.py`/`rules_registry.json` (the historical
production engine remains untouched — `redefine_coverage_full` is a
post-processing override, applied the same way `apply_page_frame_gating`
already is). Does not change `PSY_AR_SIZE_HALF_014`/`PSY_AR_SIZE_SMALL_016`'s
definitions (no equivalent structural conflict was found for those two).
Does not select 0.15 by observing which value maximizes trigger
frequency on real data — the threshold was fixed before any real-data
trigger count was examined for this specific rule.
